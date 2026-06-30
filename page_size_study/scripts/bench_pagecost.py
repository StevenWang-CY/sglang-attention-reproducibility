#!/usr/bin/env python3
"""
bench_pagecost.py -- decisive test of where (if anywhere) FlashInfer decode pays a
page_size=1 cost in a TRUE batch (distinct KV, no shared prefix, CUDA-graph-shaped).

Motivated by reading the actual sglang_CW source (flashinfer_backend.py / memory_pool.py):
  * the KV pool is TOKEN-FLAT  (memory_pool.py: (size+page_size, head_num, head_dim)),
  * the decode kv_indices list is PER-TOKEN, length = seq_lens_sum = B*ctx, and
    page_size is HARD-CODED to 1 into the FlashInfer plan (call_begin_forward),
  * so --page-size does NOT change the index length or the plan; it only changes the
    ALLOCATOR granularity -> the VALUES (contiguity) of kv_indices -> the kernel gather.

Prior microbenches (reports 7/8) used contiguous arange() indices for BOTH page sizes,
so they never tested the one thing that actually differs between ps1 and ps128.

This script has three modes:
  --mode plan    : time create-indices + FlashInfer plan() as a function of index LENGTH
                   (8k .. >1M entries). Bounds the per-step HOST cost the user hypothesized.
  --mode gather  : time the FlashInfer decode KERNEL (wrapper.run) at a fixed distinct
                   footprint with kv_indices values CONTIGUOUS vs BLOCKED(128) vs SCATTERED
                   (random perm = worst-case fragmented ps1). This is the real lever.
  --single       : run exactly ONE kernel launch wrapped in cudaProfilerStart/Stop + nvtx,
                   for ncu isolation (pair with: ncu --profile-from-start off ...).

Weightless (no model): the KV pool for B8 x L131072 distinct is ~4.3 GB, fits 16 GB,
so we reach the user's exact extreme cell.
"""
import argparse, json, time, sys
import torch
import flashinfer

# Qwen3-VL-2B text dims (match bench_xqa.py)
NUM_Q_HEADS = 16
NUM_KV_HEADS = 8           # GQA-2
HEAD_DIM = 128
DTYPE = torch.bfloat16
SM_SCALE = 1.0 / (HEAD_DIM ** 0.5)
BYTES_PER_TOK = 2 * NUM_KV_HEADS * HEAD_DIM * 2  # K+V, bf16 = 4096 B/token (one "layer"); pool is one buffer here

WARMUP = 20
ITERS = 100


def make_indptr_lastpage(B, L, device):
    kv_indptr = (torch.arange(0, B + 1, dtype=torch.int32, device=device) * L)
    last_page = torch.ones((B,), dtype=torch.int32, device=device)  # page_size=1
    return kv_indptr, last_page


def make_indices(B, L, pattern, device, gen):
    """Build per-token kv_indices of length B*L over a pool of N=B*L distinct tokens.
       pattern: 'contig' (fresh ps1/ps128), 'block128' (ps128 contiguous runs), 'scatter' (worst ps1)."""
    N = B * L
    if pattern == "contig":
        idx = torch.arange(N, dtype=torch.int32, device=device)
    elif pattern == "block128":
        # tokens grouped in contiguous 128-blocks but blocks placed in shuffled order
        P = 128
        nblk = (N + P - 1) // P
        perm = torch.randperm(nblk, generator=gen, device=device)
        base = (perm.to(torch.int64) * P).repeat_interleave(P)[:N]
        off = torch.arange(N, dtype=torch.int64, device=device) % P
        idx = (base + off).clamp_max(N - 1).to(torch.int32)
    elif pattern == "scatter":
        idx = torch.randperm(N, generator=gen, device=device).to(torch.int32)
    else:
        raise ValueError(pattern)
    return idx


def build_wrapper(device):
    ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    return flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD")


def plan_call(wrapper, kv_indptr, kv_indices, last_page):
    wrapper.plan(
        kv_indptr, kv_indices, last_page,
        NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM,
        1,  # page_size -- mirror sglang's hard-coded 1
        data_type=DTYPE, q_data_type=DTYPE, sm_scale=SM_SCALE,
    )


def time_device(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(ITERS):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


def time_host(fn):
    """Pure host-issue cost: perf_counter around the call WITHOUT synchronize (captures CPU work)."""
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(ITERS):
        t0 = time.perf_counter_ns(); fn(); t1 = time.perf_counter_ns()
        ts.append((t1 - t0) / 1e6)
    torch.cuda.synchronize()
    ts.sort()
    return ts[len(ts) // 2]


def cmd_plan(args):
    dev = "cuda"
    gen = torch.Generator(device=dev); gen.manual_seed(0)
    out = {}
    cells = [tuple(map(int, c.split())) for c in args.cells]
    for (B, L) in cells:
        N = B * L
        for ps_label, n_entries in [("ps1", N), ("ps128", (N + 127) // 128)]:
            key = f"plan_{ps_label}_bs{B}_kv{L}"
            try:
                # index list of the given length (values irrelevant for plan host cost)
                kv_indices = torch.arange(n_entries, dtype=torch.int32, device=dev)
                # indptr/last_page consistent with n_entries split across B reqs
                per = n_entries // B
                kv_indptr = (torch.arange(0, B + 1, dtype=torch.int32, device=dev) * per)
                kv_indptr[B] = n_entries
                last_page = torch.ones((B,), dtype=torch.int32, device=dev)
                w = build_wrapper(dev)
                host_ms = time_host(lambda: plan_call(w, kv_indptr, kv_indices, last_page))
                dev_ms = time_device(lambda: plan_call(w, kv_indptr, kv_indices, last_page))
                out[key] = dict(page_label=ps_label, batch_size=B, seq_len=L,
                                n_entries=int(n_entries), plan_host_ms=host_ms, plan_dev_ms=dev_ms)
                print(f"{key}: n_entries={n_entries:>9}  host={host_ms:.4f}ms  dev={dev_ms:.4f}ms", flush=True)
                del w, kv_indices, kv_indptr, last_page
                torch.cuda.empty_cache()
            except Exception as ex:
                out[key] = dict(page_label=ps_label, batch_size=B, seq_len=L,
                                n_entries=int(n_entries), error=str(ex))
                print(f"{key}: ERROR {ex}", flush=True)
                torch.cuda.empty_cache()
    json.dump(out, open(args.output, "w"), indent=1)
    print("wrote", args.output)


def cmd_gather(args):
    dev = "cuda"
    gen = torch.Generator(device=dev); gen.manual_seed(0)
    out = {}
    cells = [tuple(map(int, c.split())) for c in args.cells]
    patterns = args.patterns
    for (B, L) in cells:
        N = B * L
        kv_cache = None
        try:
            # token-flat pool, page_size=1: [N, 2, 1, n_kv, hd]  (combined K+V, mirrors bench_xqa)
            kv_cache = torch.randn(N, 2, 1, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=dev)
            q = torch.randn(B, NUM_Q_HEADS, HEAD_DIM, dtype=DTYPE, device=dev)
            kv_indptr, last_page = make_indptr_lastpage(B, L, dev)
            for pat in patterns:
                key = f"gather_{pat}_bs{B}_kv{L}"
                try:
                    kv_indices = make_indices(B, L, pat, dev, gen)
                    w = build_wrapper(dev)
                    plan_call(w, kv_indptr, kv_indices, last_page)
                    lat = time_device(lambda: w.run(q, kv_cache))
                    fp_gb = N * BYTES_PER_TOK / 1e9
                    out[key] = dict(pattern=pat, batch_size=B, seq_len=L, n_tokens=N,
                                    footprint_gb=fp_gb, latency_ms=lat)
                    print(f"{key}: tokens={N:>9} fp={fp_gb:.2f}GB  {pat:>8} = {lat:.4f}ms", flush=True)
                    del w, kv_indices
                    torch.cuda.empty_cache()
                except Exception as ex:
                    out[key] = dict(pattern=pat, batch_size=B, seq_len=L, error=str(ex))
                    print(f"{key}: ERROR {ex}", flush=True)
                    torch.cuda.empty_cache()
        except Exception as ex:
            print(f"cell B{B} L{L}: alloc ERROR {ex}", flush=True)
        finally:
            del kv_cache
            torch.cuda.empty_cache()
    # paired ratios vs contig
    for (B, L) in cells:
        base = out.get(f"gather_contig_bs{B}_kv{L}", {}).get("latency_ms")
        if base:
            for pat in patterns:
                v = out.get(f"gather_{pat}_bs{B}_kv{L}", {}).get("latency_ms")
                if v:
                    print(f"  B{B} L{L}  {pat} vs contig: {(v/base-1)*100:+.2f}%", flush=True)
    json.dump(out, open(args.output, "w"), indent=1)
    print("wrote", args.output)


def cmd_single(args):
    """One kernel launch for ncu isolation. --pattern selects index values."""
    dev = "cuda"
    gen = torch.Generator(device=dev); gen.manual_seed(0)
    B, L, pat = args.batch_size, args.seq_len, args.pattern
    N = B * L
    kv_cache = torch.randn(N, 2, 1, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=dev)
    q = torch.randn(B, NUM_Q_HEADS, HEAD_DIM, dtype=DTYPE, device=dev)
    kv_indptr, last_page = make_indptr_lastpage(B, L, dev)
    kv_indices = make_indices(B, L, pat, dev, gen)
    w = build_wrapper(dev)
    plan_call(w, kv_indptr, kv_indices, last_page)
    for _ in range(args.warmup):
        w.run(q, kv_cache)
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    torch.cuda.nvtx.range_push(f"gather_{pat}_bs{B}_kv{L}")
    w.run(q, kv_cache)
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    print(f"single gather_{pat}_bs{B}_kv{L} done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["plan", "gather"], default="gather")
    ap.add_argument("--single", action="store_true")
    ap.add_argument("--cells", nargs="+", default=["8 1024", "8 8192", "8 32768", "8 131072"])
    ap.add_argument("--patterns", nargs="+", default=["contig", "block128", "scatter"])
    ap.add_argument("--output", default="pagecost.json")
    ap.add_argument("--warmup", type=int, default=WARMUP)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=131072)
    ap.add_argument("--pattern", default="scatter")
    args = ap.parse_args()
    print("torch", torch.__version__, "flashinfer", flashinfer.__version__,
          "gpu", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), flush=True)
    if args.single:
        cmd_single(args)
    elif args.mode == "plan":
        cmd_plan(args)
    else:
        cmd_gather(args)


if __name__ == "__main__":
    main()
