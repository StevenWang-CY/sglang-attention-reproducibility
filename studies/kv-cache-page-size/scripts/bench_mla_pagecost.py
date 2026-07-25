#!/usr/bin/env python3
"""
bench_mla_pagecost.py -- the MLA caveat test for report 10.

MLA (DeepSeek-V2/V3) stores ONE compressed latent per token: head_dim_ckv=512 (+kpe=64),
= 1152 B/token (bf16). That latent is reused across ALL query heads inside the kernel
(intra-step reuse), so MLA decode is HIGH arithmetic intensity -> can be off the DRAM wall
even with distinct KV / no shared prefix. That is the regime where (unlike MHA, report 10)
the page-access pattern is NOT hidden behind the bandwidth wall -- so scatter could matter.

This times the FlashInfer MLA decode kernel (BatchMLAPagedAttentionWrapper) with kv_indices
values CONTIGUOUS vs BLOCK64 vs SCATTERED (worst-case fragmented ps1), across a footprint
ladder, for two head counts:
  V2-Lite : num_heads=16   (memory-bound)
  V2      : num_heads=128  (near/at compute roofline)
--single runs one launch for ncu isolation.

Weightless: B8 x L131072 MLA-KV (bf16) ~1.2 GB, fits 16 GB.
"""
import argparse, json, time
import torch
import flashinfer
import flashinfer.mla as mla

HEAD_DIM_CKV = 512
HEAD_DIM_KPE = 64
DTYPE = torch.bfloat16
BYTES_PER_TOK = (HEAD_DIM_CKV + HEAD_DIM_KPE) * 2  # bf16 latent = 1152 B/token
SM_SCALE = 1.0 / ((HEAD_DIM_CKV + HEAD_DIM_KPE) ** 0.5)
WARMUP = 20
ITERS = 100


def make_indices(B, L, pattern, device, gen):
    N = B * L
    if pattern == "contig":
        return torch.arange(N, dtype=torch.int32, device=device)
    if pattern == "block64":
        P = 64
        nblk = (N + P - 1) // P
        perm = torch.randperm(nblk, generator=gen, device=device)
        base = (perm.to(torch.int64) * P).repeat_interleave(P)[:N]
        off = torch.arange(N, dtype=torch.int64, device=device) % P
        return (base + off).clamp_max(N - 1).to(torch.int32)
    if pattern == "scatter":
        return torch.randperm(N, generator=gen, device=device).to(torch.int32)
    raise ValueError(pattern)


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


def build_and_run(B, L, H, pattern, device, gen, page_size=1):
    N = B * L
    ckv = torch.randn(N, page_size, HEAD_DIM_CKV, dtype=DTYPE, device=device)
    kpe = torch.randn(N, page_size, HEAD_DIM_KPE, dtype=DTYPE, device=device)
    q_nope = torch.randn(B, H, HEAD_DIM_CKV, dtype=DTYPE, device=device)
    q_pe = torch.randn(B, H, HEAD_DIM_KPE, dtype=DTYPE, device=device)
    qo_indptr = torch.arange(0, B + 1, dtype=torch.int32, device=device)          # 1 q token/req
    pages_per = (L + page_size - 1) // page_size
    kv_indptr = torch.arange(0, B + 1, dtype=torch.int32, device=device) * pages_per
    kv_indices = make_indices(B, pages_per, pattern, device, gen)                 # length B*pages_per
    kv_len = torch.full((B,), L, dtype=torch.int32, device=device)
    ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    w = mla.BatchMLAPagedAttentionWrapper(ws, backend="auto")
    w.plan(qo_indptr, kv_indptr, kv_indices, kv_len, H,
           HEAD_DIM_CKV, HEAD_DIM_KPE, page_size, False, SM_SCALE, DTYPE, DTYPE)
    run = lambda: w.run(q_nope, q_pe, ckv, kpe)
    return run, (ckv, kpe, q_nope, q_pe, kv_indices, ws, w)


def cmd_gather(args):
    dev = "cuda"
    gen = torch.Generator(device=dev); gen.manual_seed(0)
    out = {}
    cells = [tuple(map(int, c.split())) for c in args.cells]
    for H in args.num_heads:
        for (B, L) in cells:
            N = B * L
            for pat in args.patterns:
                key = f"mla_h{H}_{pat}_bs{B}_kv{L}"
                try:
                    run, keep = build_and_run(B, L, H, pat, dev, gen)
                    lat = time_device(run)
                    out[key] = dict(num_heads=H, pattern=pat, batch_size=B, seq_len=L,
                                    n_tokens=N, footprint_gb=N * BYTES_PER_TOK / 1e9, latency_ms=lat)
                    print(f"{key}: tok={N:>9} fp={N*BYTES_PER_TOK/1e9:.2f}GB {pat:>8} = {lat:.4f}ms", flush=True)
                    del run, keep
                    torch.cuda.empty_cache()
                except Exception as ex:
                    out[key] = dict(num_heads=H, pattern=pat, batch_size=B, seq_len=L, error=str(ex))
                    print(f"{key}: ERROR {ex}", flush=True)
                    torch.cuda.empty_cache()
    for H in args.num_heads:
        for (B, L) in cells:
            base = out.get(f"mla_h{H}_contig_bs{B}_kv{L}", {}).get("latency_ms")
            if base:
                for pat in args.patterns:
                    v = out.get(f"mla_h{H}_{pat}_bs{B}_kv{L}", {}).get("latency_ms")
                    if v:
                        print(f"  h{H} B{B} L{L} {pat} vs contig: {(v/base-1)*100:+.2f}%", flush=True)
    json.dump(out, open(args.output, "w"), indent=1)
    print("wrote", args.output)


def cmd_single(args):
    dev = "cuda"
    gen = torch.Generator(device=dev); gen.manual_seed(0)
    run, keep = build_and_run(args.batch_size, args.seq_len, args.num_heads_single, args.pattern, dev, gen)
    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    torch.cuda.nvtx.range_push(f"mla_h{args.num_heads_single}_{args.pattern}_bs{args.batch_size}_kv{args.seq_len}")
    run()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    print("single done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", action="store_true")
    ap.add_argument("--cells", nargs="+", default=["8 1024", "8 8192", "8 32768", "8 131072", "8 262144", "1 131072"])
    ap.add_argument("--patterns", nargs="+", default=["contig", "block64", "scatter"])
    ap.add_argument("--num-heads", nargs="+", type=int, default=[16, 128])
    ap.add_argument("--output", default="mla_pagecost.json")
    ap.add_argument("--warmup", type=int, default=WARMUP)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=131072)
    ap.add_argument("--num-heads-single", type=int, default=128)
    ap.add_argument("--pattern", default="scatter")
    args = ap.parse_args()
    print("torch", torch.__version__, "flashinfer", flashinfer.__version__,
          "gpu", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), flush=True)
    if args.single:
        cmd_single(args)
    else:
        cmd_gather(args)


if __name__ == "__main__":
    main()
