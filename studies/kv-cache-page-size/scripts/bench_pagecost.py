#!/usr/bin/env python3
"""
Synthetic physical-layout sensitivity benchmark for the studied decode kernels.

Motivated by reading the actual sglang_CW source (flashinfer_backend.py / memory_pool.py /
triton_backend.py):
  * the KV pool is TOKEN-FLAT  (memory_pool.py: (size+page_size, head_num, head_dim)),
  * the decode kv_indices list is PER-TOKEN, length = seq_lens_sum = B*ctx, and
    page_size is HARD-CODED to 1 into the FlashInfer plan (call_begin_forward),
  * so --page-size does NOT change the index length or the FlashInfer plan in this path.

The benchmark crosses contiguous, blockwise, and fully scattered index values as explicit controls.
They measure kernel sensitivity to physical layout; they are not a literal mapping from engine
``page_size`` to placement. Reports 14–15 found fresh shared prefixes contiguous at both ps1 and ps128,
and report 15 showed the historical +11–25% whole-call result was not a steady decode-kernel effect.

Report-14 controls:
  * --backend {flashinfer,triton}: the FlashInfer wrapper (reports 10/11) OR the sglang
    Triton decode kernel (decode_attention_fwd, _fwd_grouped_kernel_stage1 + _fwd_kernel_stage2)
    to bound Triton's response to controlled physical layouts.
  * --reuse {distinct,shared}: distinct = B independent KV copies (true batch, DRAM-bound);
    shared = ONE physical copy of L tokens re-read by all B seqs (shared prefix -> L2-reuse,
    off the DRAM wall).  Crosses reuse x {contig,block128,scatter} to expose the
    synthetic scatter response that reports 7/12's contiguous controls could not measure.
  * --page-size: the Triton kernel page_size CONSTEXPR {1,128}.  On a token-flat pool the
    ps>1 branch math collapses to kv_loc*stride (p_branch~=0); the VALUES are the lever.
  * Sectors/request (l1tex...op_ld.ratio) distinguishes changed transaction shape from a
    latency effect. It does not by itself establish an end-to-end page-size mechanism.

Weightless (no model): B8 x L131072 distinct is ~4.3 GB, fits 16 GB; shared L is tiny.
"""
import argparse, json, time, sys
import torch
import flashinfer

# Qwen3-VL-2B text dims (match bench_xqa.py). Overridable via --num-q-heads/--num-kv-heads
# (report 13: 16/2 = Qwen2.5-3B GQA-2 shape). BYTES_PER_TOK recomputed after override.
NUM_Q_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128
DTYPE = torch.bfloat16
SM_SCALE = 1.0 / (HEAD_DIM ** 0.5)
BYTES_PER_TOK = 2 * NUM_KV_HEADS * HEAD_DIM * 2  # K+V, bf16 = 4096 B/token (one "layer")

WARMUP = 20
ITERS = 100

TENSOR_CORES = False
BACKEND = "flashinfer"
MAX_KV_SPLITS = 16          # sglang server default: triton_attention_num_kv_splits
KV_SPLITS_MODE = "dynamic"  # dynamic = engine heuristic (get_num_kv_splits_triton); static = fill max

# --- lazy sglang Triton imports (only needed for --backend triton) -----------------
# Build-robust: the experiment build (~/latency_bench/sglang_CW) puts the kernel in
# srt/.../triton_ops/decode_attention.py and get_num_kv_splits_triton in triton_backend.py;
# a newer checkout uses kernels/ops/attention/*. Try both. Also detect whether the kernel
# takes a page_size arg (the experiment build's is PAGE-AGNOSTIC: page_size lives only in the
# allocator -> kv_indices VALUES, which is exactly the lever this bench varies).
_TRITON = None
def _import_first(paths, sym):
    import importlib
    for p in paths:
        try:
            return getattr(importlib.import_module(p), sym)
        except Exception:
            continue
    raise ImportError(f"could not import {sym} from any of {paths}")

def _triton():
    global _TRITON
    if _TRITON is None:
        import triton as _tr, inspect
        decode_fwd = _import_first(
            ["sglang.srt.layers.attention.triton_ops.decode_attention",
             "sglang.kernels.ops.attention.decode_attention"], "decode_attention_fwd")
        get_splits = _import_first(
            ["sglang.srt.layers.attention.triton_backend",
             "sglang.kernels.ops.attention.metadata"], "get_num_kv_splits_triton")
        has_ps = "page_size" in inspect.signature(decode_fwd).parameters
        _TRITON = (_tr, decode_fwd, get_splits, has_ps)
    return _TRITON


def make_indptr_lastpage(B, L, device):
    kv_indptr = (torch.arange(0, B + 1, dtype=torch.int32, device=device) * L)
    last_page = torch.ones((B,), dtype=torch.int32, device=device)  # page_size=1 (engine plan)
    return kv_indptr, last_page


def make_indices(B, L, pattern, device, gen, reuse="distinct"):
    """Per-token kv_indices of length B*L and the pool size N.
       reuse='distinct': N=B*L, pattern over all B*L tokens (each seq its own copy).
       reuse='shared'  : N=L,   pattern over the ONE L-token copy, tiled B times
                         (all B seqs point their L-index block at the SAME physical slots).
       pattern: 'contig' (fresh/coalesced), 'block128' (synthetic 128-token runs with
                blocks shuffled), 'scatter' (synthetic worst-case random permutation)."""
    unit = L if reuse == "shared" else B * L
    if pattern == "contig":
        base = torch.arange(unit, dtype=torch.int64, device=device)
    elif pattern == "block128":
        P = 128
        nblk = (unit + P - 1) // P
        perm = torch.randperm(nblk, generator=gen, device=device)
        blk = (perm.to(torch.int64) * P).repeat_interleave(P)[:unit]
        off = torch.arange(unit, dtype=torch.int64, device=device) % P
        base = (blk + off).clamp_max(unit - 1)
    elif pattern == "scatter":
        base = torch.randperm(unit, generator=gen, device=device).to(torch.int64)
    else:
        raise ValueError(pattern)
    idx = base.repeat(B) if reuse == "shared" else base   # [B*L]
    return idx.to(torch.int32), unit


# --- FlashInfer path (reports 10/11; page_size hard-coded 1 into plan, as the engine does) ---
def build_wrapper(device):
    ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    return flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD",
                                                         use_tensor_cores=TENSOR_CORES)


def plan_call(wrapper, kv_indptr, kv_indices, last_page):
    wrapper.plan(
        kv_indptr, kv_indices, last_page,
        NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM,
        1,  # page_size -- mirror sglang's hard-coded 1
        data_type=DTYPE, q_data_type=DTYPE, sm_scale=SM_SCALE,
    )


def build_flashinfer_cell(B, L, N, device):
    """Allocate the token-flat combined K+V pool + query once per cell (pattern-independent)."""
    kv_cache = torch.randn(N, 2, 1, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
    q = torch.randn(B, NUM_Q_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
    kv_indptr, last_page = make_indptr_lastpage(B, L, device)
    return dict(kv_cache=kv_cache, q=q, kv_indptr=kv_indptr, last_page=last_page)


def flashinfer_run(cell, kv_indices):
    w = build_wrapper(cell["q"].device)
    plan_call(w, cell["kv_indptr"], kv_indices, cell["last_page"])
    return lambda: w.run(cell["q"], cell["kv_cache"]), w


# --- Triton path (report 14: the sglang decode_attention_fwd, engine-faithful) -----
def build_triton_cell(B, L, N, device):
    """Allocate separate token-flat K/V pools + q/o + split buffers once per cell.
       Mirrors triton_backend.py:1758-1778 shapes/dtypes."""
    tr, _decode, get_splits, _has_ps = _triton()
    k_buf = torch.randn(N, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
    v_buf = torch.randn(N, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
    q = torch.randn(B, NUM_Q_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
    o = torch.empty(B, NUM_Q_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
    kv_indptr = torch.arange(0, B + 1, dtype=torch.int32, device=device) * L
    attn_logits = torch.empty(B, NUM_Q_HEADS, MAX_KV_SPLITS, HEAD_DIM,
                              dtype=torch.float32, device=device)
    attn_lse = torch.empty(B, NUM_Q_HEADS, MAX_KV_SPLITS, dtype=torch.float32, device=device)
    num_kv_splits = torch.empty((B,), dtype=torch.int32, device=device)
    if KV_SPLITS_MODE == "static":
        num_kv_splits.fill_(MAX_KV_SPLITS)
    else:  # engine heuristic (triton_backend.py:318-333), num_seq=B, num_group=1
        seq_lens = torch.full((B,), L, dtype=torch.int32, device=device)
        core = torch.cuda.get_device_properties(device).multi_processor_count
        SCHED = 256 if B < 256 else tr.next_power_of_2(B)
        get_splits[(1,)](num_kv_splits, seq_lens, B, 1,
                         NUM_Q_HEADS, NUM_KV_HEADS, MAX_KV_SPLITS, core, MAX_NUM_SEQ=SCHED)
    return dict(k_buf=k_buf, v_buf=v_buf, q=q, o=o, kv_indptr=kv_indptr,
                attn_logits=attn_logits, attn_lse=attn_lse, num_kv_splits=num_kv_splits,
                splits=num_kv_splits.tolist())


def triton_run(cell, kv_indices, page_size):
    _tr, decode_fwd, _get, has_ps = _triton()
    c = cell
    # kv_indices must be int64 for the sglang Triton decode kernel (matches the engine)
    kvi = kv_indices.to(torch.int64)
    kw = dict(logit_cap=0.0)
    if has_ps:                       # newer build: kernel takes the page_size constexpr
        kw["page_size"] = page_size  # experiment build ignores it (page-agnostic)
    def run():
        decode_fwd(c["q"], c["k_buf"], c["v_buf"], c["o"],
                   c["kv_indptr"], kvi, c["attn_logits"], c["attn_lse"],
                   c["num_kv_splits"], MAX_KV_SPLITS, SM_SCALE, 1.0, 1.0, **kw)
    return run, None


def build_cell(B, L, N, device):
    return build_triton_cell(B, L, N, device) if BACKEND == "triton" \
        else build_flashinfer_cell(B, L, N, device)


def cell_run(cell, kv_indices, page_size):
    """Return (run_fn, keepalive_obj). page_size ignored by the FlashInfer plan (hard-coded 1)."""
    if BACKEND == "triton":
        return triton_run(cell, kv_indices, page_size)
    return flashinfer_run(cell, kv_indices)


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
    """FlashInfer host plan() sensitivity to actual and counterfactual index lengths."""
    dev = "cuda"
    out = {}
    cells = [tuple(map(int, c.split())) for c in args.cells]
    for (B, L) in cells:
        N = B * L
        cases = [
            ("token_index_actual", N),
            ("page_divided_counterfactual", (N + 127) // 128),
        ]
        for case_label, n_entries in cases:
            key = f"plan_{case_label}_bs{B}_kv{L}"
            try:
                kv_indices = torch.arange(n_entries, dtype=torch.int32, device=dev)
                per = n_entries // B
                kv_indptr = (torch.arange(0, B + 1, dtype=torch.int32, device=dev) * per)
                kv_indptr[B] = n_entries
                last_page = torch.ones((B,), dtype=torch.int32, device=dev)
                w = build_wrapper(dev)
                host_ms = time_host(lambda: plan_call(w, kv_indptr, kv_indices, last_page))
                dev_ms = time_device(lambda: plan_call(w, kv_indptr, kv_indices, last_page))
                out[key] = dict(page_label=case_label, batch_size=B, seq_len=L,
                                n_entries=int(n_entries), plan_host_ms=host_ms, plan_dev_ms=dev_ms)
                print(f"{key}: n_entries={n_entries:>9}  host={host_ms:.4f}ms  dev={dev_ms:.4f}ms", flush=True)
                del w, kv_indices, kv_indptr, last_page
                torch.cuda.empty_cache()
            except Exception as ex:
                out[key] = dict(page_label=case_label, batch_size=B, seq_len=L,
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
    tag = f"{BACKEND}_{args.reuse}_ps{args.page_size}"
    for (B, L) in cells:
        N = L if args.reuse == "shared" else B * L
        cell = None
        try:
            cell = build_cell(B, L, N, dev)
            # Warm up (Triton JIT compile + autotune) ONCE for this shape so the first
            # pattern is not penalised by one-time compilation (reports 1/8 cold-start artifact).
            try:
                wi, _ = make_indices(B, L, "contig", dev, gen, args.reuse)
                wrun, _wk = cell_run(cell, wi, args.page_size)
                for _ in range(40):
                    wrun()
                torch.cuda.synchronize()
                del wi, wrun, _wk
            except Exception:
                pass
            for pat in patterns:
                key = f"gather_{tag}_{pat}_bs{B}_kv{L}"
                try:
                    kv_indices, _ = make_indices(B, L, pat, dev, gen, args.reuse)
                    run, keep = cell_run(cell, kv_indices, args.page_size)
                    lat = time_device(run)
                    fp_gb = N * BYTES_PER_TOK / 1e9
                    out[key] = dict(backend=BACKEND, reuse=args.reuse, page_size=args.page_size,
                                    pattern=pat, batch_size=B, seq_len=L, n_tokens=N,
                                    footprint_gb=fp_gb, latency_ms=lat,
                                    splits=cell.get("splits"))
                    print(f"{key}: N={N:>9} fp={fp_gb:.2f}GB  {pat:>8} = {lat:.4f}ms", flush=True)
                    del kv_indices, run, keep
                    torch.cuda.empty_cache()
                except Exception as ex:
                    out[key] = dict(backend=BACKEND, reuse=args.reuse, page_size=args.page_size,
                                    pattern=pat, batch_size=B, seq_len=L, error=str(ex))
                    print(f"{key}: ERROR {ex}", flush=True)
                    torch.cuda.empty_cache()
        except Exception as ex:
            print(f"cell B{B} L{L}: alloc ERROR {ex}", flush=True)
        finally:
            del cell
            torch.cuda.empty_cache()
    # Paired synthetic-layout ratios. Engine placement must be measured independently.
    for (B, L) in cells:
        base_c = out.get(f"gather_{tag}_contig_bs{B}_kv{L}", {}).get("latency_ms")
        base_b = out.get(f"gather_{tag}_block128_bs{B}_kv{L}", {}).get("latency_ms")
        for pat in patterns:
            v = out.get(f"gather_{tag}_{pat}_bs{B}_kv{L}", {}).get("latency_ms")
            if v and base_c:
                print(f"  B{B} L{L}  {pat} vs contig:   {(v/base_c-1)*100:+.2f}%", flush=True)
        sc = out.get(f"gather_{tag}_scatter_bs{B}_kv{L}", {}).get("latency_ms")
        if sc and base_b:
            print(f"  B{B} L{L}  synthetic scatter vs block128: {(sc/base_b-1)*100:+.2f}%", flush=True)
    json.dump(out, open(args.output, "w"), indent=1)
    print("wrote", args.output)


def cmd_single(args):
    """One kernel launch for ncu isolation. --pattern selects index values."""
    dev = "cuda"
    gen = torch.Generator(device=dev); gen.manual_seed(0)
    B, L, pat = args.batch_size, args.seq_len, args.pattern
    N = L if args.reuse == "shared" else B * L
    cell = build_cell(B, L, N, dev)
    kv_indices, _ = make_indices(B, L, pat, dev, gen, args.reuse)
    run, keep = cell_run(cell, kv_indices, args.page_size)
    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    torch.cuda.nvtx.range_push(f"{BACKEND}_{args.reuse}_ps{args.page_size}_{pat}_bs{B}_kv{L}")
    run()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    print(f"single {BACKEND}_{args.reuse}_ps{args.page_size}_{pat}_bs{B}_kv{L} done "
          f"(splits={cell.get('splits')})", flush=True)


def main():
    global NUM_Q_HEADS, NUM_KV_HEADS, BYTES_PER_TOK, TENSOR_CORES, BACKEND, MAX_KV_SPLITS, KV_SPLITS_MODE
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["plan", "gather"], default="gather")
    ap.add_argument("--single", action="store_true")
    ap.add_argument("--backend", choices=["flashinfer", "triton"], default="flashinfer")
    ap.add_argument("--reuse", choices=["distinct", "shared"], default="distinct")
    ap.add_argument("--page-size", type=int, default=1, help="Triton kernel page_size constexpr")
    ap.add_argument("--cells", nargs="+", default=["8 1024", "8 8192", "8 32768", "8 131072"])
    ap.add_argument("--patterns", nargs="+", default=["contig", "block128", "scatter"])
    ap.add_argument("--output", default="pagecost.json")
    ap.add_argument("--warmup", type=int, default=WARMUP)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=131072)
    ap.add_argument("--pattern", default="scatter")
    ap.add_argument("--num-q-heads", type=int, default=NUM_Q_HEADS)
    ap.add_argument("--num-kv-heads", type=int, default=NUM_KV_HEADS)
    ap.add_argument("--tensor-cores", action="store_true")
    ap.add_argument("--max-kv-splits", type=int, default=MAX_KV_SPLITS)
    ap.add_argument("--kv-splits", choices=["dynamic", "static"], default=KV_SPLITS_MODE)
    args = ap.parse_args()
    NUM_Q_HEADS = args.num_q_heads
    NUM_KV_HEADS = args.num_kv_heads
    TENSOR_CORES = args.tensor_cores
    BACKEND = args.backend
    MAX_KV_SPLITS = args.max_kv_splits
    KV_SPLITS_MODE = args.kv_splits
    BYTES_PER_TOK = 2 * NUM_KV_HEADS * HEAD_DIM * 2
    print("torch", torch.__version__, "flashinfer", flashinfer.__version__,
          "gpu", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0),
          f"heads {NUM_Q_HEADS}q/{NUM_KV_HEADS}kv backend={BACKEND} reuse={args.reuse} "
          f"ps={args.page_size} splits={KV_SPLITS_MODE}/{MAX_KV_SPLITS}", flush=True)
    if args.single:
        cmd_single(args)
    elif args.mode == "plan":
        cmd_plan(args)
    else:
        cmd_gather(args)


if __name__ == "__main__":
    main()
