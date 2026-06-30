"""
bench_xqa.py — microbench + ncu target for decode-attention kernels vs page_size.

Report 6 of the page_size study: profile the TMA-using XQA decode kernel and
compare it to the page-agnostic FlashInfer fused decode kernel, in a TRUE batch
(B independent sequences, each its own paged KV — no shared prefix).

Backends
  xqa        flashinfer.decode.trtllm_batch_decode_with_kv_cache(backend="xqa")
             -> the XQA kernel (profiler name: kernel_mha), loads KV via TMA.
  flashinfer flashinfer.BatchDecodeWithPagedKVCacheWrapper
             -> the study's "page-agnostic" fused paged decode baseline.
  (Triton's scatter-gather _fwd_kernel is profiled via the engine path,
   bench_one_batch --attention-backend triton; see profile_xqa_ncu.sh.)

Modes
  --validate            run XQA+FI on a small config, check correctness (XQA vs
                        FI reference), and probe which page_size values XQA accepts.
  --latency             latency sweep over page_size x batch x seq_len -> JSON
                        (CUDA-event timed, median of BENCH_ITERS).
  --single              one (backend,page_size,batch,seq_len): warmup then ONE
                        timed kernel launch. This is the ncu isolation target.

Model dims default to Qwen3-VL-2B text: 16 Q heads / 8 KV heads (GQA-2) / head_dim 128.
"""

import argparse
import json
import math
from pathlib import Path

import torch

# ---- model dims (Qwen3-VL-2B text) -----------------------------------------
NUM_Q_HEADS  = 16
NUM_KV_HEADS = 8
HEAD_DIM     = 128
DTYPE        = torch.bfloat16        # XQA requires bf16 query

# ---- sweep defaults --------------------------------------------------------
PAGE_SIZES  = [1, 16, 32, 64, 128]
BATCH_SIZES = [1, 8, 32, 64]
SEQ_LENS    = [1024, 4096, 16384]
WARMUP_ITERS = 20
BENCH_ITERS  = 100

SM_SCALE = 1.0 / math.sqrt(HEAD_DIM)


def _device():
    return torch.device("cuda")


def make_canonical_kv(B, L, device):
    """Canonical per-(seq,pos,kvhead,dim) K and V so every backend sees the
    SAME logical cache -> outputs are comparable for correctness."""
    K = torch.randn(B, L, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
    V = torch.randn(B, L, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
    return K, V


def make_query(B, device):
    return torch.randn(B, NUM_Q_HEADS, HEAD_DIM, dtype=DTYPE, device=device)


# ---------------------------------------------------------------------------
# FlashInfer default (BatchDecodeWithPagedKVCacheWrapper), kv_layout = NHD
# kv_cache: [total_pages, 2, page_size, num_kv_heads, head_dim]
# ---------------------------------------------------------------------------
def build_flashinfer(B, L, page_size, device, K=None, V=None, kv_mode="distinct"):
    # kv_mode: "distinct" -> B independent KV copies (true batch); each seq indexes
    #          its own ppr pages.  "shared" -> ONE physical KV copy (shared prefix);
    #          all B seqs' page indices point to the SAME ppr pages (8x reuse).
    # This single switch is the controlled variable: shared makes the working set
    # B-times smaller and re-read across the batch (cache/reuse-bound); distinct is
    # DRAM-bandwidth-bound. (No other code path differs.)
    import flashinfer
    ppr = (L + page_size - 1) // page_size
    n_phys = ppr if kv_mode == "shared" else B * ppr   # physical pages allocated

    if K is None:
        # fast path (latency/single): values irrelevant, only layout matters
        kv_cache = torch.randn(n_phys, 2, page_size, NUM_KV_HEADS, HEAD_DIM,
                               dtype=DTYPE, device=device)
    else:
        # correctness path: fill from canonical KV (slow, small configs only)
        kv_cache = torch.zeros(n_phys, 2, page_size, NUM_KV_HEADS, HEAD_DIM,
                               dtype=DTYPE, device=device)
        for b in range(B):
            for pos in range(L):
                base = 0 if kv_mode == "shared" else b * ppr   # shared: every seq writes copy 0
                pg = base + pos // page_size
                sl = pos % page_size
                kv_cache[pg, 0, sl] = (K[0] if kv_mode == "shared" else K[b])[pos]
                kv_cache[pg, 1, sl] = (V[0] if kv_mode == "shared" else V[b])[pos]

    kv_indptr = torch.arange(0, B + 1, dtype=torch.int32, device=device) * ppr
    if kv_mode == "shared":
        kv_indices = torch.arange(ppr, dtype=torch.int32, device=device).repeat(B)  # all seqs -> pages [0,ppr)
    else:
        kv_indices = torch.arange(n_phys, dtype=torch.int32, device=device)
    last = L - (ppr - 1) * page_size
    kv_last_page = torch.full((B,), last, dtype=torch.int32, device=device)

    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD")
    wrapper.plan(kv_indptr, kv_indices, kv_last_page,
                 NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, page_size,
                 data_type=DTYPE, q_data_type=DTYPE, sm_scale=SM_SCALE)

    def run(q):
        return wrapper.run(q, kv_cache)

    return run


# ---------------------------------------------------------------------------
# XQA via trtllm_batch_decode_with_kv_cache, kv_layout = HND, (k,v) tuple
# each cache: [num_pages, num_kv_heads, page_size, head_dim]
# ---------------------------------------------------------------------------
def build_xqa(B, L, page_size, device, backend="xqa", K=None, V=None, kv_mode="distinct"):
    import flashinfer
    ppr = (L + page_size - 1) // page_size
    n_phys = ppr if kv_mode == "shared" else B * ppr

    if K is None:
        k_cache = torch.randn(n_phys, NUM_KV_HEADS, page_size, HEAD_DIM,
                              dtype=DTYPE, device=device)
        v_cache = torch.randn(n_phys, NUM_KV_HEADS, page_size, HEAD_DIM,
                              dtype=DTYPE, device=device)
    else:
        k_cache = torch.zeros(n_phys, NUM_KV_HEADS, page_size, HEAD_DIM,
                              dtype=DTYPE, device=device)
        v_cache = torch.zeros(n_phys, NUM_KV_HEADS, page_size, HEAD_DIM,
                              dtype=DTYPE, device=device)
        for b in range(B):
            for pos in range(L):
                base = 0 if kv_mode == "shared" else b * ppr
                pg = base + pos // page_size
                sl = pos % page_size
                k_cache[pg, :, sl] = (K[0] if kv_mode == "shared" else K[b])[pos]
                v_cache[pg, :, sl] = (V[0] if kv_mode == "shared" else V[b])[pos]

    if kv_mode == "shared":
        block_tables = torch.arange(ppr, dtype=torch.int32, device=device).repeat(B).view(B, ppr)
    else:
        block_tables = torch.arange(n_phys, dtype=torch.int32, device=device).view(B, ppr)
    seq_lens = torch.full((B,), L, dtype=torch.int32, device=device)
    workspace = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device=device)

    def run(q):
        return flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            query=q,
            kv_cache=(k_cache, v_cache),
            workspace_buffer=workspace,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=L,
            bmm1_scale=SM_SCALE,
            bmm2_scale=1.0,
            kv_layout="HND",
            backend=backend,
            out_dtype=DTYPE,
        )

    return run


def time_run(run, q):
    for _ in range(WARMUP_ITERS):
        run(q)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(BENCH_ITERS):
        s.record()
        run(q)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


def cos(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


# ---------------------------------------------------------------------------
def cmd_validate(args):
    dev = _device()
    B, L = 4, 512
    K, V = make_canonical_kv(B, L, dev)
    q = make_query(B, dev)

    # FlashInfer reference at page_size 16
    fi = build_flashinfer(B, L, 16, dev, K=K, V=V)
    ref = fi(q)
    torch.cuda.synchronize()
    print(f"[validate] FlashInfer reference ok: out shape {tuple(ref.shape)} dtype {ref.dtype}")

    print("[validate] probing XQA-accepted page sizes (B=4, L=512):")
    accepted = []
    for ps in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        try:
            xq = build_xqa(B, L, ps, dev, backend=args.backend, K=K, V=V)
            out = xq(q)
            torch.cuda.synchronize()
            # rebuild FI ref at this page size (same canonical KV) for apples-to-apples
            fi_ps = build_flashinfer(B, L, ps, dev, K=K, V=V)
            ref_ps = fi_ps(q)
            torch.cuda.synchronize()
            c = cos(out, ref_ps)
            md = (out.float() - ref_ps.float()).abs().max().item()
            accepted.append(ps)
            print(f"    ps={ps:4d}  OK   cos(XQA,FI)={c:.5f}  max|d|={md:.4f}")
        except Exception as ex:
            msg = str(ex).splitlines()[0][:140]
            print(f"    ps={ps:4d}  REJECT  {type(ex).__name__}: {msg}")
    print(f"[validate] XQA-accepted page sizes: {accepted}")


def cmd_latency(args):
    dev = _device()
    results = {}
    for L in args.seq_lens:
        for B in args.batch_sizes:
            q = make_query(B, dev)
            for be in args.backends:
                for ps in args.page_sizes:
                    key = f"{be}_{args.kv_mode}_ps{ps}_bs{B}_kv{L}"
                    try:
                        run = build_flashinfer(B, L, ps, dev, kv_mode=args.kv_mode) if be == "flashinfer" \
                            else build_xqa(B, L, ps, dev, backend=args.backend, kv_mode=args.kv_mode)
                        lat = time_run(run, q)
                        results[key] = dict(backend=be, kv_mode=args.kv_mode, page_size=ps, batch_size=B,
                                            seq_len=L, latency_ms=lat)
                        print(f"{be:10s} {args.kv_mode:8s} ps={ps:4d} bs={B:3d} kv={L:6d} -> {lat:.4f} ms")
                        del run
                        torch.cuda.empty_cache()
                    except Exception as ex:
                        msg = str(ex).splitlines()[0][:140]
                        results[key] = dict(backend=be, kv_mode=args.kv_mode, page_size=ps, batch_size=B,
                                            seq_len=L, error=msg)
                        print(f"{be:10s} {args.kv_mode:8s} ps={ps:4d} bs={B:3d} kv={L:6d} -> ERR {msg}")
                        torch.cuda.empty_cache()
            del q
            torch.cuda.empty_cache()
    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"\nsaved {args.output}")


def cmd_single(args):
    """Warmup then exactly ONE timed launch — the kernel ncu profiles."""
    dev = _device()
    B, L, ps, be = args.batch_size, args.seq_len, args.page_size, args.backend_single
    q = make_query(B, dev)
    run = build_flashinfer(B, L, ps, dev, kv_mode=args.kv_mode) if be == "flashinfer" \
        else build_xqa(B, L, ps, dev, backend=args.backend, kv_mode=args.kv_mode)
    for _ in range(args.warmup):
        run(q)
    torch.cuda.synchronize()
    # Delimit exactly ONE attention launch with cudaProfilerStart/Stop so ncu
    # (run with --profile-from-start off) isolates this single kernel without
    # needing a kernel-name filter — robust across backends.
    import torch.cuda.profiler as cprof
    cprof.start()
    torch.cuda.nvtx.range_push(f"{be}_{args.kv_mode}_ps{ps}_bs{B}_kv{L}")
    out = run(q)
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    cprof.stop()
    print(f"[single] {be} {args.kv_mode} ps={ps} bs={B} kv={L} done; out {tuple(out.shape)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--validate", action="store_true")
    p.add_argument("--latency", action="store_true")
    p.add_argument("--single", action="store_true")
    p.add_argument("--backend", default="xqa",
                   help="flashinfer trtllm backend for the XQA path: xqa|auto|trtllm-gen")
    p.add_argument("--kv-mode", default="distinct", choices=["distinct", "shared"],
                   help="distinct = B independent KV copies (true batch); "
                        "shared = ONE KV copy re-read by all B seqs (shared prefix)")
    # latency sweep
    p.add_argument("--backends", nargs="+", default=["xqa", "flashinfer"])
    p.add_argument("--page-sizes", type=int, nargs="+", default=PAGE_SIZES)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=BATCH_SIZES)
    p.add_argument("--seq-lens", type=int, nargs="+", default=SEQ_LENS)
    p.add_argument("--output", default="xqa_latency.json")
    # single
    p.add_argument("--backend-single", default="xqa", choices=["xqa", "flashinfer"])
    p.add_argument("--page-size", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--warmup", type=int, default=WARMUP_ITERS)
    args = p.parse_args()

    print(f"torch {torch.__version__}  cuda {torch.version.cuda}  "
          f"gpu {torch.cuda.get_device_name(0)}  cap {torch.cuda.get_device_capability(0)}")
    import flashinfer
    print(f"flashinfer {flashinfer.__version__}")

    if args.validate:
        cmd_validate(args)
    elif args.latency:
        cmd_latency(args)
    elif args.single:
        cmd_single(args)
    else:
        cmd_validate(args)


if __name__ == "__main__":
    main()
