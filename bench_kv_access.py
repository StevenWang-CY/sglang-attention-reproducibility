"""
Microbenchmark: contiguous KV (single_prefill) vs paged KV (BatchPrefill)
with varying page sizes and KV lengths.

Goal: show that contiguous KV allows better async prefetch overlap,
while paged KV must chase the page table from global memory first.

L2 cache is explicitly flushed between every run to prevent cross-run
cache warming from distorting results.
"""

import torch
import flashinfer
import tabulate

# ── config ──────────────────────────────────────────────────────────────────
NUM_QO_HEADS   = 32
NUM_KV_HEADS   = 8
HEAD_DIM       = 128
QO_LEN         = 64         # 64 queries attending to the same KV (shared prefix scenario)
KV_LENGTHS     = [512, 1024, 2048, 4096, 8192, 16384]
PAGE_SIZES     = [1, 4, 16, 64, 128]   # 1 = maximally fragmented, 128 = near-contiguous
DTYPE          = torch.float16
NUM_WARMUP     = 50
NUM_ITERS      = 200
DEVICE         = "cuda:0"
WORKSPACE_MB   = 128

# B200 L2 cache = 192MB — flush buffer must exceed this
L2_FLUSH_SIZE_MB = 256
# ────────────────────────────────────────────────────────────────────────────

# Allocate the flush buffer once
_l2_flush_buf = torch.empty(
    L2_FLUSH_SIZE_MB * 1024 * 1024 // 4, dtype=torch.float32, device=DEVICE
)

def flush_l2():
    """Evict L2 cache by writing to a buffer larger than the L2 size."""
    _l2_flush_buf.fill_(0.0)
    torch.cuda.synchronize()


def timed(fn, num_warmup, num_iters):
    for _ in range(num_warmup):
        flush_l2()
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    flush_l2()
    start.record()
    for _ in range(num_iters):
        fn()
        flush_l2()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / num_iters   # ms


def bench_contiguous(kv_len):
    """
    single_prefill_with_kv_cache:
      - k, v are dense [kv_len, num_kv_heads, head_dim] tensors
      - next tile address = current_ptr + stride  (pure arithmetic)
      - cp.async fires immediately, overlaps with QK/PV compute
    """
    q = torch.randn(QO_LEN,  NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    k = torch.randn(kv_len,  NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    v = torch.randn(kv_len,  NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)

    return timed(
        lambda: flashinfer.single_prefill_with_kv_cache(q, k, v, causal=False),
        NUM_WARMUP, NUM_ITERS
    )


def bench_paged(kv_len, page_size):
    """
    BatchPrefillWithPagedKVCacheWrapper:
      - KV stored as pages, addressed via page table in global memory
      - next tile address = kv_ptr + page_table[idx]  (global mem load first)
      - cp.async is serialized behind the page table lookup
    """
    num_pages         = (kv_len + page_size - 1) // page_size
    last_page_len     = kv_len - (num_pages - 1) * page_size

    kv_cache = torch.randn(
        num_pages, 2, page_size, NUM_KV_HEADS, HEAD_DIM,
        dtype=DTYPE, device=DEVICE
    )
    q = torch.randn(QO_LEN, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)

    workspace       = torch.empty(WORKSPACE_MB * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    qo_indptr       = torch.tensor([0, QO_LEN],     dtype=torch.int32, device=DEVICE)
    kv_indptr       = torch.tensor([0, num_pages],  dtype=torch.int32, device=DEVICE)
    kv_indices      = torch.randperm(num_pages,       dtype=torch.int32, device=DEVICE)
    kv_last_page    = torch.tensor([last_page_len],  dtype=torch.int32, device=DEVICE)

    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD")
    wrapper.plan(
        qo_indptr, kv_indptr, kv_indices, kv_last_page,
        NUM_QO_HEADS, NUM_KV_HEADS, HEAD_DIM, page_size,
        causal=False,
    )

    return timed(
        lambda: wrapper.run(q, kv_cache),
        NUM_WARMUP, NUM_ITERS
    )


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"num_qo_heads={NUM_QO_HEADS}, num_kv_heads={NUM_KV_HEADS}, head_dim={HEAD_DIM}")
    print(f"qo_len={QO_LEN}, dtype={DTYPE}\n")

    # ── Table 1: contiguous vs paged (page_size=1, worst case) ──────────────
    print("=== Contiguous vs Paged (page_size=1, worst case) ===")
    rows = []
    for kv_len in KV_LENGTHS:
        t_cont  = bench_contiguous(kv_len)
        t_paged = bench_paged(kv_len, page_size=1)
        rows.append([kv_len, f"{t_cont:.4f}", f"{t_paged:.4f}", f"{t_paged/t_cont:.2f}x"])
    print(tabulate.tabulate(
        rows, headers=["kv_len", "contiguous (ms)", "paged ps=1 (ms)", "slowdown"],
        tablefmt="github"
    ))
    print()

    # ── Table 2: effect of page_size on a fixed kv_len ──────────────────────
    KV_LEN = 4096
    print(f"=== Page size sweep (kv_len={KV_LEN}) ===")
    t_cont = bench_contiguous(KV_LEN)
    rows = [["contiguous", "N/A", f"{t_cont:.4f}", "1.00x"]]
    for ps in PAGE_SIZES:
        t = bench_paged(KV_LEN, ps)
        rows.append([f"paged", ps, f"{t:.4f}", f"{t/t_cont:.2f}x"])
    print(tabulate.tabulate(
        rows, headers=["type", "page_size", "time (ms)", "vs contiguous"],
        tablefmt="github"
    ))


if __name__ == "__main__":
    main()
