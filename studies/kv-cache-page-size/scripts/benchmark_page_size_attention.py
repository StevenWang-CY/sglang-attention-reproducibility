"""
Microbenchmark: FlashInfer paged decode attention vs page size.

Measures the latency of one decode attention step (single new token attending
over a paged KV cache) for varying page sizes, batch sizes, and KV lengths.
Matches Qwen3-VL-8B-Instruct model dimensions.
"""

import argparse
import json
from pathlib import Path

import torch
import flashinfer

# Approximate Qwen3-VL-8B-Instruct attention dimensions.
# Actual model has 28Q/4KV (group_size=7) which FlashInfer doesn't support.
# Using 32Q/8KV (group_size=4) as the closest supported config.
NUM_Q_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
DTYPE = torch.float16

PAGE_SIZES   = [1, 2, 4, 8, 16, 32, 64, 128]
BATCH_SIZES  = [1, 8, 32, 64]
KV_LENGTHS   = [1024, 4096, 16384]
WARMUP_ITERS = 20
BENCH_ITERS  = 100


def bench_one(page_size: int, batch_size: int, kv_len: int, device: torch.device):
    """Return median latency in ms for one decode attention step."""
    pages_per_seq = (kv_len + page_size - 1) // page_size
    total_pages   = batch_size * pages_per_seq + 1  # +1 for workspace

    # KV cache: [total_pages, 2, page_size, num_kv_heads, head_dim]
    kv_cache = torch.randn(
        total_pages, 2, page_size, NUM_KV_HEADS, HEAD_DIM,
        dtype=DTYPE, device=device,
    )

    # Page table: each request gets its own contiguous pages
    kv_indptr    = torch.arange(0, batch_size + 1, dtype=torch.int32, device=device) * pages_per_seq
    kv_indices   = torch.arange(batch_size * pages_per_seq, dtype=torch.int32, device=device)
    kv_last_page = torch.full((batch_size,), kv_len - (pages_per_seq - 1) * page_size,
                              dtype=torch.int32, device=device)

    # Query: one new token per request
    q = torch.randn(batch_size, NUM_Q_HEADS, HEAD_DIM, dtype=DTYPE, device=device)

    workspace_buf = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace_buf, "NHD")
    wrapper.plan(
        kv_indptr,
        kv_indices,
        kv_last_page,
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        page_size,
        data_type=DTYPE,
    )

    # Warmup
    for _ in range(WARMUP_ITERS):
        wrapper.run(q, kv_cache)
    torch.cuda.synchronize()

    # Benchmark with CUDA events
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(BENCH_ITERS):
        start.record()
        wrapper.run(q, kv_cache)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    median = times[len(times) // 2]
    return median


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="page_size_attn_bench.json")
    parser.add_argument("--page-sizes",  type=int, nargs="+", default=PAGE_SIZES)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=BATCH_SIZES)
    parser.add_argument("--kv-lengths",  type=int, nargs="+", default=KV_LENGTHS)
    args = parser.parse_args()

    device = torch.device("cuda")
    results = {}

    total = len(args.page_sizes) * len(args.batch_sizes) * len(args.kv_lengths)
    done = 0
    for kv_len in args.kv_lengths:
        for batch_size in args.batch_sizes:
            for page_size in args.page_sizes:
                done += 1
                key = f"ps{page_size}_bs{batch_size}_kv{kv_len}"
                try:
                    lat = bench_one(page_size, batch_size, kv_len, device)
                    results[key] = {
                        "page_size": page_size,
                        "batch_size": batch_size,
                        "kv_len": kv_len,
                        "latency_ms": lat,
                    }
                    print(f"[{done:3d}/{total}] page_size={page_size:3d}  bs={batch_size:3d}  kv={kv_len:6d}  -> {lat:.3f} ms")
                except Exception as e:
                    print(f"[{done:3d}/{total}] page_size={page_size:3d}  bs={batch_size:3d}  kv={kv_len:6d}  -> ERROR: {e}")
                    results[key] = {"page_size": page_size, "batch_size": batch_size, "kv_len": kv_len, "error": str(e)}

    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {args.output}")

    # Print summary table per kv_len
    for kv_len in args.kv_lengths:
        print(f"\n=== kv_len={kv_len} — latency (ms) ===")
        header = f"{'page_size':>10}" + "".join(f"  bs={bs:>3}" for bs in args.batch_sizes)
        print(header)
        print("-" * len(header))
        for ps in args.page_sizes:
            row = f"{ps:>10}"
            for bs in args.batch_sizes:
                key = f"ps{ps}_bs{bs}_kv{kv_len}"
                v = results.get(key, {})
                lat = v.get("latency_ms")
                row += f"  {lat:>7.3f}" if lat is not None else f"  {'ERR':>7}"
            print(row)

    # Plot
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        for kv_len in args.kv_lengths:
            fig, ax = plt.subplots(figsize=(10, 5))
            x = np.arange(len(args.page_sizes))
            width = 0.8 / len(args.batch_sizes)
            for i, bs in enumerate(args.batch_sizes):
                lats = []
                for ps in args.page_sizes:
                    v = results.get(f"ps{ps}_bs{bs}_kv{kv_len}", {})
                    lats.append(v.get("latency_ms", float("nan")))
                offset = (i - len(args.batch_sizes) / 2 + 0.5) * width
                bars = ax.bar(x + offset, lats, width=width * 0.9, label=f"bs={bs}")

            ax.set_xticks(x)
            ax.set_xticklabels([str(ps) for ps in args.page_sizes])
            ax.set_xlabel("Page size")
            ax.set_ylabel("Attention latency (ms)")
            ax.set_title(f"FlashInfer Decode Attention Latency vs Page Size (kv_len={kv_len})")
            ax.legend(title="batch size")
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            plot_path = Path(args.output).with_name(f"page_size_attn_bench_kv{kv_len}.png")
            fig.savefig(plot_path, dpi=150)
            print(f"Plot saved: {plot_path}")
            plt.close(fig)
    except ImportError:
        print("matplotlib not available, skipping plots")


if __name__ == "__main__":
    main()
