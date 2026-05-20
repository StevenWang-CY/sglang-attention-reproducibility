#!/usr/bin/env python3
"""Benchmark contiguous vs scattered GPU memory access patterns.

Tests strided and random access to show bandwidth degradation
when memory access is not contiguous.

Usage:
    python benchmark_memory_access.py
"""

import torch
import time
import triton
import triton.language as tl


@triton.jit
def read_strided_kernel(data_ptr, output_ptr, N: tl.constexpr, stride: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mem_offsets = offsets.to(tl.int64) * stride
    mask = offsets < N
    x = tl.load(data_ptr + mem_offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(output_ptr + pid, tl.sum(x))


@triton.jit
def read_random_kernel(data_ptr, indices_ptr, output_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    idx_offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx_offsets < N
    indices = tl.load(indices_ptr + idx_offsets, mask=mask, other=0).to(tl.int64)
    x = tl.load(data_ptr + indices, mask=mask, other=0.0).to(tl.float32)
    tl.store(output_ptr + pid, tl.sum(x))


def bench(fn, grid, kwargs, num_iters=20, warmup=5):
    for _ in range(warmup):
        fn[grid](**kwargs)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iters):
        fn[grid](**kwargs)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / num_iters * 1000


def main():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    pool_gb = 80.0
    pool_elements = int(pool_gb * 1e9 / 2)
    print(f"\nAllocating {pool_gb:.0f} GB pool...")
    pool = torch.randn(pool_elements, dtype=torch.bfloat16, device='cuda')

    BLOCK = 1024
    # Fixed number of useful elements to read: 64M elements = 128 MB
    N = 64 * 1024 * 1024

    print(f"Reading {N * 2 / 1e6:.0f} MB useful data per test")
    print()

    print("=" * 85)
    print(f"{'Pattern':<55} {'Time(ms)':<10} {'Eff BW':<12} {'Slowdown':<8}")
    print("=" * 85)

    # Baseline: contiguous
    gs = (N + BLOCK - 1) // BLOCK
    out = torch.empty(gs, dtype=torch.float32, device='cuda')
    t_base = bench(read_strided_kernel, (gs,), {
        'data_ptr': pool, 'output_ptr': out, 'N': N, 'stride': 1, 'BLOCK': BLOCK
    })
    bw_base = N * 2 / (t_base / 1000) / 1e9
    print(f"{'Contiguous (stride=1)':<55} {t_base:<10.3f} {bw_base:<12.1f} {'1.00x':<8}")

    # Strided access
    for stride in [2, 4, 8, 16, 32, 64, 128, 256, 512]:
        max_offset = N * stride
        if max_offset >= pool_elements:
            # Reduce N to fit
            n = pool_elements // stride - BLOCK
            if n < BLOCK:
                break
        else:
            n = N
        gs = (n + BLOCK - 1) // BLOCK
        out = torch.empty(gs, dtype=torch.float32, device='cuda')
        t = bench(read_strided_kernel, (gs,), {
            'data_ptr': pool, 'output_ptr': out, 'N': n, 'stride': stride, 'BLOCK': BLOCK
        })
        bw = n * 2 / (t / 1000) / 1e9
        # Normalize: time per same amount of useful data
        norm_t = t * (N / n)  # scale to same N
        ratio = norm_t / t_base
        footprint = n * stride * 2 / 1e9
        label = f"Stride={stride:<4} (footprint={footprint:.1f}GB, {stride*2}B apart)"
        print(f"{label:<55} {t:<10.3f} {bw:<12.1f} {ratio:<8.2f}x")

    print()

    # Random access at different granularities
    for n_elems, desc in [(N, "element-level"), (N // 64, "64-elem chunks")]:
        if desc == "64-elem chunks":
            # For chunk-level random: pick random chunk starts, read 64 contiguous from each
            n_chunks = n_elems
            chunk_size = 64
            total_read = n_chunks * chunk_size
            indices = torch.randint(0, pool_elements - chunk_size, (total_read,),
                                     dtype=torch.int64, device='cuda')
            # Create indices: for each chunk, sequential within chunk
            base_indices = torch.randint(0, pool_elements - chunk_size, (n_chunks,),
                                          dtype=torch.int64, device='cuda')
            indices = (base_indices.unsqueeze(1) + torch.arange(chunk_size, device='cuda')).reshape(-1)
            n = len(indices)
        else:
            # Pure random element access
            n = n_elems
            indices = torch.randint(0, pool_elements, (n,), dtype=torch.int64, device='cuda')

        gs = (n + BLOCK - 1) // BLOCK
        out = torch.empty(gs, dtype=torch.float32, device='cuda')
        t = bench(read_random_kernel, (gs,), {
            'data_ptr': pool, 'indices_ptr': indices, 'output_ptr': out, 'N': n, 'BLOCK': BLOCK
        })
        bw = n * 2 / (t / 1000) / 1e9
        norm_t = t * (N / n)
        ratio = norm_t / t_base
        label = f"Random {desc} ({n * 2 / 1e6:.0f}MB, across 80GB)"
        print(f"{label:<55} {t:<10.3f} {bw:<12.1f} {ratio:<8.2f}x")

    print("=" * 85)
    del pool
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
