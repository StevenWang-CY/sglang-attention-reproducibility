#!/usr/bin/env python3
"""
Analyze PyTorch profiler Chrome trace files to extract and compare kernel names.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict


def extract_cuda_kernels(trace_file):
    """Extract CUDA kernel names and durations from Chrome trace."""

    with open(trace_file, 'r') as f:
        trace_data = json.load(f)

    # Chrome trace format has events in traceEvents array
    events = trace_data.get('traceEvents', [])

    # Extract CUDA kernels
    cuda_kernels = defaultdict(lambda: {'count': 0, 'total_dur': 0.0})

    for event in events:
        name = event.get('name', '')
        cat = event.get('cat', '')
        dur = event.get('dur', 0)  # Duration in microseconds

        # Look for CUDA kernels (category contains 'kernel' or 'cuda')
        if 'kernel' in cat.lower() or 'cuda' in cat.lower():
            cuda_kernels[name]['count'] += 1
            cuda_kernels[name]['total_dur'] += dur

    return dict(cuda_kernels)


def analyze_traces(trace16_file, trace32_file):
    """Compare CUDA kernels between batch 16 and 32."""

    print("=" * 80)
    print("Analyzing PyTorch Profiler Traces")
    print("=" * 80)
    print()

    print(f"Loading batch 16 trace: {trace16_file.name}")
    kernels_16 = extract_cuda_kernels(trace16_file)

    print(f"Loading batch 32 trace: {trace32_file.name}")
    kernels_32 = extract_cuda_kernels(trace32_file)

    print()
    print(f"Batch 16: {len(kernels_16)} unique kernel types")
    print(f"Batch 32: {len(kernels_32)} unique kernel types")
    print()

    # Filter for matrix multiplication / GEMM kernels
    def is_gemm_kernel(name):
        keywords = ['gemm', 'cutlass', 'nvjet', 'wmma', 'mma', 'tensorop',
                   'sgemm', 'hgemm', 'splitk', 'matmul']
        return any(kw in name.lower() for kw in keywords)

    gemm_16 = {k: v for k, v in kernels_16.items() if is_gemm_kernel(k)}
    gemm_32 = {k: v for k, v in kernels_32.items() if is_gemm_kernel(k)}

    print("=" * 80)
    print("GEMM / Matrix Multiplication Kernels")
    print("=" * 80)
    print()
    print(f"Batch 16: {len(gemm_16)} GEMM kernels")
    print(f"Batch 32: {len(gemm_32)} GEMM kernels")
    print()

    # Top kernels by time
    print("-" * 80)
    print("Top 10 GEMM Kernels by Time - Batch 16:")
    print("-" * 80)
    sorted_16 = sorted(gemm_16.items(), key=lambda x: x[1]['total_dur'], reverse=True)
    for i, (name, metrics) in enumerate(sorted_16[:10], 1):
        time_ms = metrics['total_dur'] / 1000
        print(f"{i:2}. {name}")
        print(f"     Time: {time_ms:.3f} ms, Calls: {metrics['count']}")
    print()

    print("-" * 80)
    print("Top 10 GEMM Kernels by Time - Batch 32:")
    print("-" * 80)
    sorted_32 = sorted(gemm_32.items(), key=lambda x: x[1]['total_dur'], reverse=True)
    for i, (name, metrics) in enumerate(sorted_32[:10], 1):
        time_ms = metrics['total_dur'] / 1000
        print(f"{i:2}. {name}")
        print(f"     Time: {time_ms:.3f} ms, Calls: {metrics['count']}")
    print()

    # Comparison
    names_16 = set(gemm_16.keys())
    names_32 = set(gemm_32.keys())

    only_16 = names_16 - names_32
    only_32 = names_32 - names_16
    common = names_16 & names_32

    print("=" * 80)
    print("KERNEL COMPARISON")
    print("=" * 80)
    print()
    print(f"Common GEMM kernels: {len(common)}")
    print(f"Only in batch 16: {len(only_16)}")
    print(f"Only in batch 32: {len(only_32)}")
    print()

    if only_16:
        print("-" * 80)
        print(f"GEMM Kernels ONLY in Batch 16: ({len(only_16)} kernels)")
        print("-" * 80)
        for name in sorted(only_16):
            metrics = gemm_16[name]
            time_ms = metrics['total_dur'] / 1000
            print(f"  {name}")
            print(f"    Time: {time_ms:.3f} ms, Calls: {metrics['count']}")
        print()

    if only_32:
        print("-" * 80)
        print(f"GEMM Kernels ONLY in Batch 32: ({len(only_32)} kernels)")
        print("-" * 80)
        for name in sorted(only_32):
            metrics = gemm_32[name]
            time_ms = metrics['total_dur'] / 1000
            print(f"  {name}")
            print(f"    Time: {time_ms:.3f} ms, Calls: {metrics['count']}")
        print()

    # Total time comparison
    total_16 = sum(m['total_dur'] for m in gemm_16.values()) / 1000
    total_32 = sum(m['total_dur'] for m in gemm_32.values()) / 1000

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total GEMM time (batch 16): {total_16:.3f} ms")
    print(f"Total GEMM time (batch 32): {total_32:.3f} ms")
    if total_16 > 0:
        change_pct = ((total_32 - total_16) / total_16) * 100
        print(f"Change: {change_pct:+.1f}%")
    print()

    # If we didn't find GEMM kernels, show all kernels
    if len(gemm_16) == 0 and len(gemm_32) == 0:
        print("WARNING: No GEMM kernels found with standard keywords!")
        print("Showing ALL kernels instead:")
        print()

        print("Batch 16 - All kernels:")
        all_16 = sorted(kernels_16.items(), key=lambda x: x[1]['total_dur'], reverse=True)
        for i, (name, metrics) in enumerate(all_16[:20], 1):
            time_ms = metrics['total_dur'] / 1000
            print(f"{i:2}. {name} - {time_ms:.3f} ms ({metrics['count']} calls)")
        print()

        print("Batch 32 - All kernels:")
        all_32 = sorted(kernels_32.items(), key=lambda x: x[1]['total_dur'], reverse=True)
        for i, (name, metrics) in enumerate(all_32[:20], 1):
            time_ms = metrics['total_dur'] / 1000
            print(f"{i:2}. {name} - {time_ms:.3f} ms ({metrics['count']} calls)")

    print("=" * 80)


def main():
    results_dir = Path('/vast/projects/liuv/pennnetworks/jiaheng/sglang_log/profile_results')

    trace16_file = results_dir / 'pytorch_trace_batch16.json'
    trace32_file = results_dir / 'pytorch_trace_batch32.json'

    if not trace16_file.exists():
        print(f"Error: {trace16_file} not found")
        sys.exit(1)

    if not trace32_file.exists():
        print(f"Error: {trace32_file} not found")
        sys.exit(1)

    analyze_traces(trace16_file, trace32_file)


if __name__ == '__main__':
    main()
