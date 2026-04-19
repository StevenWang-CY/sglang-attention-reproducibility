#!/usr/bin/env python3
"""
Analyze ncu profiling results from profile_all_kernels.sh

Extracts and compares kernel execution metrics between batch 16 and 32,
focusing on matrix multiplication (nvjet) kernels.
"""

import csv
import sys
from pathlib import Path
from collections import defaultdict


def parse_ncu_csv(csv_file):
    """Parse ncu CSV output and extract kernel metrics."""
    kernels = []

    with open(csv_file, 'r') as f:
        # Skip header lines starting with ==
        lines = f.readlines()

    # Find CSV data (skip profiler messages)
    csv_start = 0
    for i, line in enumerate(lines):
        if line.startswith('"ID"') or line.startswith('ID,'):
            csv_start = i
            break

    if csv_start == 0:
        print(f"Warning: No CSV data found in {csv_file}")
        return []

    # Parse CSV
    reader = csv.DictReader(lines[csv_start:])
    for row in reader:
        kernels.append(row)

    return kernels


def analyze_kernels(batch16_file, batch32_file):
    """Compare kernel metrics between batch 16 and 32."""

    print("=" * 80)
    print("Kernel Profile Analysis: Batch 16 vs Batch 32")
    print("=" * 80)
    print()

    # Parse profiles
    kernels_16 = parse_ncu_csv(batch16_file)
    kernels_32 = parse_ncu_csv(batch32_file)

    print(f"Batch 16: Profiled {len(kernels_16)} kernel invocations")
    print(f"Batch 32: Profiled {len(kernels_32)} kernel invocations")
    print()

    # Group by kernel name and aggregate
    def aggregate_by_kernel(kernels):
        """Group kernel invocations by name and sum metrics."""
        aggregated = defaultdict(lambda: {
            'count': 0,
            'total_time_ns': 0.0,
            'total_tensor_ops': 0.0
        })

        for kernel in kernels:
            name = kernel.get('Kernel Name', kernel.get('"Kernel Name"', 'unknown'))

            # Extract metrics
            time_str = kernel.get('gpu__time_duration.sum', kernel.get('"gpu__time_duration.sum"', '0'))
            tensor_str = kernel.get('sm__inst_executed_pipe_tensor.avg',
                                   kernel.get('"sm__inst_executed_pipe_tensor.avg"', '0'))

            # Clean and convert
            try:
                # Remove units and convert to float
                time_val = float(time_str.replace('nsecond', '').replace(',', '').strip())
                tensor_val = float(tensor_str.replace(',', '').strip()) if tensor_str else 0.0
            except (ValueError, AttributeError):
                time_val = 0.0
                tensor_val = 0.0

            aggregated[name]['count'] += 1
            aggregated[name]['total_time_ns'] += time_val
            aggregated[name]['total_tensor_ops'] += tensor_val

        return dict(aggregated)

    agg_16 = aggregate_by_kernel(kernels_16)
    agg_32 = aggregate_by_kernel(kernels_32)

    # Focus on nvjet kernels (matrix multiplication)
    print("=" * 80)
    print("NVJET Kernels (Matrix Multiplication)")
    print("=" * 80)
    print()

    nvjet_16 = {k: v for k, v in agg_16.items() if 'nvjet' in k}
    nvjet_32 = {k: v for k, v in agg_32.items() if 'nvjet' in k}

    print(f"Batch 16: {len(nvjet_16)} unique nvjet kernels")
    print(f"Batch 32: {len(nvjet_32)} unique nvjet kernels")
    print()

    # Compare top time-consuming kernels
    print("-" * 80)
    print("Top 10 Most Time-Consuming Kernels at Batch 16:")
    print("-" * 80)
    sorted_16 = sorted(nvjet_16.items(), key=lambda x: x[1]['total_time_ns'], reverse=True)
    for i, (name, metrics) in enumerate(sorted_16[:10], 1):
        time_ms = metrics['total_time_ns'] / 1e6
        print(f"{i:2}. {name}")
        print(f"     Time: {time_ms:.3f} ms (×{metrics['count']} calls)")
        print(f"     Tensor ops: {metrics['total_tensor_ops']:.0f}")
    print()

    print("-" * 80)
    print("Top 10 Most Time-Consuming Kernels at Batch 32:")
    print("-" * 80)
    sorted_32 = sorted(nvjet_32.items(), key=lambda x: x[1]['total_time_ns'], reverse=True)
    for i, (name, metrics) in enumerate(sorted_32[:10], 1):
        time_ms = metrics['total_time_ns'] / 1e6
        print(f"{i:2}. {name}")
        print(f"     Time: {time_ms:.3f} ms (×{metrics['count']} calls)")
        print(f"     Tensor ops: {metrics['total_tensor_ops']:.0f}")
    print()

    # Find kernels that appear in only one batch size
    only_16 = set(nvjet_16.keys()) - set(nvjet_32.keys())
    only_32 = set(nvjet_32.keys()) - set(nvjet_16.keys())

    if only_16:
        print("=" * 80)
        print(f"Kernels ONLY in Batch 16: {len(only_16)}")
        print("=" * 80)
        for name in sorted(only_16):
            metrics = nvjet_16[name]
            time_ms = metrics['total_time_ns'] / 1e6
            print(f"  {name}")
            print(f"    Time: {time_ms:.3f} ms, Calls: {metrics['count']}")
        print()

    if only_32:
        print("=" * 80)
        print(f"Kernels ONLY in Batch 32: {len(only_32)}")
        print("=" * 80)
        for name in sorted(only_32):
            metrics = nvjet_32[name]
            time_ms = metrics['total_time_ns'] / 1e6
            print(f"  {name}")
            print(f"    Time: {time_ms:.3f} ms, Calls: {metrics['count']}, Tensor ops: {metrics['total_tensor_ops']:.0f}")
        print()

    # Total nvjet time comparison
    total_16 = sum(m['total_time_ns'] for m in nvjet_16.values()) / 1e6
    total_32 = sum(m['total_time_ns'] for m in nvjet_32.values()) / 1e6

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total nvjet kernel time (batch 16): {total_16:.3f} ms")
    print(f"Total nvjet kernel time (batch 32): {total_32:.3f} ms")
    print(f"Change: {((total_32 - total_16) / total_16 * 100):.1f}%")
    print()

    # Tensor Core utilization
    total_tensor_16 = sum(m['total_tensor_ops'] for m in nvjet_16.values())
    total_tensor_32 = sum(m['total_tensor_ops'] for m in nvjet_32.values())

    print(f"Total Tensor Core ops (batch 16): {total_tensor_16:.0f}")
    print(f"Total Tensor Core ops (batch 32): {total_tensor_32:.0f}")
    if total_tensor_16 > 0:
        print(f"Change: {((total_tensor_32 - total_tensor_16) / total_tensor_16 * 100):.1f}%")
    print()
    print("=" * 80)


def main():
    results_dir = Path('/vast/projects/liuv/pennnetworks/jiaheng/sglang_log/profile_results')

    batch16_file = results_dir / 'all_kernels_batch16.csv'
    batch32_file = results_dir / 'all_kernels_batch32.csv'

    if not batch16_file.exists():
        print(f"Error: {batch16_file} not found")
        print("Run ./profile_all_kernels.sh first")
        sys.exit(1)

    if not batch32_file.exists():
        print(f"Error: {batch32_file} not found")
        print("Run ./profile_all_kernels.sh first")
        sys.exit(1)

    analyze_kernels(batch16_file, batch32_file)


if __name__ == '__main__':
    main()
