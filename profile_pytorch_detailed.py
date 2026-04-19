#!/usr/bin/env python3
"""
Detailed PyTorch profiling to capture kernel execution info.
This works without any special permissions and gives us concrete execution data.
"""

import argparse
import json
import os
import time
from pathlib import Path
from collections import defaultdict

os.environ["TREE_SPARSE_TIMING"] = "0"

import torch
from torch.profiler import profile, ProfilerActivity, record_function

import sglang as sgl


def profile_batch_detailed(model_path, request_file, batch_size, max_tokens=10):
    """Profile with detailed kernel capture."""

    # Load request
    with open(request_file, 'r') as f:
        request_data = json.load(f)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    messages = request_data['messages']
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    prompts = [prompt] * batch_size

    # Initialize engine
    print(f"Initializing engine for batch {batch_size}...")
    engine = sgl.Engine(
        model_path=model_path,
        tp_size=1,
        attention_backend="flashinfer",
        trust_remote_code=True,
        disable_cuda_graph=True,
    )

    # Warmup
    print("Warmup...")
    _ = engine.generate(
        prompts[:1],
        sampling_params={"max_new_tokens": 5, "temperature": 0.0, "top_p": 1.0}
    )

    torch.cuda.empty_cache()

    # Profile with maximum detail
    print(f"Profiling batch {batch_size} with detailed trace...")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        with record_function(f"decode_batch_{batch_size}"):
            _ = engine.generate(
                prompts,
                sampling_params={
                    "max_new_tokens": max_tokens,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "ignore_eos": True
                }
            )

    del engine
    torch.cuda.empty_cache()

    return prof


def extract_gemm_kernels_from_trace(trace_file):
    """Extract GEMM kernel names and metrics from Chrome trace JSON."""

    print(f"Analyzing trace file: {trace_file.name}")

    with open(trace_file, 'r') as f:
        trace_data = json.load(f)

    events = trace_data.get('traceEvents', [])

    # Look for CUDA kernel events
    kernels = []

    for event in events:
        name = event.get('name', '')
        cat = event.get('cat', '')
        dur = event.get('dur', 0)  # microseconds

        # Check if this is a CUDA kernel event
        # Look in category and name
        if 'kernel' in cat.lower() or 'cuda_runtime' in cat.lower():
            # Check if it's a GEMM/matrix mult kernel
            if any(kw in name.lower() for kw in ['gemm', 'nvjet', 'cutlass', 'wmma', 'mma', 'matmul', 'cublas']):
                kernels.append({
                    'name': name,
                    'duration_us': dur,
                    'category': cat
                })

    # Aggregate by name
    kernel_stats = defaultdict(lambda: {'count': 0, 'total_time_us': 0})

    for k in kernels:
        kernel_stats[k['name']]['count'] += 1
        kernel_stats[k['name']]['total_time_us'] += k['duration_us']

    return dict(kernel_stats)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[16, 32])
    parser.add_argument('--max-tokens', type=int, default=10)
    parser.add_argument('--output-dir', type=str, default='profile_results')
    args = parser.parse_args()

    model_path = "/vast/projects/liuv/pennnetworks/hf_models/Qwen/Qwen3-VL-8B-Instruct"
    request_file = "html_request/request_005_20260316_221014/request.json"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    all_results = {}

    for batch_size in args.batch_sizes:
        print(f"\n{'=' * 80}")
        print(f"Batch Size: {batch_size}")
        print('=' * 80)

        # Profile
        prof = profile_batch_detailed(model_path, request_file, batch_size, args.max_tokens)

        # Export trace
        trace_file = output_dir / f"detailed_trace_batch{batch_size}.json"
        prof.export_chrome_trace(str(trace_file))
        print(f"Trace exported to: {trace_file}")

        # Analyze trace
        kernel_stats = extract_gemm_kernels_from_trace(trace_file)

        if kernel_stats:
            print(f"\nFound {len(kernel_stats)} unique GEMM kernels:")
            sorted_kernels = sorted(kernel_stats.items(), key=lambda x: x[1]['total_time_us'], reverse=True)

            for i, (name, stats) in enumerate(sorted_kernels[:10], 1):
                time_ms = stats['total_time_us'] / 1000
                print(f"{i:2}. {name}")
                print(f"     Count: {stats['count']}, Time: {time_ms:.3f} ms")
        else:
            print("\n⚠️  No GEMM kernels found in trace")
            print("The trace file may not contain low-level kernel names")

        all_results[batch_size] = {
            name: {
                'count': stats['count'],
                'total_time_us': stats['total_time_us']
            }
            for name, stats in kernel_stats.items()
        }

        # Clean up
        del prof
        torch.cuda.empty_cache()
        time.sleep(3)

    # Compare
    if 16 in all_results and 32 in all_results:
        print(f"\n{'=' * 80}")
        print("Comparison: Batch 16 vs 32")
        print('=' * 80)

        kernels_16 = set(all_results[16].keys())
        kernels_32 = set(all_results[32].keys())

        common = kernels_16 & kernels_32
        only_16 = kernels_16 - kernels_32
        only_32 = kernels_32 - kernels_16

        print(f"\nKernel comparison:")
        print(f"  Common: {len(common)}")
        print(f"  Only in batch 16: {len(only_16)}")
        print(f"  Only in batch 32: {len(only_32)}")

        if only_16:
            print(f"\n  Kernels ONLY in batch 16:")
            for name in sorted(only_16):
                stats = all_results[16][name]
                print(f"    - {name} (count: {stats['count']})")

        if only_32:
            print(f"\n  Kernels ONLY in batch 32:")
            for name in sorted(only_32):
                stats = all_results[32][name]
                print(f"    - {name} (count: {stats['count']})")

    # Save
    summary_file = output_dir / "detailed_kernel_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"Summary saved to: {summary_file}")
    print('=' * 80)


if __name__ == '__main__':
    main()
