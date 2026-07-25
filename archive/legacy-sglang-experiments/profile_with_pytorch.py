#!/usr/bin/env python3
"""
Profile QKV kernels using PyTorch profiler (no special permissions needed).

This works on shared systems where ncu requires GPU performance counter permissions.
"""

import argparse
import json
import os
import time
from pathlib import Path

# Enable timing
os.environ["TREE_SPARSE_TIMING"] = "0"  # Disable to reduce noise

import torch
from torch.profiler import profile, ProfilerActivity, record_function

import sglang as sgl


def profile_batch_decode(model_path, request_file, batch_size, max_tokens=10):
    """Profile decode for a specific batch size using PyTorch profiler."""

    # Load request
    with open(request_file, 'r') as f:
        request_data = json.load(f)

    # Load tokenizer and apply chat template
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
    print(f"Initializing engine for batch size {batch_size}...")
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

    # Profile
    print(f"Profiling batch size {batch_size}...")

    with profile(
        activities=[ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
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

    return prof


def analyze_profile(prof, batch_size):
    """Analyze PyTorch profiler output for GEMM/matrix multiplication kernels."""

    print(f"\n{'=' * 80}")
    print(f"Batch Size {batch_size} - Top CUDA Kernels")
    print('=' * 80)

    # Get CUDA time sorted by duration
    cuda_events = prof.key_averages()

    # Filter for matrix multiplication kernels
    gemm_kernels = []
    for evt in cuda_events:
        name = evt.key
        # Look for GEMM, CUTLASS, nvjet, or other matrix mult kernels
        if any(kw in name.lower() for kw in ['gemm', 'cutlass', 'nvjet', 'wmma', 'mma', 'tensorop']):
            gemm_kernels.append({
                'name': name,
                'cuda_time_us': evt.cuda_time_total,
                'calls': evt.count
            })

    # Sort by CUDA time
    gemm_kernels.sort(key=lambda x: x['cuda_time_us'], reverse=True)

    # Print top 20
    print(f"\n{'Rank':<6} {'CUDA Time (ms)':<16} {'Calls':<8} {'Kernel Name'}")
    print('-' * 120)

    for i, kernel in enumerate(gemm_kernels[:20], 1):
        time_ms = kernel['cuda_time_us'] / 1000
        print(f"{i:<6} {time_ms:<16.3f} {kernel['calls']:<8} {kernel['name']}")

    return gemm_kernels


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

    results = {}

    for batch_size in args.batch_sizes:
        print(f"\n{'=' * 80}")
        print(f"Profiling Batch Size: {batch_size}")
        print('=' * 80)

        prof = profile_batch_decode(model_path, request_file, batch_size, args.max_tokens)

        # Analyze
        gemm_kernels = analyze_profile(prof, batch_size)

        # Save
        results[batch_size] = gemm_kernels

        # Export trace
        trace_file = output_dir / f"pytorch_trace_batch{batch_size}.json"
        prof.export_chrome_trace(str(trace_file))
        print(f"\nChrome trace saved to: {trace_file}")
        print("View in chrome://tracing")

        # Clean up
        del prof
        torch.cuda.empty_cache()

        print("\nWaiting 5 seconds before next batch...")
        time.sleep(5)

    # Compare
    print(f"\n{'=' * 80}")
    print("COMPARISON: Batch 16 vs Batch 32")
    print('=' * 80)

    kernels_16 = {k['name'] for k in results[16]}
    kernels_32 = {k['name'] for k in results[32]}

    only_16 = kernels_16 - kernels_32
    only_32 = kernels_32 - kernels_16
    common = kernels_16 & kernels_32

    print(f"\nCommon kernels: {len(common)}")
    print(f"Only in batch 16: {len(only_16)}")
    print(f"Only in batch 32: {len(only_32)}")

    if only_16:
        print(f"\nKernels ONLY in batch 16:")
        for name in sorted(only_16):
            print(f"  - {name}")

    if only_32:
        print(f"\nKernels ONLY in batch 32:")
        for name in sorted(only_32):
            print(f"  - {name}")

    # Save summary
    summary_file = output_dir / "pytorch_profile_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            str(k): [{'name': x['name'], 'cuda_time_us': x['cuda_time_us'], 'calls': x['calls']}
                     for x in v]
            for k, v in results.items()
        }, f, indent=2)

    print(f"\nSummary saved to: {summary_file}")
    print('=' * 80)


if __name__ == '__main__':
    main()
