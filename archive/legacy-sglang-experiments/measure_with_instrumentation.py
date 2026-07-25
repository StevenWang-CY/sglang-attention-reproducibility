#!/usr/bin/env python3
"""
Run benchmarks with kernel selection instrumentation.

This will reveal:
1. Which code paths are taken for batch 16 vs 32
2. What actual execution time is (bypassing tree-sparse timing)
3. Stack traces showing where QKV operations happen
4. Any conditional logic that switches kernels
"""

import argparse
import json
import os
from pathlib import Path

# IMPORTANT: Install instrumentation BEFORE importing SGLang
os.environ["TREE_SPARSE_TIMING"] = "0"
os.environ["KERNEL_LOG"] = "kernel_selection_log.txt"

from instrument_kernel_selection import install_instrumentation, uninstall_instrumentation

# Install patches
install_instrumentation()

import sglang as sgl


def benchmark_batch(model_path, request_file, batch_size, max_tokens=10):
    """Benchmark a single batch size with instrumentation."""

    print(f"\n{'='*80}")
    print(f"Benchmarking Batch Size: {batch_size}")
    print('='*80)

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
    print(f"Initializing engine...")
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

    # Run with instrumentation active
    print(f"\nRunning batch {batch_size} with instrumentation...")
    print("(Check kernel_selection_log.txt for detailed execution trace)")

    outputs = engine.generate(
        prompts,
        sampling_params={
            "max_new_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "ignore_eos": True
        }
    )

    print(f"✓ Batch {batch_size} complete")

    del engine
    import torch
    torch.cuda.empty_cache()

    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[16, 32])
    parser.add_argument('--max-tokens', type=int, default=10)
    args = parser.parse_args()

    model_path = "/vast/projects/liuv/pennnetworks/hf_models/Qwen/Qwen3-VL-8B-Instruct"
    request_file = "html_request/request_005_20260316_221014/request.json"

    print("="*80)
    print("KERNEL SELECTION INSTRUMENTATION BENCHMARK")
    print("="*80)
    print(f"\nThis will reveal:")
    print("  1. Which code paths execute for each batch size")
    print("  2. Actual GEMM execution times (bypassing tree-sparse timing)")
    print("  3. Stack traces showing execution flow")
    print("  4. Kernel selection logic\n")

    for batch_size in args.batch_sizes:
        benchmark_batch(model_path, request_file, batch_size, args.max_tokens)

    # Cleanup
    uninstall_instrumentation()

    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"\nDetailed log saved to: kernel_selection_log.txt")
    print("\nTo analyze:")
    print("  grep 'QKV' kernel_selection_log.txt")
    print("  grep 'batch=16' kernel_selection_log.txt")
    print("  grep 'batch=32' kernel_selection_log.txt")
    print("\nTo see timing:")
    print("  grep 'Time:' kernel_selection_log.txt")
    print("\nTo see calling stack:")
    print("  grep -A 5 'Calling stack' kernel_selection_log.txt")
    print("="*80)


if __name__ == '__main__':
    main()
