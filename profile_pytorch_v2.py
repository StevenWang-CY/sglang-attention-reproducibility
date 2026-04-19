#!/usr/bin/env python3
"""
PyTorch profiler v2 - Better inspection of kernel names.
This should work without any system permissions.
"""

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict

os.environ["TREE_SPARSE_TIMING"] = "0"

import torch
from torch.profiler import profile, ProfilerActivity

import sglang as sgl


def profile_batch(model_path, request_file, batch_size, max_tokens=10):
    """Profile a batch size with PyTorch profiler."""

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

    # Profile with detailed settings
    print(f"Profiling batch {batch_size}...")

    with profile(
        activities=[ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,  # Disable stack to reduce overhead
        profile_memory=False,
    ) as prof:
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


def analyze_profiler_events(prof, batch_size):
    """Analyze profiler events and extract kernel names."""

    print(f"\n{'=' * 80}")
    print(f"Batch {batch_size} - Analyzing Profiler Events")
    print('=' * 80)

    # Get all CUDA events
    events = prof.key_averages()

    kernel_data = defaultdict(lambda: {'count': 0, 'cuda_time_us': 0})

    print(f"\nTotal events: {len(list(events))}")

    # Print ALL events to see what we're getting
    print("\nAll CUDA events (first 50):")
    for i, evt in enumerate(events):
        if i >= 50:
            break
        # Check for CUDA time (attribute name might vary by PyTorch version)
        cuda_time = getattr(evt, 'cuda_time_total', getattr(evt, 'device_time_total', 0))

        if cuda_time > 0:
            print(f"{i+1}. {evt.key}")
            print(f"   CUDA time: {cuda_time:.0f} us, Count: {evt.count}")

            # Collect kernel data
            kernel_data[evt.key]['count'] += evt.count
            kernel_data[evt.key]['cuda_time_us'] += cuda_time

    # Look for GEMM/matrix multiplication kernels
    gemm_kernels = {}
    for name, stats in kernel_data.items():
        name_lower = name.lower()
        if any(kw in name_lower for kw in ['gemm', 'nvjet', 'cutlass', 'wmma', 'mma', 'matmul', 'cublas']):
            gemm_kernels[name] = stats

    if gemm_kernels:
        print(f"\n\nFound {len(gemm_kernels)} GEMM-related operations:")
        sorted_gemm = sorted(gemm_kernels.items(), key=lambda x: x[1]['cuda_time_us'], reverse=True)
        for name, stats in sorted_gemm[:20]:
            time_ms = stats['cuda_time_us'] / 1000
            print(f"  {name}")
            print(f"    Count: {stats['count']}, Time: {time_ms:.3f} ms")
    else:
        print("\n⚠️  No GEMM kernels found with keyword filter")
        print("This might mean:")
        print("  1. PyTorch profiler captured high-level ops, not low-level kernels")
        print("  2. Kernel names are in a different format")
        print("  3. We need to inspect the Chrome trace JSON directly")

    return dict(kernel_data)


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

        prof = profile_batch(model_path, request_file, batch_size, args.max_tokens)

        # Analyze events
        kernel_data = analyze_profiler_events(prof, batch_size)
        all_results[batch_size] = kernel_data

        # Export Chrome trace for manual inspection
        trace_file = output_dir / f"pytorch_v2_trace_batch{batch_size}.json"
        prof.export_chrome_trace(str(trace_file))
        print(f"\nChrome trace: {trace_file}")

        # Also print table view
        print("\n" + "="*80)
        print("PyTorch Profiler Table Output:")
        print("="*80)
        try:
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
        except:
            # Try alternative sort key if cuda_time_total doesn't exist
            print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30))

        del prof
        torch.cuda.empty_cache()

    # Save summary
    summary_file = output_dir / "pytorch_v2_summary.json"
    with open(summary_file, 'w') as f:
        # Convert to serializable format
        output = {}
        for batch, kernels in all_results.items():
            output[batch] = {
                name: {'count': stats['count'], 'cuda_time_us': stats['cuda_time_us']}
                for name, stats in kernels.items()
            }
        json.dump(output, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"Summary saved to: {summary_file}")
    print('=' * 80)


if __name__ == '__main__':
    main()
