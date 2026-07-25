#!/usr/bin/env python3
"""
Try profiling using pycuda and direct CUPTI callbacks.
This may have different permission requirements than ncu.
"""

import os
os.environ["TREE_SPARSE_TIMING"] = "0"

import json
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import sglang as sgl


# Store kernel launches
kernel_launches = defaultdict(int)


def setup_cuda_profiler():
    """Try to enable CUDA profiling."""
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        cuda.profiler_start()
        return True
    except:
        print("pycuda not available, trying torch profiler fallback")
        return False


def profile_batch(model_path, request_file, batch_size, max_tokens=10):
    """Profile a single batch size."""

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

    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Profile with CUDA events for timing
    print(f"Profiling batch {batch_size}...")

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    outputs = engine.generate(
        prompts,
        sampling_params={
            "max_new_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "ignore_eos": True
        }
    )
    end.record()

    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)

    # Memory stats
    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024**2)

    stats = {
        'batch_size': batch_size,
        'elapsed_ms': elapsed_ms,
        'tokens_per_request': max_tokens,
        'total_tokens': batch_size * max_tokens,
        'throughput_tokens_per_sec': (batch_size * max_tokens) / (elapsed_ms / 1000),
        'latency_per_token_ms': elapsed_ms / (batch_size * max_tokens),
        'peak_memory_mb': peak_memory_mb
    }

    del engine
    torch.cuda.empty_cache()

    return stats, outputs


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

    # Try to enable profiling
    use_pycuda = setup_cuda_profiler()

    results = {}

    for batch_size in args.batch_sizes:
        print(f"\n{'=' * 80}")
        print(f"Batch Size: {batch_size}")
        print('=' * 80)

        stats, _ = profile_batch(model_path, request_file, batch_size, args.max_tokens)
        results[batch_size] = stats

        print(f"\nResults:")
        print(f"  Total time: {stats['elapsed_ms']:.2f} ms")
        print(f"  Throughput: {stats['throughput_tokens_per_sec']:.2f} tokens/sec")
        print(f"  Latency per token: {stats['latency_per_token_ms']:.3f} ms")
        print(f"  Peak memory: {stats['peak_memory_mb']:.2f} MB")

    # Compare
    if 16 in results and 32 in results:
        print(f"\n{'=' * 80}")
        print("Comparison: Batch 16 vs 32")
        print('=' * 80)

        throughput_16 = results[16]['throughput_tokens_per_sec']
        throughput_32 = results[32]['throughput_tokens_per_sec']

        speedup = throughput_32 / throughput_16
        print(f"\nThroughput scaling:")
        print(f"  Batch 16: {throughput_16:.2f} tokens/sec")
        print(f"  Batch 32: {throughput_32:.2f} tokens/sec")
        print(f"  Speedup: {speedup:.2f}x (ideal: 2.0x)")

        if speedup < 1.5:
            print(f"  ⚠️  Sublinear scaling detected! Only {speedup:.2f}x instead of 2.0x")

    # Save
    summary_file = output_dir / "cupti_profile_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"Summary saved to: {summary_file}")
    print('=' * 80)

    if use_pycuda:
        import pycuda.driver as cuda
        cuda.profiler_stop()


if __name__ == '__main__':
    main()
