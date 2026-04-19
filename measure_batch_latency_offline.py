#!/usr/bin/env python3
"""
Batch latency measurement using sglang.Engine (offline mode).

This script provides direct control over batching without HTTP overhead.
It measures TPOT (Time Per Output Token) for different batch sizes.

Usage:
    python measure_batch_latency_offline.py \
        --request-file request.json \
        --batch-sizes 1 2 4 8 16 32 64 \
        --max-tokens 256 \
        --repeat 5
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# Enable detailed decode step timing (same profiler used by server mode)
# This activates CUDA event-based timing for qkv_proj, attention, mlp, etc.
# Respect the env var if already set (e.g. TREE_SPARSE_TIMING=0 for clean TPOT).
if "TREE_SPARSE_TIMING" not in os.environ:
    os.environ["TREE_SPARSE_TIMING"] = "1"
if "TREE_SPARSE_TIMING_INTERVAL" not in os.environ:
    os.environ["TREE_SPARSE_TIMING_INTERVAL"] = "10"

try:
    import sglang as sgl
except ImportError:
    print("Error: sglang not installed. Install with: pip install sglang")
    exit(1)

try:
    from sglang.srt.entrypoints.openai.encoding_dsv32 import encode_messages as dsv32_encode_messages
except ImportError:
    dsv32_encode_messages = None


def load_request(request_file: str) -> Dict[str, Any]:
    """Load a single request from JSON file."""
    with open(request_file, 'r') as f:
        data = json.load(f)
    return data


def prepare_prompts(
    request_data: Dict[str, Any], batch_size: int, tokenizer=None
) -> List[str]:
    """Prepare a batch of identical prompts.

    If the request contains a 'messages' array (chat format) and a tokenizer
    is provided, apply the chat template so that system/user/assistant roles
    are properly encoded — matching what the HTTP server does.
    """
    if isinstance(request_data.get('text'), str):
        prompt = request_data['text']
    elif isinstance(request_data.get('messages'), list):
        messages = request_data['messages']
        if tokenizer is not None:
            # Apply chat template to include ALL messages (system + user + ...)
            try:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                print(f"  Applied chat template: {len(messages)} messages -> {len(prompt)} chars")
            except (ValueError, AttributeError):
                # Model has no chat template — use DeepSeek V3.2 custom encoding
                # if available (matches what the server does for this model)
                if dsv32_encode_messages is None:
                    raise RuntimeError(
                        "No chat template found and DeepSeek V3.2 encoder not available."
                    )
                msgs = [m.copy() for m in messages]
                if msgs[0]["role"] != "system":
                    msgs.insert(0, {"role": "system", "content": ""})
                prompt = dsv32_encode_messages(msgs, thinking_mode="chat")
                print(f"  Applied DeepSeek V3.2 encoding: {len(messages)} messages -> {len(prompt)} chars")
        else:
            # Fallback: concatenate all message contents
            prompt = "\n".join(m.get('content', '') for m in messages)
            print(f"  WARNING: No tokenizer, concatenated {len(messages)} messages")
    else:
        raise ValueError(f"Cannot extract prompt from request: {request_data}")

    # Replicate the prompt for batching
    return [prompt] * batch_size


def measure_batch_latency(
    engine: sgl.Engine,
    prompts: List[str],
    max_tokens: int,
    min_tokens: int = 0,
    ignore_eos: bool = True,
) -> Dict[str, float]:
    """
    Measure latency for a single batch using streaming mode.

    TPOT calculation: (total_time - time_to_first_token) / (tokens_generated - 1)
    This excludes prefill time and measures pure decode performance.

    Returns:
        dict with 'prefill_time', 'decode_time', 'total_time', 'tokens_generated', 'tpot'
    """
    # Prepare sampling params
    sampling_params = {
        "max_new_tokens": max_tokens,
        "temperature": 0.0,  # Deterministic for consistency
        "top_p": 1.0,
    }

    if min_tokens > 0:
        sampling_params["min_new_tokens"] = min_tokens

    if ignore_eos:
        sampling_params["ignore_eos"] = True

    # Warmup (important for accurate timing)
    if len(prompts) == 1:
        _ = engine.generate(prompts[:1], sampling_params=sampling_params)

    # Actual measurement using streaming mode to capture TTFT
    start_time = time.perf_counter()
    first_token_time = None
    tokens_generated = 0

    # Use streaming to detect first token arrival
    output_stream = engine.generate(
        prompts,
        sampling_params=sampling_params,
        stream=True
    )

    for chunk in output_stream:
        if first_token_time is None:
            # First chunk arrives after prefill
            first_token_time = time.perf_counter()
        # Update token count from the last chunk
        if isinstance(chunk, list):
            # Batch mode returns list
            tokens_generated = len(chunk[0].get("output_ids", []))
        else:
            tokens_generated = len(chunk.get("output_ids", []))

    end_time = time.perf_counter()
    total_time = end_time - start_time

    # Calculate metrics
    if first_token_time is None or tokens_generated == 0:
        return {
            "prefill_time": total_time * 1000,
            "decode_time": 0.0,
            "total_time": total_time * 1000,
            "tokens_generated": 0,
            "tpot": 0.0,
            "ttft": total_time * 1000,
        }

    # Time to first token (prefill time)
    ttft = first_token_time - start_time

    # Decode time (everything after first token)
    decode_time = end_time - first_token_time

    # TPOT = decode_time / (tokens_generated - 1)
    # Exclude the first token since it arrives during prefill
    if tokens_generated > 1:
        tpot = decode_time / (tokens_generated - 1)
    else:
        tpot = 0.0

    return {
        "prefill_time": ttft * 1000,  # Convert to ms
        "decode_time": decode_time * 1000,
        "total_time": total_time * 1000,
        "tokens_generated": tokens_generated,
        "tpot": tpot * 1000,  # ms per token (decode only)
        "ttft": ttft * 1000,  # Time to first token
    }


def run_batch_experiments(
    model_path: str,
    request_file: str,
    batch_sizes: List[int],
    max_tokens: int,
    min_tokens: int,
    ignore_eos: bool,
    repeat: int,
    tp_size: int = 1,
    log_file: str = None,
    **engine_kwargs,
) -> Dict[int, Dict[str, Any]]:
    """
    Run batch latency experiments for different batch sizes.

    Returns:
        dict mapping batch_size -> statistics
    """
    print("=" * 80)
    print("SGLang Offline Batch Latency Experiments")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Request file: {request_file}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Max tokens: {max_tokens}")
    print(f"Min tokens: {min_tokens}")
    print(f"Ignore EOS: {ignore_eos}")
    print(f"Repetitions: {repeat}")
    print(f"TP size: {tp_size}")
    print("=" * 80)
    print()

    # Load request
    request_data = load_request(request_file)
    print(f"✓ Loaded request from {request_file}")

    # Load tokenizer for chat template application
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )
    print(f"✓ Loaded tokenizer from {model_path}")

    # Initialize SGLang engine
    print(f"\nInitializing SGLang engine...")
    print(f"  Model: {model_path}")
    print(f"  TP size: {tp_size}")

    # Configure logging to enable detailed timing output
    import logging

    # Set up logging handlers
    handlers = [logging.StreamHandler()]  # stdout
    if log_file:
        handlers.append(logging.FileHandler(log_file, mode='w'))

    logging.basicConfig(
        level=logging.DEBUG,
        format='[%(asctime)s] %(message)s',
        handlers=handlers,
    )

    print(f"Logging to: stdout" + (f" and {log_file}" if log_file else ""))
    print()

    # Extract attention_backend from kwargs (default to flashinfer)
    attention_backend = engine_kwargs.pop("attention_backend", "flashinfer")

    engine = sgl.Engine(
        model_path=model_path,
        tp_size=tp_size,
        log_level="debug",  # Enable debug logging
        decode_log_interval=1,  # Log every decode step
        show_time_cost=True,  # Show timing breakdown
        attention_backend=attention_backend,
        trust_remote_code=True,  # Required for Qwen models
        enable_layerwise_nvtx_marker=True,  # Add NVTX markers for nsys profiling
        **engine_kwargs,
    )

    print(f"  Attention backend: {attention_backend}")

    print("✓ Engine initialized")
    print()

    # Run experiments
    results = {}

    for batch_size in batch_sizes:
        print(f"\n{'=' * 80}")
        print(f"Batch Size: {batch_size}")
        print(f"{'=' * 80}")

        # Prepare prompts (apply chat template for proper tokenization)
        prompts = prepare_prompts(request_data, batch_size, tokenizer=tokenizer)
        print(f"  Prepared {len(prompts)} prompts")

        # Run multiple repetitions
        latencies = []

        for i in range(repeat):
            print(f"  Run {i+1}/{repeat}...", end=" ", flush=True)

            metrics = measure_batch_latency(
                engine=engine,
                prompts=prompts,
                max_tokens=max_tokens,
                min_tokens=min_tokens,
                ignore_eos=ignore_eos,
            )

            latencies.append(metrics)
            print(f"TPOT: {metrics['tpot']:.2f} ms, Total: {metrics['total_time']:.0f} ms")

        # Compute statistics
        tpots = [m['tpot'] for m in latencies]
        total_times = [m['total_time'] for m in latencies]
        ttfts = [m['ttft'] for m in latencies]
        decode_times = [m['decode_time'] for m in latencies]
        tokens_generated = latencies[0]['tokens_generated']

        results[batch_size] = {
            "batch_size": batch_size,
            "repetitions": repeat,
            "tokens_generated": tokens_generated,
            "tpot_mean_ms": float(np.mean(tpots)),
            "tpot_std_ms": float(np.std(tpots)),
            "tpot_min_ms": float(np.min(tpots)),
            "tpot_max_ms": float(np.max(tpots)),
            "ttft_mean_ms": float(np.mean(ttfts)),
            "ttft_std_ms": float(np.std(ttfts)),
            "decode_time_mean_ms": float(np.mean(decode_times)),
            "decode_time_std_ms": float(np.std(decode_times)),
            "total_time_mean_ms": float(np.mean(total_times)),
            "throughput_tokens_per_sec": (batch_size * tokens_generated) / (np.mean(total_times) / 1000),
        }

        print(f"\n  Summary:")
        print(f"    TTFT (prefill): {results[batch_size]['ttft_mean_ms']:.2f} ± {results[batch_size]['ttft_std_ms']:.2f} ms")
        print(f"    TPOT (decode only): {results[batch_size]['tpot_mean_ms']:.2f} ± {results[batch_size]['tpot_std_ms']:.2f} ms")
        print(f"    Decode time: {results[batch_size]['decode_time_mean_ms']:.0f} ms")
        print(f"    Total time: {results[batch_size]['total_time_mean_ms']:.0f} ms")
        print(f"    Throughput: {results[batch_size]['throughput_tokens_per_sec']:.1f} tokens/s")

    print(f"\n{'=' * 80}")
    print("All experiments completed!")
    print(f"{'=' * 80}\n")

    # Shutdown engine
    del engine

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Measure batch latency using SGLang offline mode"
    )
    parser.add_argument(
        "request_file",
        type=str,
        help="Path to request JSON file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen3-VL-8B-Instruct",
        help="Model name or path",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/vast/projects/liuv/pennnetworks/hf_models/Qwen/Qwen3-VL-8B-Instruct",
        help="Full path to model",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64],
        help="Batch sizes to test",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=0,
        help="Minimum number of tokens to generate (0 to disable)",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        default=True,
        help="Ignore EOS token (generate exactly max_tokens)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=5,
        help="Number of repetitions per batch size",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor parallel size",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="offline_batch_results.json",
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Log file for detailed timing output (default: stdout only)",
    )

    # Attention backend selection
    parser.add_argument(
        "--attention-backend",
        type=str,
        default="flashinfer",
        choices=["flashinfer", "tree_sparse", "flashinfer_tree_sparse", "nsa"],
        help="Attention backend (default: flashinfer)",
    )

    # General engine options (for quantized / MoE models like DeepSeek V3.2)
    parser.add_argument("--quantization", type=str, default=None,
                        help="Quantization method (e.g. awq_marlin)")
    parser.add_argument("--dtype", type=str, default=None,
                        help="Model dtype (e.g. bfloat16)")
    parser.add_argument("--kv-cache-dtype", type=str, default=None,
                        help="KV cache dtype (e.g. bfloat16)")
    parser.add_argument("--context-length", type=int, default=None,
                        help="Maximum context length")
    parser.add_argument("--chunked-prefill-size", type=int, default=None,
                        help="Chunked prefill size")
    parser.add_argument("--page-size", type=int, default=None,
                        help="KV cache page size")
    parser.add_argument("--mem-fraction-static", type=float, default=None,
                        help="Fraction of GPU memory for static allocation")
    parser.add_argument("--moe-runner-backend", type=str, default=None,
                        help="MoE runner backend (e.g. triton)")
    parser.add_argument("--nsa-prefill-backend", type=str, default=None,
                        help="NSA prefill backend (e.g. flashmla_sparse)")
    parser.add_argument("--nsa-decode-backend", type=str, default=None,
                        help="NSA decode backend (e.g. flashmla_sparse)")
    parser.add_argument("--model-loader-extra-config", type=str, default=None,
                        help="JSON string of extra model loader config")
    parser.add_argument("--watchdog-timeout", type=float, default=None,
                        help="Watchdog timeout in seconds")
    parser.add_argument("--enable-cuda-graph", action="store_true", default=False,
                        help="Enable CUDA graph (default: disabled for timing)")

    # Tree-sparse specific arguments
    parser.add_argument("--tree-sparse-top-k", type=int, default=8,
                        help="Number of chunks to select per query (default: 8)")
    parser.add_argument("--tree-sparse-min-seq-len", type=int, default=512,
                        help="Min sequence length to activate sparse attn (default: 512)")
    parser.add_argument("--tree-sparse-min-chunk-size", type=int, default=16,
                        help="Minimum tokens per tree chunk (default: 16)")
    parser.add_argument("--tree-sparse-max-chunk-size", type=int, default=256,
                        help="Maximum tokens per tree chunk (default: 256)")
    parser.add_argument("--tree-sparse-recent-tokens", type=int, default=128,
                        help="Always attend to last N tokens (default: 128)")
    args = parser.parse_args()

    # Build engine kwargs based on attention backend
    engine_kwargs = {
        "disable_cuda_graph": not args.enable_cuda_graph,
        "attention_backend": args.attention_backend,
    }

    # Add optional engine kwargs if specified
    optional_kwargs = {
        "quantization": args.quantization,
        "dtype": args.dtype,
        "kv_cache_dtype": getattr(args, "kv_cache_dtype", None),
        "context_length": args.context_length,
        "chunked_prefill_size": args.chunked_prefill_size,
        "page_size": args.page_size,
        "mem_fraction_static": args.mem_fraction_static,
        "moe_runner_backend": args.moe_runner_backend,
        "nsa_prefill_backend": args.nsa_prefill_backend,
        "nsa_decode_backend": args.nsa_decode_backend,
        "model_loader_extra_config": args.model_loader_extra_config,
        "watchdog_timeout": args.watchdog_timeout,
    }
    for k, v in optional_kwargs.items():
        if v is not None:
            engine_kwargs[k] = v

    if args.attention_backend in ("tree_sparse", "flashinfer_tree_sparse"):
        engine_kwargs.update({
            "enable_tree_sparse": True,
            "tree_sparse_top_k": args.tree_sparse_top_k,
            "tree_sparse_min_seq_len": args.tree_sparse_min_seq_len,
            "tree_sparse_min_chunk_size": args.tree_sparse_min_chunk_size,
            "tree_sparse_max_chunk_size": args.tree_sparse_max_chunk_size,
            "tree_sparse_recent_tokens": args.tree_sparse_recent_tokens,
        })

    # Run experiments
    results = run_batch_experiments(
        model_path=args.model_path,
        request_file=args.request_file,
        batch_sizes=args.batch_sizes,
        max_tokens=args.max_tokens,
        min_tokens=args.min_tokens,
        ignore_eos=args.ignore_eos,
        repeat=args.repeat,
        tp_size=args.tp_size,
        log_file=args.log_file,
        **engine_kwargs,
    )

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {args.output}")

    # Print summary table
    print("\n" + "=" * 100)
    print("Summary Table")
    print("=" * 100)
    print(f"{'Batch':>8} {'TTFT':>12} {'TPOT':>12} {'Decode':>12} {'Total':>12} {'Throughput':>20}")
    print(f"{'Size':>8} {'(ms)':>12} {'(ms)':>12} {'(ms)':>12} {'(ms)':>12} {'(tok/s)':>20}")
    print("-" * 100)

    for batch_size in sorted(results.keys()):
        r = results[batch_size]
        print(f"{batch_size:>8} "
              f"{r['ttft_mean_ms']:>12.2f} "
              f"{r['tpot_mean_ms']:>12.2f} "
              f"{r['decode_time_mean_ms']:>12.0f} "
              f"{r['total_time_mean_ms']:>12.0f} "
              f"{r['throughput_tokens_per_sec']:>20.1f}")

    print("=" * 100)
    print("\nNote: TPOT = (total_time - TTFT) / (tokens_generated - 1) - excludes prefill")


if __name__ == "__main__":
    main()
