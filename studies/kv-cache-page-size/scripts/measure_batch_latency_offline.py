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
import logging
import os
import re
import subprocess
import sys
import threading
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


class GPUMonitor:
    """Sample GPU stats (compute %, bandwidth %, memory MB) in a background thread."""

    def __init__(self, gpu_id: int = 0, interval_ms: int = 200):
        self.gpu_id = gpu_id
        self.interval_ms = interval_ms
        self.samples = []
        self._running = False
        self._thread = None

    def start(self):
        self.samples = []
        self._running = True
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _poll(self):
        while self._running:
            try:
                out = subprocess.check_output([
                    'nvidia-smi',
                    f'--id={self.gpu_id}',
                    '--query-gpu=utilization.gpu,utilization.memory,memory.used',
                    '--format=csv,noheader,nounits'
                ], timeout=1).decode().strip()
                parts = out.split(',')
                if len(parts) == 3:
                    self.samples.append((
                        float(parts[0]),  # sm %
                        float(parts[1]),  # mem bandwidth %
                        float(parts[2]),  # mem used MB
                    ))
            except Exception:
                pass
            time.sleep(self.interval_ms / 1000.0)

    def summary(self) -> dict:
        if not self.samples:
            return {}
        sm = [s[0] for s in self.samples]
        bw = [s[1] for s in self.samples]
        mem = [s[2] for s in self.samples]
        return {
            'gpu_compute_pct': {'mean': float(np.mean(sm)), 'max': float(np.max(sm))},
            'gpu_mem_bw_pct': {'mean': float(np.mean(bw)), 'max': float(np.max(bw))},
            'gpu_mem_used_mb': {'mean': float(np.mean(mem)), 'max': float(np.max(mem))},
            'num_samples': len(self.samples),
        }


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


class RunningReqTracker:
    """
    Intercepts sglang's 'Decode batch' log lines and tracks the max
    #running-req seen during a generation call.

    In offline engine mode, sglang writes these lines directly to stdout
    rather than through Python's logging system, so we intercept sys.stdout.

    Usage:
        tracker = RunningReqTracker()
        tracker.attach()
        engine.generate(...)
        print(tracker.max_running_req)
        tracker.reset()
    """

    _PATTERN = re.compile(r"#running-req:\s*(\d+)")

    def __init__(self):
        self.max_running_req = 0
        self._original_stdout_write = None
        self._original_stderr_write = None

    def _make_intercept(self, original_write):
        def _intercept(text):
            if "Decode batch" in text:
                m = self._PATTERN.search(text)
                if m:
                    n = int(m.group(1))
                    if n > self.max_running_req:
                        self.max_running_req = n
            original_write(text)
        return _intercept

    def reset(self):
        self.max_running_req = 0

    def attach(self):
        # Intercept both stdout and stderr — SGLang logs "Decode batch" via
        # logger.info() which goes to stderr (logging.StreamHandler default).
        if self._original_stdout_write is None:
            self._original_stdout_write = sys.stdout.write
            sys.stdout.write = self._make_intercept(self._original_stdout_write)
        if self._original_stderr_write is None:
            self._original_stderr_write = sys.stderr.write
            sys.stderr.write = self._make_intercept(self._original_stderr_write)

    def detach(self):
        if self._original_stdout_write is not None:
            sys.stdout.write = self._original_stdout_write
            self._original_stdout_write = None
        if self._original_stderr_write is not None:
            sys.stderr.write = self._original_stderr_write
            self._original_stderr_write = None


def measure_batch_latency(
    engine: sgl.Engine,
    prompts: List[str],
    max_tokens: int,
    min_tokens: int = 0,
    ignore_eos: bool = True,
    enable_gpu_monitor: bool = False,
    running_req_tracker: "RunningReqTracker" = None,
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

    # Start GPU monitoring (nvidia-smi polling — coarse but always available)
    gpu_monitor = GPUMonitor(gpu_id=0, interval_ms=200) if enable_gpu_monitor else None
    if gpu_monitor:
        gpu_monitor.start()

    # Reset running-req tracker for this measurement window
    if running_req_tracker:
        running_req_tracker.reset()

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

    if gpu_monitor:
        gpu_monitor.stop()
    gpu_stats = gpu_monitor.summary() if gpu_monitor else {}

    # Calculate metrics
    if first_token_time is None or tokens_generated == 0:
        return {
            "prefill_time": total_time * 1000,
            "decode_time": 0.0,
            "total_time": total_time * 1000,
            "tokens_generated": 0,
            "tpot": 0.0,
            "ttft": total_time * 1000,
            "gpu_stats": gpu_stats,
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

    max_running_req = running_req_tracker.max_running_req if running_req_tracker else None

    return {
        "prefill_time": ttft * 1000,  # Convert to ms
        "decode_time": decode_time * 1000,
        "total_time": total_time * 1000,
        "tokens_generated": tokens_generated,
        "tpot": tpot * 1000,  # ms per token (decode only)
        "ttft": ttft * 1000,  # Time to first token
        "gpu_stats": gpu_stats,
        "max_running_req": max_running_req,
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
    enable_gpu_monitor: bool = False,
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
        enable_layerwise_nvtx_marker=engine_kwargs.pop("enable_layerwise_nvtx_marker", True),
        **engine_kwargs,
    )

    print(f"  Attention backend: {attention_backend}")

    print("✓ Engine initialized")
    print()

    # Attach log interceptor to track actual #running-req during decode
    tracker = RunningReqTracker()
    tracker.attach()

    # Run experiments
    results = {}

    # NVTX helper — no-op if torch not available or not under nsys
    try:
        import torch.cuda.nvtx as _nvtx
        def nvtx_range_push(s): _nvtx.range_push(s)
        def nvtx_range_pop(): _nvtx.range_pop()
    except Exception:
        def nvtx_range_push(s): pass
        def nvtx_range_pop(): pass

    # nvidia-smi free-memory probe — lets a long multi-batch launch record the
    # GPU-idle state per batch, so shared-GPU contamination is localized to one
    # batch instead of silently tainting the whole launch.
    def _gpu_free_mb():
        try:
            import subprocess
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                timeout=10).decode().strip().splitlines()[0]
            return int(out.strip())
        except Exception:
            return None

    for batch_size in batch_sizes:
        nvtx_range_push(f"bs_{batch_size}")
        print(f"\n{'=' * 80}")
        print(f"Batch Size: {batch_size}")
        print(f"{'=' * 80}")
        free_before_mb = _gpu_free_mb()
        print(f"  GPU free before: {free_before_mb} MiB")

        # Flush KV cache before each batch size to ensure a clean slate
        engine.flush_cache()
        print(f"  KV cache flushed")

        # Prepare prompts (apply chat template for proper tokenization)
        prompts = prepare_prompts(request_data, batch_size, tokenizer=tokenizer)
        print(f"  Prepared {len(prompts)} prompts")

        # Count actual prompt tokens
        prompt_token_ids = tokenizer(prompts[0], return_tensors=None)["input_ids"]
        prompt_len = len(prompt_token_ids)
        print(f"  Prompt tokens: {prompt_len:,}")

        # Run multiple repetitions
        latencies = []

        for i in range(repeat):
            print(f"  Run {i+1}/{repeat}...", end=" ", flush=True)
            engine.flush_cache()

            metrics = measure_batch_latency(
                engine=engine,
                prompts=prompts,
                max_tokens=max_tokens,
                min_tokens=min_tokens,
                ignore_eos=ignore_eos,
                enable_gpu_monitor=enable_gpu_monitor,
                running_req_tracker=tracker,
            )

            # ── Sanity check: actual concurrent requests vs requested batch size ──
            max_rr = metrics.get("max_running_req")
            if max_rr is not None and max_rr != batch_size:
                print(f"\n  !! WARNING: batch_size={batch_size} but max #running-req={max_rr} !!")
                print(f"  !! Sglang ran only {max_rr} requests concurrently (KV pool limit). !!")
                print(f"  !! TPOT measurement is NOT representative of true BS={batch_size}. !!")
                print(f"  !! Reduce batch size or increase --mem-fraction-static.           !!")

            latencies.append(metrics)
            gpu = metrics.get('gpu_stats', {})
            gpu_str = ""
            if gpu:
                gpu_str = f", GPU: {gpu['gpu_compute_pct']['mean']:.0f}%sm/{gpu['gpu_mem_bw_pct']['mean']:.0f}%bw/{gpu['gpu_mem_used_mb']['mean']:.0f}MB"
            rr_str = f", max_running_req={max_rr}" if max_rr is not None else ""
            print(f"TPOT: {metrics['tpot']:.2f} ms, Total: {metrics['total_time']:.0f} ms{gpu_str}{rr_str}")

        free_after_mb = _gpu_free_mb()
        print(f"  GPU free after: {free_after_mb} MiB")

        # Compute statistics
        tpots = [m['tpot'] for m in latencies]
        total_times = [m['total_time'] for m in latencies]
        ttfts = [m['ttft'] for m in latencies]
        decode_times = [m['decode_time'] for m in latencies]
        tokens_generated = latencies[0]['tokens_generated']

        # Aggregate GPU stats across repetitions
        all_gpu = [m.get('gpu_stats', {}) for m in latencies if m.get('gpu_stats')]
        gpu_summary = {}
        if all_gpu:
            gpu_summary = {
                "gpu_compute_pct_mean": float(np.mean([g['gpu_compute_pct']['mean'] for g in all_gpu])),
                "gpu_mem_bw_pct_mean": float(np.mean([g['gpu_mem_bw_pct']['mean'] for g in all_gpu])),
                "gpu_mem_used_mb_mean": float(np.mean([g['gpu_mem_used_mb']['mean'] for g in all_gpu])),
                "gpu_compute_pct_max": float(np.max([g['gpu_compute_pct']['max'] for g in all_gpu])),
                "gpu_mem_bw_pct_max": float(np.max([g['gpu_mem_bw_pct']['max'] for g in all_gpu])),
                "gpu_mem_used_mb_max": float(np.max([g['gpu_mem_used_mb']['max'] for g in all_gpu])),
            }

        max_running_reqs = [m['max_running_req'] for m in latencies if m.get('max_running_req') is not None]
        max_running_req_observed = max(max_running_reqs) if max_running_reqs else None
        concurrency_ok = (max_running_req_observed == batch_size) if max_running_req_observed is not None else None

        results[batch_size] = {
            "batch_size": batch_size,
            "repetitions": repeat,
            "tokens_generated": tokens_generated,
            "tpot_median_ms": float(np.median(tpots)),
            "tpot_mean_ms": float(np.mean(tpots)),
            "tpot_std_ms": float(np.std(tpots)),
            "tpot_min_ms": float(np.min(tpots)),
            "tpot_max_ms": float(np.max(tpots)),
            "ttft_median_ms": float(np.median(ttfts)),
            "ttft_mean_ms": float(np.mean(ttfts)),
            "ttft_std_ms": float(np.std(ttfts)),
            "decode_time_median_ms": float(np.median(decode_times)),
            "decode_time_mean_ms": float(np.mean(decode_times)),
            "decode_time_std_ms": float(np.std(decode_times)),
            "total_time_median_ms": float(np.median(total_times)),
            "total_time_mean_ms": float(np.mean(total_times)),
            "throughput_tokens_per_sec": (batch_size * tokens_generated) / (np.median(total_times) / 1000),
            "gpu_stats": gpu_summary,
            "max_running_req_observed": max_running_req_observed,
            "concurrency_ok": concurrency_ok,
            "free_before_mb": free_before_mb,
            "free_after_mb": free_after_mb,
        }

        print(f"\n  Summary:")
        print(f"    TTFT (prefill): median={results[batch_size]['ttft_median_ms']:.2f} ms  mean={results[batch_size]['ttft_mean_ms']:.2f} ± {results[batch_size]['ttft_std_ms']:.2f} ms")
        print(f"    TPOT (decode only): median={results[batch_size]['tpot_median_ms']:.2f} ms  mean={results[batch_size]['tpot_mean_ms']:.2f} ± {results[batch_size]['tpot_std_ms']:.2f} ms")
        print(f"    Decode time: {results[batch_size]['decode_time_median_ms']:.0f} ms (median)")
        print(f"    Total time: {results[batch_size]['total_time_median_ms']:.0f} ms (median)")
        print(f"    Throughput: {results[batch_size]['throughput_tokens_per_sec']:.1f} tokens/s")
        if max_running_req_observed is not None:
            flag = "OK" if concurrency_ok else f"!! MISMATCH — only {max_running_req_observed} ran concurrently, TPOT invalid !!"
            print(f"    Concurrency check: max_running_req={max_running_req_observed} vs batch_size={batch_size} → {flag}")
        if gpu_summary:
            print(f"    GPU compute:  {gpu_summary['gpu_compute_pct_mean']:.1f}% avg, {gpu_summary['gpu_compute_pct_max']:.1f}% max")
            print(f"    GPU mem BW:   {gpu_summary['gpu_mem_bw_pct_mean']:.1f}% avg, {gpu_summary['gpu_mem_bw_pct_max']:.1f}% max")
            print(f"    GPU mem used: {gpu_summary['gpu_mem_used_mb_mean']:.0f} MB avg, {gpu_summary['gpu_mem_used_mb_max']:.0f} MB max")

        nvtx_range_pop()

    print(f"\n{'=' * 80}")
    print("All experiments completed!")
    print(f"{'=' * 80}\n")

    tracker.detach()

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
        default=os.environ.get("MODEL_PATH", "Qwen/Qwen3-VL-8B-Instruct"),
        help="Model path or Hugging Face identifier (default: MODEL_PATH or Qwen/Qwen3-VL-8B-Instruct)",
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
        choices=["flashinfer", "tree_sparse", "flashinfer_tree_sparse", "nsa", "triton", "trtllm_mha"],
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
    parser.add_argument("--cuda-graph-max-bs", type=int, default=None,
                        help="Max batch size to capture CUDA graphs for. On a 16GB card SGLang auto-caps this at 8, "
                             "so larger batches silently run EAGER even with --enable-cuda-graph. Raise it (e.g. 48) "
                             "so high-batch cells are actually captured.")
    parser.add_argument("--disable-radix-cache", action="store_true", default=False,
                        help="Disable radix cache (prefix sharing) to simulate different prompts per request")
    parser.add_argument("--disable-layerwise-nvtx-marker", action="store_true", default=False,
                        help="Disable per-layer NVTX markers (reduces nsys trace size from ~100MB to ~3MB)")
    parser.add_argument("--gpu-monitor", action="store_true", default=False,
                        help="Enable nvidia-smi GPU monitoring during measurement (disabled by default)")

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
        "disable_radix_cache": args.disable_radix_cache,
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
        "cuda_graph_max_bs": args.cuda_graph_max_bs,
    }
    for k, v in optional_kwargs.items():
        if v is not None:
            engine_kwargs[k] = v

    # report-14: env-gated override to match gray's build default (triton_attention_num_kv_splits=8)
    import os as _os
    _kvs = _os.environ.get("TRITON_KV_SPLITS")
    if _kvs:
        engine_kwargs["triton_attention_num_kv_splits"] = int(_kvs)
    # report-14: gray-era commit renamed cuda_graph_max_bs -> cuda_graph_max_bs_decode
    if _os.environ.get("SGLANG_CG_DECODE_FIELD") and "cuda_graph_max_bs" in engine_kwargs:
        engine_kwargs["cuda_graph_max_bs_decode"] = engine_kwargs.pop("cuda_graph_max_bs")

    engine_kwargs["enable_layerwise_nvtx_marker"] = not args.disable_layerwise_nvtx_marker

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
        enable_gpu_monitor=args.gpu_monitor,
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
    print(f"{'Size':>8} {'median(ms)':>12} {'median(ms)':>12} {'median(ms)':>12} {'median(ms)':>12} {'(tok/s)':>20}")
    print("-" * 100)

    for batch_size in sorted(results.keys()):
        r = results[batch_size]
        print(f"{batch_size:>8} "
              f"{r['ttft_median_ms']:>12.2f} "
              f"{r['tpot_median_ms']:>12.2f} "
              f"{r['decode_time_median_ms']:>12.0f} "
              f"{r['total_time_median_ms']:>12.0f} "
              f"{r['throughput_tokens_per_sec']:>20.1f}")

    print("=" * 100)
    print("\nNote: TPOT = (total_time - TTFT) / (tokens_generated - 1) - excludes prefill")
    print("      All reported values are medians across repetitions.")


if __name__ == "__main__":
    main()
