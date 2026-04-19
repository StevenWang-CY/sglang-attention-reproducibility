#!/usr/bin/env python3
"""
Measure TTFT (time to first token) and ITL (inter-token latency)
for SGLang server using streaming responses.

Usage:
    python measure_latency.py [request_json_path] [--url URL] [--max-tokens N] [--repeat N]

Examples:
    python measure_latency.py                                    # simple text prompt
    python measure_latency.py html_request/request_005_.../request.json  # existing request
    python measure_latency.py --repeat 3                         # run 3 times and average
    python measure_latency.py --repeat 5 --log-dir latency_logs  # save to custom folder
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import requests


def get_server_info(url: str, retries: int = 6, retry_interval: int = 10) -> dict:
    """Query SGLang server for backend and config info (retries during JIT warmup)."""
    for attempt in range(retries):
        try:
            resp = requests.get(f"{url}/get_server_info", timeout=30)
            resp.raise_for_status()
            info = resp.json()
            return {
                "attention_backend": info.get("attention_backend", "unknown"),
                "enable_tree_sparse": info.get("enable_tree_sparse", False),
                "tree_sparse_top_k": info.get("tree_sparse_top_k"),
                "tree_sparse_min_seq_len": info.get("tree_sparse_min_seq_len"),
                "tree_sparse_recent_tokens": info.get("tree_sparse_recent_tokens"),
                "tree_sparse_shared_selection": info.get("tree_sparse_shared_selection", False),
                "disable_cuda_graph": info.get("disable_cuda_graph", False),
                "context_length": info.get("context_length"),
                "dtype": info.get("dtype"),
            }
        except Exception as e:
            if attempt < retries - 1:
                print(f"  Server info request failed (attempt {attempt + 1}/{retries}): {e}")
                print(f"  Retrying in {retry_interval}s (server may be JIT compiling)...")
                time.sleep(retry_interval)
            else:
                return {"error": str(e)}


def stream_and_measure(url: str, payload: dict) -> dict:
    """
    Send a streaming request and measure token-level timing.

    Returns dict with:
        ttft: time to first token (seconds)
        itl_list: list of inter-token latencies (seconds)
        itl_mean: mean ITL
        itl_p50: median ITL
        itl_p99: P99 ITL
        total_time: total generation time (seconds)
        num_tokens: number of generated tokens
        tokens_per_sec: throughput
        output_text: generated text
        prompt_tokens: number of prompt tokens (from server response)
        completion_tokens: number of completion tokens (from server response)
    """
    # Ensure streaming is on
    payload["stream"] = True

    headers = {"Content-Type": "application/json"}

    t_start = time.perf_counter()
    t_first_token = None
    token_times = []
    output_chunks = []
    num_tokens = 0
    prompt_tokens = None
    completion_tokens = None

    # Request usage stats in the final SSE chunk so we get prompt_tokens
    payload.setdefault("stream_options", {})["include_usage"] = True

    resp = requests.post(
        f"{url}/v1/chat/completions",
        json=payload,
        headers=headers,
        stream=True,
        timeout=600,
    )
    resp.raise_for_status()

    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        # SSE format: "data: {...}" or "data: [DONE]"
        if not line.startswith("data: "):
            continue
        data_str = line[len("data: "):]
        if data_str.strip() == "[DONE]":
            break

        t_now = time.perf_counter()

        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        # Try to get usage info from the chunk
        usage = chunk.get("usage")
        if usage:
            prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
            completion_tokens = usage.get("completion_tokens", completion_tokens)

        # Extract token content
        choices = chunk.get("choices", [])
        if not choices:
            continue
        delta = choices[0].get("delta", {})
        content = delta.get("content", "")

        if content:
            if t_first_token is None:
                t_first_token = t_now
            token_times.append(t_now)
            output_chunks.append(content)
            num_tokens += 1

    t_end = time.perf_counter()

    if t_first_token is None:
        return {
            "error": "No tokens received",
            "total_time": t_end - t_start,
        }

    # Compute metrics
    ttft = t_first_token - t_start
    total_time = t_end - t_start

    # Inter-token latencies (between consecutive tokens)
    itl_list = []
    for i in range(1, len(token_times)):
        itl_list.append(token_times[i] - token_times[i - 1])

    itl_mean = sum(itl_list) / len(itl_list) if itl_list else 0
    sorted_itl = sorted(itl_list)
    itl_p50 = sorted_itl[len(sorted_itl) // 2] if sorted_itl else 0
    itl_p99 = sorted_itl[int(len(sorted_itl) * 0.99)] if sorted_itl else 0

    output_text = "".join(output_chunks)

    return {
        "ttft": ttft,
        "itl_list": itl_list,
        "itl_mean": itl_mean,
        "itl_p50": itl_p50,
        "itl_p99": itl_p99,
        "total_time": total_time,
        "num_tokens": num_tokens,
        "tokens_per_sec": num_tokens / total_time if total_time > 0 else 0,
        "output_text": output_text,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


def make_simple_payload(model_name: str, max_tokens: int) -> dict:
    """Create a simple test payload."""
    return {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": "Write a short paragraph about the history of the internet.",
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }


def load_request_payload(path: str) -> dict:
    """Load an existing request.json file."""
    with open(path) as f:
        data = json.load(f)

    # The OpenAI API uses top-level fields, not nested "parameters"
    # Merge parameters into top level if present
    if "parameters" in data:
        params = data.pop("parameters")
        for k, v in params.items():
            if k not in data:
                data[k] = v

    # Normalize max_completion_tokens -> max_tokens (OpenAI compat)
    if "max_completion_tokens" in data and "max_tokens" not in data:
        val = data.pop("max_completion_tokens")
        if val is not None:
            data["max_tokens"] = val

    return data


def format_results(results: dict, run_idx: int = None) -> str:
    """Format measurement results as a string."""
    prefix = f"[Run {run_idx}] " if run_idx is not None else ""
    lines = []

    if "error" in results:
        lines.append(f"{prefix}ERROR: {results['error']}")
        return "\n".join(lines)

    lines.append(f"{prefix}TTFT (time to first token):  {results['ttft']*1000:8.1f} ms")
    lines.append(f"{prefix}ITL mean:                    {results['itl_mean']*1000:8.2f} ms")
    lines.append(f"{prefix}ITL median (P50):            {results['itl_p50']*1000:8.2f} ms")
    lines.append(f"{prefix}ITL P99:                     {results['itl_p99']*1000:8.2f} ms")
    lines.append(f"{prefix}Total time:                  {results['total_time']:8.2f} s")
    lines.append(f"{prefix}Tokens generated:            {results['num_tokens']:8d}")
    if results.get("prompt_tokens"):
        lines.append(f"{prefix}Prompt tokens (input):       {results['prompt_tokens']:8d}")
    if results.get("completion_tokens"):
        lines.append(f"{prefix}Completion tokens (output):  {results['completion_tokens']:8d}")
    lines.append(f"{prefix}Throughput:                  {results['tokens_per_sec']:8.1f} tok/s")
    lines.append(f"{prefix}Output preview: {results['output_text'][:120]}...")
    return "\n".join(lines)


def save_results(log_dir: str, all_results: list, config: dict):
    """Save detailed results to JSON and a human-readable summary."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Build summary
    valid = [r for r in all_results if "error" not in r]

    summary = {
        "timestamp": timestamp,
        "config": config,
        "num_runs": len(all_results),
        "num_successful": len(valid),
        "runs": [],
    }

    for i, r in enumerate(all_results):
        run_data = {k: v for k, v in r.items() if k != "itl_list"}
        run_data["run_idx"] = i + 1
        if "itl_list" in r:
            run_data["itl_count"] = len(r["itl_list"])
        summary["runs"].append(run_data)

    if valid:
        summary["aggregate"] = {
            "median_ttft_ms": statistics.median(r["ttft"] for r in valid) * 1000,
            "median_itl_mean_ms": statistics.median(r["itl_mean"] for r in valid) * 1000,
            "median_itl_p50_ms": statistics.median(r["itl_p50"] for r in valid) * 1000,
            "median_itl_p99_ms": statistics.median(r["itl_p99"] for r in valid) * 1000,
            "median_throughput_tps": statistics.median(r["tokens_per_sec"] for r in valid),
            "median_total_time_s": statistics.median(r["total_time"] for r in valid),
            "median_tokens": statistics.median(r["num_tokens"] for r in valid),
            "min_ttft_ms": min(r["ttft"] for r in valid) * 1000,
            "max_ttft_ms": max(r["ttft"] for r in valid) * 1000,
        }

    # Save JSON
    json_path = os.path.join(log_dir, f"latency_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Save human-readable text
    txt_path = os.path.join(log_dir, f"latency_{timestamp}.txt")
    with open(txt_path, "w") as f:
        f.write(f"Latency Measurement - {timestamp}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Server: {config.get('url', 'unknown')}\n")
        f.write(f"Model: {config.get('model', 'unknown')}\n")
        f.write(f"Attention backend: {config.get('attention_backend', 'unknown')}\n")
        f.write(f"CUDA graph: {'disabled' if config.get('disable_cuda_graph') else 'enabled'}\n")
        if config.get("enable_tree_sparse"):
            selection = "shared" if config.get("tree_sparse_shared_selection") else "per-layer"
            f.write(f"Tree sparse: top_k={config.get('tree_sparse_top_k')}, "
                    f"min_seq_len={config.get('tree_sparse_min_seq_len')}, "
                    f"recent={config.get('tree_sparse_recent_tokens')}, "
                    f"selection={selection}\n")
        f.write(f"Request: {config.get('request_json', 'simple prompt')}\n")
        f.write(f"Runs: {len(all_results)} ({len(valid)} successful)\n")

        # Show input/output token counts from first valid run
        if valid:
            r0 = valid[0]
            if r0.get("prompt_tokens"):
                f.write(f"Input tokens: {r0['prompt_tokens']}\n")
            f.write(f"Output tokens (median): {int(statistics.median(r['num_tokens'] for r in valid))}\n")

        f.write("=" * 60 + "\n\n")

        for i, r in enumerate(all_results):
            f.write(format_results(r, run_idx=i + 1) + "\n\n")

        if valid and len(valid) > 1:
            agg = summary["aggregate"]
            f.write("=" * 60 + "\n")
            f.write(f"SUMMARY ({len(valid)} successful runs)\n")
            f.write("=" * 60 + "\n")
            f.write(f"Median TTFT:       {agg['median_ttft_ms']:8.1f} ms  (min={agg['min_ttft_ms']:.1f}, max={agg['max_ttft_ms']:.1f})\n")
            f.write(f"Median ITL mean:   {agg['median_itl_mean_ms']:8.2f} ms\n")
            f.write(f"Median ITL P50:    {agg['median_itl_p50_ms']:8.2f} ms\n")
            f.write(f"Median ITL P99:    {agg['median_itl_p99_ms']:8.2f} ms\n")
            f.write(f"Median throughput: {agg['median_throughput_tps']:8.1f} tok/s\n")
            f.write(f"Median total time: {agg['median_total_time_s']:8.2f} s\n")
            f.write(f"Median tokens:     {agg['median_tokens']:8.0f}\n")

    # Also save per-token ITL data for detailed analysis
    itl_path = os.path.join(log_dir, f"itl_details_{timestamp}.json")
    itl_data = []
    for i, r in enumerate(all_results):
        if "itl_list" in r:
            itl_data.append({
                "run_idx": i + 1,
                "itl_ms": [t * 1000 for t in r["itl_list"]],
            })
    with open(itl_path, "w") as f:
        json.dump(itl_data, f)

    return json_path, txt_path, itl_path


def main():
    parser = argparse.ArgumentParser(description="Measure TTFT and ITL for SGLang")
    parser.add_argument(
        "request_json",
        nargs="?",
        default=None,
        help="Path to request.json (optional, uses simple prompt if omitted)",
    )
    parser.add_argument(
        "--url", default="http://localhost:8000", help="Server URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=None, help="Max tokens to generate (overrides request JSON if specified)"
    )
    parser.add_argument(
        "--model",
        default="Qwen3-VL-8B-Instruct",
        help="Model name (default: Qwen3-VL-8B-Instruct)",
    )
    parser.add_argument(
        "--repeat", type=int, default=1, help="Number of runs (default: 1)"
    )
    parser.add_argument(
        "--log-dir",
        default="latency_logs",
        help="Directory to save results (default: latency_logs)",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Don't save results to files"
    )
    parser.add_argument(
        "--warmup", type=int, default=0, help="Number of warmup runs to discard (default: 0)"
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Ignore EOS tokens to force exact token count generation (default: False)",
    )
    args = parser.parse_args()

    # Query server info for backend detection
    print(f"Querying server info from {args.url}...")
    server_info = get_server_info(args.url)
    if "error" in server_info:
        print(f"Warning: Could not get server info: {server_info['error']}")
    else:
        backend = server_info["attention_backend"]
        tree_sparse = server_info["enable_tree_sparse"]
        print(f"Attention backend: {backend}")
        if tree_sparse:
            selection = "shared" if server_info.get("tree_sparse_shared_selection") else "per-layer"
            print(f"Tree sparse: top_k={server_info['tree_sparse_top_k']}, "
                  f"min_seq_len={server_info['tree_sparse_min_seq_len']}, "
                  f"recent={server_info['tree_sparse_recent_tokens']}, "
                  f"selection={selection}")
        print(f"CUDA graph: {'disabled' if server_info.get('disable_cuda_graph') else 'enabled'}")

    # Build payload
    if args.request_json:
        payload = load_request_payload(args.request_json)
        print(f"Loaded request from: {args.request_json}")
    else:
        payload = make_simple_payload(args.model, args.max_tokens)
        print(f"Using simple test prompt")

    # Apply max_tokens override if explicitly specified on command line
    if args.max_tokens is not None:
        payload["max_tokens"] = args.max_tokens
        print(f"Overriding max_tokens with command-line value: {args.max_tokens}")

    # Ensure max_tokens is set (safety default only if not in payload)
    if "max_tokens" not in payload or payload.get("max_tokens") is None:
        payload["max_tokens"] = 256  # safety default for runaway generation
        print(f"Using safety default max_tokens: 256")

    # Override model name
    if args.model:
        payload["model"] = args.model

    # Force exact token generation by ignoring EOS
    if args.ignore_eos:
        payload["ignore_eos"] = True
        print(f"Forcing exact token generation (ignore_eos=True)")

    total_runs = args.warmup + args.repeat
    print(f"Server: {args.url}")
    print(f"Model: {payload.get('model', 'unknown')}")
    if args.warmup > 0:
        print(f"Warmup runs: {args.warmup} (discarded)")
    print(f"Measured runs: {args.repeat}")
    if not args.no_save:
        print(f"Log dir: {args.log_dir}")
    print("=" * 60)

    all_results = []
    for run in range(total_runs):
        is_warmup = run < args.warmup
        label = f"Warmup {run + 1}/{args.warmup}" if is_warmup else f"Run {run - args.warmup + 1}/{args.repeat}"
        print(f"\n--- {label} ---")

        result = stream_and_measure(args.url, payload.copy())

        if is_warmup:
            print(format_results(result, run_idx=None))
            print("(warmup, discarded)")
        else:
            all_results.append(result)
            print(format_results(result, run_idx=run - args.warmup + 1))

    # Summary for multiple runs
    valid = [r for r in all_results if "error" not in r]
    if len(valid) > 1:
        print("\n" + "=" * 60)
        print(f"SUMMARY ({len(valid)} successful runs)")
        print("=" * 60)
        med_ttft = statistics.median(r["ttft"] for r in valid)
        min_ttft = min(r["ttft"] for r in valid)
        max_ttft = max(r["ttft"] for r in valid)
        med_itl = statistics.median(r["itl_mean"] for r in valid)
        med_tps = statistics.median(r["tokens_per_sec"] for r in valid)
        med_tokens = statistics.median(r["num_tokens"] for r in valid)
        print(f"Median TTFT:       {med_ttft*1000:8.1f} ms  (min={min_ttft*1000:.1f}, max={max_ttft*1000:.1f})")
        print(f"Median ITL mean:   {med_itl*1000:8.2f} ms")
        print(f"Median throughput: {med_tps:8.1f} tok/s")
        print(f"Median tokens:     {med_tokens:8.0f}")
        if valid[0].get("prompt_tokens"):
            print(f"Input tokens:      {valid[0]['prompt_tokens']:8d}")

    # Save results
    if not args.no_save and all_results:
        config = {
            "url": args.url,
            "model": payload.get("model", "unknown"),
            "request_json": args.request_json or "simple_prompt",
            "max_tokens": payload.get("max_tokens"),
            "repeat": args.repeat,
            "warmup": args.warmup,
            **server_info,  # includes attention_backend, tree_sparse config, etc.
        }

        # Build backend subfolder name: e.g. "trtllm_mha", "tree_sparse", "flashinfer_no_cudagraph"
        backend_name = server_info.get("attention_backend", "unknown")
        if server_info.get("disable_cuda_graph"):
            backend_name += "_no_cudagraph"
        log_dir = os.path.join(args.log_dir, backend_name)

        json_path, txt_path, itl_path = save_results(log_dir, all_results, config)
        print(f"\nResults saved to:")
        print(f"  Summary:    {txt_path}")
        print(f"  JSON:       {json_path}")
        print(f"  ITL detail: {itl_path}")


if __name__ == "__main__":
    main()
