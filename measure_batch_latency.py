#!/usr/bin/env python3
"""
Measure batch inference throughput and per-request latency for SGLang server.

Sends N concurrent requests (a "batch"), repeats for multiple rounds,
and reports aggregate throughput + per-request metrics.

Supports single or multiple request files:
  - Single file:    all N concurrent requests use the same payload
  - Multiple files: payloads are distributed round-robin across the N requests
  - Directory:      loads all *.json files found (recursively)

Usage:
    # 10 copies of the same request
    python measure_batch_latency.py request.json --concurrency 10 --repeat 3

    # 10 concurrent requests, cycling through 3 different request files
    python measure_batch_latency.py req1.json req2.json req3.json --concurrency 10 --repeat 3

    # Load all request.json files under a directory
    python measure_batch_latency.py html_request/ --concurrency 10 --repeat 3
"""

import argparse
import glob
import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


def get_server_info(url: str, retries: int = 6, retry_interval: int = 10) -> dict:
    """Query SGLang server for backend and config info."""
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


def load_request_payload(path: str) -> dict:
    """Load a request.json file and normalize it for the OpenAI API."""
    with open(path) as f:
        data = json.load(f)

    if "parameters" in data:
        params = data.pop("parameters")
        for k, v in params.items():
            if k not in data:
                data[k] = v

    if "max_completion_tokens" in data and "max_tokens" not in data:
        val = data.pop("max_completion_tokens")
        if val is not None:
            data["max_tokens"] = val

    return data


def send_request_streaming(url: str, payload: dict, request_id: int) -> dict:
    """Send a single streaming request and measure per-request metrics."""
    payload = payload.copy()
    payload["stream"] = True
    payload.setdefault("stream_options", {})["include_usage"] = True

    t_start = time.perf_counter()
    t_first_token = None
    token_times = []
    output_chunks = []
    num_tokens = 0
    prompt_tokens = None
    completion_tokens = None

    try:
        resp = requests.post(
            f"{url}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            stream=True,
            timeout=600,
        )
        resp.raise_for_status()

        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data_str = line[len("data: "):]
            if data_str.strip() == "[DONE]":
                break

            t_now = time.perf_counter()
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            usage = chunk.get("usage")
            if usage:
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                completion_tokens = usage.get("completion_tokens", completion_tokens)

            choices = chunk.get("choices", [])
            if not choices:
                continue
            content = choices[0].get("delta", {}).get("content", "")
            if content:
                if t_first_token is None:
                    t_first_token = t_now
                token_times.append(t_now)
                output_chunks.append(content)
                num_tokens += 1

    except Exception as e:
        return {
            "request_id": request_id,
            "error": str(e),
            "total_time": time.perf_counter() - t_start,
        }

    t_end = time.perf_counter()

    if t_first_token is None:
        return {
            "request_id": request_id,
            "error": "No tokens received",
            "total_time": t_end - t_start,
        }

    ttft = t_first_token - t_start
    total_time = t_end - t_start
    itl_list = [token_times[i] - token_times[i - 1] for i in range(1, len(token_times))]
    itl_mean = sum(itl_list) / len(itl_list) if itl_list else 0

    return {
        "request_id": request_id,
        "ttft": ttft,
        "itl_mean": itl_mean,
        "total_time": total_time,
        "num_tokens": num_tokens,
        "tokens_per_sec": num_tokens / total_time if total_time > 0 else 0,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "output_text": "".join(output_chunks),
    }


def send_request_non_streaming(url: str, payload: dict, request_id: int) -> dict:
    """Send a single non-streaming request and measure total latency."""
    payload = payload.copy()
    payload["stream"] = False

    t_start = time.perf_counter()
    try:
        resp = requests.post(
            f"{url}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=600,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return {
            "request_id": request_id,
            "error": str(e),
            "total_time": time.perf_counter() - t_start,
        }

    t_end = time.perf_counter()
    total_time = t_end - t_start

    usage = data.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    num_tokens = completion_tokens or 0

    choices = data.get("choices", [])
    output_text = choices[0]["message"]["content"] if choices else ""

    return {
        "request_id": request_id,
        "total_time": total_time,
        "num_tokens": num_tokens,
        "tokens_per_sec": num_tokens / total_time if total_time > 0 else 0,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "output_text": output_text,
    }


def run_batch(url: str, payloads: list[dict], concurrency: int, streaming: bool) -> dict:
    """Send a batch of concurrent requests and return aggregate results.

    payloads: list of request dicts. Distributed round-robin across the
              concurrency slots (so if you pass 3 payloads with concurrency=10,
              slots get payloads [0,1,2,0,1,2,0,1,2,0]).
    """
    send_fn = send_request_streaming if streaming else send_request_non_streaming

    batch_start = time.perf_counter()
    results = []

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(send_fn, url, payloads[i % len(payloads)], i): i
            for i in range(concurrency)
        }
        for future in as_completed(futures):
            results.append(future.result())

    batch_end = time.perf_counter()
    batch_time = batch_end - batch_start

    results.sort(key=lambda r: r["request_id"])

    valid = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]

    total_output_tokens = sum(r["num_tokens"] for r in valid)
    total_prompt_tokens = sum(r.get("prompt_tokens") or 0 for r in valid)

    batch_summary = {
        "batch_time": batch_time,
        "concurrency": concurrency,
        "num_successful": len(valid),
        "num_errors": len(errors),
        "total_output_tokens": total_output_tokens,
        "total_prompt_tokens": total_prompt_tokens,
        "batch_output_throughput": total_output_tokens / batch_time if batch_time > 0 else 0,
        "batch_total_throughput": (total_prompt_tokens + total_output_tokens) / batch_time if batch_time > 0 else 0,
        "per_request": results,
    }

    if valid:
        batch_summary["mean_request_latency"] = statistics.mean(r["total_time"] for r in valid)
        batch_summary["median_request_latency"] = statistics.median(r["total_time"] for r in valid)
        batch_summary["mean_tokens_per_request"] = statistics.mean(r["num_tokens"] for r in valid)
        if streaming:
            ttfts = [r["ttft"] for r in valid if "ttft" in r]
            if ttfts:
                batch_summary["mean_ttft"] = statistics.mean(ttfts)
                batch_summary["median_ttft"] = statistics.median(ttfts)

    return batch_summary


def print_batch_summary(batch: dict, round_idx: int = None):
    """Print a human-readable summary of one batch round."""
    prefix = f"[Round {round_idx}] " if round_idx is not None else ""
    print(f"{prefix}Batch time:             {batch['batch_time']:8.2f} s")
    print(f"{prefix}Requests:               {batch['num_successful']} ok, {batch['num_errors']} errors")
    print(f"{prefix}Total output tokens:    {batch['total_output_tokens']:8d}")
    print(f"{prefix}Output throughput:      {batch['batch_output_throughput']:8.1f} tok/s")
    print(f"{prefix}Total throughput:       {batch['batch_total_throughput']:8.1f} tok/s  (prompt+output)")
    if "mean_request_latency" in batch:
        print(f"{prefix}Mean request latency:   {batch['mean_request_latency']:8.2f} s")
        print(f"{prefix}Median request latency: {batch['median_request_latency']:8.2f} s")
    if "mean_ttft" in batch:
        print(f"{prefix}Mean TTFT:              {batch['mean_ttft']*1000:8.1f} ms")
        print(f"{prefix}Median TTFT:            {batch['median_ttft']*1000:8.1f} ms")


def save_results(log_dir: str, all_batches: list, config: dict):
    """Save results to JSON and human-readable text."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Build saveable summary (strip full output text to keep file small)
    save_batches = []
    for b in all_batches:
        b_copy = {k: v for k, v in b.items() if k != "per_request"}
        per_req = []
        for r in b["per_request"]:
            r_copy = dict(r)
            if "output_text" in r_copy:
                r_copy["output_text"] = r_copy["output_text"][:200] + "..."
            per_req.append(r_copy)
        b_copy["per_request"] = per_req
        save_batches.append(b_copy)

    summary = {
        "timestamp": timestamp,
        "config": config,
        "num_rounds": len(all_batches),
        "batches": save_batches,
    }

    valid_batches = [b for b in all_batches if b["num_successful"] > 0]
    if valid_batches:
        summary["aggregate"] = {
            "median_batch_time": statistics.median(b["batch_time"] for b in valid_batches),
            "median_output_throughput": statistics.median(b["batch_output_throughput"] for b in valid_batches),
            "median_total_throughput": statistics.median(b["batch_total_throughput"] for b in valid_batches),
            "median_request_latency": statistics.median(b["mean_request_latency"] for b in valid_batches if "mean_request_latency" in b),
        }

    json_path = os.path.join(log_dir, f"batch_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    txt_path = os.path.join(log_dir, f"batch_{timestamp}.txt")
    with open(txt_path, "w") as f:
        f.write(f"Batch Inference Measurement - {timestamp}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Server:      {config.get('url')}\n")
        f.write(f"Model:       {config.get('model')}\n")
        f.write(f"Concurrency: {config.get('concurrency')}\n")
        f.write(f"Rounds:      {config.get('repeat')}\n")
        f.write(f"Streaming:   {config.get('streaming')}\n")
        req_files = config.get("request_files", [])
        f.write(f"Requests:    {config.get('num_request_files')} file(s)\n")
        for rf in req_files:
            f.write(f"             - {rf}\n")
        backend = config.get("attention_backend", "unknown")
        f.write(f"Backend:     {backend}\n")
        f.write("=" * 60 + "\n\n")

        for i, batch in enumerate(all_batches):
            f.write(f"--- Round {i + 1} ---\n")
            f.write(f"Batch time:             {batch['batch_time']:8.2f} s\n")
            f.write(f"Requests:               {batch['num_successful']} ok, {batch['num_errors']} errors\n")
            f.write(f"Total output tokens:    {batch['total_output_tokens']:8d}\n")
            f.write(f"Output throughput:      {batch['batch_output_throughput']:8.1f} tok/s\n")
            f.write(f"Total throughput:       {batch['batch_total_throughput']:8.1f} tok/s\n")
            if "mean_request_latency" in batch:
                f.write(f"Mean request latency:   {batch['mean_request_latency']:8.2f} s\n")
            if "mean_ttft" in batch:
                f.write(f"Mean TTFT:              {batch['mean_ttft']*1000:8.1f} ms\n")
            f.write("\n")

        if valid_batches and len(valid_batches) > 1:
            agg = summary["aggregate"]
            f.write("=" * 60 + "\n")
            f.write(f"SUMMARY ({len(valid_batches)} rounds)\n")
            f.write("=" * 60 + "\n")
            f.write(f"Median batch time:        {agg['median_batch_time']:8.2f} s\n")
            f.write(f"Median output throughput: {agg['median_output_throughput']:8.1f} tok/s\n")
            f.write(f"Median total throughput:  {agg['median_total_throughput']:8.1f} tok/s\n")
            f.write(f"Median request latency:  {agg['median_request_latency']:8.2f} s\n")

    return json_path, txt_path


def main():
    parser = argparse.ArgumentParser(
        description="Measure batch inference throughput for SGLang",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 10 concurrent identical requests, repeated 3 times
  python measure_batch_latency.py request.json --concurrency 10 --repeat 3

  # Compare sequential vs batch
  python measure_batch_latency.py request.json --concurrency 1 --repeat 5
  python measure_batch_latency.py request.json --concurrency 10 --repeat 5

  # Non-streaming (simpler, less overhead)
  python measure_batch_latency.py request.json --concurrency 10 --no-stream

  # Custom model and server
  python measure_batch_latency.py request.json --model Qwen3-VL-8B-Instruct --url http://localhost:8000
""",
    )
    parser.add_argument(
        "request_json",
        nargs="+",
        help="Path(s) to request.json file(s), or a directory containing them",
    )
    parser.add_argument(
        "--url", default="http://localhost:8000",
        help="Server URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=10,
        help="Number of concurrent requests per batch (default: 10)",
    )
    parser.add_argument(
        "--repeat", type=int, default=3,
        help="Number of batch rounds (default: 3)",
    )
    parser.add_argument(
        "--warmup", type=int, default=0,
        help="Number of warmup rounds to discard (default: 0)",
    )
    parser.add_argument(
        "--model", default=None,
        help="Override model name in the request",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=None,
        help="Override max_tokens in the request",
    )
    parser.add_argument(
        "--no-stream", action="store_true",
        help="Use non-streaming requests (no TTFT/ITL, but less client overhead)",
    )
    parser.add_argument(
        "--log-dir", default="batch_logs",
        help="Directory to save results (default: batch_logs)",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Don't save results to files",
    )
    args = parser.parse_args()

    streaming = not args.no_stream

    # Server info
    print(f"Querying server info from {args.url}...")
    server_info = get_server_info(args.url)
    if "error" in server_info:
        print(f"Warning: Could not get server info: {server_info['error']}")
    else:
        print(f"Attention backend: {server_info['attention_backend']}")
        print(f"CUDA graph: {'disabled' if server_info.get('disable_cuda_graph') else 'enabled'}")

    # Resolve request file paths: expand directories and globs
    request_paths = []
    for p in args.request_json:
        if os.path.isdir(p):
            found = sorted(glob.glob(os.path.join(p, "**", "*.json"), recursive=True))
            if not found:
                print(f"Warning: no .json files found in {p}")
            request_paths.extend(found)
        else:
            request_paths.append(p)

    if not request_paths:
        print("Error: no request files found.")
        return

    # Load all payloads
    payloads = []
    for rp in request_paths:
        payload = load_request_payload(rp)
        if args.model:
            payload["model"] = args.model
        if args.max_tokens is not None:
            payload["max_tokens"] = args.max_tokens
        elif "max_tokens" not in payload:
            payload["max_tokens"] = 256
        payloads.append(payload)

    print(f"\nLoaded {len(payloads)} request file(s):")
    for rp in request_paths:
        print(f"  - {rp}")

    if len(payloads) == 1:
        print(f"Mode: all {args.concurrency} slots use the same request")
    elif len(payloads) >= args.concurrency:
        print(f"Mode: each slot gets a different request ({len(payloads)} files, {args.concurrency} slots)")
    else:
        print(f"Mode: {len(payloads)} files distributed round-robin across {args.concurrency} slots")

    print(f"\nServer:      {args.url}")
    print(f"Model:       {payloads[0].get('model', 'unknown')}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Streaming:   {streaming}")
    if args.warmup > 0:
        print(f"Warmup:      {args.warmup} rounds (discarded)")
    print(f"Rounds:      {args.repeat}")
    print("=" * 60)

    total_rounds = args.warmup + args.repeat
    all_batches = []

    for rnd in range(total_rounds):
        is_warmup = rnd < args.warmup
        label = (
            f"Warmup {rnd + 1}/{args.warmup}"
            if is_warmup
            else f"Round {rnd - args.warmup + 1}/{args.repeat}"
        )
        print(f"\n--- {label}: sending {args.concurrency} concurrent requests ---")

        batch = run_batch(args.url, payloads, args.concurrency, streaming)

        if is_warmup:
            print_batch_summary(batch)
            print("(warmup, discarded)")
        else:
            all_batches.append(batch)
            print_batch_summary(batch, round_idx=rnd - args.warmup + 1)

    # Overall summary
    valid_batches = [b for b in all_batches if b["num_successful"] > 0]
    if len(valid_batches) > 1:
        print("\n" + "=" * 60)
        print(f"OVERALL SUMMARY ({len(valid_batches)} rounds, concurrency={args.concurrency})")
        print("=" * 60)
        med_bt = statistics.median(b["batch_time"] for b in valid_batches)
        med_ot = statistics.median(b["batch_output_throughput"] for b in valid_batches)
        med_tt = statistics.median(b["batch_total_throughput"] for b in valid_batches)
        med_rl = statistics.median(b["mean_request_latency"] for b in valid_batches if "mean_request_latency" in b)
        print(f"Median batch time:        {med_bt:8.2f} s")
        print(f"Median output throughput: {med_ot:8.1f} tok/s")
        print(f"Median total throughput:  {med_tt:8.1f} tok/s  (prompt+output)")
        print(f"Median request latency:  {med_rl:8.2f} s")

        if streaming:
            ttft_batches = [b for b in valid_batches if "mean_ttft" in b]
            if ttft_batches:
                med_ttft = statistics.median(b["mean_ttft"] for b in ttft_batches)
                print(f"Median TTFT:             {med_ttft*1000:8.1f} ms")

    # Save
    if not args.no_save and all_batches:
        config = {
            "url": args.url,
            "model": payloads[0].get("model", "unknown"),
            "request_files": request_paths,
            "num_request_files": len(payloads),
            "concurrency": args.concurrency,
            "repeat": args.repeat,
            "warmup": args.warmup,
            "streaming": streaming,
            "max_tokens": payloads[0].get("max_tokens"),
            **server_info,
        }

        backend_name = server_info.get("attention_backend", "unknown")
        if server_info.get("disable_cuda_graph"):
            backend_name += "_no_cudagraph"
        log_dir = os.path.join(args.log_dir, backend_name)

        json_path, txt_path = save_results(log_dir, all_batches, config)
        print(f"\nResults saved to:")
        print(f"  Summary: {txt_path}")
        print(f"  JSON:    {json_path}")


if __name__ == "__main__":
    main()
