#!/usr/bin/env python3
"""
Async version of measure_batch_latency.py for tighter request timing.
Uses asyncio + aiohttp to send truly concurrent requests with minimal timing skew.
"""

import argparse
import asyncio
import glob
import json
import os
import statistics
import time
from pathlib import Path

import aiohttp


async def send_request_async(session, url, payload, request_id):
    """Send a single async request."""
    payload = payload.copy()
    payload["stream"] = False

    t_start = time.perf_counter()
    try:
        async with session.post(
            f"{url}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=600)
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
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


async def run_batch_async(url: str, payloads: list, concurrency: int):
    """Send all requests truly concurrently using asyncio.gather."""
    batch_start = time.perf_counter()

    # Create connector with high connection limit for large batches
    connector = aiohttp.TCPConnector(limit=max(concurrency, 100), limit_per_host=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            send_request_async(session, url, payloads[i % len(payloads)], i)
            for i in range(concurrency)
        ]
        # gather ensures all requests start as simultaneously as possible
        results = await asyncio.gather(*tasks)

    batch_end = time.perf_counter()
    batch_time = batch_end - batch_start

    results = sorted(results, key=lambda r: r["request_id"])

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

    return batch_summary


def print_batch_summary(batch: dict, round_idx: int = None):
    """Print a human-readable summary."""
    prefix = f"[Round {round_idx}] " if round_idx is not None else ""
    print(f"{prefix}Batch time:             {batch['batch_time']:8.2f} s")
    print(f"{prefix}Requests:               {batch['num_successful']} ok, {batch['num_errors']} errors")
    print(f"{prefix}Total output tokens:    {batch['total_output_tokens']:8d}")
    print(f"{prefix}Output throughput:      {batch['batch_output_throughput']:8.1f} tok/s")
    print(f"{prefix}Total throughput:       {batch['batch_total_throughput']:8.1f} tok/s")
    if "mean_request_latency" in batch:
        print(f"{prefix}Mean request latency:   {batch['mean_request_latency']:8.2f} s")


def load_request_payload(path: str) -> dict:
    """Load a request.json file."""
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


async def main_async():
    parser = argparse.ArgumentParser(description="Async batch inference benchmark")
    parser.add_argument("request_json", nargs="+", help="Request file(s) or directory")
    parser.add_argument("--url", default="http://localhost:8000", help="Server URL")
    parser.add_argument("--concurrency", type=int, default=10, help="Concurrent requests")
    parser.add_argument("--repeat", type=int, default=3, help="Number of rounds")
    parser.add_argument("--model", default=None, help="Override model name")
    parser.add_argument("--max-tokens", type=int, default=None, help="Override max_tokens")
    parser.add_argument("--min-tokens", type=int, default=None, help="Override min_tokens (minimum generation length)")
    parser.add_argument("--ignore-eos", action="store_true", help="Ignore EOS token and generate until max_tokens")
    args = parser.parse_args()

    # Resolve request paths
    request_paths = []
    for p in args.request_json:
        if os.path.isdir(p):
            found = sorted(glob.glob(os.path.join(p, "**", "*.json"), recursive=True))
            request_paths.extend(found)
        else:
            request_paths.append(p)

    if not request_paths:
        print("Error: no request files found.")
        return

    # Load payloads
    payloads = []
    for rp in request_paths:
        payload = load_request_payload(rp)
        if args.model:
            payload["model"] = args.model
        if args.max_tokens is not None:
            payload["max_tokens"] = args.max_tokens
        elif "max_tokens" not in payload:
            payload["max_tokens"] = 256
        if args.min_tokens is not None:
            payload["min_tokens"] = args.min_tokens
        if args.ignore_eos:
            payload["ignore_eos"] = True
        payloads.append(payload)

    print(f"\nLoaded {len(payloads)} request file(s)")
    print(f"Server:      {args.url}")
    print(f"Model:       {payloads[0].get('model', 'unknown')}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Rounds:      {args.repeat}")
    print("=" * 60)

    all_batches = []
    for rnd in range(args.repeat):
        print(f"\n--- Round {rnd + 1}/{args.repeat}: sending {args.concurrency} requests ---")
        batch = await run_batch_async(args.url, payloads, args.concurrency)
        all_batches.append(batch)
        print_batch_summary(batch, round_idx=rnd + 1)

    # Summary
    valid_batches = [b for b in all_batches if b["num_successful"] > 0]
    if len(valid_batches) > 1:
        print("\n" + "=" * 60)
        print(f"SUMMARY ({len(valid_batches)} rounds, concurrency={args.concurrency})")
        print("=" * 60)
        med_ot = statistics.median(b["batch_output_throughput"] for b in valid_batches)
        med_tt = statistics.median(b["batch_total_throughput"] for b in valid_batches)
        print(f"Median output throughput: {med_ot:8.1f} tok/s")
        print(f"Median total throughput:  {med_tt:8.1f} tok/s")


if __name__ == "__main__":
    asyncio.run(main_async())
