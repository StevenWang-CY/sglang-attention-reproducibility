#!/usr/bin/env python3
"""
Parse qwen3vl-log files that contain mixed batch sizes.
Automatically extracts batch size from context and groups timing data accordingly.
"""

import re
import json
import os
import sys
from pathlib import Path
from collections import defaultdict


def parse_decode_timing_block(lines):
    """
    Parse a DECODE STEP TIMING block and extract timing information.
    """
    timing_data = {
        'layer_0': {},
        'layers_avg': {},
        'total': {},
        'breakdown': {}
    }

    i = 0
    while i < len(lines):
        line = lines[i]

        # Parse Layer 0 timings
        if 'Layer 0:' in line:
            i += 1
            while i < len(lines) and 'Layers 1-' not in lines[i] and '──────' not in lines[i]:
                # Parse individual component lines
                match = re.search(r'([\w_]+)\s+([\d.]+)\s*ms', lines[i])
                if match:
                    component = match.group(1)
                    time_ms = float(match.group(2))
                    timing_data['layer_0'][component] = time_ms
                i += 1
            continue

        # Parse averaged layers timings (Layers 1-35)
        if 'Layers 1-' in line:
            # Extract number of layers
            num_layers = int(re.search(r'x (\d+)', line).group(1))
            timing_data['num_layers'] = num_layers

            i += 1
            while i < len(lines) and 'Scheduler:' not in lines[i] and '──────' not in lines[i]:
                # Parse lines like: "qkv_proj                       0.088 x 35 =    3.085 ms"
                if 'x ' + str(num_layers) in lines[i]:
                    # Try to match component name with underscore (e.g., kv_cache_save)
                    match = re.search(r'([\w_]+)\s+([\d.]+)\s+x\s+\d+\s+=\s+([\d.]+)\s*ms', lines[i])
                    if match:
                        component = match.group(1)
                        total_time = float(match.group(3))
                        timing_data['layers_avg'][component] = total_time
                i += 1

        # Parse summary breakdown
        elif 'Attention block (all layers)' in line:
            match = re.search(r'([\d.]+)\s*ms', line)
            if match:
                timing_data['breakdown']['attention_block'] = float(match.group(1))
        elif 'MLP block (all layers)' in line:
            match = re.search(r'([\d.]+)\s*ms', line)
            if match:
                timing_data['breakdown']['mlp_block'] = float(match.group(1))
        elif 'Untimed (embed/logits/overhead)' in line:
            match = re.search(r'([\d.]+)\s*ms', line)
            if match:
                timing_data['breakdown']['others'] = float(match.group(1))
        elif 'TOTAL DECODE STEP (model forward)' in line:
            match = re.search(r'([\d.]+)\s*ms', line)
            if match:
                timing_data['total']['decode_step'] = float(match.group(1))
        elif 'FULL LOOP ITERATION' in line:
            match = re.search(r'([\d.]+)\s*ms', line)
            if match:
                timing_data['total']['full_iteration'] = float(match.group(1))

        i += 1

    return timing_data


def extract_batch_size_before_timing(lines, timing_block_start):
    """
    Extract the batch size from the nearest Decode batch line before the timing block.
    """
    # Search backwards from timing block to find the most recent "Decode batch" line
    for i in range(timing_block_start - 1, max(0, timing_block_start - 100), -1):
        if 'Decode batch' in lines[i] and '#running-req:' in lines[i]:
            match = re.search(r'#running-req:\s*(\d+)', lines[i])
            if match:
                return int(match.group(1))

    return None


def parse_log_file_with_batch_detection(log_path):
    """
    Parse a single log file and extract all decode timing blocks with their batch sizes.
    """
    print(f"  Reading: {log_path.name}")

    with open(log_path, 'r') as f:
        lines = f.readlines()

    # Find all DECODE STEP TIMING blocks
    timing_blocks_with_batch = []

    i = 0
    while i < len(lines):
        if 'DECODE STEP TIMING' in lines[i]:
            # Find the end of this timing block (next "=====" line after some content)
            block_start = i
            i += 2  # Skip the line and the ===== separator
            block_end = i

            # Find the end (look for the next section or end of timing block)
            # Timing blocks end with another ===== line after content
            while block_end < len(lines):
                if '==============' in lines[block_end] and block_end > i + 10:
                    break
                block_end += 1

            # Parse this block
            block_lines = lines[block_start:block_end]
            timing_data = parse_decode_timing_block(block_lines)

            # Extract batch size from context
            batch_size = extract_batch_size_before_timing(lines, block_start)

            if timing_data['breakdown'] and batch_size is not None:
                timing_blocks_with_batch.append({
                    'batch_size': batch_size,
                    'timing': timing_data
                })

            i = block_end
        i += 1

    print(f"    Found {len(timing_blocks_with_batch)} timing blocks")
    return timing_blocks_with_batch


def aggregate_timing_data_by_batch(log_dir):
    """
    Process all log files in a directory and aggregate timing data by batch size.
    """
    log_dir = Path(log_dir)

    # Dictionary to store results: batch_size -> list of timing data
    results_by_batch = defaultdict(list)

    # Process all .log files
    log_files = sorted(log_dir.glob('*.log'))

    if not log_files:
        print(f"No .log files found in {log_dir}")
        return {}

    print(f"\nProcessing {len(log_files)} log file(s) from {log_dir}:")
    print("=" * 70)

    for log_file in log_files:
        timing_blocks = parse_log_file_with_batch_detection(log_file)

        # Group by batch size
        for block in timing_blocks:
            batch_size = block['batch_size']
            timing = block['timing']
            results_by_batch[batch_size].append(timing)

    print("\n" + "=" * 70)
    print(f"Batch sizes found: {sorted(results_by_batch.keys())}")
    print("=" * 70)

    # Aggregate data for each batch size
    aggregated = {}

    for batch_size, timing_list in sorted(results_by_batch.items()):
        if not timing_list:
            continue

        print(f"\nBatch size {batch_size}: {len(timing_list)} samples")

        # Average across all timing blocks
        avg_breakdown = {
            'attention_block': sum(t['breakdown'].get('attention_block', 0) for t in timing_list) / len(timing_list),
            'mlp_block': sum(t['breakdown'].get('mlp_block', 0) for t in timing_list) / len(timing_list),
            'others': sum(t['breakdown'].get('others', 0) for t in timing_list) / len(timing_list),
        }

        avg_total = sum(t['total'].get('decode_step', 0) for t in timing_list) / len(timing_list)

        # Detailed breakdown (from Layer 0 + averaged layers)
        detailed = {}
        if timing_list[0]['layers_avg'] or timing_list[0]['layer_0']:
            # Collect all unique component keys from all timing blocks
            all_component_keys = set()
            for t in timing_list:
                all_component_keys.update(t.get('layer_0', {}).keys())
                all_component_keys.update(t.get('layers_avg', {}).keys())

            # Process each component found
            for key in sorted(all_component_keys):
                # Sum Layer 0 + Layers 1-35 for each component
                layer0_values = [t['layer_0'].get(key, 0) for t in timing_list if key in t.get('layer_0', {})]
                layers_avg_values = [t['layers_avg'].get(key, 0) for t in timing_list if key in t.get('layers_avg', {})]

                # Average Layer 0 contribution
                layer0_avg = sum(layer0_values) / len(layer0_values) if layer0_values else 0
                # Average Layers 1-35 contribution
                layers_avg = sum(layers_avg_values) / len(layers_avg_values) if layers_avg_values else 0

                # Total = Layer 0 + Layers 1-35
                detailed[key] = layer0_avg + layers_avg

        aggregated[batch_size] = {
            'batch_size': batch_size,
            'decode_latency_ms': avg_total,
            'breakdown_simple': avg_breakdown,
            'breakdown_detailed': detailed,
            'num_samples': len(timing_list)
        }

    return aggregated


def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_qwen3vl_logs.py <log_directory>")
        print("\nExample:")
        print("  python parse_qwen3vl_logs.py qwen3vl-log/")
        sys.exit(1)

    log_dir = sys.argv[1]

    if not os.path.isdir(log_dir):
        print(f"Error: {log_dir} is not a valid directory")
        sys.exit(1)

    print(f"\n{'=' * 70}")
    print(f"Parsing qwen3vl logs from: {log_dir}")
    print(f"{'=' * 70}")

    results = aggregate_timing_data_by_batch(log_dir)

    if not results:
        print("\nNo timing data found!")
        sys.exit(1)

    # Sort by batch size
    sorted_results = dict(sorted(results.items()))

    # Save to JSON
    output_file = Path(log_dir) / 'timing_data.json'
    with open(output_file, 'w') as f:
        json.dump(sorted_results, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Results saved to: {output_file}")
    print(f"\n{'=' * 70}")
    print(f"SUMMARY TABLE")
    print(f"{'=' * 70}")
    print(f"{'Batch':<8} {'Latency':<12} {'Attn':<10} {'MLP':<10} {'Others':<10} {'Samples':<8}")
    print(f"{'Size':<8} {'(ms)':<12} {'(ms)':<10} {'(ms)':<10} {'(ms)':<10} {'#':<8}")
    print("-" * 70)

    for batch_size, data in sorted_results.items():
        latency = data['decode_latency_ms']
        attn = data['breakdown_simple']['attention_block']
        mlp = data['breakdown_simple']['mlp_block']
        others = data['breakdown_simple']['others']
        samples = data['num_samples']

        print(f"{batch_size:<8} {latency:<12.2f} {attn:<10.2f} {mlp:<10.2f} {others:<10.2f} {samples:<8}")

    print("=" * 70)
    print("\nNext steps:")
    print(f"  python plot_latency_breakdown.py {output_file}")


if __name__ == '__main__':
    main()
