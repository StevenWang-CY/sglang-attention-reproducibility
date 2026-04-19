#!/usr/bin/env python3
"""
Parse SGLang timing logs to extract decode latency breakdown.
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
            while i < len(lines) and 'Layers 1-' not in lines[i]:
                if 'input_ln' in lines[i]:
                    timing_data['layer_0']['input_ln'] = float(re.search(r'([\d.]+)\s*ms', lines[i]).group(1))
                elif 'qkv_proj' in lines[i]:
                    timing_data['layer_0']['qkv_proj'] = float(re.search(r'([\d.]+)\s*ms', lines[i]).group(1))
                elif 'attention' in lines[i] and 'post_attn' not in lines[i]:
                    timing_data['layer_0']['attention'] = float(re.search(r'([\d.]+)\s*ms', lines[i]).group(1))
                elif 'o_proj' in lines[i]:
                    timing_data['layer_0']['o_proj'] = float(re.search(r'([\d.]+)\s*ms', lines[i]).group(1))
                elif 'post_attn_ln' in lines[i]:
                    timing_data['layer_0']['post_attn_ln'] = float(re.search(r'([\d.]+)\s*ms', lines[i]).group(1))
                elif line.strip().startswith('mlp') and '[MLP' not in line:
                    timing_data['layer_0']['mlp'] = float(re.search(r'([\d.]+)\s*ms', lines[i]).group(1))
                i += 1

        # Parse averaged layers timings (Layers 1-35)
        elif 'Layers 1-' in line:
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


def parse_log_file(log_path):
    """
    Parse a single log file and extract all decode timing blocks.
    """
    with open(log_path, 'r') as f:
        content = f.read()

    # Find all DECODE STEP TIMING blocks
    timing_blocks = []
    lines = content.split('\n')

    i = 0
    while i < len(lines):
        if 'DECODE STEP TIMING' in lines[i]:
            # Find the end of this timing block
            block_start = i
            i += 1
            while i < len(lines) and '==============' not in lines[i]:
                i += 1

            # Parse this block
            block_lines = lines[block_start:i+1]
            timing_data = parse_decode_timing_block(block_lines)

            if timing_data['breakdown']:  # Only add if we got valid data
                timing_blocks.append(timing_data)
        i += 1

    return timing_blocks


def extract_batch_size_from_filename(filename):
    """
    Extract batch size from filename like 'batch_10_20260329_145205.log'
    """
    match = re.search(r'batch_(\d+)_', filename)
    if match:
        return int(match.group(1))
    return None


def aggregate_timing_data(log_dir):
    """
    Process all log files in a directory and aggregate timing data by batch size.
    """
    log_dir = Path(log_dir)

    # Dictionary to store results: batch_size -> list of timing data
    results = defaultdict(list)

    # Process all .log files
    for log_file in sorted(log_dir.glob('*.log')):
        print(f"Processing: {log_file.name}")

        batch_size = extract_batch_size_from_filename(log_file.name)
        if batch_size is None:
            print(f"  Warning: Could not extract batch size from {log_file.name}")
            continue

        timing_blocks = parse_log_file(log_file)

        if not timing_blocks:
            print(f"  Warning: No timing data found in {log_file.name}")
            continue

        print(f"  Found {len(timing_blocks)} timing blocks for batch size {batch_size}")
        results[batch_size].extend(timing_blocks)

    # Aggregate data for each batch size
    aggregated = {}

    for batch_size, timing_list in results.items():
        if not timing_list:
            continue

        # Average across all timing blocks
        avg_breakdown = {
            'attention_block': sum(t['breakdown'].get('attention_block', 0) for t in timing_list) / len(timing_list),
            'mlp_block': sum(t['breakdown'].get('mlp_block', 0) for t in timing_list) / len(timing_list),
            'others': sum(t['breakdown'].get('others', 0) for t in timing_list) / len(timing_list),
        }

        avg_total = sum(t['total'].get('decode_step', 0) for t in timing_list) / len(timing_list)

        # Detailed breakdown (from averaged layers)
        detailed = {}
        if timing_list[0]['layers_avg']:
            # All components from the timing logs
            component_keys = [
                'input_ln',
                'qkv_proj',
                'kv_cache_save',
                'kv_cache_load',
                'attention',
                'o_proj',
                'post_attn_ln',
                'mlp'
            ]
            for key in component_keys:
                values = [t['layers_avg'].get(key, 0) for t in timing_list if key in t['layers_avg']]
                if values:
                    detailed[key] = sum(values) / len(values)

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
        print("Usage: python parse_timing_logs.py <log_directory>")
        sys.exit(1)

    log_dir = sys.argv[1]

    if not os.path.isdir(log_dir):
        print(f"Error: {log_dir} is not a valid directory")
        sys.exit(1)

    print(f"Parsing logs from: {log_dir}")
    print("=" * 60)

    results = aggregate_timing_data(log_dir)

    if not results:
        print("\nNo timing data found!")
        sys.exit(1)

    # Sort by batch size
    sorted_results = dict(sorted(results.items()))

    # Save to JSON
    output_file = Path(log_dir) / 'timing_data.json'
    with open(output_file, 'w') as f:
        json.dump(sorted_results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Results saved to: {output_file}")
    print(f"\nSummary:")
    print(f"{'Batch Size':<12} {'Latency (ms)':<15} {'Attention':<12} {'MLP':<12} {'Others':<12} {'Samples':<8}")
    print("-" * 80)

    for batch_size, data in sorted_results.items():
        latency = data['decode_latency_ms']
        attn = data['breakdown_simple']['attention_block']
        mlp = data['breakdown_simple']['mlp_block']
        others = data['breakdown_simple']['others']
        samples = data['num_samples']

        print(f"{batch_size:<12} {latency:<15.2f} {attn:<12.2f} {mlp:<12.2f} {others:<12.2f} {samples:<8}")


if __name__ == '__main__':
    main()
