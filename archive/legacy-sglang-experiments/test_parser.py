#!/usr/bin/env python3
"""
Quick test to verify we're extracting all components from the logs.
"""

import sys
import re
from pathlib import Path

def show_components_in_log(log_file):
    """Show what components are in a log file."""
    with open(log_file, 'r') as f:
        content = f.read()

    # Find a DECODE STEP TIMING block
    lines = content.split('\n')

    for i, line in enumerate(lines):
        if 'DECODE STEP TIMING' in line:
            # Print the next 50 lines to see the structure
            print(f"Found DECODE STEP TIMING at line {i}")
            print("=" * 70)

            for j in range(i, min(i + 60, len(lines))):
                if 'Scheduler:' in lines[j] or ('──────' in lines[j] and j > i + 10):
                    break
                if 'x 35' in lines[j] or 'x 36' in lines[j]:  # Show the averaged layers
                    print(lines[j])

            print("=" * 70)
            break

if __name__ == '__main__':
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    else:
        # Use the most recent log
        log_files = list(Path('qwen3vl-log').glob('*.log'))
        if not log_files:
            print("No log files found in qwen3vl-log/")
            sys.exit(1)
        log_file = sorted(log_files)[-1]

    print(f"Analyzing: {log_file}")
    print()
    show_components_in_log(log_file)
    print()
    print("Expected components to extract:")
    print("  - input_ln")
    print("  - qkv_proj")
    print("  - kv_cache_save")
    print("  - kv_cache_load")
    print("  - attention")
    print("  - o_proj")
    print("  - post_attn_ln")
    print("  - mlp")
    print("  - others (calculated as remainder)")
