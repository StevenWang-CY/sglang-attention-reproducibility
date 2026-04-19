#!/usr/bin/env python3
"""
Plot decode latency breakdown from parsed timing data.
Creates two figures:
1. High-level: Attention block, MLP block, Others
2. Detailed: qkv_proj, attention, o_proj, mlp, etc.
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_simple_breakdown(data, output_file='latency_breakdown_simple.png'):
    """
    Plot high-level breakdown: Attention, MLP, Others
    """
    batch_sizes = []
    attention_times = []
    mlp_times = []
    others_times = []

    # Extract data
    for batch_size, metrics in sorted(data.items()):
        batch_sizes.append(batch_size)
        breakdown = metrics['breakdown_simple']
        attention_times.append(breakdown['attention_block'])
        mlp_times.append(breakdown['mlp_block'])
        others_times.append(breakdown['others'])

    # Convert to numpy arrays
    batch_sizes = np.array(batch_sizes)
    attention_times = np.array(attention_times)
    mlp_times = np.array(mlp_times)
    others_times = np.array(others_times)
    total_times = attention_times + mlp_times + others_times

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define colors (matching the example)
    colors = {
        'others': '#6B7C93',      # Dark blue-gray
        'attention': '#7BAF7E',    # Green
        'mlp': '#7FC8D4'           # Light blue
    }

    # Create stacked bars
    width = 0.6
    x = np.arange(len(batch_sizes))

    bars_others = ax.bar(x, others_times, width, label='others', color=colors['others'])
    bars_attention = ax.bar(x, attention_times, width, bottom=others_times,
                           label='attention', color=colors['attention'])
    bars_mlp = ax.bar(x, mlp_times, width,
                     bottom=others_times + attention_times,
                     label='mlp', color=colors['mlp'])

    # Add percentage labels
    for i in range(len(batch_sizes)):
        total = total_times[i]

        # Others percentage
        if others_times[i] > 0:
            pct = 100 * others_times[i] / total
            y_pos = others_times[i] / 2
            if pct > 3:  # Only show if significant
                ax.text(i, y_pos, f'{pct:.1f}%', ha='center', va='center',
                       fontsize=9, color='white', fontweight='bold')

        # Attention percentage
        if attention_times[i] > 0:
            pct = 100 * attention_times[i] / total
            y_pos = others_times[i] + attention_times[i] / 2
            if pct > 3:
                ax.text(i, y_pos, f'{pct:.1f}%', ha='center', va='center',
                       fontsize=9, color='white', fontweight='bold')

        # MLP percentage
        if mlp_times[i] > 0:
            pct = 100 * mlp_times[i] / total
            y_pos = others_times[i] + attention_times[i] + mlp_times[i] / 2
            if pct > 3:
                ax.text(i, y_pos, f'{pct:.1f}%', ha='center', va='center',
                       fontsize=9, color='white', fontweight='bold')

    # Customize plot
    ax.set_xlabel('Batch Size', fontsize=14, fontweight='bold')
    ax.set_ylabel('Decode Latency (ms)', fontsize=14, fontweight='bold')
    ax.set_title('(a) Transformer decode latency breakdown', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(batch_sizes)
    ax.legend(title='Module', loc='upper left', fontsize=11, title_fontsize=12)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Set background color
    ax.set_facecolor('#E8EAF6')
    fig.patch.set_facecolor('white')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved simple breakdown plot to: {output_file}")
    plt.close()


def plot_detailed_breakdown(data, output_file='latency_breakdown_detailed.png'):
    """
    Plot detailed breakdown with ALL components:
    - Attention: input_ln, qkv_proj, kv_cache_save, kv_cache_load, attention, o_proj
    - MLP: post_attn_ln, mlp
    - Others
    """
    batch_sizes = []
    input_ln_times = []
    qkv_proj_times = []
    kv_cache_save_times = []
    kv_cache_load_times = []
    attention_times = []
    o_proj_times = []
    post_attn_ln_times = []
    mlp_times = []
    others_times = []

    # Extract data
    for batch_size, metrics in sorted(data.items()):
        batch_sizes.append(batch_size)

        detailed = metrics['breakdown_detailed']

        # Get all detailed components
        input_ln_times.append(detailed.get('input_ln', 0))
        qkv_proj_times.append(detailed.get('qkv_proj', 0))
        kv_cache_save_times.append(detailed.get('kv_cache_save', 0))
        kv_cache_load_times.append(detailed.get('kv_cache_load', 0))
        attention_times.append(detailed.get('attention', 0))
        o_proj_times.append(detailed.get('o_proj', 0))
        post_attn_ln_times.append(detailed.get('post_attn_ln', 0))
        mlp_times.append(detailed.get('mlp', 0))

        # Calculate "others" as remainder
        detailed_sum = (detailed.get('input_ln', 0) +
                       detailed.get('qkv_proj', 0) +
                       detailed.get('kv_cache_save', 0) +
                       detailed.get('kv_cache_load', 0) +
                       detailed.get('attention', 0) +
                       detailed.get('o_proj', 0) +
                       detailed.get('post_attn_ln', 0) +
                       detailed.get('mlp', 0))

        total = metrics['decode_latency_ms']
        others_times.append(max(0, total - detailed_sum))

    # Convert to numpy arrays
    batch_sizes = np.array(batch_sizes)
    others_times = np.array(others_times)
    input_ln_times = np.array(input_ln_times)
    qkv_proj_times = np.array(qkv_proj_times)
    kv_cache_save_times = np.array(kv_cache_save_times)
    kv_cache_load_times = np.array(kv_cache_load_times)
    attention_times = np.array(attention_times)
    o_proj_times = np.array(o_proj_times)
    post_attn_ln_times = np.array(post_attn_ln_times)
    mlp_times = np.array(mlp_times)

    total_times = (others_times + input_ln_times + qkv_proj_times +
                   kv_cache_save_times + kv_cache_load_times + attention_times +
                   o_proj_times + post_attn_ln_times + mlp_times)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    # Define colors for all components
    colors = {
        'others': '#6B7C93',        # Dark blue-gray
        'input_ln': '#9B59B6',      # Purple
        'qkv_proj': '#E74C3C',      # Red
        'kv_cache_save': '#F39C12', # Orange
        'kv_cache_load': '#F1C40F', # Yellow
        'attention': '#27AE60',     # Green
        'o_proj': '#16A085',        # Teal
        'post_attn_ln': '#3498DB',  # Blue
        'mlp': '#5DADE2'            # Light blue
    }

    # Create stacked bars
    width = 0.6
    x = np.arange(len(batch_sizes))

    # Stack from bottom to top
    cumulative = np.zeros(len(batch_sizes))

    bars = {}
    components = [
        ('others', others_times),
        ('input_ln', input_ln_times),
        ('qkv_proj', qkv_proj_times),
        ('kv_cache_save', kv_cache_save_times),
        ('kv_cache_load', kv_cache_load_times),
        ('attention', attention_times),
        ('o_proj', o_proj_times),
        ('post_attn_ln', post_attn_ln_times),
        ('mlp', mlp_times)
    ]

    for name, values in components:
        bars[name] = ax.bar(x, values, width, bottom=cumulative,
                           label=name, color=colors[name])
        cumulative += values

    # Add percentage labels
    for i in range(len(batch_sizes)):
        total = total_times[i]
        cumulative_pos = 0

        # Helper function to add label
        def add_label(value, cumulative_pos, name):
            if value > 0:
                pct = 100 * value / total
                y_pos = cumulative_pos + value / 2
                # Only show percentage if > 2% to avoid clutter
                if pct > 2.0:
                    ax.text(i, y_pos, f'{pct:.1f}%', ha='center', va='center',
                           fontsize=8, color='white', fontweight='bold')
            return cumulative_pos + value

        # Add labels for each component
        for name, values in components:
            cumulative_pos = add_label(values[i], cumulative_pos, name)

    # Customize plot
    ax.set_xlabel('Batch Size', fontsize=14, fontweight='bold')
    ax.set_ylabel('Decode Latency (ms)', fontsize=14, fontweight='bold')
    ax.set_title('(b) Transformer decode latency breakdown (detailed)', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(batch_sizes)

    # Create legend with better formatting
    # Reverse order so it matches the visual stacking
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1],
             title='Module', loc='upper left',
             fontsize=10, title_fontsize=11,
             ncol=1)

    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Set background color
    ax.set_facecolor('#E8EAF6')
    fig.patch.set_facecolor('white')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved detailed breakdown plot to: {output_file}")
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_latency_breakdown.py <timing_data.json>")
        print("       python plot_latency_breakdown.py <log_directory>")
        sys.exit(1)

    input_path = Path(sys.argv[1])

    # Check if input is a directory or a JSON file
    if input_path.is_dir():
        json_file = input_path / 'timing_data.json'
    elif input_path.suffix == '.json':
        json_file = input_path
    else:
        print(f"Error: {input_path} is not a valid directory or JSON file")
        sys.exit(1)

    if not json_file.exists():
        print(f"Error: {json_file} not found")
        print(f"Please run parse_timing_logs.py first to generate timing data")
        sys.exit(1)

    # Load data
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Convert string keys to integers
    data = {int(k): v for k, v in data.items()}

    if not data:
        print("Error: No data found in JSON file")
        sys.exit(1)

    # Create output directory
    output_dir = json_file.parent
    output_simple = output_dir / 'latency_breakdown_simple.png'
    output_detailed = output_dir / 'latency_breakdown_detailed.png'

    print(f"Generating plots from: {json_file}")
    print(f"Found data for batch sizes: {sorted(data.keys())}")
    print()

    # Generate plots
    plot_simple_breakdown(data, output_simple)
    plot_detailed_breakdown(data, output_detailed)

    print()
    print("=" * 60)
    print("Done! Generated plots:")
    print(f"  1. Simple breakdown: {output_simple}")
    print(f"  2. Detailed breakdown: {output_detailed}")


if __name__ == '__main__':
    main()
