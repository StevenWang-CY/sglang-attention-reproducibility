#!/usr/bin/env python3
"""
Plot decode latency breakdown from parsed timing data.
This version allows filtering to specific batch sizes.
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def filter_batch_sizes(data, target_sizes=None):
    """
    Filter data to only include specific batch sizes.
    """
    if target_sizes is None:
        return data

    filtered = {}
    for batch_size, metrics in data.items():
        if int(batch_size) in target_sizes:
            filtered[batch_size] = metrics

    return filtered


def plot_simple_breakdown(data, output_file='latency_breakdown_simple.png'):
    """
    Plot high-level breakdown: Attention, MLP, Sparse Selection, Others.
    Automatically detects tree-sparse components and groups them.
    """
    # Tree-sparse component names to aggregate into "sparse_selection"
    TREE_SPARSE_COMPONENTS = [
        'centroid_update', 'sparse_get_centroids', 'sparse_score_topk',
        'sparse_build_mask', 'sparse_extract_indices', 'sparse_batch_meta',
        'sparse_build_indices',  # Unified fused component
        'begin_forward'
    ]

    batch_sizes = []
    attention_times = []
    mlp_times = []
    sparse_times = []
    others_times = []

    # Check if any tree-sparse components exist in the data
    has_tree_sparse = False
    for metrics in data.values():
        detailed = metrics.get('breakdown_detailed', {})
        if any(c in detailed for c in TREE_SPARSE_COMPONENTS):
            has_tree_sparse = True
            break

    # Extract data
    for batch_size, metrics in sorted(data.items(), key=lambda x: int(x[0])):
        batch_sizes.append(int(batch_size))
        breakdown = metrics['breakdown_simple']
        attention_times.append(breakdown['attention_block'])
        mlp_times.append(breakdown['mlp_block'])

        if has_tree_sparse:
            # Sum up tree-sparse components from detailed breakdown
            detailed = metrics.get('breakdown_detailed', {})
            sparse_total = sum(detailed.get(c, 0) for c in TREE_SPARSE_COMPONENTS)
            sparse_times.append(sparse_total)
            # Subtract sparse from others (since they were previously lumped in)
            others_times.append(max(0, breakdown['others'] - sparse_total))
        else:
            sparse_times.append(0)
            others_times.append(breakdown['others'])

    # Convert to numpy arrays
    batch_sizes = np.array(batch_sizes)
    attention_times = np.array(attention_times)
    mlp_times = np.array(mlp_times)
    sparse_times = np.array(sparse_times)
    others_times = np.array(others_times)
    total_times = attention_times + mlp_times + sparse_times + others_times

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define colors
    colors = {
        'others': '#6B7C93',           # Dark blue-gray
        'sparse_selection': '#E91E63', # Pink
        'attention': '#7BAF7E',        # Green
        'mlp': '#7FC8D4'               # Light blue
    }

    # Create stacked bars
    width = 0.6
    x = np.arange(len(batch_sizes))

    # Build component stack
    components = [('others', others_times, colors['others'])]
    if has_tree_sparse:
        components.append(('sparse_selection', sparse_times, colors['sparse_selection']))
    components.append(('attention', attention_times, colors['attention']))
    components.append(('mlp', mlp_times, colors['mlp']))

    cumulative = np.zeros(len(batch_sizes))
    for name, values, color in components:
        ax.bar(x, values, width, bottom=cumulative, label=name, color=color)
        cumulative += values

    # Add percentage labels
    for i in range(len(batch_sizes)):
        total = total_times[i]
        cumulative_pos = 0
        for name, values, color in components:
            value = values[i]
            if value > 0:
                pct = 100 * value / total
                y_pos = cumulative_pos + value / 2
                if pct > 3:
                    ax.text(i, y_pos, f'{pct:.1f}%', ha='center', va='center',
                           fontsize=9, color='white', fontweight='bold')
            cumulative_pos += value

    # Customize plot
    ax.set_xlabel('Batch Size', fontsize=14, fontweight='bold')
    ax.set_ylabel('Decode Latency (ms)', fontsize=14, fontweight='bold')
    title_suffix = ' (tree-sparse)' if has_tree_sparse else ''
    ax.set_title(f'(a) Transformer decode latency breakdown{title_suffix}',
                 fontsize=16, fontweight='bold')
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
    Plot detailed breakdown with ALL components.
    Dynamically detects tree-sparse components from the data.
    """
    # Standard component order (always present)
    STANDARD_COMPONENTS = [
        'input_ln', 'qkv_linear', 'qk_norm', 'rope',
        'kv_cache_save', 'kv_cache_load',
        'attention', 'o_proj', 'post_attn_ln', 'mlp'
    ]

    # Tree-sparse specific components (in display order)
    TREE_SPARSE_COMPONENTS = [
        'centroid_update', 'sparse_get_centroids', 'sparse_score_topk',
        'sparse_build_mask', 'sparse_extract_indices', 'sparse_batch_meta',
        'sparse_build_indices',  # Unified fused component (replaces build_mask + extract_indices + batch_meta)
        'begin_forward'
    ]

    # Collect all component keys that appear in the data
    all_keys = set()
    for metrics in data.values():
        all_keys.update(metrics.get('breakdown_detailed', {}).keys())

    # Determine which tree-sparse components are present
    tree_sparse_present = [c for c in TREE_SPARSE_COMPONENTS if c in all_keys]
    has_tree_sparse = len(tree_sparse_present) > 0

    # Build ordered component list: standard + tree-sparse (interleaved logically)
    # Place tree-sparse components between kv_cache_save and kv_cache_load
    # (since sparse selection happens after saving KV and before loading)
    if has_tree_sparse:
        ordered_components = ['input_ln', 'qkv_linear', 'qk_norm', 'rope', 'kv_cache_save']
        ordered_components.extend(tree_sparse_present)
        ordered_components.extend(['kv_cache_load', 'attention', 'o_proj',
                                   'post_attn_ln', 'mlp'])
    else:
        ordered_components = STANDARD_COMPONENTS[:]

    # Extract data
    batch_sizes = []
    component_data = {c: [] for c in ordered_components}
    others_times = []

    for batch_size, metrics in sorted(data.items(), key=lambda x: int(x[0])):
        batch_sizes.append(int(batch_size))
        detailed = metrics.get('breakdown_detailed', {})

        detailed_sum = 0
        for comp in ordered_components:
            val = detailed.get(comp, 0)
            component_data[comp].append(val)
            detailed_sum += val

        total = metrics['decode_latency_ms']
        others_times.append(max(0, total - detailed_sum))

    # Convert to numpy arrays
    batch_sizes = np.array(batch_sizes)
    others_times = np.array(others_times)
    for comp in ordered_components:
        component_data[comp] = np.array(component_data[comp])

    total_times = others_times + sum(component_data[c] for c in ordered_components)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    # Colors for standard components
    colors = {
        'others': '#6B7C93',            # Dark blue-gray
        'input_ln': '#9B59B6',          # Purple
        'qkv_linear': '#E74C3C',        # Red
        'qk_norm': '#C0392B',           # Dark red
        'rope': '#E67E22',              # Dark orange
        'kv_cache_save': '#F39C12',     # Orange
        'kv_cache_load': '#F1C40F',     # Yellow
        'attention': '#27AE60',         # Green
        'o_proj': '#16A085',            # Teal
        'post_attn_ln': '#3498DB',      # Blue
        'mlp': '#5DADE2',              # Light blue
        # Tree-sparse components (warm/distinct colors)
        'centroid_update': '#E91E63',   # Pink
        'sparse_get_centroids': '#FF5722',  # Deep orange
        'sparse_score_topk': '#FF9800',     # Amber
        'sparse_build_mask': '#CDDC39',     # Lime
        'sparse_extract_indices': '#8BC34A', # Light green
        'sparse_batch_meta': '#00BCD4',     # Cyan
        'begin_forward': '#673AB7',         # Deep purple
    }

    # Create stacked bars
    width = 0.6
    x = np.arange(len(batch_sizes))

    # Stack from bottom to top
    cumulative = np.zeros(len(batch_sizes))

    # Build components list: others first, then ordered components
    components = [('others', others_times)]
    for comp in ordered_components:
        components.append((comp, component_data[comp]))

    for name, values in components:
        color = colors.get(name, '#AAAAAA')
        ax.bar(x, values, width, bottom=cumulative, label=name, color=color)
        cumulative += values

    # Add percentage labels
    for i in range(len(batch_sizes)):
        total = total_times[i]
        cumulative_pos = 0

        for name, values in components:
            value = values[i]
            if value > 0:
                pct = 100 * value / total
                y_pos = cumulative_pos + value / 2
                if pct > 2.0:
                    ax.text(i, y_pos, f'{pct:.1f}%', ha='center', va='center',
                           fontsize=8, color='white', fontweight='bold')
            cumulative_pos += value

    # Customize plot
    ax.set_xlabel('Batch Size', fontsize=14, fontweight='bold')
    ax.set_ylabel('Decode Latency (ms)', fontsize=14, fontweight='bold')
    title_suffix = ' (tree-sparse)' if has_tree_sparse else ''
    ax.set_title(f'(b) Transformer decode latency breakdown (detailed){title_suffix}',
                 fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(batch_sizes)

    # Create legend with better formatting
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1],
             title='Module', loc='upper left',
             fontsize=9, title_fontsize=11,
             ncol=2 if has_tree_sparse else 1)

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
        print("Usage: python plot_latency_breakdown_filtered.py <timing_data.json> [batch_sizes]")
        print("\nExamples:")
        print("  # Plot all batch sizes")
        print("  python plot_latency_breakdown_filtered.py qwen3vl-log/timing_data.json")
        print("")
        print("  # Plot only specific batch sizes")
        print("  python plot_latency_breakdown_filtered.py qwen3vl-log/timing_data.json 1,2,4,8,16,32,64")
        sys.exit(1)

    json_file = Path(sys.argv[1])

    # Parse target batch sizes if provided
    target_sizes = None
    if len(sys.argv) >= 3:
        target_sizes = [int(x.strip()) for x in sys.argv[2].split(',')]
        print(f"Filtering to batch sizes: {target_sizes}")

    if not json_file.exists():
        print(f"Error: {json_file} not found")
        sys.exit(1)

    # Load data
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Convert string keys to integers
    data = {int(k): v for k, v in data.items()}

    if not data:
        print("Error: No data found in JSON file")
        sys.exit(1)

    # Filter if requested
    if target_sizes:
        data = filter_batch_sizes(data, target_sizes)
        if not data:
            print(f"Error: No data found for batch sizes {target_sizes}")
            print(f"Available batch sizes: {sorted([int(k) for k in json.load(open(json_file)).keys()])}")
            sys.exit(1)

    # Create output directory
    output_dir = json_file.parent
    suffix = "_filtered" if target_sizes else ""
    output_simple = output_dir / f'latency_breakdown_simple{suffix}.png'
    output_detailed = output_dir / f'latency_breakdown_detailed{suffix}.png'

    print(f"\nGenerating plots from: {json_file}")
    print(f"Batch sizes in plot: {sorted(data.keys())}")
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
