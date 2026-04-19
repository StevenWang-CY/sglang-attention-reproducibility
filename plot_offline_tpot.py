#!/usr/bin/env python3
"""
Plot TPOT (Time Per Output Token) from offline batch experiment results.

Usage:
    python plot_offline_tpot.py results.json [--output output.png]
"""

import argparse
import json
import sys
import matplotlib.pyplot as plt
import numpy as np


def plot_tpot(results_file: str, output_file: str = None):
    """Plot TPOT vs batch size from offline experiment results."""
    with open(results_file, 'r') as f:
        results = json.load(f)

    # Extract data sorted by batch size
    batch_sizes = sorted([int(k) for k in results.keys()])
    tpot_means = [results[str(bs)]['tpot_mean_ms'] for bs in batch_sizes]
    tpot_stds = [results[str(bs)]['tpot_std_ms'] for bs in batch_sizes]
    throughputs = [results[str(bs)]['throughput_tokens_per_sec'] for bs in batch_sizes]
    total_times = [results[str(bs)]['total_time_mean_ms'] for bs in batch_sizes]

    if output_file is None:
        output_file = results_file.replace('.json', '_tpot.png')

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('SGLang Offline Batch Latency Analysis', fontsize=16, fontweight='bold')

    # --- Plot 1: TPOT vs Batch Size ---
    ax1 = axes[0]
    ax1.errorbar(
        batch_sizes, tpot_means, yerr=tpot_stds,
        fmt='o-', color='#2196F3', capsize=5, capthick=2,
        linewidth=2, markersize=8, label='TPOT'
    )

    # Add value labels
    for x, y in zip(batch_sizes, tpot_means):
        ax1.annotate(f'{y:.1f}', (x, y), textcoords="offset points",
                     xytext=(0, 12), ha='center', fontsize=9, fontweight='bold')

    ax1.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax1.set_ylabel('TPOT (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Time Per Output Token', fontsize=14, fontweight='bold')
    ax1.set_xscale('log', base=2)
    ax1.set_xticks(batch_sizes)
    ax1.set_xticklabels([str(bs) for bs in batch_sizes])
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # --- Plot 2: Throughput vs Batch Size ---
    ax2 = axes[1]
    bars = ax2.bar(
        range(len(batch_sizes)), throughputs,
        color='#4CAF50', alpha=0.8, edgecolor='#388E3C', linewidth=1.5
    )

    # Add value labels on bars
    for bar, val in zip(bars, throughputs):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(throughputs) * 0.02,
                 f'{val:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax2.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Throughput (tokens/s)', fontsize=12, fontweight='bold')
    ax2.set_title('(b) Generation Throughput', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(batch_sizes)))
    ax2.set_xticklabels([str(bs) for bs in batch_sizes])
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(bottom=0)

    # --- Plot 3: Total Time vs Batch Size ---
    ax3 = axes[2]
    ax3.plot(
        batch_sizes, total_times,
        'o-', color='#FF5722', linewidth=2, markersize=8, label='Total Time'
    )

    # Add value labels
    for x, y in zip(batch_sizes, total_times):
        ax3.annotate(f'{y:.0f}', (x, y), textcoords="offset points",
                     xytext=(0, 12), ha='center', fontsize=9, fontweight='bold')

    ax3.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Total Time (ms)', fontsize=12, fontweight='bold')
    ax3.set_title('(c) Total Generation Time', fontsize=14, fontweight='bold')
    ax3.set_xscale('log', base=2)
    ax3.set_xticks(batch_sizes)
    ax3.set_xticklabels([str(bs) for bs in batch_sizes])
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to: {output_file}")

    # Also print summary table
    print("\n" + "=" * 70)
    print(f"{'Batch':>8} {'TPOT (ms)':>12} {'Std (ms)':>10} {'Throughput':>14} {'Total (ms)':>12}")
    print("-" * 70)
    for bs, tpot, std, tp, tt in zip(batch_sizes, tpot_means, tpot_stds, throughputs, total_times):
        print(f"{bs:>8} {tpot:>12.2f} {std:>10.2f} {tp:>14.1f} {tt:>12.0f}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot TPOT from offline batch results")
    parser.add_argument("results_file", type=str, help="Path to results JSON file")
    parser.add_argument("--output", type=str, default=None, help="Output plot file path")
    args = parser.parse_args()

    plot_tpot(args.results_file, args.output)
