#!/usr/bin/env python3
"""Plot GPU utilization stats (compute %, bandwidth %, memory usage) from results JSON.

Usage:
    python plot_gpu_stats.py results.json --output output_dir/
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_gpu_stats(results_file: str, output_dir: str):
    with open(results_file) as f:
        results = json.load(f)

    batch_sizes = sorted([int(k) for k in results.keys()])

    # Extract GPU stats
    compute_avg = []
    compute_max = []
    bw_avg = []
    bw_max = []
    mem_avg = []
    tpots = []

    for bs in batch_sizes:
        gpu = results[str(bs)].get('gpu_stats', {})
        tpots.append(results[str(bs)].get('tpot_mean_ms', 0))
        if gpu:
            compute_avg.append(gpu.get('gpu_compute_pct_mean', 0))
            compute_max.append(gpu.get('gpu_compute_pct_max', 0))
            bw_avg.append(gpu.get('gpu_mem_bw_pct_mean', 0))
            bw_max.append(gpu.get('gpu_mem_bw_pct_max', 0))
            mem_avg.append(gpu.get('gpu_mem_used_mb_mean', 0) / 1024)  # convert to GB
        else:
            compute_avg.append(0)
            compute_max.append(0)
            bw_avg.append(0)
            bw_max.append(0)
            mem_avg.append(0)

    if not any(compute_avg):
        print("No GPU stats found in results, skipping plot")
        return

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    fig.suptitle('GPU Utilization vs Batch Size', fontsize=16, fontweight='bold')

    x = np.arange(len(batch_sizes))
    width = 0.35

    # --- Panel 1: TPOT ---
    ax1 = axes[0]
    ax1.plot(batch_sizes, tpots, 'o-', color='#2196F3', linewidth=2, markersize=8)
    for bx, y in zip(batch_sizes, tpots):
        ax1.annotate(f'{y:.1f}', (bx, y), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=8, fontweight='bold')
    ax1.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax1.set_ylabel('TPOT (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Time Per Output Token', fontsize=13, fontweight='bold')
    ax1.set_xscale('log', base=2)
    ax1.set_xticks(batch_sizes)
    ax1.set_xticklabels([str(bs) for bs in batch_sizes], fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # --- Panel 2: Compute utilization ---
    ax2 = axes[1]
    bars1 = ax2.bar(x - width/2, compute_avg, width, label='avg', color='#E74C3C', alpha=0.8)
    bars2 = ax2.bar(x + width/2, compute_max, width, label='max', color='#C0392B', alpha=0.6)
    for bar, val in zip(bars1, compute_avg):
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f'{val:.0f}%', ha='center', fontsize=7, fontweight='bold')
    ax2.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Utilization (%)', fontsize=12, fontweight='bold')
    ax2.set_title('(b) GPU Compute (SM)', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(bs) for bs in batch_sizes], fontsize=9)
    ax2.set_ylim(0, 105)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')

    # --- Panel 3: Memory bandwidth utilization ---
    ax3 = axes[2]
    bars3 = ax3.bar(x - width/2, bw_avg, width, label='avg', color='#27AE60', alpha=0.8)
    bars4 = ax3.bar(x + width/2, bw_max, width, label='max', color='#1E8449', alpha=0.6)
    for bar, val in zip(bars3, bw_avg):
        if val > 0:
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f'{val:.0f}%', ha='center', fontsize=7, fontweight='bold')
    ax3.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Utilization (%)', fontsize=12, fontweight='bold')
    ax3.set_title('(c) Memory Bandwidth', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([str(bs) for bs in batch_sizes], fontsize=9)
    ax3.set_ylim(0, 105)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y')

    # --- Panel 4: Memory usage ---
    ax4 = axes[3]
    ax4.bar(x, mem_avg, width * 2, color='#3498DB', alpha=0.8)
    for i, val in enumerate(mem_avg):
        if val > 0:
            ax4.text(i, val + 0.5, f'{val:.1f}', ha='center', fontsize=8, fontweight='bold')
    ax4.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Memory Used (GB)', fontsize=12, fontweight='bold')
    ax4.set_title('(d) GPU Memory Usage', fontsize=13, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([str(bs) for bs in batch_sizes], fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim(bottom=0)

    plt.tight_layout()
    output_path = Path(output_dir) / "gpu_utilization.png"
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot GPU utilization stats")
    parser.add_argument("results_file", type=str, help="Path to results JSON")
    parser.add_argument("--output", type=str, default=".", help="Output directory")
    args = parser.parse_args()

    plot_gpu_stats(args.results_file, args.output)
