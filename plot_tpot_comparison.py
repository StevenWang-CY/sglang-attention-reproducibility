#!/usr/bin/env python3
"""Plot TPOT comparison bar chart across attention methods."""

import matplotlib.pyplot as plt
import numpy as np
import sys


def plot_comparison(output_path: str = "tpot_comparison.png"):
    methods = ['FlashInfer', 'Quest', 'Tree-Sparse']
    tpots = [13.0, 10.8, 10.0]
    colors = ['#E74C3C', '#3498DB', '#27AE60']

    fig, ax = plt.subplots(figsize=(8, 6))

    x = np.arange(len(methods))
    width = 0.5

    bars = ax.bar(x, tpots, width, color=colors, edgecolor='white', linewidth=1.5)

    # Add value labels
    for bar, val in zip(bars, tpots):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{val:.1f} ms', ha='center', va='bottom',
                fontsize=14, fontweight='bold')

    # Speedup annotations
    baseline = tpots[0]
    for i in range(1, len(tpots)):
        speedup = baseline / tpots[i]
        ax.text(x[i], tpots[i] / 2, f'{speedup:.2f}×',
                ha='center', va='center', fontsize=12,
                color='white', fontweight='bold')

    ax.set_xlabel('Attention Method', fontsize=14, fontweight='bold')
    ax.set_ylabel('TPOT (ms)', fontsize=14, fontweight='bold')
    ax.set_title('Decode Latency Comparison (BS=1, Qwen3-VL-8B)',
                 fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=13)
    ax.set_ylim(0, max(tpots) * 1.25)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == '__main__':
    output = sys.argv[1] if len(sys.argv) > 1 else "tpot_comparison.png"
    plot_comparison(output)
