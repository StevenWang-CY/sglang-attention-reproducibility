#!/usr/bin/env python3
"""Plot TPOT vs input length from sweep results.

Usage:
    python plot_input_length_sweep.py <results_dir>
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_sweep_results(results_dir: Path) -> dict:
    """Load all results_input*.json files and combine."""
    combined = {}  # {input_len: {batch_size: {tpot, throughput, ...}}}

    for f in sorted(results_dir.glob("results_input*.json")):
        # Extract input length from filename
        name = f.stem  # e.g. "results_input10000"
        input_len = int(name.replace("results_input", ""))

        with open(f) as fh:
            data = json.load(fh)

        combined[input_len] = {}
        for bs_str, metrics in data.items():
            combined[input_len][int(bs_str)] = metrics

    return combined


def plot_sweep(combined: dict, output_dir: Path):
    """Generate sweep plots."""
    input_lengths = sorted(combined.keys())
    all_batch_sizes = sorted(set(
        bs for data in combined.values() for bs in data.keys()
    ))

    # --- Plot 1: TPOT vs Input Length (one line per batch size) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('TPOT vs Input Sequence Length', fontsize=16, fontweight='bold')

    ax1 = axes[0]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(all_batch_sizes)))

    for i, bs in enumerate(all_batch_sizes):
        tpots = []
        lens = []
        for il in input_lengths:
            if bs in combined[il]:
                tpots.append(combined[il][bs]['tpot_mean_ms'])
                lens.append(il)
        if tpots:
            ax1.plot(lens, tpots, 'o-', color=colors[i], linewidth=2,
                     markersize=7, label=f'BS={bs}')
            for x, y in zip(lens, tpots):
                ax1.annotate(f'{y:.1f}', (x, y), textcoords="offset points",
                             xytext=(0, 8), ha='center', fontsize=7, fontweight='bold')

    ax1.set_xlabel('Input Length (tokens)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('TPOT (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Time Per Output Token', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # Format x-axis with K suffix
    ax1.set_xticks(input_lengths)
    ax1.set_xticklabels([f'{l//1000}K' for l in input_lengths], fontsize=9)

    # --- Plot 2: Throughput vs Input Length ---
    ax2 = axes[1]
    for i, bs in enumerate(all_batch_sizes):
        throughputs = []
        lens = []
        for il in input_lengths:
            if bs in combined[il]:
                throughputs.append(combined[il][bs]['throughput_tokens_per_sec'])
                lens.append(il)
        if throughputs:
            ax2.plot(lens, throughputs, 'o-', color=colors[i], linewidth=2,
                     markersize=7, label=f'BS={bs}')
            for x, y in zip(lens, throughputs):
                ax2.annotate(f'{y:.0f}', (x, y), textcoords="offset points",
                             xytext=(0, 8), ha='center', fontsize=7, fontweight='bold')

    ax2.set_xlabel('Input Length (tokens)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Throughput (tokens/s)', fontsize=12, fontweight='bold')
    ax2.set_title('(b) Generation Throughput', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)
    ax2.set_xticks(input_lengths)
    ax2.set_xticklabels([f'{l//1000}K' for l in input_lengths], fontsize=9)

    plt.tight_layout()
    out_path = output_dir / "tpot_vs_input_length.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()

    # --- Plot 3: Heatmap (TPOT: input_length x batch_size) ---
    fig, ax = plt.subplots(figsize=(10, 6))

    tpot_matrix = np.full((len(input_lengths), len(all_batch_sizes)), np.nan)
    for i, il in enumerate(input_lengths):
        for j, bs in enumerate(all_batch_sizes):
            if bs in combined[il]:
                tpot_matrix[i, j] = combined[il][bs]['tpot_mean_ms']

    im = ax.imshow(tpot_matrix, aspect='auto', cmap='YlOrRd', origin='lower')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('TPOT (ms)', fontsize=12, fontweight='bold')

    ax.set_xticks(range(len(all_batch_sizes)))
    ax.set_xticklabels(all_batch_sizes)
    ax.set_yticks(range(len(input_lengths)))
    ax.set_yticklabels([f'{l//1000}K' for l in input_lengths])
    ax.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Input Length (tokens)', fontsize=12, fontweight='bold')
    ax.set_title('TPOT Heatmap: Input Length × Batch Size', fontsize=14, fontweight='bold')

    # Add text annotations
    for i in range(len(input_lengths)):
        for j in range(len(all_batch_sizes)):
            val = tpot_matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                        fontsize=9, fontweight='bold',
                        color='white' if val > np.nanmedian(tpot_matrix) else 'black')

    plt.tight_layout()
    out_path = output_dir / "tpot_heatmap.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()

    # Save combined data
    serializable = {}
    for il in input_lengths:
        serializable[str(il)] = {
            str(bs): metrics for bs, metrics in combined[il].items()
        }
    combined_path = output_dir / "sweep_combined.json"
    with open(combined_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"Saved: {combined_path}")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <results_dir>")
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    combined = load_sweep_results(results_dir)

    if not combined:
        print(f"No results_input*.json files found in {results_dir}")
        sys.exit(1)

    print(f"Loaded {len(combined)} input lengths: {sorted(combined.keys())}")
    plot_sweep(combined, results_dir)


if __name__ == '__main__':
    main()
