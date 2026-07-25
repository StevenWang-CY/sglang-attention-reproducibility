#!/usr/bin/env python3
"""
Debug script to check what the attention weights actually look like.
"""
import csv
from pathlib import Path
import sys


def read_attention_weights(csv_file: Path, focus_token: int = -1):
    """Read attention weights for a specific token."""
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header

        rows = list(reader)
        if focus_token < 0:
            focus_token = len(rows) + focus_token

        row = rows[focus_token]
        weights = [float(w) for w in row[1:] if w]
        return weights, focus_token, len(rows)


def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_attention_weights.py <attention_csv>")
        return 1

    attention_file = Path(sys.argv[1])

    weights, focus_idx, total_tokens = read_attention_weights(attention_file, -1)

    print(f"Focus token index: {focus_idx}")
    print(f"Total tokens in file: {total_tokens}")
    print(f"Length of attention weights array: {len(weights)}")
    print(f"\nFirst 20 attention weights:")
    for i, w in enumerate(weights[:20]):
        print(f"  [{i:>4}] {w:.10f}")

    print(f"\nLast 20 attention weights:")
    for i, w in enumerate(weights[-20:], len(weights)-20):
        print(f"  [{i:>4}] {w:.10f}")

    print(f"\nAttention weights in range [3999..4020]:")
    for i in range(3999, min(4021, len(weights))):
        print(f"  [{i:>4}] {weights[i]:.10f}")

    # Sum check
    total_sum = sum(weights)
    print(f"\nTotal sum of all attention weights: {total_sum:.10f}")
    print(f"Expected sum (should be ≈1.0): {total_sum:.10f}")

    # Check range [3999..8417]
    if len(weights) > 8417:
        range_sum = sum(weights[3999:8418])
        print(f"\nSum of attention weights in range [3999..8417]: {range_sum:.10f}")

    return 0


if __name__ == '__main__':
    exit(main())
