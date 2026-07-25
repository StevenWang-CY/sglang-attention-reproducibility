#!/usr/bin/env python3
"""
Script to find significant changes in attention patterns and identify tokens.

Usage:
    python find_attention_change.py <csv_file> [--model MODEL] [--top-n N]
"""

import argparse
import csv
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer


def analyze_attention_changes(csv_path, tokenizer=None, top_n=10):
    """
    Analyze attention weights to find rows with significant pattern changes.

    Returns:
        List of (row_idx, token_id, token_text, avg_attention, max_attention, change_score)
    """
    print(f"Reading CSV file: {csv_path}")

    token_ids = []
    attention_matrix = []

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header

        for row in reader:
            if not row or not row[0]:
                continue

            try:
                token_id = int(row[0])
                token_ids.append(token_id)

                # Extract weights
                weights = []
                for val in row[1:]:
                    if val and val.strip():
                        weights.append(float(val))
                    else:
                        break

                attention_matrix.append(weights)
            except (ValueError, IndexError):
                continue

    # Convert to numpy array
    max_len = max(len(row) for row in attention_matrix)
    padded_matrix = np.zeros((len(attention_matrix), max_len), dtype=np.float32)

    for i, row in enumerate(attention_matrix):
        padded_matrix[i, :len(row)] = row

    print(f"Loaded matrix: {padded_matrix.shape}")

    # Calculate statistics for each row
    row_means = padded_matrix.mean(axis=1)  # Average attention per row
    row_maxs = padded_matrix.max(axis=1)    # Max attention per row
    row_sums = padded_matrix.sum(axis=1)    # Total attention per row

    # Use a more robust method: compare average before vs average after each split point
    # This helps identify true pattern changes rather than single-line outliers
    window_size = 10  # Look at 10 rows before and after

    changes = np.zeros(len(row_means))
    for i in range(window_size, len(row_means) - window_size):
        # Calculate average of window BEFORE this row
        before_avg = row_means[i - window_size:i].mean()

        # Calculate average of window AFTER this row
        after_avg = row_means[i:i + window_size].mean()

        # Calculate change between before and after windows
        changes[i] = np.abs(after_avg - before_avg)

    # Find rows with largest changes (biggest difference between before/after windows)
    change_indices = np.argsort(changes)[::-1][:top_n]

    results = []

    print("\n" + "="*80)
    print("TOP ATTENTION PATTERN CHANGES:")
    print("="*80)

    for rank, idx in enumerate(change_indices, 1):
        row_idx = idx + 1  # The row after the change
        token_id = token_ids[row_idx] if row_idx < len(token_ids) else None

        if token_id is None:
            continue

        # Decode token
        if tokenizer:
            try:
                token_text = tokenizer.decode([token_id])
            except:
                token_text = f"ID:{token_id}"
        else:
            token_text = f"ID:{token_id}"

        prev_mean = row_means[idx]
        curr_mean = row_means[row_idx]
        change_magnitude = changes[idx]

        print(f"\n{rank}. Row {row_idx} (token: {repr(token_text)})")
        print(f"   Token ID: {token_id}")
        print(f"   Change magnitude: {change_magnitude:.6f}")
        print(f"   Before (row {idx}): avg={prev_mean:.6f}, max={row_maxs[idx]:.6f}")
        print(f"   After  (row {row_idx}): avg={curr_mean:.6f}, max={row_maxs[row_idx]:.6f}")

        results.append({
            'rank': rank,
            'row_idx': row_idx,
            'token_id': token_id,
            'token_text': token_text,
            'change': change_magnitude,
            'avg_before': prev_mean,
            'avg_after': curr_mean,
            'max_before': row_maxs[idx],
            'max_after': row_maxs[row_idx]
        })

    # Also show tokens before the biggest change
    if results:
        biggest_change = results[0]
        row_idx = biggest_change['row_idx']

        print("\n" + "="*80)
        print(f"TOKENS BEFORE THE BIGGEST CHANGE (before row {row_idx}):")
        print("="*80)

        # Show previous 20 tokens
        start_idx = max(0, row_idx - 20)

        # Collect all tokens for context
        context_tokens = []
        for i in range(start_idx, row_idx):
            token_id = token_ids[i]
            if tokenizer:
                try:
                    token_text = tokenizer.decode([token_id])
                except:
                    token_text = f"[ID:{token_id}]"
            else:
                token_text = f"[ID:{token_id}]"

            context_tokens.append(token_text)
            print(f"Row {i}: {repr(token_text)} (ID: {token_id}, avg_attn: {row_means[i]:.6f})")

        # Show concatenated context
        print("\n" + "-"*80)
        print("FULL CONTEXT (concatenated tokens):")
        print("-"*80)
        full_context = "".join(context_tokens)
        print(full_context)
        print("\n" + "-"*80)

    # Find rows with darkest (highest average attention)
    print("\n" + "="*80)
    print("ROWS WITH HIGHEST AVERAGE ATTENTION (darkest rows):")
    print("="*80)

    dark_indices = np.argsort(row_means)[::-1][:top_n]

    for rank, idx in enumerate(dark_indices, 1):
        token_id = token_ids[idx]

        if tokenizer:
            try:
                token_text = tokenizer.decode([token_id])
            except:
                token_text = f"ID:{token_id}"
        else:
            token_text = f"ID:{token_id}"

        print(f"{rank}. Row {idx}: {repr(token_text)}")
        print(f"   Token ID: {token_id}")
        print(f"   Avg attention: {row_means[idx]:.6f}")
        print(f"   Max attention: {row_maxs[idx]:.6f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Find attention pattern changes in CSV file'
    )
    parser.add_argument(
        'csv_file',
        type=str,
        help='Path to attention weights CSV file'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='Qwen/Qwen3-VL-30B-A3B-Instruct',
        help='Model name for tokenizer'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=10,
        help='Number of top changes to show (default: 10)'
    )
    parser.add_argument(
        '--no-tokenizer',
        action='store_true',
        help='Skip loading tokenizer'
    )

    args = parser.parse_args()

    # Load tokenizer
    tokenizer = None
    if not args.no_tokenizer:
        try:
            print(f"Loading tokenizer: {args.model}")
            tokenizer = AutoTokenizer.from_pretrained(args.model)
        except Exception as e:
            print(f"Warning: Could not load tokenizer: {e}")

    # Analyze
    analyze_attention_changes(args.csv_file, tokenizer=tokenizer, top_n=args.top_n)


if __name__ == '__main__':
    main()