#!/usr/bin/env python3
"""
Query attention weights from CSV file.
Get the sum of attention values for a specific query token over a specific range of source tokens.
"""
import argparse
import csv
from pathlib import Path
from typing import List, Tuple


def read_attention_weights(csv_file: Path) -> Tuple[List[int], List[List[float]]]:
    """
    Read attention weights from CSV file.
    Returns (token_ids, attention_matrix) where attention_matrix[i] contains
    attention weights from token i to all previous tokens (including itself).
    """
    token_ids = []
    attention_matrix = []

    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header

        for row in reader:
            if not row:
                continue
            token_id = int(row[0])
            # attention weights to all previous tokens
            weights = [float(w) for w in row[1:] if w]

            token_ids.append(token_id)
            attention_matrix.append(weights)

    return token_ids, attention_matrix


def query_attention_sum(
    attention_matrix: List[List[float]],
    query_token: int,
    start_range: int = None,
    end_range: int = None
) -> float:
    """
    Get the sum of attention weights for a specific query token over a range.

    Args:
        attention_matrix: The attention matrix where attention_matrix[i] contains
                         weights from token i to all previous tokens
        query_token: The token index doing the querying (row index)
        start_range: Start of the range to sum (inclusive). If None, defaults to 0
        end_range: End of the range to sum (inclusive). If None, defaults to query_token

    Returns:
        Sum of attention weights in the specified range
    """
    if query_token < 0:
        query_token = len(attention_matrix) + query_token

    if query_token < 0 or query_token >= len(attention_matrix):
        raise ValueError(f"Query token {query_token} out of range [0..{len(attention_matrix)-1}]")

    weights = attention_matrix[query_token]

    if start_range is None:
        start_range = 0
    if end_range is None:
        end_range = len(weights) - 1

    # Handle negative indices
    if start_range < 0:
        start_range = len(weights) + start_range
    if end_range < 0:
        end_range = len(weights) + end_range

    # Validate range
    # if start_range < 0 or start_range >= len(weights):
    #     raise ValueError(f"Start range {start_range} out of bounds [0..{len(weights)-1}]")
    # if end_range < 0 or end_range >= len(weights):
    #     raise ValueError(f"End range {end_range} out of bounds [0..{len(weights)-1}]")
    # if start_range > end_range:
    #     raise ValueError(f"Start range {start_range} must be <= end range {end_range}")

    # Calculate sum
    total = sum(weights[start_range:end_range + 1])

    return total


def main():
    parser = argparse.ArgumentParser(
        description='Query attention weights for a specific token over a range',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Get total attention from last token to all tokens
  python query_attention_range.py attention.csv -q -1

  # Get attention from token 100 to tokens 50-75
  python query_attention_range.py attention.csv -q 100 -s 50 -e 75

  # Get attention from last token to tokens in range [3999..8417]
  python query_attention_range.py attention.csv -q -1 -s 3999 -e 8417

  # Get attention from token 8652 to single token 4000
  python query_attention_range.py attention.csv -q 8652 -s 4000 -e 4000

  # Multiple ranges for the same query token
  python query_attention_range.py attention.csv -q -1 -s 0 -e 100 -s 3999 -e 8417
        """
    )

    parser.add_argument(
        'csv_file',
        type=Path,
        help='Path to attention weights CSV file'
    )

    parser.add_argument(
        '-q', '--query-token',
        type=int,
        default=-1,
        help='Query token index (use -1 for last token, default: -1)'
    )

    parser.add_argument(
        '-s', '--start',
        type=int,
        action='append',
        help='Start of range (inclusive). Can be specified multiple times for multiple ranges.'
    )

    parser.add_argument(
        '-e', '--end',
        type=int,
        action='append',
        help='End of range (inclusive). Can be specified multiple times for multiple ranges.'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed information about each token in the range'
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.csv_file.exists():
        print(f"Error: File not found: {args.csv_file}")
        return 1

    # Read attention weights
    print(f"Reading attention weights from: {args.csv_file}")
    token_ids, attention_matrix = read_attention_weights(args.csv_file)

    print(f"Loaded {len(attention_matrix)} tokens")

    # Handle query token
    query_token = args.query_token
    if query_token < 0:
        query_token = len(attention_matrix) + query_token

    query_token_id = token_ids[query_token]
    print(f"Query token: {args.query_token} (absolute index: {query_token}, token_id: {query_token_id})")
    print()

    # Handle ranges
    if args.start is None and args.end is None:
        # Default: entire range
        ranges = [(0, len(attention_matrix[query_token]) - 1)]
    elif args.start is None or args.end is None:
        print("Error: Both --start and --end must be specified together")
        return 1
    elif len(args.start) != len(args.end):
        print("Error: Number of --start and --end arguments must match")
        return 1
    else:
        ranges = list(zip(args.start, args.end))

    # Process each range
    total_sum = 0.0
    for i, (start, end) in enumerate(ranges, 1):
        print(f"Range {i}: [{start}..{end}]")

        # Get sum
        range_sum = query_attention_sum(attention_matrix, args.query_token, start, end)
        total_sum += range_sum

        # Calculate actual indices if negative
        weights = attention_matrix[query_token]
        actual_start = start if start >= 0 else len(weights) + start
        actual_end = end if end >= 0 else len(weights) + end

        num_tokens = actual_end - actual_start + 1

        print(f"  Actual range: [{actual_start}..{actual_end}]")
        print(f"  Number of tokens: {num_tokens}")
        print(f"  Sum of attention: {range_sum:.10f}")
        print(f"  Average attention: {range_sum / num_tokens:.10f}")

        # Show details if verbose
        if args.verbose and num_tokens <= 100:
            print(f"  Individual token attentions:")
            for idx in range(actual_start, actual_end + 1):
                print(f"    [{idx:>5}] {weights[idx]:.10f}")

        print()

    # Summary
    if len(ranges) > 1:
        print(f"{'='*80}")
        print(f"TOTAL across all {len(ranges)} ranges: {total_sum:.10f}")

    # Also show full sum for reference
    full_sum = sum(attention_matrix[query_token])
    print(f"\nFull attention sum for token {query_token}: {full_sum:.10f}")

    if len(ranges) == 1:
        percentage = (total_sum / full_sum * 100) if full_sum > 0 else 0
        print(f"Query range represents {percentage:.2f}% of total attention")

    return 0


if __name__ == '__main__':
    exit(main())
