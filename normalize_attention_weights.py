#!/usr/bin/env python3
"""
Script to perform min-max normalization on attention weight CSV files.

Normalizes each row independently using min-max normalization:
    normalized_value = (value - min) / (max - min)

Usage:
    python normalize_attention_weights.py <folder_path> [--output-suffix _normalized]

Example:
    python normalize_attention_weights.py attention_weights/
    python normalize_attention_weights.py attention_weights/ --output-suffix _norm
"""

import argparse
import csv
import numpy as np
from pathlib import Path
import time


def normalize_csv_file(input_path, output_path, method='minmax'):
    """
    Normalize a single CSV file row by row.

    Args:
        input_path: Path to input CSV file
        output_path: Path to output normalized CSV file
        method: Normalization method ('minmax' or 'max')
    """
    print(f"Processing: {input_path.name} (method: {method})")
    t_start = time.time()

    # Read the CSV file
    rows_data = []
    token_ids = []

    with open(input_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header

        for row in reader:
            if not row or not row[0]:
                continue

            try:
                token_id = row[0]
                token_ids.append(token_id)

                # Extract weights (skip empty values)
                weights = []
                for val in row[1:]:
                    if val and val.strip():
                        weights.append(float(val))
                    else:
                        break

                rows_data.append(weights)
            except (ValueError, IndexError) as e:
                print(f"  Warning: Could not parse row with token_id {row[0]}: {e}")
                continue

    print(f"  Read {len(rows_data)} rows")

    # Normalize each row based on the specified method
    normalized_rows = []
    for i, weights in enumerate(rows_data):
        if len(weights) == 0:
            normalized_rows.append([])
            continue

        weights_array = np.array(weights, dtype=np.float32)

        # Create mask for non-zero values
        non_zero_mask = weights_array > 1e-10

        # If all values are zero, keep as is
        if not non_zero_mask.any():
            normalized_rows.append(weights_array.tolist())
            continue

        # Extract non-zero values for calculating min/max
        non_zero_values = weights_array[non_zero_mask]
        max_val = non_zero_values.max()

        if method == 'max':
            # Max normalization: x / max(x) for non-zero values only
            # Zero values remain zero
            normalized = np.zeros_like(weights_array)
            normalized[non_zero_mask] = weights_array[non_zero_mask] / max_val
        else:  # minmax
            # Min-max normalization: (x - min) / (max - min) for non-zero values only
            # Zero values remain zero
            min_val = non_zero_values.min()

            # Handle case where all non-zero values are the same
            if max_val - min_val < 1e-10:
                # All non-zero values are the same, normalize to 1.0
                normalized = np.zeros_like(weights_array)
                normalized[non_zero_mask] = 1.0
            else:
                normalized = np.zeros_like(weights_array)
                normalized[non_zero_mask] = (weights_array[non_zero_mask] - min_val) / (max_val - min_val)

        normalized_rows.append(normalized.tolist())

    # Write normalized data to output file
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Write header
        writer.writerow(['query_token_id'])

        # Write normalized rows
        for token_id, normalized_weights in zip(token_ids, normalized_rows):
            writer.writerow([token_id] + normalized_weights)

    t_end = time.time()
    print(f"  ✓ Saved to: {output_path.name} ({t_end - t_start:.2f}s)")


def main():
    parser = argparse.ArgumentParser(
        description='Normalize attention weight CSV files using min-max normalization'
    )
    parser.add_argument(
        'folder',
        type=str,
        help='Folder containing CSV files to normalize'
    )
    parser.add_argument(
        '--output-suffix',
        type=str,
        default='_normalized',
        help='Suffix to add to output filenames (default: _normalized)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: same as input folder)'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='attn_weights_*.csv',
        help='File pattern to match (default: attn_weights_*.csv)'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='minmax',
        choices=['minmax', 'max'],
        help='Normalization method: "minmax" for (x-min)/(max-min) or "max" for x/max(x) (default: minmax)'
    )

    args = parser.parse_args()

    # Validate input folder
    folder_path = Path(args.folder)
    if not folder_path.exists():
        print(f"Error: Folder not found: {folder_path}")
        return 1

    if not folder_path.is_dir():
        print(f"Error: Not a directory: {folder_path}")
        return 1

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = folder_path

    # Find all CSV files matching the pattern
    all_csv_files = list(folder_path.glob(args.pattern))

    # Filter out already-normalized files (those with common normalization suffixes)
    normalization_suffixes = ['_normalized', '_max_norm', '_minmax_norm', '_norm']
    csv_files = [
        f for f in all_csv_files
        if not any(suffix in f.stem for suffix in normalization_suffixes)
    ]

    if len(csv_files) == 0:
        if len(all_csv_files) > 0:
            print(f"Found {len(all_csv_files)} CSV files, but they all appear to be already normalized")
            print(f"Skipping files with suffixes: {normalization_suffixes}")
        else:
            print(f"No CSV files found matching pattern '{args.pattern}' in {folder_path}")
        return 1

    if len(all_csv_files) > len(csv_files):
        print(f"Filtered out {len(all_csv_files) - len(csv_files)} already-normalized files")

    print(f"Found {len(csv_files)} CSV files to process")
    print(f"Output directory: {output_dir}")
    print()

    # Process each file
    total_start = time.time()
    success_count = 0
    error_count = 0

    for csv_file in csv_files:
        try:
            # Generate output filename
            stem = csv_file.stem
            output_filename = f"{stem}{args.output_suffix}.csv"
            output_path = output_dir / output_filename

            # Normalize the file
            normalize_csv_file(csv_file, output_path, method=args.method)
            success_count += 1

        except Exception as e:
            print(f"  ✗ Error processing {csv_file.name}: {e}")
            import traceback
            traceback.print_exc()
            error_count += 1

    total_end = time.time()

    # Summary
    print()
    print("=" * 60)
    print(f"Processing complete!")
    print(f"  Success: {success_count} files")
    print(f"  Errors:  {error_count} files")
    print(f"  Total time: {total_end - total_start:.2f}s")
    print("=" * 60)

    return 0 if error_count == 0 else 1


if __name__ == '__main__':
    exit(main())
