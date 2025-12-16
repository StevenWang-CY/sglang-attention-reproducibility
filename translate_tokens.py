#!/usr/bin/env python3
"""
Script to translate token IDs in attention weight CSV files to readable text.

This script:
1. Reads attention weight CSV files with token IDs
2. Translates token IDs to their text representation using a tokenizer
3. Saves the full token sequence to a text file

Usage:
    python translate_tokens.py <csv_file> [--model MODEL] [--text-output OUTPUT]

Example:
    python translate_tokens.py attention_weights/attn_weights_request_000002_layer_01_head_01.csv
    python translate_tokens.py attention_weights/attn_weights_request_000002_layer_01_head_01.csv -t custom_output.txt
"""

import argparse
import csv
from pathlib import Path
from transformers import AutoTokenizer


def translate_csv(input_path, output_csv_path=None, output_txt_path=None, output_token_map_path=None, tokenizer=None):
    """
    Translate token IDs in CSV to text and save results.

    Args:
        input_path: Path to input CSV file
        output_csv_path: Path to output CSV with translated tokens (optional)
        output_txt_path: Path to output text file with full context (optional)
        output_token_map_path: Path to output CSV with token ID to text mapping (optional)
        tokenizer: Tokenizer instance
    """
    print(f"Reading CSV file: {input_path}")

    token_ids = []
    token_texts = []
    rows_data = []

    # Read the CSV file
    with open(input_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header

        for row in reader:
            if not row or not row[0]:
                continue

            try:
                token_id = int(row[0])
                token_ids.append(token_id)

                # Decode token
                if tokenizer:
                    try:
                        token_text = tokenizer.decode([token_id])
                    except:
                        token_text = f"[ID:{token_id}]"
                else:
                    token_text = f"[ID:{token_id}]"

                token_texts.append(token_text)

                # Store row data
                rows_data.append(row)

            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse row: {row[0]}")
                continue

    print(f"Processed {len(token_ids)} tokens")

    # Write token mapping CSV (new feature!)
    if output_token_map_path:
        print(f"Writing token mapping to: {output_token_map_path}")
        with open(output_token_map_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)

            # Write header
            writer.writerow(['row_index', 'token_id', 'token_text'])

            # Write token mappings
            for i, (token_id, token_text) in enumerate(zip(token_ids, token_texts)):
                # Use repr() to escape special characters (newlines, tabs, etc.)
                # This makes them visible as \n, \t instead of actual control characters
                escaped_text = repr(token_text)[1:-1]  # Remove outer quotes from repr()

                writer.writerow([i, token_id, escaped_text])

        print(f"✓ Token mapping saved to: {output_token_map_path}")

    # Skip CSV output - only generating TXT file
    # if output_csv_path:
    #     print(f"Writing translated CSV to: {output_csv_path}")
    #     with open(output_csv_path, 'w', newline='') as f:
    #         writer = csv.writer(f)
    #
    #         # Write header with token_text column
    #         new_header = ['query_token_id', 'token_text'] + header[1:]
    #         writer.writerow(new_header)
    #
    #         # Write data rows
    #         for i, row in enumerate(rows_data):
    #             # Insert token_text as second column
    #             new_row = [row[0], token_texts[i]] + row[1:]
    #             writer.writerow(new_row)
    #
    #     print(f"✓ Translated CSV saved to: {output_csv_path}")

    # Write full context to text file
    if output_txt_path:
        print(f"Writing full context to: {output_txt_path}")
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            # Write header
            f.write("="*80 + "\n")
            f.write("FULL TOKEN SEQUENCE\n")
            f.write("="*80 + "\n\n")

            # Write concatenated tokens
            full_context = "".join(token_texts)
            f.write("CONCATENATED:\n")
            f.write("-"*80 + "\n")
            f.write(full_context)
            f.write("\n\n")

            # Write individual tokens with IDs
            f.write("="*80 + "\n")
            f.write("INDIVIDUAL TOKENS:\n")
            f.write("="*80 + "\n\n")

            for i, (token_id, token_text) in enumerate(zip(token_ids, token_texts)):
                f.write(f"Row {i}: {repr(token_text)} (ID: {token_id})\n")

        print(f"✓ Full context saved to: {output_txt_path}")

    return token_ids, token_texts


def main():
    parser = argparse.ArgumentParser(
        description='Translate token IDs in attention weight CSV files to readable text'
    )
    parser.add_argument(
        'csv_file',
        type=str,
        help='Path to input CSV file with token IDs'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='Qwen/Qwen3-VL-30B-A3B-Instruct',
        help='Model name for tokenizer (default: Qwen/Qwen3-VL-30B-A3B-Instruct)'
    )
    # Removed --output option since we only generate TXT files now
    parser.add_argument(
        '--text-output',
        '-t',
        type=str,
        default=None,
        help='Output text file path (default: auto-generate from input name)'
    )
    parser.add_argument(
        '--token-map',
        '-m',
        type=str,
        default=None,
        help='Output CSV file with token mappings (default: auto-generate from input name)'
    )
    parser.add_argument(
        '--no-tokenizer',
        action='store_true',
        help='Skip loading tokenizer (will just show token IDs)'
    )

    args = parser.parse_args()

    # Verify input file exists
    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        return 1

    # Auto-generate text output filename if not specified
    if args.text_output is None:
        output_txt = csv_path.parent / f"{csv_path.stem}_context.txt"
    else:
        output_txt = Path(args.text_output)

    # Auto-generate token mapping CSV filename if not specified
    if args.token_map is None:
        output_token_map = csv_path.parent / f"{csv_path.stem}_tokens.csv"
    else:
        output_token_map = Path(args.token_map)

    # Load tokenizer
    tokenizer = None
    if not args.no_tokenizer:
        try:
            print(f"Loading tokenizer: {args.model}")
            tokenizer = AutoTokenizer.from_pretrained(args.model)
        except Exception as e:
            print(f"Warning: Could not load tokenizer: {e}")
            print("Continuing without tokenizer...")

    # Translate (generate both TXT and token mapping CSV)
    translate_csv(
        csv_path,
        output_csv_path=None,  # Skip CSV output
        output_txt_path=output_txt,
        output_token_map_path=output_token_map,
        tokenizer=tokenizer
    )

    print("\n" + "="*80)
    print("TRANSLATION COMPLETE!")
    print("="*80)
    print(f"Full context text: {output_txt}")
    print(f"Token mapping CSV: {output_token_map}")

    return 0


if __name__ == '__main__':
    exit(main())
