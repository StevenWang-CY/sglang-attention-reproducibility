#!/usr/bin/env python3
"""
Script to read attention weight CSV files and convert token IDs to human-readable text.

Usage:
    python verify_attention_token.py <csv_file> [--model MODEL_NAME]

Example:
    python verify_attention_token.py attention_weights/attn_weights_request_000001_layer_00_head_00.csv
    python verify_attention_token.py attention_weights/attn_weights_request_000001_layer_00_head_00.csv --model meta-llama/Llama-3.1-8B-Instruct
"""

import csv
import argparse
from pathlib import Path
from transformers import AutoTokenizer


def load_tokenizer(model_name):
    """Load the tokenizer for the specified model."""
    print(f"Loading tokenizer for model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def read_and_decode_csv(csv_path, tokenizer):
    """Read CSV file and decode token IDs to text."""
    print(f"\nReading CSV file: {csv_path}")

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)

        print("\n" + "="*80)
        print(f"CSV Header: {header[0]} (token ID � decoded text)")
        print("="*80)

        rows_data = []
        for row in reader:
            if not row or not row[0]:  # Skip empty rows
                continue

            token_id_str = row[0]
            try:
                token_id = int(token_id_str)
                # Decode single token
                token_text = tokenizer.decode([token_id])
                # Also get the raw token string representation
                try:
                    token_repr = tokenizer.convert_ids_to_tokens([token_id])[0]
                except:
                    token_repr = "N/A"

                rows_data.append({
                    'token_id': token_id,
                    'token_text': token_text,
                    'token_repr': token_repr,
                    'attention_weights': row[1:]
                })
            except ValueError:
                print(f"Warning: Could not parse token ID: {token_id_str}")
                continue

        return header, rows_data


def display_tokens(rows_data, show_weights=False, max_weights=5):
    """Display the decoded tokens in a readable format."""
    print("\n" + "="*80)
    print("DECODED TOKENS:")
    print("="*80)
    print(f"{'Index':<6} {'Token ID':<10} {'Token Repr':<20} {'Decoded Text':<30}")
    print("-"*80)

    for idx, row in enumerate(rows_data):
        token_id = row['token_id']
        token_text = repr(row['token_text'])  # Use repr to show special chars
        token_repr = row['token_repr']

        print(f"{idx:<6} {token_id:<10} {token_repr:<20} {token_text:<30}")

        if show_weights:
            # Show first few attention weights
            weights = row['attention_weights'][:max_weights]
            weights_str = ', '.join([f"{float(w):.4f}" if w else "empty" for w in weights])
            print(f"       Attention (first {max_weights}): [{weights_str}, ...]")
            print()

    print("="*80)
    print(f"Total tokens: {len(rows_data)}")
    print("="*80)


def reconstruct_text(rows_data, tokenizer):
    """Reconstruct the full text from all tokens."""
    token_ids = [row['token_id'] for row in rows_data]
    full_text = tokenizer.decode(token_ids)

    print("\n" + "="*80)
    print("RECONSTRUCTED FULL TEXT:")
    print("="*80)
    print(full_text)
    print("="*80)

    return full_text


def main():
    parser = argparse.ArgumentParser(
        description='Convert token IDs in attention CSV to readable text'
    )
    parser.add_argument(
        'csv_file',
        nargs='?',
        type=str,
        default='attention_weights/attn_weights_request_000001_layer_00_head_00.csv',
        help='Path to the attention weights CSV file (default: attention_weights/attn_weights_request_000001_layer_00_head_00.csv)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='Qwen/Qwen3-VL-30B-A3B-Instruct',
        help='Model name for tokenizer (default: Qwen/Qwen3-VL-30B-A3B-Instruct)'
    )
    parser.add_argument(
        '--show-weights',
        action='store_true',
        help='Show first few attention weights for each token'
    )
    parser.add_argument(
        '--max-weights',
        type=int,
        default=5,
        help='Number of attention weights to show (default: 5)'
    )
    parser.add_argument(
        '--no-reconstruct',
        action='store_true',
        help='Skip reconstructing full text'
    )

    args = parser.parse_args()

    # Verify CSV file exists
    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        return 1

    # Load tokenizer
    try:
        tokenizer = load_tokenizer(args.model)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return 1

    # Read and decode CSV
    try:
        header, rows_data = read_and_decode_csv(csv_path, tokenizer)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Display results
    display_tokens(rows_data, show_weights=args.show_weights, max_weights=args.max_weights)

    # Reconstruct full text
    if not args.no_reconstruct:
        try:
            reconstruct_text(rows_data, tokenizer)
        except Exception as e:
            print(f"Warning: Could not reconstruct full text: {e}")

    return 0


if __name__ == '__main__':
    exit(main())
