#!/usr/bin/env python3
"""
Extract a specific row from a CSV file for easier viewing.
"""

import csv
import sys

def extract_row(input_file, row_number, output_file=None):
    """
    Extract a specific row from a CSV file.

    Args:
        input_file: Path to the input CSV file
        row_number: Row number to extract (0-based, excluding header)
        output_file: Optional path to save the extracted row
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        # Read all rows to find the specific one
        for i, row in enumerate(reader):
            if i == row_number:
                print(f"\n{'='*60}")
                print(f"Row {row_number} from {input_file}")
                print(f"{'='*60}")

                # Print row data in a readable format
                for key, value in row.items():
                    # Handle None values for both key and value
                    display_key = key if key is not None else "None"
                    display_value = value if value is not None else ""
                    print(f"{display_key:15s}: {display_value}")

                # Save to file if specified
                if output_file:
                    with open(output_file, 'w', encoding='utf-8') as out_f:
                        writer = csv.DictWriter(out_f, fieldnames=row.keys())
                        writer.writeheader()
                        writer.writerow(row)
                    print(f"\n✓ Row saved to: {output_file}")

                return row

        print(f"Error: Row {row_number} not found in file.")
        return None


if __name__ == "__main__":
    # Example usage
    input_csv = "attention_weights/archive_20251215_112126/attn_weights_request_000001_layer_01_head_01.csv"

    # Change this to the row you want to extract (0-based index, excluding header)
    row_to_extract = 8425

    # Optional: specify an output file
    output_csv = "extracted_row.csv"

    # Extract and display the row
    extract_row(input_csv, row_to_extract, output_csv)

    print("\n" + "="*60)
    print("To extract a different row, modify 'row_to_extract' in the script")
    print("or run: python extract_row.py <input_file> <row_number> [output_file]")
