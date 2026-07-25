#!/usr/bin/env python3
"""
Script to visualize attention weights as a heatmap.

Usage:
    python visualize_attention_heatmap.py [csv_file] [--output OUTPUT] [--model MODEL]

Example:
    python visualize_attention_heatmap.py
    python visualize_attention_heatmap.py attention_weights/attn_weights_request_000001_layer_00_head_00.csv
    python visualize_attention_heatmap.py --output heatmap.png --model Qwen/Qwen3-VL-30B-A3B-Instruct
"""

import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from transformers import AutoTokenizer
import time
import pandas as pd


def load_tokenizer(model_name):
    """Load the tokenizer for the specified model."""
    print(f"Loading tokenizer for model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def read_attention_csv(csv_path, tokenizer=None, max_tokens=None, start_row=None, end_row=None, start_col=None, end_col=None):
    """Read attention weights from CSV file."""
    print(f"Reading CSV file: {csv_path}")

    t_start = time.time()

    token_ids = []
    token_labels = []
    attention_matrix = []

    t_csv_start = time.time()
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header

        # Apply row range
        start_idx = start_row if start_row is not None else 0
        end_idx = end_row if end_row is not None else float('inf')

        for row_idx, row in enumerate(reader):
            # Skip rows before start_idx
            if row_idx < start_idx:
                continue
            # Stop at end_idx
            if row_idx >= end_idx:
                break

            if not row or not row[0]:
                continue

            try:
                token_id = int(row[0])
                token_ids.append(token_id)

                # Decode token for label
                if tokenizer:
                    try:
                        token_text = tokenizer.decode([token_id])
                        # Use repr to show special characters, limit length
                        token_label = repr(token_text)[:20]
                    except:
                        token_label = f"ID:{token_id}"
                else:
                    token_label = f"ID:{token_id}"

                token_labels.append(token_label)

                # Extract attention weights (skip empty values)
                weights = []
                for val in row[1:]:
                    if val and val.strip():
                        weights.append(float(val))
                    else:
                        break  # Stop at first empty value

                attention_matrix.append(weights)

                # Limit number of tokens if specified
                if max_tokens and len(token_ids) >= max_tokens:
                    break

            except ValueError as e:
                print(f"Warning: Could not parse row: {row[0]}")
                continue

    t_csv_end = time.time()
    print(f"⏱️  CSV parsing time: {t_csv_end - t_csv_start:.2f}s")

    # Convert to numpy array and pad if necessary
    t_numpy_start = time.time()
    max_len = max(len(row) for row in attention_matrix)

    # Apply column range if specified
    col_start = start_col if start_col is not None else 0
    col_end = end_col if end_col is not None else max_len

    # Preallocate numpy array (much faster than list padding)
    num_rows = len(attention_matrix)
    selected_cols = col_end - col_start
    padded_matrix = np.zeros((num_rows, selected_cols), dtype=np.float32)

    # Fill in the actual values with column slicing
    for i, row in enumerate(attention_matrix):
        # Get the slice of columns we want
        row_slice = row[col_start:col_end]
        padded_matrix[i, :len(row_slice)] = row_slice

    attention_matrix = padded_matrix
    t_numpy_end = time.time()
    print(f"⏱️  Numpy conversion time: {t_numpy_end - t_numpy_start:.2f}s")

    print(f"Loaded attention matrix: {attention_matrix.shape}")
    print(f"Query tokens (rows): {len(token_labels)}")
    print(f"Key positions (cols): {attention_matrix.shape[1]}")

    t_total = time.time() - t_start
    print(f"⏱️  Total loading time: {t_total:.2f}s")

    return attention_matrix, token_ids, token_labels


def plot_heatmap(attention_matrix, token_labels, output_path=None,
                 title="Attention Weights Heatmap", figsize=None,
                 show_labels=True, vmin=0, vmax=1, dpi=150, compress_level=1, square_size=None):
    """Plot attention weights as a heatmap."""

    t_plot_start = time.time()

    num_rows = attention_matrix.shape[0]
    num_cols = attention_matrix.shape[1]

    # Auto-size figure based on number of tokens
    if figsize is None:
        if square_size is not None:
            # Calculate figure size based on square size
            width = num_cols * square_size
            height = num_rows * square_size
            print(f"Using square_size={square_size}, calculated figure size: ({width:.1f}, {height:.1f})")
        else:
            # Scale figure size proportionally based on data dimensions
            # Use a base size per cell, with reasonable min/max limits
            # Use larger cells when showing labels (tokenizer mode) to accommodate text
            if show_labels:
                # For tokenizer mode: each cell should be at least 0.15 inches (to fit text), but cap at 0.3 inches
                cell_width = min(0.3, max(0.15, 50.0 / num_cols))
                cell_height = min(0.3, max(0.15, 50.0 / num_rows))
            else:
                # For no-tokenizer mode: each cell should be at least 0.01 inches, but cap at 0.1 inches
                cell_width = min(0.1, max(0.01, 20.0 / num_cols))
                cell_height = min(0.1, max(0.01, 20.0 / num_rows))

            width = num_cols * cell_width
            height = num_rows * cell_height

            # Apply hard limits to prevent excessive memory usage
            max_width = 100  # Max 100 inches width
            max_height = 100  # Max 100 inches height

            if width > max_width or height > max_height:
                # Scale down proportionally
                scale = min(max_width / width, max_height / height)
                width *= scale
                height *= scale

            print(f"Auto-calculated figure size: ({width:.1f}, {height:.1f}) for {num_rows}×{num_cols} matrix (show_labels={show_labels})")
        figsize = (width, height)

    print(f"Creating heatmap with figure size: {figsize}")

    t_render_start = time.time()
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    # When using square_size, disable ticks/labels to avoid rendering issues with large figures
    use_labels = show_labels and square_size is None

    # Use imshow for pixel-perfect rendering (each cell = 1 pixel when interpolation='nearest')
    im = ax.imshow(
        attention_matrix,
        cmap='YlOrRd',
        aspect='equal',  # Equal aspect ratio - makes cells square
        interpolation='nearest',  # No interpolation - each cell is a distinct pixel
        vmin=vmin,
        vmax=vmax
    )

    # Add colorbar (much smaller and with consistent font size)
    cbar = plt.colorbar(im, ax=ax, fraction=0.015, pad=0.02, shrink=0.8)  # Much smaller colorbar
    if square_size is None:
        cbar.set_label('Attention Weight', fontsize=4)  # Match other font sizes
        cbar.ax.tick_params(labelsize=4)  # Make colorbar tick labels smaller

    # Set y-tick labels if requested
    if use_labels and token_labels:
        # Set y-ticks to show token labels
        # For large matrices, only show a subset of labels
        if num_rows <= 500:
            # Show all labels for small matrices with smaller font
            ax.set_yticks(range(num_rows))
            ax.set_yticklabels(token_labels, fontsize=4)  # Reduced from 8 to 4
        else:
            # Show every Nth label for large matrices
            step = max(1, num_rows // 50)
            tick_indices = range(0, num_rows, step)
            ax.set_yticks(tick_indices)
            ax.set_yticklabels([token_labels[i] for i in tick_indices], fontsize=3)  # Reduced from 6 to 3

    # Set x-axis tick label font size to match everything else
    ax.tick_params(axis='x', labelsize=4)
    ax.tick_params(axis='y', labelsize=4)

    # Only add labels for reasonably-sized figures
    if square_size is None:
        ax.set_xlabel('Key Position', fontsize=4)  # Consistent font size
        ax.set_ylabel('Query Token', fontsize=4)  # Consistent font size
        # ax.set_title(title, fontsize=14, pad=20)  # Title disabled
    else:
        # For large figures, turn off all decorations
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title('')
        ax.tick_params(left=False, bottom=False)

    t_render_end = time.time()
    print(f"⏱️  Heatmap rendering time: {t_render_end - t_render_start:.2f}s")

    # Adjust layout (skip for large figures to save time)
    if square_size is None:
        plt.tight_layout()

    # Save or show
    if output_path:
        t_save_start = time.time()
        print(f"Saving heatmap to: {output_path}")

        # Faster saving options:
        # - Use lower DPI (controllable via parameter)
        # - Remove bbox_inches='tight' (saves recalculation)
        # - For PNG, reduce compression for faster saving
        save_kwargs = {'dpi': dpi}
        if str(output_path).endswith('.png'):
            # Use pil_kwargs to pass compression level to PIL
            save_kwargs['pil_kwargs'] = {'compress_level': compress_level}  # 0-9, lower is faster

        plt.savefig(output_path, **save_kwargs)
        t_save_end = time.time()
        print(f"⏱️  Image saving time: {t_save_end - t_save_start:.2f}s")
        print(f"✓ Heatmap saved successfully!")
    else:
        print("Displaying heatmap...")
        plt.show()

    plt.close()

    t_plot_total = time.time() - t_plot_start
    print(f"⏱️  Total plotting time: {t_plot_total:.2f}s")


def plot_multi_view_heatmap(attention_matrix, token_labels, output_path=None,
                            title_prefix="Attention Weights"):
    """Create multiple views of the attention matrix."""

    fig = plt.figure(figsize=(20, 10))

    # Full heatmap
    ax1 = plt.subplot(2, 2, 1)
    sns.heatmap(
        attention_matrix,
        cmap='YlOrRd',
        cbar=True,
        square=False,
        xticklabels=False,
        yticklabels=token_labels,
        vmin=0,
        vmax=1,
        ax=ax1
    )
    # ax1.set_title(f"{title_prefix} - Full Matrix", fontsize=12)
    ax1.set_xlabel('Key Position')
    ax1.set_ylabel('Query Token')

    # Log scale heatmap (to see small values)
    ax2 = plt.subplot(2, 2, 2)
    log_matrix = np.log10(attention_matrix + 1e-10)  # Add small value to avoid log(0)
    sns.heatmap(
        log_matrix,
        cmap='viridis',
        cbar=True,
        square=False,
        xticklabels=False,
        yticklabels=token_labels,
        ax=ax2
    )
    ax2.set_title(f"{title_prefix} - Log Scale", fontsize=12)
    ax2.set_xlabel('Key Position')
    ax2.set_ylabel('Query Token')

    # Attention pattern (top 20 tokens)
    max_display = min(20, len(token_labels))
    ax3 = plt.subplot(2, 2, 3)
    subset = attention_matrix[:max_display, :max_display]
    sns.heatmap(
        subset,
        cmap='YlOrRd',
        cbar=True,
        annot=True,
        fmt='.2f',
        square=True,
        xticklabels=range(max_display),
        yticklabels=token_labels[:max_display],
        vmin=0,
        vmax=1,
        ax=ax3
    )
    ax3.set_title(f"{title_prefix} - First {max_display}×{max_display} (Annotated)", fontsize=12)
    ax3.set_xlabel('Key Position')
    ax3.set_ylabel('Query Token')

    # Attention distribution statistics
    ax4 = plt.subplot(2, 2, 4)
    mean_attention = attention_matrix.mean(axis=1)
    max_attention = attention_matrix.max(axis=1)
    ax4.plot(mean_attention, label='Mean Attention', marker='o', markersize=3)
    ax4.plot(max_attention, label='Max Attention', marker='s', markersize=3)
    ax4.set_xlabel('Query Token Index')
    ax4.set_ylabel('Attention Weight')
    ax4.set_title('Attention Statistics per Query Token', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        # Change extension to multi_view
        output_path = Path(output_path)
        multi_view_path = output_path.parent / f"{output_path.stem}_multi_view{output_path.suffix}"
        print(f"Saving multi-view heatmap to: {multi_view_path}")
        plt.savefig(multi_view_path, dpi=150, bbox_inches='tight')
        print(f"✓ Multi-view heatmap saved successfully!")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize attention weights as heatmap'
    )
    parser.add_argument(
        'csv_file',
        nargs='?',
        type=str,
        default='attention_weights/attn_weights_request_000007_layer_00_head_00.csv',
        help='Path to attention weights CSV file (default: attention_weights/attn_weights_request_000001_layer_00_head_00.csv)'
    )
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        default=None,
        help='Output file path (e.g., heatmap.png or heatmap.pdf). If not specified, auto-generates from CSV filename.'
    )
    parser.add_argument(
        '--format',
        '-f',
        type=str,
        choices=['png', 'pdf'],
        default='png',
        help='Output format (png or pdf, default: png)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='Qwen/Qwen3-VL-30B-A3B-Instruct',
        help='Model name for tokenizer (default: Qwen/Qwen3-VL-30B-A3B-Instruct)'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=None,
        help='Maximum number of tokens to display (default: all)'
    )
    parser.add_argument(
        '--no-labels',
        action='store_true',
        help='Hide token labels on y-axis'
    )
    parser.add_argument(
        '--no-tokenizer',
        action='store_true',
        help='Skip loading tokenizer (use token IDs only)'
    )
    parser.add_argument(
        '--multi-view',
        action='store_true',
        help='Create multi-view visualization with different perspectives'
    )
    parser.add_argument(
        '--figsize',
        type=str,
        default=None,
        help='Figure size as "width,height" (e.g., "15,12")'
    )
    parser.add_argument(
        '--start-row',
        type=int,
        default=None,
        help='Start row index (0-based) for visualization'
    )
    parser.add_argument(
        '--end-row',
        type=int,
        default=None,
        help='End row index (exclusive) for visualization'
    )
    parser.add_argument(
        '--start-col',
        type=int,
        default=None,
        help='Start column index (0-based) for visualization'
    )
    parser.add_argument(
        '--end-col',
        type=int,
        default=None,
        help='End column index (exclusive) for visualization'
    )
    parser.add_argument(
        '--compress-level',
        type=int,
        default=1,
        choices=range(0, 10),
        help='PNG compression level (0-9, lower is faster but larger file, default: 1)'
    )
    parser.add_argument(
        '--square-size',
        type=float,
        default=None,
        help='Size of each square in the heatmap (in inches). If set, figure size will be calculated to accommodate all squares. E.g., 0.1 means each cell is 0.1 inches.'
    )

    args = parser.parse_args()

    # Verify CSV file exists
    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        return 1

    # Load tokenizer unless disabled
    tokenizer = None
    if not args.no_tokenizer:
        try:
            tokenizer = load_tokenizer(args.model)
        except Exception as e:
            print(f"Warning: Could not load tokenizer: {e}")
            print("Continuing without tokenizer...")

    # Read attention data
    try:
        attention_matrix, token_ids, token_labels = read_attention_csv(
            csv_path,
            tokenizer=tokenizer,
            max_tokens=args.max_tokens,
            start_row=args.start_row,
            end_row=args.end_row,
            start_col=args.start_col,
            end_col=args.end_col
        )
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Auto-generate output filename if not specified
    output_path = args.output
    if output_path is None:
        # Generate filename from CSV name: replace .csv with specified format
        output_path = csv_path.with_suffix(f'.{args.format}')
        print(f"Auto-generated output filename: {output_path}")
    else:
        # Convert to Path object
        output_path = Path(output_path)

    # Parse figsize if provided
    figsize = None
    if args.figsize:
        try:
            width, height = map(float, args.figsize.split(','))
            figsize = (width, height)
        except:
            print(f"Warning: Invalid figsize format '{args.figsize}', using auto size")

    # Extract layer and head info from filename
    title = f"Attention Weights: {csv_path.name}"

    # Create visualization
    if args.multi_view:
        plot_multi_view_heatmap(
            attention_matrix,
            token_labels,
            output_path=output_path,
            title_prefix=title
        )

    # Always create the main heatmap
    # If --no-tokenizer is used, automatically hide labels (unless --no-labels is explicitly set)
    show_labels = not args.no_labels and not args.no_tokenizer

    plot_heatmap(
        attention_matrix,
        token_labels,
        output_path=output_path,
        title=title,
        figsize=figsize,
        show_labels=show_labels,
        compress_level=args.compress_level,
        square_size=args.square_size
    )

    return 0


if __name__ == '__main__':
    exit(main())
