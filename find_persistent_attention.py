#!/usr/bin/env python3
"""
Find elements that consistently receive high attention across multiple generated tokens.
This helps identify which parts of the context remain important throughout generation.
"""
import argparse
import csv
import json
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
import codecs


def load_tokens(token_file: Path) -> List[str]:
    """Load token mapping from CSV file."""
    tokens = []
    with open(token_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if not row or len(row) < 3:
                continue
            escaped_text = row[2]
            try:
                token_text = codecs.decode(escaped_text, 'unicode_escape')
            except Exception:
                token_text = escaped_text
            tokens.append(token_text)
    return tokens


def read_attention_weights(csv_file: Path) -> Tuple[List[int], List[List[float]]]:
    """Read attention weights from CSV file."""
    token_ids = []
    attention_matrix = []

    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header

        for row in reader:
            if not row:
                continue
            token_id = int(row[0])
            weights = [float(w) for w in row[1:] if w]
            token_ids.append(token_id)
            attention_matrix.append(weights)

    return token_ids, attention_matrix


def parse_chatml_structure(tokens: List[str]):
    """Parse the structure to identify ranges for different sections."""
    # Find key sections
    sections = {}

    # Find assistant section (generation starts here)
    assistant_start = -1
    for i, token in enumerate(tokens):
        if "<|im_start|>" in token:
            # Look ahead for role
            role_idx = i + 1
            while role_idx < len(tokens) and tokens[role_idx] not in ['\n', '<', '|']:
                if 'assistant' in tokens[role_idx]:
                    assistant_start = i
                    break
                role_idx += 1
            if assistant_start >= 0:
                break

    if assistant_start >= 0:
        sections['assistant_start'] = assistant_start

        # Find where actual generation starts (after the role declaration)
        gen_start = assistant_start
        while gen_start < len(tokens) and tokens[gen_start] != '\n':
            gen_start += 1
        gen_start += 1  # Skip the newline
        sections['generation_start'] = gen_start

    # Find system and user sections
    for i, token in enumerate(tokens):
        if "<|im_start|>" in token:
            role_idx = i + 1
            while role_idx < len(tokens) and tokens[role_idx] not in ['\n', '<', '|']:
                if 'system' in tokens[role_idx]:
                    sections['system_start'] = i
                elif 'user' in tokens[role_idx]:
                    sections['user_start'] = i
                role_idx += 1

    return sections


def find_high_attention_ranges(
    attention_weights: List[float],
    min_attention: float = 0.001
) -> List[Tuple[int, int, float]]:
    """
    Find continuous ranges with high attention.
    Returns list of (start_idx, end_idx, total_attention).
    """
    ranges = []
    current_start = None
    current_sum = 0.0

    for i, weight in enumerate(attention_weights):
        if weight >= min_attention:
            if current_start is None:
                current_start = i
                current_sum = weight
            else:
                current_sum += weight
        else:
            if current_start is not None:
                ranges.append((current_start, i - 1, current_sum))
                current_start = None
                current_sum = 0.0

    # Handle last range
    if current_start is not None:
        ranges.append((current_start, len(attention_weights) - 1, current_sum))

    return ranges


def analyze_attention_across_tokens(
    csv_file: Path,
    token_file: Path,
    start_token: int,
    end_token: int,
    min_attention: float = 0.001,
    top_n: int = 20,
    exclude_generated: bool = True
):
    """Analyze which ranges consistently receive high attention."""

    print(f"Loading tokens from: {token_file}")
    tokens = load_tokens(token_file)

    print(f"Loading attention weights from: {csv_file}")
    token_ids, attention_matrix = read_attention_weights(csv_file)

    print(f"Analyzing tokens {start_token} to {end_token}")
    print(f"Minimum attention threshold: {min_attention}")
    if exclude_generated:
        print(f"Excluding generated tokens from analysis (focus on context)")
    print()

    # Parse structure
    sections = parse_chatml_structure(tokens)
    print(f"Detected sections: {sections}")
    print()

    # Track attention to each token position across all generation steps
    position_attention = defaultdict(list)  # position -> [attention values across tokens]

    # Analyze each generated token
    for query_token_idx in range(start_token, min(end_token + 1, len(attention_matrix))):
        weights = attention_matrix[query_token_idx]

        # Record attention for each position (excluding generated tokens if requested)
        max_pos = sections.get('generation_start', len(weights)) if exclude_generated else len(weights)

        for pos in range(min(max_pos, len(weights))):
            weight = weights[pos]
            if weight >= min_attention:
                position_attention[pos].append(weight)

    # Calculate statistics for each position
    position_stats = []
    for pos, attention_values in position_attention.items():
        if len(attention_values) == 0:
            continue

        stats = {
            'position': pos,
            'count': len(attention_values),  # How many tokens attended to this
            'total': sum(attention_values),
            'mean': sum(attention_values) / len(attention_values),
            'max': max(attention_values),
            'min': min(attention_values),
            'frequency': len(attention_values) / (end_token - start_token + 1)  # % of tokens that attended
        }
        position_stats.append(stats)

    # Sort by total attention
    position_stats.sort(key=lambda x: x['total'], reverse=True)

    # Display results
    print("=" * 120)
    print(f"TOP {top_n} POSITIONS BY TOTAL ATTENTION ACROSS GENERATED TOKENS")
    print("=" * 120)
    print(f"{'Rank':<6} {'Pos':<8} {'Token':<40} {'Freq':<8} {'Total':<12} {'Mean':<12} {'Max':<12} {'Min':<12}")
    print("-" * 120)

    for i, stats in enumerate(position_stats[:top_n], 1):
        pos = stats['position']
        token_text = tokens[pos] if pos < len(tokens) else '???'

        # Truncate/escape token text for display
        display_token = repr(token_text)
        if len(display_token) > 37:
            display_token = display_token[:34] + "..."

        print(f"{i:<6} {pos:<8} {display_token:<40} {stats['frequency']:>6.1%} "
              f"{stats['total']:>11.6f} {stats['mean']:>11.6f} "
              f"{stats['max']:>11.6f} {stats['min']:>11.6f}")

    # Find continuous ranges that are consistently attended
    print("\n" + "=" * 120)
    print("CONTINUOUS HIGH-ATTENTION RANGES")
    print("=" * 120)

    # Group consecutive positions
    ranges = []
    current_range = []

    for i, stats in enumerate(position_stats):
        if not current_range:
            current_range = [stats]
        elif stats['position'] == current_range[-1]['position'] + 1:
            current_range.append(stats)
        else:
            if len(current_range) >= 3:  # Only consider ranges of 3+ tokens
                ranges.append(current_range)
            current_range = [stats]

    if len(current_range) >= 3:
        ranges.append(current_range)

    # Sort ranges by total attention
    ranges.sort(key=lambda r: sum(s['total'] for s in r), reverse=True)

    print(f"\nFound {len(ranges)} continuous ranges with 3+ tokens\n")

    for i, range_stats in enumerate(ranges[:20], 1):
        start_pos = range_stats[0]['position']
        end_pos = range_stats[-1]['position']
        total_attention = sum(s['total'] for s in range_stats)
        avg_frequency = sum(s['frequency'] for s in range_stats) / len(range_stats)

        # Get token text
        range_tokens = tokens[start_pos:end_pos + 1]
        range_text = "".join(range_tokens)
        if len(range_text) > 100:
            range_text = range_text[:97] + "..."

        print(f"{i}. Range [{start_pos}..{end_pos}] ({end_pos - start_pos + 1} tokens)")
        print(f"   Total attention: {total_attention:.6f}")
        print(f"   Avg frequency: {avg_frequency:.1%}")
        print(f"   Text: {repr(range_text)}")
        print()

    # Analyze by section
    if sections:
        print("=" * 120)
        print("ATTENTION BY SECTION")
        print("=" * 120)

        section_attention = defaultdict(float)
        section_counts = defaultdict(int)

        for pos, attention_values in position_attention.items():
            total_attn = sum(attention_values)

            if 'system_start' in sections and pos >= sections['system_start']:
                if 'user_start' in sections and pos < sections['user_start']:
                    section_attention['system'] += total_attn
                    section_counts['system'] += 1

            if 'user_start' in sections and pos >= sections['user_start']:
                if 'generation_start' in sections and pos < sections['generation_start']:
                    section_attention['user'] += total_attn
                    section_counts['user'] += 1

            if 'generation_start' in sections and pos >= sections['generation_start']:
                section_attention['generated'] += total_attn
                section_counts['generated'] += 1

        print(f"\n{'Section':<20} {'Total Attention':<20} {'Unique Positions':<20} {'Avg per Position':<20}")
        print("-" * 80)
        for section in ['system', 'user', 'generated']:
            if section in section_attention:
                avg = section_attention[section] / section_counts[section] if section_counts[section] > 0 else 0
                print(f"{section:<20} {section_attention[section]:<20.6f} "
                      f"{section_counts[section]:<20} {avg:<20.6f}")


def main():
    parser = argparse.ArgumentParser(
        description='Find elements with persistent high attention across generated tokens',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze tokens 8420-8450 in the generation
  python find_persistent_attention.py \\
      --csv attention_weights.csv \\
      --tokens tokens.csv \\
      --start 8420 --end 8450

  # Use lower threshold to catch more positions
  python find_persistent_attention.py \\
      --csv attention_weights.csv \\
      --tokens tokens.csv \\
      --start 8420 --end 8450 \\
      --min-attention 0.0005 \\
      --top-n 50
        """
    )

    parser.add_argument('--csv', type=Path, required=True,
                        help='Path to attention weights CSV file')
    parser.add_argument('--tokens', type=Path, required=True,
                        help='Path to tokens CSV file')
    parser.add_argument('--start', type=int, required=True,
                        help='Start token index for generation')
    parser.add_argument('--end', type=int, required=True,
                        help='End token index for generation')
    parser.add_argument('--min-attention', type=float, default=0.001,
                        help='Minimum attention threshold (default: 0.001)')
    parser.add_argument('--top-n', type=int, default=20,
                        help='Number of top positions to display (default: 20)')

    args = parser.parse_args()

    analyze_attention_across_tokens(
        args.csv,
        args.tokens,
        args.start,
        args.end,
        args.min_attention,
        args.top_n
    )

    return 0


if __name__ == '__main__':
    exit(main())
