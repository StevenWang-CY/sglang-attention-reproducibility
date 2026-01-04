#!/usr/bin/env python3
"""
Show the actual tokens that are uncaptured by child nodes.
This helps understand where attention weights are "lost".
"""
import json
import sys
import csv
from pathlib import Path


def load_tokens(token_file: Path):
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
                import codecs
                token_text = codecs.decode(escaped_text, 'unicode_escape')
            except Exception:
                token_text = escaped_text
            tokens.append(token_text)
    return tokens


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
        return weights


def get_child_ranges(node):
    """Get all child token ranges."""
    children = node.get("children", [])
    ranges = []
    for child in children:
        start = child.get("start", -1)
        end = child.get("end", -1)
        if start >= 0 and end >= 0:
            ranges.append((start, end))
    return sorted(ranges)


def find_uncaptured_ranges(parent_start, parent_end, child_ranges):
    """Find token ranges not covered by any child."""
    uncaptured = []
    current_pos = parent_start

    for child_start, child_end in child_ranges:
        if child_start > current_pos:
            uncaptured.append((current_pos, child_start - 1))
        current_pos = max(current_pos, child_end + 1)

    if current_pos <= parent_end:
        uncaptured.append((current_pos, parent_end))

    return uncaptured


def main():
    if len(sys.argv) < 4:
        print("Usage: python show_uncaptured_tokens.py <html_file> <tokens_csv> <attention_csv>")
        print("\nExample:")
        print("  python show_uncaptured_tokens.py flamegraph.html tokens.csv attention.csv")
        return 1

    html_file = Path(sys.argv[1])
    token_file = Path(sys.argv[2])
    attention_file = Path(sys.argv[3])

    # Load data
    print("Loading data...")
    tokens = load_tokens(token_file)
    attention_weights = read_attention_weights(attention_file, -1)

    # Load HTML and extract JSON
    with open(html_file, 'r') as f:
        content = f.read()

    start_marker = "const data = "
    start_idx = content.find(start_marker)
    if start_idx < 0:
        print("Error: Could not find JSON data in HTML file")
        return 1

    start_idx += len(start_marker)
    end_idx = content.find(";\n", start_idx)
    json_str = content[start_idx:end_idx]
    data = json.loads(json_str)

    # Find the browser_state node (the one in range 3999-8417)
    def find_node_by_range(node, start_range, end_range):
        node_start = node.get("start", -1)
        node_end = node.get("end", -1)
        node_name = node.get("name", "")

        if "<browser_state>" in node_name and node_start == start_range and node_end == end_range:
            return node

        for child in node.get("children", []):
            result = find_node_by_range(child, start_range, end_range)
            if result:
                return result
        return None

    browser_state = find_node_by_range(data, 3999, 8417)

    if not browser_state:
        print("Error: Could not find <browser_state> node")
        return 1

    print("\n" + "="*80)
    print(f"BROWSER_STATE NODE ANALYSIS")
    print("="*80)
    print(f"Node: {browser_state['name']}")
    print(f"Range: [{browser_state['start']}..{browser_state['end']}]")
    print(f"Total attention: {browser_state['value']:.6f}")

    # Calculate children sum
    children_sum = sum(child.get('value', 0.0) for child in browser_state.get('children', []))
    uncaptured_value = browser_state['value'] - children_sum

    print(f"Children sum: {children_sum:.6f}")
    print(f"UNCAPTURED: {uncaptured_value:.6f} ({uncaptured_value/browser_state['value']*100:.2f}%)")

    # Find uncaptured token ranges
    child_ranges = get_child_ranges(browser_state)
    uncaptured_ranges = find_uncaptured_ranges(
        browser_state['start'],
        browser_state['end'],
        child_ranges
    )

    print(f"\nFound {len(uncaptured_ranges)} uncaptured token ranges:")
    print("-"*80)

    total_uncaptured_attention = 0.0

    for i, (start, end) in enumerate(uncaptured_ranges, 1):
        # Calculate attention for this range
        range_attention = sum(attention_weights[start:end+1])
        total_uncaptured_attention += range_attention

        # Get the tokens
        range_tokens = tokens[start:end+1]
        token_text = "".join(range_tokens)

        # Limit display length
        display_text = token_text if len(token_text) <= 100 else token_text[:97] + "..."

        print(f"\n{i}. Range [{start}..{end}] ({end-start+1} tokens)")
        print(f"   Attention: {range_attention:.6f}")
        print(f"   Tokens: {repr(display_text)}")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total uncaptured attention from gaps: {total_uncaptured_attention:.6f}")
    print(f"Expected uncaptured (parent - children): {uncaptured_value:.6f}")
    print(f"Difference: {abs(total_uncaptured_attention - uncaptured_value):.6f}")

    if abs(total_uncaptured_attention - uncaptured_value) < 0.0001:
        print("\n✓ Gap analysis matches! All uncaptured attention accounted for.")
    else:
        print("\n✗ Mismatch detected - there may be overlapping child ranges or other issues.")

    return 0


if __name__ == '__main__':
    exit(main())
