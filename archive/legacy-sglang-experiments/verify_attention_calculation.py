#!/usr/bin/env python3
"""
Verify that the attention calculation in the flamegraph is correct.
Check if node.value actually matches the sum of attention weights in its range.
"""
import json
import csv
from pathlib import Path
import sys


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


def find_node_by_range(node, start_range, end_range):
    """Find a node by its exact range."""
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


def verify_node_attention(node, attention_weights, depth=0, max_depth=3):
    """
    Verify that a node's value matches the actual sum of attention in its range.
    Also check children recursively.
    """
    name = node.get("name", "unknown")
    start = node.get("start", -1)
    end = node.get("end", -1)
    node_value = node.get("value", 0.0)

    # Calculate actual attention from raw weights
    if start >= 0 and end >= 0 and start < len(attention_weights):
        actual_sum = sum(attention_weights[start:min(end+1, len(attention_weights))])
    else:
        actual_sum = 0.0

    difference = abs(node_value - actual_sum)
    matches = difference < 0.000001

    indent = "  " * depth
    status = "✓" if matches else "✗"

    print(f"{status} {indent}{name[:70]}")
    print(f"  {indent}  Range: [{start}..{end}] ({end-start+1 if start >= 0 and end >= 0 else 0} tokens)")
    print(f"  {indent}  Node value:   {node_value:.10f}")
    print(f"  {indent}  Actual sum:   {actual_sum:.10f}")
    print(f"  {indent}  Difference:   {difference:.10f}")

    if not matches:
        print(f"  {indent}  ⚠ MISMATCH DETECTED!")

    # Check children sum
    children = node.get("children", [])
    if children:
        children_sum = sum(child.get("value", 0.0) for child in children)
        print(f"  {indent}  Children sum: {children_sum:.10f}")
        print(f"  {indent}  Uncaptured:   {node_value - children_sum:.10f} ({(node_value - children_sum)/node_value*100 if node_value > 0 else 0:.2f}%)")

    print()

    # Recursively check children (up to max_depth)
    if depth < max_depth and children:
        for child in children:
            verify_node_attention(child, attention_weights, depth + 1, max_depth)


def main():
    if len(sys.argv) < 4:
        print("Usage: python verify_attention_calculation.py <html_file> <tokens_csv> <attention_csv>")
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
    start_idx += len(start_marker)
    end_idx = content.find(";\n", start_idx)
    json_str = content[start_idx:end_idx]
    data = json.loads(json_str)

    # Find the browser_state node
    browser_state = find_node_by_range(data, 3999, 8417)

    if not browser_state:
        print("Error: Could not find <browser_state> node at range [3999..8417]")
        return 1

    print("\n" + "="*100)
    print("VERIFYING ATTENTION CALCULATIONS")
    print("="*100)
    print("\nChecking if node.value matches the actual sum of attention weights in each node's range...")
    print()

    verify_node_attention(browser_state, attention_weights, depth=0, max_depth=2)

    # Also check the <html /> child specifically
    html_node = None
    for child in browser_state.get("children", []):
        if "<html />" in child.get("name", ""):
            html_node = child
            break

    if html_node:
        print("\n" + "="*100)
        print("DETAILED CHECK OF <html /> NODE")
        print("="*100)
        print()
        verify_node_attention(html_node, attention_weights, depth=0, max_depth=1)

    return 0


if __name__ == '__main__':
    exit(main())
