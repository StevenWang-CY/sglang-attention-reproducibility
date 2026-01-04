#!/usr/bin/env python3
"""
Detailed analysis of where attention is lost in the browser_state hierarchy.
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


def analyze_node_recursive(node, attention_weights, tokens, depth=0):
    """Recursively analyze node to understand attention distribution."""
    indent = "  " * depth
    name = node.get("name", "unknown")
    start = node.get("start", -1)
    end = node.get("end", -1)
    node_value = node.get("value", 0.0)

    # Calculate actual attention sum from raw weights
    actual_attention = 0.0
    if start >= 0 and end >= 0 and start < len(attention_weights):
        actual_attention = sum(attention_weights[start:min(end+1, len(attention_weights))])

    children = node.get("children", [])
    children_sum = sum(child.get("value", 0.0) for child in children)

    uncaptured = node_value - children_sum
    uncaptured_pct = (uncaptured / node_value * 100) if node_value > 0 else 0

    # Get covered ranges
    child_ranges = []
    for child in children:
        child_start = child.get("start", -1)
        child_end = child.get("end", -1)
        if child_start >= 0 and child_end >= 0:
            child_ranges.append((child_start, child_end))

    # Find uncovered tokens within this node's range
    child_ranges.sort()
    uncovered_ranges = []
    current_pos = start

    for child_start, child_end in child_ranges:
        if child_start > current_pos:
            uncovered_ranges.append((current_pos, child_start - 1))
        current_pos = max(current_pos, child_end + 1)

    if current_pos <= end:
        uncovered_ranges.append((current_pos, end))

    # Calculate attention in gaps
    gap_attention = 0.0
    for gap_start, gap_end in uncovered_ranges:
        if gap_start < len(attention_weights):
            gap_attention += sum(attention_weights[gap_start:min(gap_end+1, len(attention_weights))])

    result = {
        "depth": depth,
        "name": name[:80],
        "range": f"[{start}..{end}]",
        "tokens_count": end - start + 1 if start >= 0 and end >= 0 else 0,
        "node_value": node_value,
        "actual_attention": actual_attention,
        "children_sum": children_sum,
        "gap_attention": gap_attention,
        "uncaptured": uncaptured,
        "uncaptured_pct": uncaptured_pct,
        "num_children": len(children),
        "has_issue": uncaptured > 0.001,
    }

    return result


def main():
    if len(sys.argv) < 4:
        print("Usage: python analyze_browser_state_detail.py <html_file> <tokens_csv> <attention_csv>")
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
        print("Error: Could not find <browser_state> node at range [3999..8417]")
        return 1

    print("\n" + "="*100)
    print(f"DETAILED BROWSER_STATE ANALYSIS")
    print("="*100)

    # Analyze browser_state itself
    bs_analysis = analyze_node_recursive(browser_state, attention_weights, tokens, 0)
    print(f"\nBrowser State Node:")
    print(f"  Range: {bs_analysis['range']} ({bs_analysis['tokens_count']} tokens)")
    print(f"  Node value: {bs_analysis['node_value']:.6f}")
    print(f"  Actual attention (raw sum): {bs_analysis['actual_attention']:.6f}")
    print(f"  Children sum: {bs_analysis['children_sum']:.6f}")
    print(f"  Gap attention: {bs_analysis['gap_attention']:.6f}")
    print(f"  UNCAPTURED: {bs_analysis['uncaptured']:.6f} ({bs_analysis['uncaptured_pct']:.2f}%)")

    # Analyze direct children
    print(f"\n{'='*100}")
    print(f"DIRECT CHILDREN OF BROWSER_STATE:")
    print(f"{'='*100}")
    print(f"{'Name':<50} {'Range':<18} {'Value':>12} {'Children':>12} {'Gap':>12} {'Uncap':>12} {'%':>8}")
    print("-"*100)

    children = browser_state.get("children", [])
    for child in children:
        analysis = analyze_node_recursive(child, attention_weights, tokens, 1)
        flag = "⚠" if analysis['has_issue'] else " "
        print(f"{flag} {analysis['name']:<48} {analysis['range']:<18} "
              f"{analysis['node_value']:>12.6f} {analysis['children_sum']:>12.6f} "
              f"{analysis['gap_attention']:>12.6f} {analysis['uncaptured']:>12.6f} "
              f"{analysis['uncaptured_pct']:>7.2f}%")

    # Now drill down into the <html /> node specifically
    html_node = None
    for child in children:
        if "<html />" in child.get("name", ""):
            html_node = child
            break

    if html_node:
        print(f"\n{'='*100}")
        print(f"DRILLING INTO <html /> NODE:")
        print(f"{'='*100}")

        html_analysis = analyze_node_recursive(html_node, attention_weights, tokens, 1)
        print(f"\n<html /> Node:")
        print(f"  Range: {html_analysis['range']} ({html_analysis['tokens_count']} tokens)")
        print(f"  Node value: {html_analysis['node_value']:.6f}")
        print(f"  Children sum: {html_analysis['children_sum']:.6f}")
        print(f"  Gap attention: {html_analysis['gap_attention']:.6f}")
        print(f"  UNCAPTURED: {html_analysis['uncaptured']:.6f} ({html_analysis['uncaptured_pct']:.2f}%)")

        print(f"\nDirect children of <html />:")
        print(f"{'Name':<50} {'Range':<18} {'Value':>12} {'Children':>12} {'Gap':>12} {'Uncap':>12} {'%':>8}")
        print("-"*100)

        for child in html_node.get("children", []):
            analysis = analyze_node_recursive(child, attention_weights, tokens, 2)
            flag = "⚠" if analysis['has_issue'] else " "
            print(f"{flag} {analysis['name']:<48} {analysis['range']:<18} "
                  f"{analysis['node_value']:>12.6f} {analysis['children_sum']:>12.6f} "
                  f"{analysis['gap_attention']:>12.6f} {analysis['uncaptured']:>12.6f} "
                  f"{analysis['uncaptured_pct']:>7.2f}%")

    print(f"\n{'='*100}")
    print("SUMMARY")
    print(f"{'='*100}")
    print(f"The 'missing' 0.15 attention is distributed across:")
    print(f"  1. Gap tokens (opening/closing tags): {bs_analysis['gap_attention']:.6f}")
    print(f"  2. Recursive gaps in child hierarchies: {bs_analysis['uncaptured'] - bs_analysis['gap_attention']:.6f}")
    print(f"Total uncaptured in browser_state: {bs_analysis['uncaptured']:.6f}")

    return 0


if __name__ == '__main__':
    exit(main())
