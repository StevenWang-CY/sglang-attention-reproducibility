#!/usr/bin/env python3
"""
Analyze where attention weights are "lost" between parent and children nodes.
This helps identify tokens that aren't captured by any child node.
"""
import json
import sys
from pathlib import Path


def analyze_node_coverage(node, depth=0, parent_name="root"):
    """
    Recursively analyze a node to find missing attention weights.
    Returns: (node_value, children_sum, uncaptured_value, details)
    """
    node_name = node.get("name", "unknown")
    node_value = node.get("value", 0.0)
    node_type = node.get("type", "unknown")
    start = node.get("start", -1)
    end = node.get("end", -1)

    children = node.get("children", [])

    if not children:
        # Leaf node - no children to sum
        return {
            "name": node_name,
            "type": node_type,
            "range": f"[{start}..{end}]",
            "tokens": end - start + 1 if start >= 0 and end >= 0 else 0,
            "value": node_value,
            "children_sum": 0.0,
            "uncaptured": 0.0,
            "uncaptured_pct": 0.0,
            "depth": depth,
            "has_issue": False
        }

    # Sum up children's values
    children_sum = sum(child.get("value", 0.0) for child in children)
    uncaptured = node_value - children_sum
    uncaptured_pct = (uncaptured / node_value * 100) if node_value > 0 else 0

    result = {
        "name": node_name,
        "type": node_type,
        "range": f"[{start}..{end}]",
        "tokens": end - start + 1 if start >= 0 and end >= 0 else 0,
        "value": node_value,
        "children_sum": children_sum,
        "uncaptured": uncaptured,
        "uncaptured_pct": uncaptured_pct,
        "depth": depth,
        "has_issue": abs(uncaptured) > 0.001,  # Flag if uncaptured > 0.1%
        "children_details": []
    }

    # Recursively analyze children
    for child in children:
        child_analysis = analyze_node_coverage(child, depth + 1, node_name)
        result["children_details"].append(child_analysis)

    return result


def find_token_gaps(node, all_ranges=None):
    """
    Find gaps in token coverage - tokens in parent range not covered by any child.
    """
    if all_ranges is None:
        all_ranges = []

    start = node.get("start", -1)
    end = node.get("end", -1)

    if start < 0 or end < 0:
        return []

    children = node.get("children", [])
    if not children:
        return []

    # Get all child ranges
    child_ranges = []
    for child in children:
        child_start = child.get("start", -1)
        child_end = child.get("end", -1)
        if child_start >= 0 and child_end >= 0:
            child_ranges.append((child_start, child_end))

    # Sort child ranges by start position
    child_ranges.sort()

    # Find gaps
    gaps = []
    current_pos = start

    for child_start, child_end in child_ranges:
        if child_start > current_pos:
            # There's a gap
            gaps.append({
                "parent": node.get("name", "unknown"),
                "parent_range": f"[{start}..{end}]",
                "gap_range": f"[{current_pos}..{child_start-1}]",
                "gap_size": child_start - current_pos
            })
        current_pos = max(current_pos, child_end + 1)

    # Check if there's a gap at the end
    if current_pos <= end:
        gaps.append({
            "parent": node.get("name", "unknown"),
            "parent_range": f"[{start}..{end}]",
            "gap_range": f"[{current_pos}..{end}]",
            "gap_size": end - current_pos + 1
        })

    # Recursively find gaps in children
    for child in children:
        child_gaps = find_token_gaps(child)
        gaps.extend(child_gaps)

    return gaps


def print_analysis(analysis, indent=0, show_all=False):
    """Print analysis results in a tree format."""
    prefix = "  " * indent

    if analysis["has_issue"] or show_all:
        print(f"{prefix}{'[!] ' if analysis['has_issue'] else ''}{analysis['name']}")
        print(f"{prefix}    Type: {analysis['type']}, Range: {analysis['range']} ({analysis['tokens']} tokens)")
        print(f"{prefix}    Node value: {analysis['value']:.6f}")
        print(f"{prefix}    Children sum: {analysis['children_sum']:.6f}")
        print(f"{prefix}    UNCAPTURED: {analysis['uncaptured']:.6f} ({analysis['uncaptured_pct']:.2f}%)")
        print()

    # Recursively print children
    for child in analysis.get("children_details", []):
        print_analysis(child, indent + 1, show_all)


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_missing_attention.py <flamegraph_json_or_html>")
        print("\nThis script analyzes where attention weights are 'lost' between parent and child nodes.")
        return 1

    input_file = Path(sys.argv[1])

    if not input_file.exists():
        print(f"Error: File not found: {input_file}")
        return 1

    # Load the data
    if input_file.suffix == '.json':
        with open(input_file, 'r') as f:
            data = json.load(f)
    elif input_file.suffix == '.html':
        # Extract JSON from HTML
        with open(input_file, 'r') as f:
            content = f.read()

        # Find the data = {...}; line
        start_marker = "const data = "
        start_idx = content.find(start_marker)
        if start_idx < 0:
            print("Error: Could not find JSON data in HTML file")
            return 1

        start_idx += len(start_marker)
        end_idx = content.find(";\n", start_idx)

        json_str = content[start_idx:end_idx]
        data = json.loads(json_str)
    else:
        print(f"Error: Unsupported file format: {input_file.suffix}")
        return 1

    print("="*80)
    print("ATTENTION WEIGHT COVERAGE ANALYSIS")
    print("="*80)
    print()

    # Analyze the hierarchy
    analysis = analyze_node_coverage(data)

    # Print nodes with uncaptured attention
    print("NODES WITH UNCAPTURED ATTENTION (>0.1%):")
    print("-"*80)
    print_analysis(analysis, show_all=False)

    # Find token gaps
    print("\n" + "="*80)
    print("TOKEN COVERAGE GAPS")
    print("="*80)
    gaps = find_token_gaps(data)

    if gaps:
        print(f"\nFound {len(gaps)} gaps in token coverage:\n")
        for i, gap in enumerate(gaps, 1):
            print(f"{i}. Parent: {gap['parent']}")
            print(f"   Parent range: {gap['parent_range']}")
            print(f"   Gap: {gap['gap_range']} ({gap['gap_size']} tokens)")
            print()
    else:
        print("\nNo gaps found - all tokens are covered by children!")

    # Summary statistics
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Root node value: {analysis['value']:.6f}")
    print(f"Children sum: {analysis['children_sum']:.6f}")
    print(f"Total uncaptured: {analysis['uncaptured']:.6f} ({analysis['uncaptured_pct']:.2f}%)")
    print(f"Total token coverage gaps: {len(gaps)}")

    return 0


if __name__ == '__main__':
    exit(main())
