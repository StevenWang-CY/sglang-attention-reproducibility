#!/usr/bin/env python3
"""
Analyze which tree structure elements (nodes) in the flamegraph maintain high attention
across multiple generated tokens. This helps identify which parts of the context
(e.g., specific HTML elements, browser state sections) are consistently important.
"""
import argparse
import json
import re
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any


def load_flamegraph_from_html(html_file: Path) -> Dict[str, Any]:
    """Extract the flamegraph JSON data from an HTML file."""
    with open(html_file, 'r', encoding='utf-8') as f:
        content = f.read()

    start_marker = "const data = "
    start_idx = content.find(start_marker)
    if start_idx < 0:
        raise ValueError(f"Could not find JSON data in {html_file}")

    start_idx += len(start_marker)

    # Find the closing }; pattern
    marker_pos = content.find("\n};", start_idx)
    if marker_pos < 0:
        raise ValueError(f"Could not find end of JSON data in {html_file}")

    # Include everything up to and including the closing }
    # marker_pos points to the \n before };
    # We want to include the \n and the }, but not the ;
    end_idx = marker_pos + len("\n}")
    json_str = content[start_idx:end_idx]

    return json.loads(json_str)


def extract_node_path(node: Dict[str, Any], path: str = "") -> str:
    """
    Create a path identifier for a node based on its type and position in the tree.
    This allows us to match the same structural element across different flamegraphs.
    """
    node_type = node.get("type", "unknown")
    name = node.get("name", "unknown")
    start = node.get("start", -1)
    end = node.get("end", -1)

    # Create a unique but comparable identifier
    # We use the range to identify nodes since the structure should be the same
    node_id = f"{node_type}[{start}:{end}]:{name[:50]}"

    if path:
        return f"{path}/{node_id}"
    return node_id


def collect_all_nodes(node: Dict[str, Any], path: str = "", nodes_dict: Dict[str, Dict] = None) -> Dict[str, Dict]:
    """
    Recursively collect all nodes in the tree with their paths and metadata.
    Returns a dictionary: path -> node_info
    """
    if nodes_dict is None:
        nodes_dict = {}

    node_path = extract_node_path(node, path)

    # Store node information
    nodes_dict[node_path] = {
        'name': node.get('name', 'unknown'),
        'type': node.get('type', 'unknown'),
        'start': node.get('start', -1),
        'end': node.get('end', -1),
        'value': node.get('value', 0.0),
        'token_count': node.get('end', -1) - node.get('start', -1) + 1 if node.get('start', -1) >= 0 else 0
    }

    # Recursively process children
    for child in node.get('children', []):
        collect_all_nodes(child, node_path, nodes_dict)

    return nodes_dict


def categorize_node(node_info: Dict) -> str:
    """Categorize a node based on its name and type."""
    name = node_info['name']
    node_type = node_info['type']

    if node_type == 'chatml':
        return f"ChatML:{name}"
    elif node_type == 'html_tag':
        # Extract tag name
        if '<browser_state>' in name:
            return "HTML:browser_state"
        elif '<html' in name:
            return "HTML:html_root"
        elif '<page_stats>' in name:
            return "HTML:page_stats"
        elif '<page_info>' in name:
            return "HTML:page_info"
        else:
            return f"HTML:{node_type}"
    elif node_type == 'browser_element':
        # Extract element type
        match = re.search(r'<(\w+)', name)
        if match:
            tag = match.group(1)
            # Check if it's an article
            if 'article' in name.lower():
                return "BrowserElement:article"
            elif 'button' in name.lower():
                return "BrowserElement:button"
            elif 'form' in name.lower():
                return "BrowserElement:form"
            elif tag in ['div', 'span', 'a', 'li']:
                return f"BrowserElement:{tag}"
            else:
                return f"BrowserElement:{tag}"
        return "BrowserElement:other"
    elif node_type == 'text':
        return "Text"
    else:
        return f"Other:{node_type}"


def analyze_tree_persistence(
    html_files: List[Path],
    min_frequency: float = 0.5,
    top_n: int = 30
):
    """
    Analyze which tree nodes appear with high attention across multiple flamegraphs.
    """
    print(f"Analyzing {len(html_files)} flamegraph files...")
    print()

    # Collect nodes from all files
    all_nodes_by_file = []
    file_info = []

    for html_file in html_files:
        try:
            data = load_flamegraph_from_html(html_file)
            nodes = collect_all_nodes(data)
            all_nodes_by_file.append(nodes)

            # Extract token index from filename
            match = re.search(r'token(\d+)', html_file.name)
            token_idx = int(match.group(1)) if match else -1

            file_info.append({
                'file': html_file.name,
                'token_idx': token_idx,
                'num_nodes': len(nodes)
            })
        except Exception as e:
            print(f"Error loading {html_file}: {e}")
            continue

    if not all_nodes_by_file:
        print("No valid flamegraph files found!")
        return

    print(f"Loaded {len(all_nodes_by_file)} files successfully")
    print()

    # Track node statistics across files
    node_stats = defaultdict(lambda: {
        'appearances': 0,
        'values': [],
        'example_name': '',
        'example_type': '',
        'start': -1,
        'end': -1,
        'token_count': 0
    })

    # Analyze each node path
    for nodes_dict in all_nodes_by_file:
        for path, node_info in nodes_dict.items():
            stats = node_stats[path]
            stats['appearances'] += 1
            stats['values'].append(node_info['value'])
            if not stats['example_name']:
                stats['example_name'] = node_info['name']
                stats['example_type'] = node_info['type']
                stats['start'] = node_info['start']
                stats['end'] = node_info['end']
                stats['token_count'] = node_info['token_count']

    # Calculate statistics
    num_files = len(all_nodes_by_file)
    node_analysis = []

    for path, stats in node_stats.items():
        frequency = stats['appearances'] / num_files
        if frequency < min_frequency:
            continue

        values = stats['values']

        # Build a node_info dict for categorization
        node_info = {
            'name': stats['example_name'],
            'type': stats['example_type']
        }
        category = categorize_node(node_info)

        analysis = {
            'path': path,
            'name': stats['example_name'],
            'type': stats['example_type'],
            'category': category,
            'start': stats['start'],
            'end': stats['end'],
            'token_count': stats['token_count'],
            'frequency': frequency,
            'appearances': stats['appearances'],
            'mean_value': sum(values) / len(values),
            'max_value': max(values),
            'min_value': min(values),
            'total_value': sum(values),
            'std_dev': (sum((x - sum(values) / len(values)) ** 2 for x in values) / len(values)) ** 0.5
        }
        node_analysis.append(analysis)

    # Sort by total value
    node_analysis.sort(key=lambda x: x['total_value'], reverse=True)

    # Display results
    print("=" * 140)
    print(f"TOP {top_n} TREE NODES BY TOTAL ATTENTION (Frequency >= {min_frequency:.0%})")
    print("=" * 140)
    print(f"{'Rank':<6} {'Range':<16} {'Tokens':<8} {'Freq':<7} {'Total':<12} {'Mean':<12} {'Max':<12} {'Category':<25} {'Name':<40}")
    print("-" * 140)

    for i, analysis in enumerate(node_analysis[:top_n], 1):
        name_short = analysis['name'][:37] + "..." if len(analysis['name']) > 40 else analysis['name']

        print(f"{i:<6} [{analysis['start']:>5}:{analysis['end']:<5}] {analysis['token_count']:<8} "
              f"{analysis['frequency']:>6.0%} {analysis['total_value']:>11.6f} "
              f"{analysis['mean_value']:>11.6f} {analysis['max_value']:>11.6f} "
              f"{analysis['category']:<25} {name_short:<40}")

    # Group by category
    print("\n" + "=" * 140)
    print("ATTENTION BY NODE CATEGORY")
    print("=" * 140)

    category_stats = defaultdict(lambda: {'total': 0.0, 'count': 0, 'mean_freq': 0.0})
    for analysis in node_analysis:
        cat = analysis['category']
        category_stats[cat]['total'] += analysis['total_value']
        category_stats[cat]['count'] += 1
        category_stats[cat]['mean_freq'] += analysis['frequency']

    category_list = []
    for cat, stats in category_stats.items():
        category_list.append({
            'category': cat,
            'total': stats['total'],
            'count': stats['count'],
            'mean_freq': stats['mean_freq'] / stats['count']
        })

    category_list.sort(key=lambda x: x['total'], reverse=True)

    print(f"\n{'Category':<30} {'Total Attention':<20} {'Num Nodes':<15} {'Avg Frequency':<15}")
    print("-" * 80)
    for cat_stat in category_list[:20]:
        print(f"{cat_stat['category']:<30} {cat_stat['total']:<20.6f} "
              f"{cat_stat['count']:<15} {cat_stat['mean_freq']:<15.1%}")

    # Find most persistent high-value nodes
    print("\n" + "=" * 140)
    print("MOST CONSISTENT HIGH-ATTENTION NODES (appear in all files)")
    print("=" * 140)

    consistent_nodes = [a for a in node_analysis if a['frequency'] == 1.0]
    consistent_nodes.sort(key=lambda x: x['mean_value'], reverse=True)

    print(f"{'Rank':<6} {'Range':<16} {'Tokens':<8} {'Mean':<12} {'StdDev':<12} {'Category':<25} {'Name':<40}")
    print("-" * 140)

    for i, analysis in enumerate(consistent_nodes[:20], 1):
        name_short = analysis['name'][:37] + "..." if len(analysis['name']) > 40 else analysis['name']

        print(f"{i:<6} [{analysis['start']:>5}:{analysis['end']:<5}] {analysis['token_count']:<8} "
              f"{analysis['mean_value']:>11.6f} {analysis['std_dev']:>11.6f} "
              f"{analysis['category']:<25} {name_short:<40}")

    # Identify nodes with high variance (attention changes significantly)
    print("\n" + "=" * 140)
    print("NODES WITH HIGHEST VARIANCE (attention changes most across generation)")
    print("=" * 140)

    # Filter nodes that appear frequently
    frequent_nodes = [a for a in node_analysis if a['appearances'] >= max(3, num_files * 0.3)]
    frequent_nodes.sort(key=lambda x: x['std_dev'], reverse=True)

    print(f"{'Rank':<6} {'Range':<16} {'Tokens':<8} {'Mean':<12} {'StdDev':<12} {'Category':<25} {'Name':<40}")
    print("-" * 140)

    for i, analysis in enumerate(frequent_nodes[:20], 1):
        name_short = analysis['name'][:37] + "..." if len(analysis['name']) > 40 else analysis['name']

        print(f"{i:<6} [{analysis['start']:>5}:{analysis['end']:<5}] {analysis['token_count']:<8} "
              f"{analysis['mean_value']:>11.6f} {analysis['std_dev']:>11.6f} "
              f"{analysis['category']:<25} {name_short:<40}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze tree structure persistence across multiple flamegraphs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all HTML files matching a pattern
  python analyze_persistent_tree_nodes.py "attention_flamegraph_L01_H01_token*.html"

  # Analyze specific range of tokens
  python analyze_persistent_tree_nodes.py "attention_flamegraph_L01_H01_token842*.html" --min-freq 0.8

  # Show top 50 nodes
  python analyze_persistent_tree_nodes.py "*.html" --top-n 50
        """
    )

    parser.add_argument(
        '--pattern',
        type=str,
        default="flamegraph/attention_flamegraph_*.html",
        help='Glob pattern for HTML files to analyze (e.g., "attention_flamegraph_*.html")'
    )

    parser.add_argument(
        '--min-freq',
        type=float,
        default=0.5,
        help='Minimum frequency (0.0-1.0) for a node to be included (default: 0.5)'
    )

    parser.add_argument(
        '--top-n',
        type=int,
        default=30,
        help='Number of top nodes to display (default: 30)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default="flamegraph/flamegraph_analysis.txt",
        help='Output file path to save results (default: flamegraph_analysis.txt)'
    )

    args = parser.parse_args()

    # Find matching files
    from glob import glob
    html_files = sorted([Path(f) for f in glob(args.pattern)])

    if not html_files:
        print(f"No files found matching pattern: {args.pattern}")
        return 1

    # Redirect output to file
    import sys
    original_stdout = sys.stdout

    with open(args.output, 'w', encoding='utf-8') as f:
        sys.stdout = f

        print(f"Found {len(html_files)} files matching pattern: {args.pattern}")
        print()

        analyze_tree_persistence(html_files, args.min_freq, args.top_n)

    # Restore stdout
    sys.stdout = original_stdout
    print(f"Results written to: {args.output}")

    return 0


if __name__ == '__main__':
    exit(main())
