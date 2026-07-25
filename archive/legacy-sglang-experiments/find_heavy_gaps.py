#!/usr/bin/env python3
"""
Find the specific gap tokens that have the highest attention weights.
This will show us WHY the missing 0.15 is so large.
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


def get_all_child_ranges(node):
    """Recursively get ALL token ranges covered by any descendant."""
    ranges = set()

    def collect_ranges(n):
        start = n.get("start", -1)
        end = n.get("end", -1)
        if start >= 0 and end >= 0:
            for i in range(start, end + 1):
                ranges.add(i)

        for child in n.get("children", []):
            collect_ranges(child)

    # Collect from all children (not including the node itself)
    for child in node.get("children", []):
        collect_ranges(child)

    return ranges


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


def main():
    if len(sys.argv) < 4:
        print("Usage: python find_heavy_gaps.py <html_file> <tokens_csv> <attention_csv>")
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
    print("FINDING HIGH-ATTENTION GAP TOKENS IN <browser_state>")
    print("="*100)

    # Get all tokens covered by children (recursively)
    child_covered_tokens = get_all_child_ranges(browser_state)

    # Find gap tokens (tokens in parent range but not in any child)
    parent_start = browser_state['start']
    parent_end = browser_state['end']

    gap_tokens = []
    for i in range(parent_start, parent_end + 1):
        if i not in child_covered_tokens:
            gap_tokens.append({
                'index': i,
                'token': tokens[i] if i < len(tokens) else '???',
                'attention': attention_weights[i] if i < len(attention_weights) else 0.0
            })

    # Sort by attention weight (descending)
    gap_tokens.sort(key=lambda x: x['attention'], reverse=True)

    total_gap_attention = sum(t['attention'] for t in gap_tokens)

    print(f"\nTotal gap tokens: {len(gap_tokens)}")
    print(f"Total gap attention: {total_gap_attention:.6f}")
    print(f"Expected uncaptured: {browser_state['value'] - sum(c.get('value', 0) for c in browser_state.get('children', [])):.6f}")

    print(f"\n{'='*100}")
    print(f"TOP 50 GAP TOKENS BY ATTENTION WEIGHT:")
    print(f"{'='*100}")
    print(f"{'Rank':<6} {'Index':<8} {'Attention':<15} {'Token (repr)':<70}")
    print("-"*100)

    for i, gap in enumerate(gap_tokens[:50], 1):
        token_repr = repr(gap['token'])
        if len(token_repr) > 67:
            token_repr = token_repr[:64] + "..."
        print(f"{i:<6} {gap['index']:<8} {gap['attention']:<15.8f} {token_repr:<70}")

    # Group by token patterns
    print(f"\n{'='*100}")
    print("ATTENTION BY TOKEN PATTERN:")
    print(f"{'='*100}")

    patterns = {
        'newline': 0.0,
        'tab': 0.0,
        'space': 0.0,
        'element_marker': 0.0,  # [1234] or *[1234]
        'tags': 0.0,  # < > /
        'other': 0.0
    }

    pattern_counts = {k: 0 for k in patterns.keys()}

    for gap in gap_tokens:
        tok = gap['token']
        att = gap['attention']

        if tok == '\n':
            patterns['newline'] += att
            pattern_counts['newline'] += 1
        elif tok == '\t':
            patterns['tab'] += att
            pattern_counts['tab'] += 1
        elif tok == ' ':
            patterns['space'] += att
            pattern_counts['space'] += 1
        elif tok in ['[', ']', '*'] or tok.isdigit():
            patterns['element_marker'] += att
            pattern_counts['element_marker'] += 1
        elif tok in ['<', '>', '/']:
            patterns['tags'] += att
            pattern_counts['tags'] += 1
        else:
            patterns['other'] += att
            pattern_counts['other'] += 1

    for pattern, attention in sorted(patterns.items(), key=lambda x: x[1], reverse=True):
        pct = (attention / total_gap_attention * 100) if total_gap_attention > 0 else 0
        print(f"{pattern:<20} Count: {pattern_counts[pattern]:>6}  Attention: {attention:>12.6f}  ({pct:>6.2f}%)")

    # Show context around highest attention gaps
    print(f"\n{'='*100}")
    print("CONTEXT AROUND TOP 10 HIGHEST-ATTENTION GAP TOKENS:")
    print(f"{'='*100}")

    for i, gap in enumerate(gap_tokens[:10], 1):
        idx = gap['index']
        # Show 5 tokens before and after
        context_start = max(0, idx - 5)
        context_end = min(len(tokens), idx + 6)

        context_tokens = []
        for j in range(context_start, context_end):
            if j == idx:
                context_tokens.append(f">>>{repr(tokens[j])}<<<")
            elif j in child_covered_tokens:
                context_tokens.append(f"[{repr(tokens[j])}]")  # Child token
            else:
                context_tokens.append(repr(tokens[j]))  # Another gap

        print(f"\n{i}. Index {idx}, Attention: {gap['attention']:.8f}")
        print(f"   Context: {' '.join(context_tokens)}")

    return 0


if __name__ == '__main__':
    exit(main())
