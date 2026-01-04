#!/usr/bin/env python3
"""
Validate the flamegraph HTML tree structure by comparing it with the original context.txt.
This helps identify parsing issues and verify the tree structure is correct.
"""
import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple


def load_html_tree(html_file: Path) -> Dict[str, Any]:
    """Extract the data tree from the generated HTML file."""
    print(f"Loading tree structure from: {html_file}")

    with open(html_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find the JavaScript data section
    start = content.find('const data = ')
    if start == -1:
        raise ValueError("Could not find 'const data = ' in HTML file")

    start += len('const data = ')
    end = content.find(';', start)
    if end == -1:
        raise ValueError("Could not find end of data section")

    data_str = content[start:end]
    tree = json.loads(data_str)

    print(f"Successfully loaded tree structure")
    return tree


def load_context_txt(context_file: Path) -> str:
    """Load the original context.txt file."""
    print(f"Loading context from: {context_file}")

    with open(context_file, 'r', encoding='utf-8') as f:
        content = f.read()

    print(f"Loaded context ({len(content)} characters)")
    return content


def extract_tree_nodes(tree: Dict[str, Any], depth: int = 0) -> List[Tuple[str, str, int]]:
    """
    Extract all nodes from the tree as a flat list.
    Returns list of (node_name, node_type, depth) tuples.
    """
    nodes = []

    name = tree.get('name', '')
    node_type = tree.get('type', '')

    # Skip root node
    if node_type != 'root':
        nodes.append((name, node_type, depth))

    # Recursively process children
    for child in tree.get('children', []):
        nodes.extend(extract_tree_nodes(child, depth + 1))

    return nodes


def validate_chatml_structure(nodes: List[Tuple[str, str, int]], context: str) -> Dict[str, Any]:
    """Validate ChatML markers are correctly parsed."""
    print("\n" + "="*80)
    print("VALIDATING ChatML STRUCTURE")
    print("="*80)

    issues = []
    stats = {
        'chatml_nodes_found': 0,
        'expected_chatml_roles': [],
        'found_chatml_roles': []
    }

    # Find expected ChatML roles in context
    chatml_pattern = r'<\|im_start\|>(\w+)'
    expected_roles = re.findall(chatml_pattern, context)
    stats['expected_chatml_roles'] = expected_roles

    # Find ChatML nodes in tree
    chatml_nodes = [(name, depth) for name, node_type, depth in nodes if node_type == 'chatml']
    stats['chatml_nodes_found'] = len(chatml_nodes)
    stats['found_chatml_roles'] = [name for name, _ in chatml_nodes]

    print(f"\nExpected ChatML roles: {expected_roles}")
    print(f"Found ChatML nodes: {stats['found_chatml_roles']}")

    if len(expected_roles) != len(chatml_nodes):
        issues.append(f"ChatML count mismatch: expected {len(expected_roles)}, found {len(chatml_nodes)}")

    # Check if roles match
    for expected, (found, _) in zip(expected_roles, chatml_nodes):
        if expected != found:
            issues.append(f"ChatML role mismatch: expected '{expected}', found '{found}'")

    if not issues:
        print("✓ ChatML structure is correct!")
    else:
        print("✗ ChatML structure has issues:")
        for issue in issues:
            print(f"  - {issue}")

    return {'issues': issues, 'stats': stats}


def validate_html_tags(nodes: List[Tuple[str, str, int]], context: str) -> Dict[str, Any]:
    """Validate HTML tags are correctly parsed."""
    print("\n" + "="*80)
    print("VALIDATING HTML TAGS")
    print("="*80)

    issues = []
    stats = {
        'html_tag_nodes_found': 0,
        'expected_tags': [],
        'found_tags': []
    }

    # Find expected HTML tags in context (excluding ChatML)
    html_pattern = r'<([a-z_][a-z0-9_]*)\s*(?:/?>|>)'
    expected_tags = re.findall(html_pattern, context)
    # Filter out ChatML markers
    expected_tags = [tag for tag in expected_tags if tag not in ['|im_start|>', '|im_end|>']]
    stats['expected_tags'] = expected_tags[:20]  # First 20 for readability

    # Find HTML tag nodes in tree
    html_nodes = [(name, depth) for name, node_type, depth in nodes if node_type == 'html_tag']
    stats['html_tag_nodes_found'] = len(html_nodes)
    stats['found_tags'] = [name for name, _ in html_nodes[:20]]

    print(f"\nTotal HTML tags in context: {len(expected_tags)}")
    print(f"HTML tag nodes in tree: {len(html_nodes)}")
    print(f"\nFirst 10 expected tags: {expected_tags[:10]}")
    print(f"First 10 found tags: {[name for name, _ in html_nodes[:10]]}")

    # Check for critical tags
    critical_tags = ['user_request', 'browser_state', 'html']
    for tag in critical_tags:
        expected_count = expected_tags.count(tag)
        found_count = sum(1 for name, _ in html_nodes if tag in name.lower())

        if expected_count > 0 and found_count == 0:
            issues.append(f"Critical tag '<{tag}>' not found in tree (expected {expected_count})")
        elif expected_count != found_count:
            print(f"  Note: '<{tag}>' count: expected {expected_count}, found {found_count}")

    if not issues:
        print("✓ HTML tags look good!")
    else:
        print("✗ HTML tag issues:")
        for issue in issues:
            print(f"  - {issue}")

    return {'issues': issues, 'stats': stats}


def validate_browser_elements(nodes: List[Tuple[str, str, int]], context: str) -> Dict[str, Any]:
    """Validate browser elements are correctly parsed."""
    print("\n" + "="*80)
    print("VALIDATING BROWSER ELEMENTS")
    print("="*80)

    issues = []
    stats = {
        'browser_element_nodes_found': 0,
        'expected_browser_elements': 0,
        'found_elements': []
    }

    # Find expected browser elements in context
    browser_pattern = r'\*?\[\d+\]<\w+'
    expected_elements = re.findall(browser_pattern, context)
    stats['expected_browser_elements'] = len(expected_elements)

    # Find browser element nodes in tree
    browser_nodes = [(name, depth) for name, node_type, depth in nodes if node_type == 'browser_element']
    stats['browser_element_nodes_found'] = len(browser_nodes)
    stats['found_elements'] = [name[:40] for name, _ in browser_nodes[:10]]

    print(f"\nExpected browser elements in context: {len(expected_elements)}")
    print(f"Browser element nodes in tree: {len(browser_nodes)}")
    print(f"\nFirst 10 expected elements: {expected_elements[:10]}")
    print(f"First 10 found elements: {stats['found_elements']}")

    # Check if counts are reasonably close
    if len(browser_nodes) == 0 and len(expected_elements) > 0:
        issues.append(f"No browser elements found in tree, but {len(expected_elements)} expected in context")
    elif abs(len(browser_nodes) - len(expected_elements)) > len(expected_elements) * 0.2:
        # Allow 20% difference
        issues.append(f"Browser element count differs significantly: expected ~{len(expected_elements)}, found {len(browser_nodes)}")

    if not issues:
        print("✓ Browser elements look good!")
    else:
        print("✗ Browser element issues:")
        for issue in issues:
            print(f"  - {issue}")

    return {'issues': issues, 'stats': stats}


def validate_hierarchy_depth(nodes: List[Tuple[str, str, int]]) -> Dict[str, Any]:
    """Validate the tree hierarchy makes sense."""
    print("\n" + "="*80)
    print("VALIDATING HIERARCHY DEPTH")
    print("="*80)

    issues = []
    stats = {
        'max_depth': 0,
        'depth_distribution': {}
    }

    # Analyze depth distribution
    for name, node_type, depth in nodes:
        stats['max_depth'] = max(stats['max_depth'], depth)
        if depth not in stats['depth_distribution']:
            stats['depth_distribution'][depth] = 0
        stats['depth_distribution'][depth] += 1

    print(f"\nMax depth: {stats['max_depth']}")
    print(f"Depth distribution:")
    for depth in sorted(stats['depth_distribution'].keys()):
        count = stats['depth_distribution'][depth]
        print(f"  Depth {depth}: {count} nodes")

    # Check for suspiciously deep nesting
    if stats['max_depth'] > 20:
        issues.append(f"Very deep nesting detected (depth {stats['max_depth']}), might indicate parsing issue")

    # Check for browser elements at wrong depth
    browser_elements = [(name, depth) for name, node_type, depth in nodes if node_type == 'browser_element']
    if browser_elements:
        browser_depths = [depth for _, depth in browser_elements]
        min_browser_depth = min(browser_depths)
        max_browser_depth = max(browser_depths)
        print(f"\nBrowser element depths: {min_browser_depth} to {max_browser_depth}")

        if min_browser_depth < 2:
            issues.append(f"Browser elements found at shallow depth {min_browser_depth}, expected to be nested deeper")

    if not issues:
        print("✓ Hierarchy depth looks reasonable!")
    else:
        print("✗ Hierarchy depth issues:")
        for issue in issues:
            print(f"  - {issue}")

    return {'issues': issues, 'stats': stats}


def check_text_line_splitting(nodes: List[Tuple[str, str, int]], context: str) -> Dict[str, Any]:
    """Check if text nodes are properly split by newlines."""
    print("\n" + "="*80)
    print("VALIDATING TEXT LINE SPLITTING")
    print("="*80)

    issues = []
    stats = {
        'text_nodes_found': 0,
        'text_nodes_with_newlines': 0,
        'avg_text_length': 0
    }

    # Find text nodes
    text_nodes = [(name, depth) for name, node_type, depth in nodes if node_type == 'text']
    stats['text_nodes_found'] = len(text_nodes)

    # Check for newlines in text nodes (should be rare now)
    newline_count = sum(1 for name, _ in text_nodes if '\n' in name)
    stats['text_nodes_with_newlines'] = newline_count

    # Calculate average text length
    if text_nodes:
        total_length = sum(len(name) for name, _ in text_nodes)
        stats['avg_text_length'] = total_length / len(text_nodes)

    print(f"\nTotal text nodes: {len(text_nodes)}")
    print(f"Text nodes containing newlines: {newline_count}")
    print(f"Average text length: {stats['avg_text_length']:.1f} characters")

    if newline_count > len(text_nodes) * 0.1:  # More than 10% have newlines
        issues.append(f"Many text nodes ({newline_count}) still contain newlines, expected to be split by newlines")

    # Show sample text nodes
    print(f"\nSample text nodes:")
    for i, (name, depth) in enumerate(text_nodes[:5]):
        display_name = name[:60] + "..." if len(name) > 60 else name
        print(f"  {i+1}. [{depth}] {repr(display_name)}")

    if not issues:
        print("✓ Text splitting looks good!")
    else:
        print("✗ Text splitting issues:")
        for issue in issues:
            print(f"  - {issue}")

    return {'issues': issues, 'stats': stats}


def load_tokens(tokens_file: Path) -> List[str]:
    """Load tokens from CSV file."""
    import csv
    tokens = []
    with open(tokens_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
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


def validate_token_coverage(tree: Dict[str, Any], tokens: List[str]) -> Dict[str, Any]:
    """Validate that all tokens are covered by the tree nodes with no gaps or overlaps."""
    print("\n" + "="*80)
    print("VALIDATING TOKEN COVERAGE")
    print("="*80)

    issues = []
    stats = {
        'total_tokens': len(tokens),
        'covered_tokens': set(),
        'gaps': [],
        'overlaps': []
    }

    def collect_token_ranges(node, ranges):
        """Recursively collect all token ranges from tree."""
        start = node.get('start')
        end = node.get('end')
        node_type = node.get('type')

        # Only collect leaf nodes (text, html_tag, browser_element)
        # Skip container nodes (root, chatml)
        if node_type in ['text', 'html_tag', 'browser_element'] and start is not None and end is not None:
            ranges.append((start, end, node_type, node.get('name', '')[:40]))

        for child in node.get('children', []):
            collect_token_ranges(child, ranges)

    ranges = []
    collect_token_ranges(tree, ranges)
    ranges.sort(key=lambda x: x[0])

    print(f"\nTotal tokens: {len(tokens)}")
    print(f"Total leaf nodes: {len(ranges)}")

    # Check coverage
    for start, end, node_type, name in ranges:
        for token_idx in range(start, end + 1):
            if token_idx in stats['covered_tokens']:
                stats['overlaps'].append((token_idx, name))
            stats['covered_tokens'].add(token_idx)

    # Find gaps
    for i in range(len(tokens)):
        if i not in stats['covered_tokens']:
            stats['gaps'].append(i)

    # Report gaps
    if stats['gaps']:
        gap_ranges = []
        start = stats['gaps'][0]
        prev = start
        for gap in stats['gaps'][1:]:
            if gap != prev + 1:
                gap_ranges.append((start, prev))
                start = gap
            prev = gap
        gap_ranges.append((start, prev))

        print(f"\n⚠️  Found {len(stats['gaps'])} uncovered tokens in {len(gap_ranges)} gap(s):")
        for gap_start, gap_end in gap_ranges[:5]:
            gap_tokens = ''.join(tokens[gap_start:gap_end+1])
            issues.append(f"Gap at tokens {gap_start}-{gap_end}: {repr(gap_tokens[:60])}")
            print(f"  Tokens {gap_start}-{gap_end}: {repr(gap_tokens[:60])}")

    # Report overlaps
    if stats['overlaps']:
        print(f"\n⚠️  Found {len(stats['overlaps'])} overlapping tokens:")
        for token_idx, name in stats['overlaps'][:5]:
            issues.append(f"Overlap at token {token_idx} in node: {name}")
            print(f"  Token {token_idx}: {repr(tokens[token_idx][:40])} in node: {name}")

    coverage_pct = len(stats['covered_tokens']) / len(tokens) * 100
    print(f"\nCoverage: {len(stats['covered_tokens'])}/{len(tokens)} tokens ({coverage_pct:.1f}%)")

    if not issues:
        print("✓ Token coverage is complete with no gaps or overlaps!")

    return {'issues': issues, 'stats': stats}


def validate_text_reconstruction(tree: Dict[str, Any], tokens: List[str], context: str) -> Dict[str, Any]:
    """Validate by reconstructing text from token ranges and comparing with original."""
    print("\n" + "="*80)
    print("VALIDATING TEXT RECONSTRUCTION")
    print("="*80)

    issues = []
    stats = {
        'nodes_checked': 0,
        'mismatches': []
    }

    def check_node_text(node, depth=0):
        """Recursively check if node's text matches its token range."""
        start = node.get('start')
        end = node.get('end')
        node_type = node.get('type')
        node_name = node.get('name', '')

        if start is not None and end is not None and node_type in ['text', 'browser_element']:
            stats['nodes_checked'] += 1

            # Reconstruct text from tokens
            if start < len(tokens) and end < len(tokens):
                reconstructed = ''.join(tokens[start:end+1])

                # For text nodes, the name might be truncated at 50 chars
                # So we check if the name is a prefix of reconstructed text
                if node_type == 'text':
                    # Remove "..." if present
                    expected_name = node_name.replace('...', '').strip()
                    actual_text = reconstructed.strip()

                    if not actual_text.startswith(expected_name) and expected_name not in actual_text:
                        stats['mismatches'].append({
                            'type': node_type,
                            'expected': expected_name[:60],
                            'actual': actual_text[:60],
                            'tokens': f'{start}-{end}'
                        })

        for child in node.get('children', []):
            check_node_text(child, depth + 1)

    check_node_text(tree)

    print(f"\nChecked {stats['nodes_checked']} text/browser_element nodes")

    if stats['mismatches']:
        print(f"\n⚠️  Found {len(stats['mismatches'])} text mismatches:")
        for mismatch in stats['mismatches'][:5]:
            issues.append(f"Mismatch in {mismatch['type']} at tokens {mismatch['tokens']}")
            print(f"  Expected: {repr(mismatch['expected'])}")
            print(f"  Actual: {repr(mismatch['actual'])}")
    else:
        print("✓ All text nodes match their token ranges!")

    return {'issues': issues, 'stats': stats}


def validate_tree_structure_logic(tree: Dict[str, Any], context: str) -> Dict[str, Any]:
    """Validate logical correctness of tree structure."""
    print("\n" + "="*80)
    print("VALIDATING TREE STRUCTURE LOGIC")
    print("="*80)

    issues = []
    stats = {
        'tag_pairs': [],
        'mismatched_tags': []
    }

    # Check that ChatML tags are properly paired
    chatml_start_count = context.count('<|im_start|>')
    chatml_end_count = context.count('<|im_end|>')

    if chatml_start_count != chatml_end_count:
        issues.append(f"ChatML tag mismatch: {chatml_start_count} starts vs {chatml_end_count} ends")

    # Check for common parsing mistakes
    # 1. Text nodes should not contain HTML tags (unless they're examples in documentation)
    def check_text_nodes(node):
        if node.get('type') == 'text':
            name = node.get('name', '')
            # Count < and > - if there are unclosed tags, it's suspicious
            open_count = name.count('<')
            close_count = name.count('>')
            if open_count != close_count:
                issues.append(f"Text node has unbalanced < > : {repr(name[:60])}")

        for child in node.get('children', []):
            check_text_nodes(child)

    check_text_nodes(tree)

    print(f"\nChatML tags: {chatml_start_count} starts, {chatml_end_count} ends")

    if not issues:
        print("✓ Tree structure logic looks correct!")
    else:
        print(f"⚠️  Found {len(issues)} structural issues")

    return {'issues': issues, 'stats': stats}


def build_reference_tree(context: str, tokens: List[str]) -> Dict[str, Any]:
    """
    Build a reference tree structure from context.txt by parsing it line by line.
    This serves as ground truth for comparison.
    """
    print("\nBuilding reference tree from context.txt...")

    # Simple reference tree based on structure markers in context
    reference = {
        'type': 'root',
        'name': 'all',
        'children': [],
        'start': 0,
        'end': len(tokens) - 1
    }

    lines = context.split('\n')
    current_token_idx = 0
    chatml_stack = [reference]  # Stack for ChatML nesting

    for line_num, line in enumerate(lines):
        # Find this line in tokens
        line_start_idx = current_token_idx

        # Count tokens for this line (find next \n token)
        line_tokens = 0
        temp_idx = current_token_idx
        while temp_idx < len(tokens):
            line_tokens += 1
            if '\n' in tokens[temp_idx]:
                current_token_idx = temp_idx + 1
                break
            temp_idx += 1
        else:
            current_token_idx = temp_idx

        # Check for ChatML markers
        if '<|im_start|>' in line:
            role_match = re.search(r'<\|im_start\|>(\w+)', line)
            if role_match:
                role = role_match.group(1)
                chatml_node = {
                    'type': 'chatml',
                    'name': role,
                    'children': [],
                    'start': line_start_idx,
                    'end': current_token_idx - 1
                }
                reference['children'].append(chatml_node)
                chatml_stack = [reference, chatml_node]
                continue

        if '<|im_end|>' in line:
            if len(chatml_stack) > 1:
                chatml_stack.pop()
            continue

        # Check for HTML tags
        html_tag_match = re.match(r'<([a-z_][a-z0-9_]*)', line)
        if html_tag_match and len(chatml_stack) > 1:
            tag_name = html_tag_match.group(1)
            tag_node = {
                'type': 'html_tag',
                'name': f'<{tag_name}>',
                'children': [],
                'start': line_start_idx,
                'end': current_token_idx - 1
            }
            chatml_stack[-1]['children'].append(tag_node)
            continue

        # Check for browser elements [1234]<tag>
        browser_match = re.match(r'\s*(\*?\[\d+\]<[^>]+>)', line)
        if browser_match and len(chatml_stack) > 1:
            elem_str = browser_match.group(1)
            browser_node = {
                'type': 'browser_element',
                'name': elem_str,
                'children': [],
                'start': line_start_idx,
                'end': current_token_idx - 1
            }
            chatml_stack[-1]['children'].append(browser_node)
            continue

        # Regular text line
        if line.strip() and len(chatml_stack) > 1:
            text_node = {
                'type': 'text',
                'name': line.strip()[:50],
                'children': [],
                'start': line_start_idx,
                'end': current_token_idx - 1
            }
            chatml_stack[-1]['children'].append(text_node)

    print(f"Built reference tree with {count_nodes(reference)} nodes")
    return reference


def count_nodes(tree: Dict[str, Any]) -> int:
    """Count total nodes in tree."""
    count = 1
    for child in tree.get('children', []):
        count += count_nodes(child)
    return count


def compare_trees(html_tree: Dict[str, Any], reference_tree: Dict[str, Any], tokens: List[str]) -> Dict[str, Any]:
    """
    Compare the HTML tree against the reference tree to find structural differences.
    """
    print("\n" + "="*80)
    print("COMPARING TREE STRUCTURES")
    print("="*80)

    issues = []
    stats = {
        'html_node_count': count_nodes(html_tree),
        'reference_node_count': count_nodes(reference_tree),
        'structural_differences': []
    }

    print(f"\nHTML tree nodes: {stats['html_node_count']}")
    print(f"Reference tree nodes: {stats['reference_node_count']}")

    # Compare top-level structure
    html_children = html_tree.get('children', [])
    ref_children = reference_tree.get('children', [])

    print(f"\nHTML tree has {len(html_children)} top-level children")
    print(f"Reference tree has {len(ref_children)} top-level children")

    # Compare ChatML nodes
    html_chatml = [c for c in html_children if c.get('type') == 'chatml']
    ref_chatml = [c for c in ref_children if c.get('type') == 'chatml']

    print(f"\nChatML nodes: HTML={len(html_chatml)}, Reference={len(ref_chatml)}")

    if len(html_chatml) != len(ref_chatml):
        issues.append(f"ChatML count mismatch: HTML has {len(html_chatml)}, reference has {len(ref_chatml)}")

    # Compare token coverage
    def get_token_coverage(node, covered_set):
        """Recursively collect covered tokens."""
        start = node.get('start')
        end = node.get('end')
        if start is not None and end is not None:
            for i in range(start, end + 1):
                covered_set.add(i)
        for child in node.get('children', []):
            get_token_coverage(child, covered_set)

    html_coverage = set()
    ref_coverage = set()
    get_token_coverage(html_tree, html_coverage)
    get_token_coverage(reference_tree, ref_coverage)

    html_only = html_coverage - ref_coverage
    ref_only = ref_coverage - html_coverage

    if html_only:
        print(f"\n⚠️  HTML tree covers {len(html_only)} tokens not in reference")
        issues.append(f"HTML tree covers tokens not in reference: {len(html_only)} tokens")

    if ref_only:
        print(f"⚠️  Reference tree covers {len(ref_only)} tokens not in HTML")
        issues.append(f"Reference tree missing tokens: {len(ref_only)} tokens")

    if not issues:
        print("\n✓ Tree structures match!")
    else:
        print(f"\n⚠️  Found {len(issues)} structural differences")

    return {'issues': issues, 'stats': stats}


def generate_summary_report(results: Dict[str, Dict[str, Any]]) -> None:
    """Generate a summary report of all validation results."""
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    total_issues = sum(len(r['issues']) for r in results.values())

    print(f"\nTotal validation checks: {len(results)}")
    print(f"Total issues found: {total_issues}\n")

    for check_name, result in results.items():
        issue_count = len(result['issues'])
        status = "✓ PASS" if issue_count == 0 else f"✗ FAIL ({issue_count} issues)"
        print(f"  {check_name}: {status}")

    if total_issues == 0:
        print("\n🎉 All validation checks passed! The tree structure looks correct.")
    else:
        print(f"\n⚠️  Found {total_issues} issues that need attention.")
        print("\nAll issues:")
        for check_name, result in results.items():
            for issue in result['issues']:
                print(f"  [{check_name}] {issue}")


def main():
    parser = argparse.ArgumentParser(
        description='Validate flamegraph HTML tree structure against context.txt',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python validate_flamegraph_structure.py \\
      --html test_final_token8652.html \\
      --context attention_weights/archive_20251215_112126/attn_weights_request_000001_layer_01_head_01_context.txt
        """
    )

    # Get the directory where this script is located
    script_dir = Path(__file__).parent.absolute()

    parser.add_argument(
        '--html',
        default=str(script_dir / "attention_flamegraph_token8425.html"),
        help='Path to generated HTML flamegraph file'
    )

    parser.add_argument(
        '--context',
        default=str(script_dir / "attention_weights/archive_20251215_112126/attn_weights_request_000001_layer_01_head_01_context.txt"),
        help='Path to original context.txt file'
    )

    parser.add_argument(
        '--tokens',
        default=str(script_dir / "attention_weights/archive_20251215_112126/attn_weights_request_000001_layer_01_head_01_tokens.csv"),
        help='Path to tokens CSV file (optional, enables deeper validation)'
    )

    args = parser.parse_args()

    # Validate inputs
    html_file = Path(args.html)
    context_file = Path(args.context)
    tokens_file = Path(args.tokens)

    if not html_file.exists():
        print(f"Error: HTML file not found: {html_file}")
        return 1

    if not context_file.exists():
        print(f"Error: Context file not found: {context_file}")
        return 1

    # Load data
    tree = load_html_tree(html_file)
    context = load_context_txt(context_file)

    # Load tokens if available
    tokens = None
    if tokens_file.exists():
        tokens = load_tokens(tokens_file)
        print(f"Loaded {len(tokens)} tokens from {tokens_file.name}")
    else:
        print(f"Tokens file not found: {tokens_file}")
        print("Skipping token-based validation checks")

    # Extract nodes
    nodes = extract_tree_nodes(tree)
    print(f"\nExtracted {len(nodes)} nodes from tree structure")

    # Run validation checks
    results = {}
    results['ChatML Structure'] = validate_chatml_structure(nodes, context)
    results['HTML Tags'] = validate_html_tags(nodes, context)
    results['Browser Elements'] = validate_browser_elements(nodes, context)
    results['Hierarchy Depth'] = validate_hierarchy_depth(nodes)
    results['Text Splitting'] = check_text_line_splitting(nodes, context)
    results['Tree Structure Logic'] = validate_tree_structure_logic(tree, context)

    # Token-based validation (only if tokens are available)
    if tokens:
        results['Token Coverage'] = validate_token_coverage(tree, tokens)
        results['Text Reconstruction'] = validate_text_reconstruction(tree, tokens, context)

        # Build reference tree and compare
        reference_tree = build_reference_tree(context, tokens)
        results['Tree Comparison'] = compare_trees(tree, reference_tree, tokens)

        # Save reference tree to file for inspection
        reference_file = html_file.parent / f"{html_file.stem}_reference_tree.json"
        with open(reference_file, 'w', encoding='utf-8') as f:
            json.dump(reference_tree, f, indent=2)
        print(f"\n✓ Saved reference tree to: {reference_file}")

    # Generate summary
    generate_summary_report(results)

    return 0


if __name__ == '__main__':
    main()