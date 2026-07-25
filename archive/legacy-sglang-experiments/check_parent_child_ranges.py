#!/usr/bin/env python3
"""
Check if all parent nodes fully cover their children's token ranges.
This helps identify parsing bugs where parent ranges are incorrect.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple


def load_html_tree(html_file: Path) -> Dict[str, Any]:
    """Extract the data tree from the generated HTML file."""
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

    return tree


def check_node_ranges(node: Dict[str, Any], path: str = "root", issues: List[Tuple[str, str, int, int, int, int]] = None) -> List[Tuple[str, str, int, int, int, int]]:
    """
    Recursively check if parent node ranges fully cover children.
    Returns list of (path, node_name, parent_start, parent_end, child_start, child_end) for violations.
    """
    if issues is None:
        issues = []

    node_name = node.get('name', 'unnamed')[:50]
    node_start = node.get('start')
    node_end = node.get('end')
    children = node.get('children', [])

    if children and node_start is not None and node_end is not None:
        # Check each child
        for i, child in enumerate(children):
            child_name = child.get('name', 'unnamed')[:50]
            child_start = child.get('start')
            child_end = child.get('end')

            if child_start is not None and child_end is not None:
                # Check if parent range fully covers child
                if child_start < node_start or child_end > node_end:
                    issues.append((
                        f"{path} → child[{i}]",
                        f"Parent '{node_name}' ({node_start}-{node_end}) doesn't cover child '{child_name}' ({child_start}-{child_end})",
                        node_start,
                        node_end,
                        child_start,
                        child_end
                    ))

    # Recursively check children
    for i, child in enumerate(children):
        child_path = f"{path} → [{i}]{child.get('name', 'unnamed')[:30]}"
        check_node_ranges(child, child_path, issues)

    return issues


def check_sibling_overlaps(node: Dict[str, Any], path: str = "root", issues: List[Tuple[str, str, int, int, int, int]] = None) -> List[Tuple[str, str, int, int, int, int]]:
    """
    Check if sibling nodes have overlapping token ranges.
    Returns list of (path, description, sib1_start, sib1_end, sib2_start, sib2_end) for overlaps.
    """
    if issues is None:
        issues = []

    children = node.get('children', [])

    # Check siblings for overlaps
    for i in range(len(children)):
        for j in range(i + 1, len(children)):
            child1 = children[i]
            child2 = children[j]

            start1 = child1.get('start')
            end1 = child1.get('end')
            start2 = child2.get('start')
            end2 = child2.get('end')

            if all(x is not None for x in [start1, end1, start2, end2]):
                # Check for overlap
                if not (end1 < start2 or end2 < start1):
                    name1 = child1.get('name', 'unnamed')[:30]
                    name2 = child2.get('name', 'unnamed')[:30]
                    issues.append((
                        path,
                        f"Siblings overlap: '{name1}' ({start1}-{end1}) and '{name2}' ({start2}-{end2})",
                        start1,
                        end1,
                        start2,
                        end2
                    ))

    # Recursively check children
    for i, child in enumerate(children):
        child_path = f"{path} → [{i}]{child.get('name', 'unnamed')[:30]}"
        check_sibling_overlaps(child, child_path, issues)

    return issues


def main():
    parser = argparse.ArgumentParser(
        description='Check parent-child token range coverage in flamegraph tree',
    )

    script_dir = Path(__file__).parent.absolute()

    parser.add_argument(
        '--html',
        default=str(script_dir / "attention_flamegraph_token8425_with_steps_token8425.html"),
        help='Path to generated HTML flamegraph file'
    )

    args = parser.parse_args()

    html_file = Path(args.html)

    if not html_file.exists():
        print(f"Error: HTML file not found: {html_file}")
        return 1

    # Load tree
    print(f"Loading tree from: {html_file.name}")
    tree = load_html_tree(html_file)
    print(f"✓ Loaded tree structure\n")

    # Check parent-child ranges
    print("="*80)
    print("CHECKING PARENT-CHILD RANGE COVERAGE")
    print("="*80)

    coverage_issues = check_node_ranges(tree)

    if coverage_issues:
        print(f"\n❌ Found {len(coverage_issues)} parent-child range violations:\n")
        for i, (path, desc, p_start, p_end, c_start, c_end) in enumerate(coverage_issues[:10]):
            print(f"{i+1}. {desc}")
            print(f"   Path: {path}")
            print(f"   Parent range: {p_start}-{p_end} (span={p_end-p_start+1})")
            print(f"   Child range:  {c_start}-{c_end} (span={c_end-c_start+1})")
            if c_start < p_start:
                print(f"   → Child starts {p_start - c_start} tokens before parent")
            if c_end > p_end:
                print(f"   → Child ends {c_end - p_end} tokens after parent")
            print()

        if len(coverage_issues) > 10:
            print(f"   ... and {len(coverage_issues) - 10} more issues\n")
    else:
        print("\n✓ All parent nodes fully cover their children's ranges!\n")

    # Check sibling overlaps
    print("="*80)
    print("CHECKING SIBLING OVERLAP")
    print("="*80)

    overlap_issues = check_sibling_overlaps(tree)

    if overlap_issues:
        print(f"\n❌ Found {len(overlap_issues)} sibling overlap violations:\n")
        for i, (path, desc, s1_start, s1_end, s2_start, s2_end) in enumerate(overlap_issues[:10]):
            print(f"{i+1}. {desc}")
            print(f"   Path: {path}")
            overlap_start = max(s1_start, s2_start)
            overlap_end = min(s1_end, s2_end)
            print(f"   Overlap: tokens {overlap_start}-{overlap_end} ({overlap_end-overlap_start+1} tokens)")
            print()

        if len(overlap_issues) > 10:
            print(f"   ... and {len(overlap_issues) - 10} more issues\n")
    else:
        print("\n✓ No sibling overlaps found!\n")

    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    total_issues = len(coverage_issues) + len(overlap_issues)
    print(f"\nTotal issues found: {total_issues}")
    print(f"  - Parent-child coverage violations: {len(coverage_issues)}")
    print(f"  - Sibling overlaps: {len(overlap_issues)}")

    if total_issues == 0:
        print("\n✅ All token ranges are correct!\n")
        return 0
    else:
        print(f"\n⚠️  Found {total_issues} range issues that need fixing\n")
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
