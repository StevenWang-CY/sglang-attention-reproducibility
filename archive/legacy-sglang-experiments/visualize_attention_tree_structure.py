#!/usr/bin/env python3
"""
Create hierarchical tree visualization based on ChatML/HTML structure in context.
Shows the actual conversation and HTML DOM tree, with attention weights overlaid.
"""
import argparse
import csv
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from html.parser import HTMLParser
from collections import defaultdict


def parse_token_mapping(token_map_file: Path) -> List[str]:
    """Parse token mapping CSV file."""
    print(f"Reading token mapping from: {token_map_file}")

    tokens = []

    with open(token_map_file, 'r', encoding='utf-8') as f:
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

    print(f"Loaded {len(tokens)} tokens from mapping file")
    return tokens


def read_attention_weights(csv_file: Path) -> Tuple[List[int], List[List[float]]]:
    """Read attention weights from CSV."""
    print(f"Reading attention weights from: {csv_file}")

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

    print(f"Read attention weights for {len(token_ids)} tokens")
    return token_ids, attention_matrix


class StructureNode:
    """Represents a node in the tree structure."""
    def __init__(self, node_type: str, content: str, start_idx: int, end_idx: int):
        self.node_type = node_type  # 'chatml', 'html_tag', 'text'
        self.content = content  # Tag name or text content
        self.start_idx = start_idx  # Token start index
        self.end_idx = end_idx  # Token end index
        self.children = []
        self.attention_weight = 0.0  # Average attention to this node

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "name": self.content,
            "type": self.node_type,
            "start_idx": self.start_idx,
            "end_idx": self.end_idx,
            "value": self.attention_weight,
            "children": [child.to_dict() for child in self.children]
        }


def parse_chatml_structure(tokens: List[str]) -> StructureNode:
    """
    Parse ChatML structure from tokens (optimized version).

    ChatML format uses tags like:
    <|im_start|>system
    <|im_end|>
    <|im_start|>user
    <|im_end|>
    """
    root = StructureNode("root", "Conversation", 0, len(tokens))
    current_parent = root
    stack = [root]

    i = 0
    n = len(tokens)

    print(f"Parsing {n} tokens...")

    while i < n:
        token = tokens[i]

        # Fast check for ChatML markers - check single token first
        if "<|im_start|>" in token or (i + 1 < n and "<|im_start|>" in token + tokens[i+1]):
            # Find role
            role_parts = []
            role_idx = i + 1
            # Skip the marker token(s)
            while role_idx < n and "<|im_start|>" in tokens[role_idx]:
                role_idx += 1

            # Collect role name
            while role_idx < n and tokens[role_idx] not in ['\n', '<', '|']:
                role_parts.append(tokens[role_idx])
                role_idx += 1
                if role_idx >= n or len(role_parts) > 10:  # Limit role length
                    break

            role = "".join(role_parts).strip() or "unknown"
            node = StructureNode("chatml", f"<{role}>", i, role_idx)
            current_parent.children.append(node)
            stack.append(node)
            current_parent = node
            i = role_idx
            continue

        elif "<|im_end|>" in token or (i + 1 < n and "<|im_end|>" in token + tokens[i+1]):
            # Close current ChatML block
            if len(stack) > 1:
                current_parent.end_idx = i
                stack.pop()
                current_parent = stack[-1]
            i += 1
            continue

        # Check for HTML tags (if token contains '<')
        if '<' in token:
            # Collect all tokens until we find one containing '>'
            j = i
            tag_end = min(i + 20, n)  # Limit tag search length

            # Find closing '>' (check if token contains '>')
            while j < tag_end and '>' not in tokens[j]:
                j += 1

            if j < n and '>' in tokens[j]:
                # Build the full tag string
                tag_str = "".join(tokens[i:j+1])

                # Check if it's a closing tag (contains '</')
                if '</' in tag_str:
                    # Extract closing tag name
                    closing_tag_match = re.match(r'.*?</(\w+)', tag_str)
                    if closing_tag_match and len(stack) > 1:
                        closing_tag_name = closing_tag_match.group(1)

                        # Find matching opening tag in stack
                        # Search from top of stack downward
                        for idx in range(len(stack) - 1, 0, -1):
                            if stack[idx].node_type == "html_tag" and stack[idx].content == closing_tag_name:
                                # Found matching tag - close it
                                stack[idx].end_idx = j

                                # Pop all tags from this one to the top
                                while len(stack) > idx:
                                    stack.pop()

                                current_parent = stack[-1]
                                break
                else:
                    # Opening tag - extract tag name (find first <tagname pattern)
                    tag_match = re.match(r'.*?<(\w+)', tag_str)
                    if tag_match:
                        tag_name = tag_match.group(1)
                        node = StructureNode("html_tag", tag_name, i, j)
                        current_parent.children.append(node)

                        # Check if self-closing
                        if not tag_str.endswith('/>') and '/>' not in tag_str:
                            stack.append(node)
                            current_parent = node

                i = j + 1
                continue

        # Regular text - skip if just whitespace
        if not token.strip():
            i += 1
            continue

        # Accumulate text tokens
        text_start = i
        text_parts = [token]
        i += 1

        # Collect consecutive non-tag tokens (limit to avoid slowdown)
        text_limit = min(i + 100, n)
        while i < text_limit:
            if tokens[i] == '<' or '>' in tokens[i]:
                break
            if "<|im" in tokens[i]:
                break
            text_parts.append(tokens[i])
            i += 1

        text = "".join(text_parts).strip()
        if text:
            # Truncate long text
            if len(text) > 40:
                text = text[:37] + "..."
            node = StructureNode("text", text, text_start, i - 1)
            current_parent.children.append(node)

    print(f"Parsed {n} tokens successfully")
    return root


def calculate_attention_for_nodes(
    root: StructureNode,
    attention_weights: List[float],
    focus_token_idx: int
) -> None:
    """
    Calculate average attention weight for each node in the tree.
    """
    def calc_node_attention(node: StructureNode):
        if node.start_idx >= len(attention_weights):
            node.attention_weight = 0.0
            return

        # Average attention to all tokens in this node's range
        weights_in_range = []
        for idx in range(node.start_idx, min(node.end_idx + 1, len(attention_weights))):
            if idx < len(attention_weights):
                weights_in_range.append(attention_weights[idx])

        if weights_in_range:
            node.attention_weight = sum(weights_in_range) / len(weights_in_range)
        else:
            node.attention_weight = 0.0

        # Recursively calculate for children
        for child in node.children:
            calc_node_attention(child)

    calc_node_attention(root)


def generate_html_tree_viz(
    tree: StructureNode,
    output_file: Path,
    layer: int,
    head: int,
    focus_token_idx: int
):
    """Generate interactive HTML tree visualization."""

    html_template = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Attention Tree Structure - Layer {layer} Head {head}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}

        #header {{
            background: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        h1 {{
            margin: 0 0 10px 0;
            color: #333;
        }}

        .info {{
            color: #666;
            font-size: 14px;
            margin-top: 5px;
        }}

        #chart {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow: auto;
        }}

        .node circle {{
            fill: #fff;
            stroke: steelblue;
            stroke-width: 2px;
        }}

        .node text {{
            font-size: 12px;
            font-family: monospace;
        }}

        .link {{
            fill: none;
            stroke: #ccc;
            stroke-width: 2px;
        }}

        #tooltip {{
            position: absolute;
            background: rgba(0, 0, 0, 0.9);
            color: white;
            padding: 12px;
            border-radius: 6px;
            font-size: 13px;
            pointer-events: none;
            display: none;
            z-index: 1000;
            max-width: 300px;
        }}

        #controls {{
            background: white;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        button {{
            padding: 8px 16px;
            margin-right: 10px;
            background: #2196F3;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }}

        button:hover {{
            background: #1976D2;
        }}

        .legend {{
            margin-top: 10px;
            padding: 10px;
            background: #f9f9f9;
            border-radius: 4px;
        }}

        .legend-item {{
            display: inline-block;
            margin-right: 15px;
            font-size: 12px;
        }}

        .legend-color {{
            display: inline-block;
            width: 15px;
            height: 15px;
            margin-right: 5px;
            vertical-align: middle;
            border: 1px solid #ccc;
            border-radius: 3px;
        }}
    </style>
</head>
<body>
    <div id="header">
        <h1>Attention Weights Tree Structure</h1>
        <div class="info">
            <strong>Layer:</strong> {layer} |
            <strong>Head:</strong> {head} |
            <strong>Focus Token:</strong> {focus_token_idx}
        </div>
        <div class="info">
            This visualization shows the ChatML/HTML structure with attention weights.
            Node color intensity represents attention weight.
        </div>

        <div class="legend">
            <strong>Node Types:</strong>
            <div class="legend-item">
                <span class="legend-color" style="background-color: #4CAF50;"></span>
                ChatML Tags
            </div>
            <div class="legend-item">
                <span class="legend-color" style="background-color: #2196F3;"></span>
                HTML Tags
            </div>
            <div class="legend-item">
                <span class="legend-color" style="background-color: #FF9800;"></span>
                Text Content
            </div>
        </div>
    </div>

    <div id="controls">
        <button onclick="expandAll()">Expand All</button>
        <button onclick="collapseAll()">Collapse All</button>
        <button onclick="resetView()">Reset View</button>
    </div>

    <div id="chart"></div>
    <div id="tooltip"></div>

    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script>
        const treeData = {data_json};

        const width = 1400;
        const height = 800;

        const svg = d3.select("#chart")
            .append("svg")
            .attr("width", width)
            .attr("height", height)
            .append("g")
            .attr("transform", "translate(40,40)");

        const tooltip = d3.select("#tooltip");

        const tree = d3.tree()
            .size([height - 100, width - 200]);

        const root = d3.hierarchy(treeData);

        function getNodeColor(d) {{
            const type = d.data.type;
            if (type === "chatml") return "#4CAF50";
            if (type === "html_tag") return "#2196F3";
            if (type === "text") return "#FF9800";
            return "#999";
        }}

        function getNodeOpacity(d) {{
            // Opacity based on attention weight
            const weight = d.data.value || 0;
            return 0.3 + (weight * 0.7);  // Range from 0.3 to 1.0
        }}

        function update(source) {{
            const treeData = tree(root);
            const nodes = treeData.descendants();
            const links = treeData.links();

            // Update nodes
            const node = svg.selectAll(".node")
                .data(nodes, d => d.id || (d.id = ++i));

            const nodeEnter = node.enter()
                .append("g")
                .attr("class", "node")
                .attr("transform", d => `translate(${{d.y}},${{d.x}})`);

            nodeEnter.append("circle")
                .attr("r", 6)
                .style("fill", d => getNodeColor(d))
                .style("opacity", d => getNodeOpacity(d))
                .on("mouseover", function(event, d) {{
                    d3.select(this).attr("r", 8);

                    let html = `<strong>${{d.data.name}}</strong><br>`;
                    html += `Type: ${{d.data.type}}<br>`;
                    html += `Tokens: ${{d.data.start_idx}} - ${{d.data.end_idx}}<br>`;
                    html += `Attention: ${{d.data.value.toFixed(6)}}`;

                    tooltip
                        .style("display", "block")
                        .style("left", (event.pageX + 10) + "px")
                        .style("top", (event.pageY - 10) + "px")
                        .html(html);
                }})
                .on("mouseout", function(event, d) {{
                    d3.select(this).attr("r", 6);
                    tooltip.style("display", "none");
                }});

            nodeEnter.append("text")
                .attr("dy", "0.31em")
                .attr("x", d => d.children ? -10 : 10)
                .attr("text-anchor", d => d.children ? "end" : "start")
                .text(d => d.data.name)
                .style("fill", "#333")
                .style("font-size", "11px");

            // Update links
            svg.selectAll(".link")
                .data(links)
                .enter()
                .append("path")
                .attr("class", "link")
                .attr("d", d3.linkHorizontal()
                    .x(d => d.y)
                    .y(d => d.x));
        }}

        let i = 0;
        update(root);

        function expandAll() {{
            root.each(d => d._children = null);
            update(root);
        }}

        function collapseAll() {{
            root.each(collapse);
            update(root);
        }}

        function collapse(d) {{
            if (d.children) {{
                d._children = d.children;
                d._children.forEach(collapse);
                d.children = null;
            }}
        }}

        function resetView() {{
            expandAll();
        }}
    </script>
</body>
</html>"""

    data_json = json.dumps(tree.to_dict(), indent=2)

    html_content = html_template.format(
        layer=layer,
        head=head,
        focus_token_idx=focus_token_idx,
        data_json=data_json
    )

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"Generated tree structure visualization: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate tree structure visualization based on ChatML/HTML',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize ChatML/HTML tree structure with attention weights
  python visualize_attention_tree_structure.py \\
      --csv attn_weights_request_000001_layer_00_head_00.csv \\
      --tokens attn_weights_request_000001_layer_00_head_00_tokens.csv \\
      --layer 0 --head 0
        """
    )

    parser.add_argument('--csv', required=True, help='Path to CSV file with attention weights')
    parser.add_argument('--tokens', required=True, help='Path to token mapping CSV file')
    parser.add_argument('--output', default=None, help='Output HTML file (default: auto-generate)')
    parser.add_argument('--focus-token', type=int, default=-1, help='Token index to focus on (-1 for last)')
    parser.add_argument('--layer', type=int, default=0, help='Layer number (for display)')
    parser.add_argument('--head', type=int, default=0, help='Head number (for display)')

    args = parser.parse_args()

    csv_file = Path(args.csv)
    token_map_file = Path(args.tokens)

    if not csv_file.exists():
        print(f"Error: CSV file not found: {csv_file}")
        return 1

    if not token_map_file.exists():
        print(f"Error: Token mapping file not found: {token_map_file}")
        return 1

    tokens = parse_token_mapping(token_map_file)
    token_ids, attention_matrix = read_attention_weights(csv_file)

    # Calculate focus token index
    focus_token_idx = args.focus_token
    if focus_token_idx < 0:
        focus_token_idx = len(token_ids) + focus_token_idx

    # Auto-generate output filename
    if args.output is None:
        csv_stem = csv_file.stem
        output_file = csv_file.parent / f"{csv_stem}_focus{focus_token_idx}_tree.html"
    else:
        output_file = Path(args.output)

    print(f"Parsing ChatML/HTML structure...")
    tree = parse_chatml_structure(tokens)

    print(f"Calculating attention weights for tree nodes...")
    if focus_token_idx < len(attention_matrix):
        calculate_attention_for_nodes(tree, attention_matrix[focus_token_idx], focus_token_idx)

    generate_html_tree_viz(
        tree,
        output_file,
        args.layer,
        args.head,
        focus_token_idx
    )

    print(f"\nDone! Open {output_file} in your browser to view the tree structure.")

    return 0


if __name__ == '__main__':
    exit(main())
