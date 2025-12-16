#!/usr/bin/env python3
"""
Create an interactive HTML flame graph visualization of attention weights.
Similar to profiler flame graphs but shows attention flow through token hierarchy.
"""
import argparse
import csv
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
import html


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
    """
    Read attention weights from CSV file.
    Returns (token_ids, attention_matrix) where attention_matrix[i] contains
    attention weights from token i to all previous tokens.
    """
    print(f"Reading attention weights from: {csv_file}")

    token_ids = []
    attention_matrix = []

    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header

        for row in reader:
            if not row:
                continue
            token_id = int(row[0])
            # attention weights to all previous tokens
            weights = [float(w) for w in row[1:] if w]

            token_ids.append(token_id)
            attention_matrix.append(weights)

    print(f"Read attention weights for {len(token_ids)} tokens")
    return token_ids, attention_matrix


class StructureNode:
    """Represents a node in the ChatML/HTML tree structure."""
    def __init__(self, node_type: str, content: str, start_idx: int, end_idx: int):
        self.node_type = node_type  # 'root', 'chatml', 'html_tag', 'text'
        self.content = content
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.children = []
        self.attention_weight = 0.0

    def to_flamegraph_dict(self) -> Dict[str, Any]:
        """Convert to flamegraph format."""
        # Use attention weight as the value
        return {
            "name": self.content,
            "value": self.attention_weight,
            "type": self.node_type,
            "start": self.start_idx,
            "end": self.end_idx,
            "children": [child.to_flamegraph_dict() for child in self.children] if self.children else []
        }


def parse_chatml_structure(tokens: List[str]) -> StructureNode:
    """Parse ChatML/HTML structure from tokens to build tree hierarchy."""
    root = StructureNode("root", "all", 0, len(tokens))
    current_chatml_node = None  # Track current ChatML block (system/user/assistant)
    current_parent = root
    stack = [root]

    i = 0
    n = len(tokens)

    print(f"Parsing {n} tokens into tree structure...")

    while i < n:
        token = tokens[i]

        # Check for ChatML start markers
        if "<|im_start|>" in token or (i + 1 < n and "<|im_start|>" in token + tokens[i+1]):
            role_parts = []
            role_idx = i + 1
            while role_idx < n and "<|im_start|>" in tokens[role_idx]:
                role_idx += 1

            while role_idx < n and tokens[role_idx] not in ['\n', '<', '|']:
                role_parts.append(tokens[role_idx])
                role_idx += 1
                if role_idx >= n or len(role_parts) > 10:
                    break

            role = "".join(role_parts).strip() or "unknown"
            # ChatML nodes are siblings under root, not nested
            node = StructureNode("chatml", role, i, role_idx)
            root.children.append(node)  # Always add to root
            current_chatml_node = node
            current_parent = node  # Content goes inside this ChatML block
            stack = [root, node]
            i = role_idx
            continue

        elif "<|im_end|>" in token or (i + 1 < n and "<|im_end|>" in token + tokens[i+1]):
            if current_chatml_node:
                current_chatml_node.end_idx = i + 1
                current_chatml_node = None
                current_parent = root
                stack = [root]
            i += 1
            continue

        # Check for HTML tags - look for < at start of token or as standalone token
        if token.startswith('<') or token == '<':
            j = i
            tag_end = min(i + 20, n)

            # Find the end of the tag (look for >)
            while j < tag_end and '>' not in tokens[j]:
                j += 1

            if j < n and '>' in tokens[j]:
                tag_str = "".join(tokens[i:j+1])

                # Check if it's a closing tag
                if '</' in tag_str:
                    closing_tag_match = re.match(r'.*?</(\w+)', tag_str)
                    if closing_tag_match and len(stack) > 1:
                        closing_tag_name = closing_tag_match.group(1)

                        # Find matching opening tag and close it
                        for idx in range(len(stack) - 1, 0, -1):
                            if stack[idx].node_type == "html_tag" and closing_tag_name in stack[idx].content:
                                stack[idx].end_idx = j
                                while len(stack) > idx:
                                    stack.pop()
                                current_parent = stack[-1]
                                break
                # It's an opening tag
                else:
                    tag_match = re.match(r'.*?<(\w+)', tag_str)
                    if tag_match:
                        tag_name = tag_match.group(1)
                        display_name = tag_str[:40] if len(tag_str) < 40 else tag_str[:37] + "..."
                        node = StructureNode("html_tag", display_name, i, j)
                        current_parent.children.append(node)

                        # Self-closing tag check
                        if not tag_str.endswith('/>') and '/>' not in tag_str:
                            stack.append(node)
                            current_parent = node

                i = j + 1
                continue

        if not token.strip():
            i += 1
            continue

        # Text content
        text_start = i
        text_parts = [token]
        i += 1

        text_limit = min(i + 30, n)
        while i < text_limit:
            # Stop if we hit a new HTML tag (starts with < or is </)
            if tokens[i].startswith('<') or tokens[i] == '<':
                break
            # Stop if we hit ChatML markers
            if "<|im" in tokens[i]:
                break
            text_parts.append(tokens[i])
            i += 1

        text = "".join(text_parts).strip()
        if text:
            if len(text) > 50:
                display_text = text[:47] + "..."
            else:
                display_text = text
            node = StructureNode("text", display_text, text_start, i - 1)
            current_parent.children.append(node)

    print(f"Parsed tree structure with {len(root.children)} top-level nodes")
    return root


def calculate_attention_for_nodes(root: StructureNode, attention_weights: List[float]) -> None:
    """Calculate attention weights for each node in the tree."""
    def calc_node_attention(node: StructureNode):
        if node.start_idx >= len(attention_weights):
            node.attention_weight = 0.0
            return

        weights_in_range = []
        for idx in range(node.start_idx, min(node.end_idx + 1, len(attention_weights))):
            if idx < len(attention_weights):
                weights_in_range.append(attention_weights[idx])

        if weights_in_range:
            # Use sum instead of average for flamegraph - represents total attention to this span
            node.attention_weight = sum(weights_in_range)
        else:
            node.attention_weight = 0.0

        for child in node.children:
            calc_node_attention(child)

    calc_node_attention(root)


def get_max_depth(node: Dict[str, Any]) -> int:
    """Compute maximum depth of the flamegraph hierarchy."""
    if not node.get("children"):
        return 1
    return 1 + max(get_max_depth(child) for child in node["children"])


def build_hierarchical_structure(
    tokens: List[str],
    token_ids: List[int],
    attention_matrix: List[List[float]],
    focus_token_idx: int = -1
) -> Dict[str, Any]:
    """Build hierarchical structure based on ChatML/HTML tree."""
    if focus_token_idx < 0:
        focus_token_idx = len(token_ids) + focus_token_idx

    if focus_token_idx >= len(attention_matrix):
        raise ValueError(f"Focus token index {focus_token_idx} out of range")

    attention_weights = attention_matrix[focus_token_idx]

    # Parse the tree structure
    tree = parse_chatml_structure(tokens)

    # Calculate attention weights for each node
    calculate_attention_for_nodes(tree, attention_weights)

    return tree.to_flamegraph_dict()


def generate_svg_flamegraph(
    hierarchy: Dict[str, Any],
    output_file: Path,
    layer: int,
    head: int,
    focus_token_idx: int,
    width: int = 1400,
    cell_height: int = 18
):
    """Generate static SVG flamegraph."""
    from xml.etree import ElementTree as ET

    # Compute max depth to invert children above parents
    def depth(node):
        if not node.get("children"):
            return 1
        return 1 + max(depth(child) for child in node["children"])

    max_depth = depth(hierarchy)

    # Calculate layout
    def layout_node(node, x0, x1, depth_idx, nodes_list):
        # Invert vertical ordering so children appear above parents
        y_top = (max_depth - depth_idx - 1) * cell_height
        y_bottom = y_top + cell_height

        # Get color based on type
        color_map = {
            "root": "#888",
            "chatml": "#4CAF50",
            "html_tag": "#2196F3",
            "text": "#FF9800"
        }
        color = color_map.get(node.get("type", ""), "#999")

        nodes_list.append({
            "name": node["name"],
            "x": x0,
            "y": y_top,
            "width": x1 - x0,
            "height": cell_height,
            "color": color,
            "value": node.get("value", 0)
        })

        if "children" in node and node["children"]:
            total = sum(child.get("value", 0) for child in node["children"])
            if total == 0:
                total = len(node["children"])

            x = x0
            for child in node["children"]:
                child_value = child.get("value", 1.0 / len(node["children"]))
                child_width = (x1 - x0) * child_value / total
                layout_node(child, x, x + child_width, depth_idx + 1, nodes_list)
                x += child_width

    nodes = []
    layout_node(hierarchy, 0, width, 0, nodes)

    # Find max depth
    max_y = (max_depth * cell_height)

    # Create SVG
    svg = ET.Element("svg", {
        "xmlns": "http://www.w3.org/2000/svg",
        "width": str(width),
        "height": str(max_y + 50),
        "viewBox": f"0 0 {width} {max_y + 50}"
    })

    # Add title
    title = ET.SubElement(svg, "text", {
        "x": str(width / 2),
        "y": "20",
        "text-anchor": "middle",
        "font-size": "16",
        "font-family": "Arial, sans-serif"
    })
    title.text = f"Attention Flamegraph - Layer {layer}, Head {head}, Token {focus_token_idx}"

    g = ET.SubElement(svg, "g", {
        "id": "viewport"
    })

    # Add rectangles
    for node in nodes:
        if node["width"] < 1:
            continue

        rect = ET.SubElement(g, "rect", {
            "x": str(node["x"]),
            "y": str(node["y"]),
            "width": str(node["width"]),
            "height": str(node["height"]),
            "fill": node["color"],
            "stroke": "white",
            "stroke-width": "0.5"
        })

        # Add text (need to flip back)
        if node["width"] > 40:
            text = ET.SubElement(g, "text", {
                "x": str(node["x"] + 3),
                "y": str(node["y"] + (node["height"] * 0.65)),
                "font-size": "11",
                "font-family": "Arial, sans-serif",
                "fill": "white"
            })
            # Truncate long names
            name = node["name"]
            max_chars = int(node["width"] / 7)
            if len(name) > max_chars:
                name = name[:max_chars - 3] + "..."
            text.text = name

    # Add lightweight pan/zoom interactivity directly in the SVG
    script = ET.SubElement(svg, "script", {"type": "application/ecmascript"})
    script.text = r"""
        (function() {
        const svg = document.documentElement;
        const viewport = document.getElementById('viewport');
        if (!svg || !viewport) return;

        const initVB = {
            x: svg.viewBox.baseVal.x,
            y: svg.viewBox.baseVal.y,
            width: svg.viewBox.baseVal.width,
            height: svg.viewBox.baseVal.height
        };
        let vb = { ...initVB };
        let isPanning = false;
        let last = null;
        let startPoint = null;
        let hasMoved = false;

        function toSvgPoint(evt) {
            const pt = svg.createSVGPoint();
            pt.x = evt.clientX;
            pt.y = evt.clientY;
            const ctm = svg.getScreenCTM();
            return pt.matrixTransform(ctm.inverse());
        }

        function applyViewBox() {
            svg.setAttribute('viewBox', `${vb.x} ${vb.y} ${vb.width} ${vb.height}`);
        }

        svg.addEventListener('wheel', (evt) => {
            evt.preventDefault();
            const factor = evt.deltaY < 0 ? 0.9 : 1.1;
            const mouse = toSvgPoint(evt);

            const newWidth = vb.width * factor;
            const newHeight = vb.height * factor;
            const dx = (mouse.x - vb.x) * (1 - factor);
            const dy = (mouse.y - vb.y) * (1 - factor);

            vb = { x: vb.x + dx, y: vb.y + dy, width: newWidth, height: newHeight };
            applyViewBox();
        }, { passive: false });

        svg.addEventListener('mousedown', (evt) => {
            if (evt.button !== 0) return;
            evt.preventDefault();
            isPanning = true;
            last = toSvgPoint(evt);
            startPoint = { x: evt.clientX, y: evt.clientY };
            hasMoved = false;
        });

        svg.addEventListener('mousemove', (evt) => {
            if (!isPanning) return;
            const pt = toSvgPoint(evt);
            const dx = pt.x - last.x;
            const dy = pt.y - last.y;

            if (startPoint) {
            const distX = Math.abs(evt.clientX - startPoint.x);
            const distY = Math.abs(evt.clientY - startPoint.y);
            if (distX > 3 || distY > 3) hasMoved = true;
            }

            vb.x -= dx;
            vb.y -= dy;
            last = pt;
            applyViewBox();
        });

        window.addEventListener('mouseup', () => {
            isPanning = false;
            last = null;
            startPoint = null;
        });

        function zoomToRect(rect) {
            const bbox = rect.getBBox();
            const pad = Math.max(bbox.width, bbox.height) * 0.2 + 10;
            vb = { x: bbox.x - pad, y: bbox.y - pad, width: bbox.width + 2*pad, height: bbox.height + 2*pad };
            applyViewBox();
        }

        viewport.querySelectorAll('rect').forEach(rect => {
            rect.style.cursor = 'pointer';
            rect.addEventListener('mouseup', (evt) => {
            evt.stopPropagation();
            if (!hasMoved) {
                zoomToRect(rect);
                isPanning = false;
                last = null;
                startPoint = null;
            }
            });
        });

        svg.addEventListener('dblclick', () => { vb = { ...initVB }; applyViewBox(); });
        })();
    """


    # Write to file
    tree = ET.ElementTree(svg)
    ET.indent(tree, space="  ")
    tree.write(output_file, encoding="utf-8", xml_declaration=True)
    print(f"Generated SVG flamegraph: {output_file}")


def generate_html_flamegraph(
    hierarchy: Dict[str, Any],
    output_file: Path,
    layer: int,
    head: int,
    focus_token_idx: int
):
    """
    Generate interactive HTML flame graph visualization.
    Uses d3-flame-graph library for interactive visualization.
    """

    html_template = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Attention Weights Flame Graph - Layer {layer} Head {head}</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/d3-flame-graph@4.1.3/dist/d3-flamegraph.css">
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }

        #header {
            background: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        h1 {
            margin: 0 0 10px 0;
            color: #333;
        }

        .info {
            color: #666;
            font-size: 14px;
        }

        #chart {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow-x: auto;
        }

        .node {
            cursor: pointer;
        }

        .node rect {
            stroke: #fff;
            stroke-width: 1.5px;
        }

        .node text {
            font-size: 11px;
            fill: #333;
            pointer-events: none;
        }

        #tooltip {
            position: absolute;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 10px;
            border-radius: 4px;
            font-size: 12px;
            pointer-events: none;
            display: none;
            z-index: 1000;
            max-width: 520px;
        }

        #controls {
            background: white;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        button {
            padding: 8px 16px;
            margin-right: 10px;
            background: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }

        button:hover {
            background: #45a049;
        }

        input[type="range"] {
            width: 200px;
            vertical-align: middle;
        }

        label {
            margin-right: 10px;
            font-weight: bold;
        }

        /* Hide the built-in d3-flame-graph tooltip; we render our own stable tooltip */
        .d3-flame-graph-tooltip,
        .d3-flame-graph-tip,
        .d3-flamegraph-tooltip {
            display: none !important;
        }
    </style>
</head>
<body>
    <div id="header">
        <h1>Attention Weights Flame Graph</h1>
        <div class="info">
            <strong>Layer:</strong> {layer} |
            <strong>Head:</strong> {head} |
            <strong>Focus Token Index:</strong> {focus_token_idx}
        </div>
        <div class="info" style="margin-top: 10px;">
            Hover over boxes to see attention weights. Click to zoom in/out.
            Width represents attention weight magnitude.
        </div>
    </div>

    <div id="controls">
        <label>Min Attention Weight: <span id="threshold-value">0.001</span></label>
        <input type="range" id="threshold-slider" min="0" max="0.1" step="0.001" value="0.001">
        <button onclick="resetZoom()">Reset Zoom</button>
    </div>

    <div id="chart"></div>
    <div id="tooltip"></div>

    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/d3-flame-graph@4.1.3/dist/d3-flamegraph.min.js"></script>
    <script>
        const data = {data_json};

        let filteredData = data;
        let currentThreshold = 0.001;

        const width = Math.max(1400, window.innerWidth - 100);
        const cellHeight = 22;
        const globalTotal = (data && typeof data.value === "number" && data.value > 0)
            ? data.value
            : (data && data.children ? d3.sum(data.children, c => (c && c.value) ? c.value : 0) : 1);

        // Color mapping function
        function getColor(d) {
            if (d.data.type === "root") return "#888";
            if (d.data.type === "chatml") return "#4CAF50";
            if (d.data.type === "html_tag") return "#2196F3";
            if (d.data.type === "text") return "#FF9800";
            return "#999";
        }

        // Create flamegraph using d3-flame-graph library
        const chart = flamegraph()
            .width(width)
            .cellHeight(cellHeight)
            .transitionDuration(750)
            .minFrameSize(5)
            .transitionEase(d3.easeCubic)
            .sort(true)
            .title("")
            .selfValue(false)
            // Flame graph convention: root at bottom, children above
            .inverted(false)
            .color(getColor);

        // Disable built-in tooltip if supported (varies by version)
        if (typeof chart.tooltip === "function") {
            chart.tooltip(false);
        }

        function filterData(data, threshold) {
            if (!data.children) return data;
            return {
                ...data,
                children: data.children
                    .filter(child => child.value >= threshold)
                    .map(child => filterData(child, threshold))
            };
        }

        function maxDepth(node) {
            if (!node || !node.children || node.children.length === 0) return 1;
            let m = 1;
            for (const c of node.children) {
                m = Math.max(m, 1 + maxDepth(c));
            }
            return m;
        }

        function render(rootData) {
            const depth = maxDepth(rootData);
            const height = Math.max((depth + 1) * cellHeight + 40, 180);
            chart.height(height);
            d3.select("#chart").selectAll("*").remove();
            d3.select("#chart")
                .datum(rootData)
                .call(chart);
        }

        // Stable tooltip that always reports GLOBAL % (not re-based after zoom).
        const tooltip = document.getElementById("tooltip");
        const chartEl = document.getElementById("chart");

        function hideTooltip() {
            tooltip.style.display = "none";
        }

        function showTooltip(evt, d) {
            const value = (d && d.data && typeof d.data.value === "number") ? d.data.value
                : (d && typeof d.value === "number") ? d.value
                : 0;

            const globalPct = globalTotal > 0 ? (value / globalTotal) * 100 : 0;

            // Percent within current zoomed view (approximate using rendered rect width).
            let viewPct = null;
            const svg = chartEl.querySelector("svg");
            if (svg && evt.target && evt.target.getBoundingClientRect) {
                const svgW = svg.getBoundingClientRect().width;
                const rectW = evt.target.getBoundingClientRect().width;
                if (svgW > 0 && rectW >= 0) {
                    viewPct = (rectW / svgW) * 100;
                }
            }

            const name = (d && d.data && d.data.name) ? d.data.name : "(unknown)";
            const type = (d && d.data && d.data.type) ? d.data.type : "";
            const start = (d && d.data && typeof d.data.start === "number") ? d.data.start : null;
            const end = (d && d.data && typeof d.data.end === "number") ? d.data.end : null;

            let html = `<div><strong>${name}</strong></div>`;
            html += `<div>Global: ${globalPct.toFixed(3)}% (value=${value.toFixed(6)})</div>`;
            if (viewPct !== null) {
                html += `<div>In View: ${viewPct.toFixed(3)}%</div>`;
            }
            if (type) {
                html += `<div>Type: ${type}</div>`;
            }
            if (start !== null || end !== null) {
                html += `<div>Token span: ${start ?? "?"}..${end ?? "?"}</div>`;
            }

            tooltip.innerHTML = html;
            tooltip.style.left = (evt.pageX + 10) + "px";
            tooltip.style.top = (evt.pageY - 10) + "px";
            tooltip.style.display = "block";
        }

        // Works because d3 binds hierarchy nodes as __data__ on the SVG elements.
        chartEl.addEventListener("mousemove", (evt) => {
            const d = evt.target && evt.target.__data__;
            if (!d || !d.data) {
                hideTooltip();
                return;
            }
            showTooltip(evt, d);
        });
        chartEl.addEventListener("mouseleave", hideTooltip);

        // Custom zoom (re-root) so clicked node becomes the bottom row and the
        // chart height collapses to the subtree (avoids huge blank space).
        const zoomStack = [data];

        function currentRoot() {
            return zoomStack[zoomStack.length - 1];
        }

        function resetZoom() {
            zoomStack.length = 0;
            zoomStack.push(filteredData);
            render(currentRoot());
        }

        function zoomIn(nodeData) {
            if (!nodeData || !nodeData.children || nodeData.children.length === 0) return;
            if (nodeData === currentRoot()) return;
            zoomStack.push(nodeData);
            render(currentRoot());
        }

        // Initial render
        render(data);

        // Threshold slider
        d3.select("#threshold-slider").on("input", function() {
            currentThreshold = +this.value;
            d3.select("#threshold-value").text(currentThreshold.toFixed(4));
            filteredData = filterData(data, currentThreshold);
            zoomStack.length = 0;
            zoomStack.push(filteredData);
            render(currentRoot());
        });

        // Capture clicks on rects and use our zoom instead of the library's,
        // otherwise the library keeps the original depth baseline.
        chartEl.addEventListener("click", (evt) => {
            const d = evt.target && evt.target.__data__;
            if (!d || !d.data) return;
            evt.preventDefault();
            evt.stopPropagation();
            evt.stopImmediatePropagation();
            zoomIn(d.data);
        }, true);

        // Double-click anywhere to go up one level
        chartEl.addEventListener("dblclick", (evt) => {
            evt.preventDefault();
            evt.stopPropagation();
            evt.stopImmediatePropagation();
            if (zoomStack.length > 1) {
                zoomStack.pop();
                render(currentRoot());
            }
        }, true);

        window.resetZoom = resetZoom;
    </script>
</body>
</html>"""

    data_json = json.dumps(hierarchy, indent=2)
    max_depth = get_max_depth(hierarchy)

    # Replace placeholders manually to avoid brace escaping issues
    html_content = html_template.replace("{layer}", str(layer))
    html_content = html_content.replace("{head}", str(head))
    html_content = html_content.replace("{focus_token_idx}", str(focus_token_idx))
    html_content = html_content.replace("{data_json}", data_json)
    html_content = html_content.replace("{max_depth}", str(max_depth))

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"Generated HTML flame graph: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate HTML flame graph visualization of attention weights',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize attention for last token (default)
  python visualize_attention_flamegraph.py \\
      --csv attn_weights_request_000001_layer_00_head_00.csv \\
      --context attn_weights_request_000001_layer_00_head_00_context.txt \\
      --output attention_flamegraph.html

  # Visualize specific token
  python visualize_attention_flamegraph.py \\
      --csv attn_weights.csv \\
      --context context.txt \\
      --focus-token 100 \\
      --output output.html
        """
    )

    parser.add_argument(
        '--csv',
        required=True,
        help='Path to CSV file with attention weights'
    )

    parser.add_argument(
        '--tokens',
        required=True,
        help='Path to token mapping CSV file'
    )

    parser.add_argument(
        '--output',
        default='attention_flamegraph.html',
        help='Output file (default: attention_flamegraph.html)'
    )

    parser.add_argument(
        '--format',
        choices=['html', 'svg', 'both'],
        default='both',
        help='Output format: html (interactive), svg (static), or both (default: both)'
    )

    parser.add_argument(
        '--focus-token',
        type=int,
        default=-1,
        help='Token index to focus on (-1 for last token, default: -1)'
    )

    parser.add_argument(
        '--layer',
        type=int,
        default=0,
        help='Layer number (for display only)'
    )

    parser.add_argument(
        '--head',
        type=int,
        default=0,
        help='Head number (for display only)'
    )

    args = parser.parse_args()

    # Convert paths
    csv_file = Path(args.csv)
    token_map_file = Path(args.tokens)
    output_file = Path(args.output)

    # Validate inputs
    if not csv_file.exists():
        print(f"Error: CSV file not found: {csv_file}")
        return 1

    if not token_map_file.exists():
        print(f"Error: Token mapping file not found: {token_map_file}")
        return 1

    # Parse token mapping file
    tokens = parse_token_mapping(token_map_file)

    # Read attention weights
    token_ids, attention_matrix = read_attention_weights(csv_file)

    # Build hierarchical structure
    print(f"Building hierarchical structure for token index {args.focus_token}...")
    hierarchy = build_hierarchical_structure(
        tokens,
        token_ids,
        attention_matrix,
        args.focus_token
    )

    # Generate output based on format
    focus_idx = args.focus_token if args.focus_token >= 0 else len(token_ids) + args.focus_token

    if args.format in ['html', 'both']:
        html_file = output_file if args.format == 'html' else output_file.with_suffix('.html')
        generate_html_flamegraph(hierarchy, html_file, args.layer, args.head, focus_idx)
        print(f"\nDone! Open {html_file} in your browser to view the interactive visualization.")

    if args.format in ['svg', 'both']:
        svg_file = output_file if args.format == 'svg' else output_file.with_suffix('.svg')
        generate_svg_flamegraph(hierarchy, svg_file, args.layer, args.head, focus_idx)
        print(f"\nDone! View {svg_file} for the static SVG flamegraph.")

    return 0


if __name__ == '__main__':
    exit(main())
