#!/usr/bin/env python3
"""
Standalone verification script for the tree parser.

Usage:
    # Basic: parse tree and show results
    python verify_tree_parser.py --request-json html_test_request.json --model-path $HF_MODELS/Qwen/Qwen3-VL-8B-Instruct

    # Compare token-level parser against BeautifulSoup ground truth
    python verify_tree_parser.py --request-json html_test_request.json --model-path $HF_MODELS/Qwen/Qwen3-VL-8B-Instruct --compare

    # Compare against a runtime trace JSON from tree_sparse_traces/
    python verify_tree_parser.py --request-json html_test_request.json --model-path $HF_MODELS/Qwen/Qwen3-VL-8B-Instruct --compare --trace-json qwen3vl-log/tree_sparse_traces/req_0_20260315_231847.json
"""

import argparse
import io
import json
import os
import re
import sys
from collections import defaultdict

# Add sglang source to path so we can import the tree parser
SGLANG_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "sglang_my", "python",
)
sys.path.insert(0, os.path.abspath(SGLANG_ROOT))

from sglang.srt.layers.attention.tree_sparse.tree_parser import (
    FlatChunk,
    TreeNode,
    extract_leaf_chunks,
    format_chunks,
    format_tree,
    parse_chatml_tree,
)


# ============================================================
# Helpers
# ============================================================

class TeeWriter:
    """Write to both stdout and a StringIO buffer simultaneously."""

    def __init__(self, original_stdout):
        self._stdout = original_stdout
        self._buf = io.StringIO()

    def write(self, text):
        self._stdout.write(text)
        self._buf.write(text)

    def flush(self):
        self._stdout.flush()

    def getvalue(self):
        return self._buf.getvalue()


def load_request_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def apply_chat_template_and_tokenize(messages, tokenizer):
    """Apply the chat template and return (token_ids, full_prompt_text)."""
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    return token_ids, prompt


def decode_per_token(token_ids, tokenizer):
    """Decode each token ID individually to get per-token text strings."""
    token_texts = []
    for tid in token_ids:
        text = tokenizer.decode([tid])
        token_texts.append(text)
    return token_texts


def print_tree_full(node: TreeNode, indent: int = 0):
    """Print the full tree with no depth limit."""
    prefix = "  " * indent
    label = node.label[:80] + "..." if len(node.label) > 80 else node.label
    print(
        f"{prefix}[{node.node_type}] {label} "
        f"[{node.start_idx}-{node.end_idx}] ({node.token_count} tokens)"
    )
    for child in node.children:
        print_tree_full(child, indent + 1)


def print_all_chunks(chunks):
    """Print all chunks with no truncation."""
    for c in chunks:
        print(
            f"  chunk_{c.chunk_id:3d}: [{c.start_idx:5d}-{c.end_idx:5d}] "
            f"({c.token_count:4d} tokens) {c.label}"
        )


def tree_to_dict_full(node: TreeNode) -> dict:
    """Convert tree to dict with no truncation, for JSON saving."""
    d = {
        "node_type": node.node_type,
        "label": node.label,
        "start_idx": node.start_idx,
        "end_idx": node.end_idx,
        "token_count": node.token_count,
    }
    if node.children:
        d["children"] = [tree_to_dict_full(c) for c in node.children]
    return d


# ============================================================
# BeautifulSoup ground truth comparison
# ============================================================

def build_char_to_token_map(prompt_text, token_ids, tokenizer):
    """
    Map character positions in prompt_text to token indices.

    Tries tokenizer offset_mapping first; falls back to cumulative alignment.
    Returns a list of length len(prompt_text) where char_to_tok[c] = token_index.
    """
    # Approach 1: use tokenizer's offset_mapping (fast tokenizer)
    try:
        encoding = tokenizer(
            prompt_text, return_offsets_mapping=True, add_special_tokens=False
        )
        offsets = encoding.get("offset_mapping")
        if offsets and len(offsets) == len(token_ids):
            char_to_tok = [None] * len(prompt_text)
            for tok_idx, (start, end) in enumerate(offsets):
                for c in range(start, min(end, len(prompt_text))):
                    char_to_tok[c] = tok_idx
            return char_to_tok
    except Exception:
        pass

    # Approach 2: cumulative alignment of decoded token texts
    token_texts = [tokenizer.decode([tid]) for tid in token_ids]
    char_to_tok = [None] * len(prompt_text)
    char_pos = 0
    for tok_idx, tok_text in enumerate(token_texts):
        # Try to find this token's text near the current position
        search_window = prompt_text[char_pos : char_pos + len(tok_text) + 20]
        pos_in_window = search_window.find(tok_text)
        if pos_in_window >= 0 and pos_in_window < 10:
            abs_pos = char_pos + pos_in_window
            for c in range(abs_pos, min(abs_pos + len(tok_text), len(prompt_text))):
                char_to_tok[c] = tok_idx
            char_pos = abs_pos + len(tok_text)
        else:
            # Fallback: assume contiguous
            for c in range(char_pos, min(char_pos + len(tok_text), len(prompt_text))):
                char_to_tok[c] = tok_idx
            char_pos = min(char_pos + len(tok_text), len(prompt_text))
    return char_to_tok


def char_span_to_token_span(char_start, char_end, char_to_tok):
    """Convert a character span to a token span using the char-to-token map."""
    tok_start = None
    tok_end = None
    for c in range(char_start, min(char_end, len(char_to_tok))):
        t = char_to_tok[c]
        if t is not None:
            if tok_start is None:
                tok_start = t
            tok_end = t
    return tok_start, tok_end


def find_tags_with_regex(text):
    """
    Find all XML/HTML opening and closing tags in the text with character positions.

    Returns a list of dicts with tag info. Skips ChatML special markers.
    """
    # Match <tag>, </tag>, <tag />, <tag attr="val">, etc.
    # But skip ChatML markers like <|im_start|>
    tag_pattern = re.compile(
        r'<(?!\|)'           # opening < but not <|
        r'(/?)(\w[\w_.-]*)'  # optional / + tag name
        r'([^>]*?)'          # attributes
        r'(/?)>'             # optional / + >
    )
    tags = []
    for m in tag_pattern.finditer(text):
        is_closing = bool(m.group(1))
        tag_name = m.group(2)
        is_self_closing = bool(m.group(4)) or "/>" in m.group(0)
        tags.append({
            "tag_name": tag_name,
            "char_start": m.start(),
            "char_end": m.end(),
            "is_closing": is_closing,
            "is_self_closing": is_self_closing,
            "full_match": m.group(0)[:80],
        })
    return tags


def build_bs4_tree(text):
    """
    Parse the prompt text with BeautifulSoup to get the structural hierarchy.

    Returns a nested dict representation of the BS4 parse tree.
    """
    from bs4 import BeautifulSoup, Tag

    # Sanitize ChatML markers so BS4 doesn't choke on them
    clean = text
    clean = clean.replace("<|im_start|>", "__CHATML_START__")
    clean = clean.replace("<|im_end|>", "__CHATML_END__")
    clean = clean.replace("<|vision_start|>", "__VISION_START__")
    clean = clean.replace("<|vision_end|>", "__VISION_END__")
    clean = clean.replace("<|endoftext|>", "__ENDOFTEXT__")

    soup = BeautifulSoup(clean, "html.parser")

    def _to_dict(element, depth=0):
        result = {
            "tag": element.name if isinstance(element, Tag) else "[text]",
            "depth": depth,
            "children": [],
        }
        if isinstance(element, Tag):
            text_len = len(element.get_text())
            result["text_length"] = text_len
            for child in element.children:
                if isinstance(child, Tag):
                    result["children"].append(_to_dict(child, depth + 1))
        return result

    # Collect top-level tags
    nodes = []
    for child in soup.children:
        if isinstance(child, Tag):
            nodes.append(_to_dict(child))
    return nodes


def collect_all_tags_from_bs4(bs4_nodes, prefix=""):
    """Flatten the BS4 tree into a list of (tag_name, depth) tuples."""
    result = []
    for node in bs4_nodes:
        result.append((node["tag"], node["depth"]))
        result.extend(collect_all_tags_from_bs4(node["children"], prefix + "  "))
    return result


def collect_all_nodes_from_token_tree(node, depth=0):
    """Flatten the token-level tree into a list of (label, node_type, depth, start, end)."""
    result = []
    if node.node_type != "root":
        result.append({
            "label": node.label,
            "node_type": node.node_type,
            "depth": depth,
            "start_idx": node.start_idx,
            "end_idx": node.end_idx,
            "token_count": node.token_count,
        })
    for child in node.children:
        result.extend(collect_all_nodes_from_token_tree(child, depth + 1))
    return result


def run_comparison(prompt_text, token_ids, token_texts, tokenizer, token_tree, chunks):
    """
    Compare the token-level tree parser against a BeautifulSoup ground truth.

    Prints a detailed comparison report.
    """
    print("\n" + "=" * 80)
    print("BEAUTIFULSOUP GROUND TRUTH COMPARISON")
    print("=" * 80)

    # 1. Build char-to-token mapping
    print("\nBuilding character-to-token mapping...")
    char_to_tok = build_char_to_token_map(prompt_text, token_ids, tokenizer)
    mapped_count = sum(1 for x in char_to_tok if x is not None)
    print(f"  Mapped {mapped_count}/{len(prompt_text)} chars to tokens")

    # 2. Find all tags with regex (gives us character positions)
    print("\nFinding tags with regex...")
    regex_tags = find_tags_with_regex(prompt_text)
    opening_tags = [t for t in regex_tags if not t["is_closing"]]
    closing_tags = [t for t in regex_tags if t["is_closing"]]
    print(f"  Found {len(regex_tags)} total tags ({len(opening_tags)} opening, {len(closing_tags)} closing)")

    # 3. Map regex tags to token positions
    print("\nMapping regex tags to token positions...")
    for tag in regex_tags:
        tok_start, tok_end = char_span_to_token_span(
            tag["char_start"], tag["char_end"], char_to_tok
        )
        tag["tok_start"] = tok_start
        tag["tok_end"] = tok_end

    # 4. Build BS4 structural tree
    print("\nParsing with BeautifulSoup...")
    bs4_nodes = build_bs4_tree(prompt_text)

    # 5. Collect unique tag names from regex
    regex_tag_names = defaultdict(int)
    for t in opening_tags:
        regex_tag_names[t["tag_name"]] += 1

    # 6. Collect unique tag/node types from token tree
    token_nodes = collect_all_nodes_from_token_tree(token_tree)
    token_html_tags = [n for n in token_nodes if n["node_type"] == "html_tag"]
    token_tag_names = defaultdict(int)
    for n in token_html_tags:
        # Extract tag name from label (e.g., "<intro>" -> "intro")
        m = re.match(r"<?(\w+)", n["label"])
        if m:
            token_tag_names[m.group(1)] += 1

    # 7. Print comparison: tag name sets
    all_tag_names = set(regex_tag_names.keys()) | set(token_tag_names.keys())

    print(f"\n{'Tag Name':<25} {'Regex(GT)':>10} {'TokenParser':>12} {'Match':>6}")
    print("-" * 60)

    matched = 0
    missed = 0
    extra = 0
    for name in sorted(all_tag_names):
        gt_count = regex_tag_names.get(name, 0)
        tp_count = token_tag_names.get(name, 0)
        if gt_count > 0 and tp_count > 0:
            status = "OK" if gt_count == tp_count else f"~"
            matched += 1
        elif gt_count > 0 and tp_count == 0:
            status = "MISS"
            missed += 1
        else:
            status = "EXTRA"
            extra += 1
        print(f"  {name:<23} {gt_count:>10} {tp_count:>12} {status:>6}")

    print("-" * 60)
    print(f"  Matched: {matched}, Missed: {missed}, Extra: {extra}")

    # 8. Detailed per-tag boundary comparison
    # For each opening tag found by regex, find the closest token-level node
    print(f"\n{'='*80}")
    print("DETAILED TAG BOUNDARY COMPARISON (opening tags)")
    print(f"{'='*80}")
    print(f"  {'Tag':<20} {'Char Pos':>10} {'Regex Tok':>10} {'Parser Tok':>11} {'Delta':>6}")
    print("  " + "-" * 60)

    boundary_deltas = []
    for tag in opening_tags:
        if tag["tok_start"] is None:
            continue
        tag_name = tag["tag_name"]
        # Find the closest token-tree html_tag node with matching name
        best_match = None
        best_dist = float("inf")
        for n in token_html_tags:
            m = re.match(r"<?(\w+)", n["label"])
            if m and m.group(1) == tag_name:
                dist = abs(n["start_idx"] - tag["tok_start"])
                if dist < best_dist:
                    best_dist = dist
                    best_match = n

        if best_match:
            delta = best_match["start_idx"] - tag["tok_start"]
            boundary_deltas.append(abs(delta))
            delta_str = f"{delta:+d}" if delta != 0 else "0"
            print(
                f"  {tag_name:<20} {tag['char_start']:>10} "
                f"{tag['tok_start']:>10} {best_match['start_idx']:>11} {delta_str:>6}"
            )
        else:
            print(
                f"  {tag_name:<20} {tag['char_start']:>10} "
                f"{tag['tok_start']:>10} {'---':>11} {'MISS':>6}"
            )

    # 9. Summary
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"  Regex ground truth tags (opening):  {len(opening_tags)}")
    print(f"  Token parser html_tag nodes:        {len(token_html_tags)}")
    print(f"  Unique tag names in regex:           {len(regex_tag_names)}")
    print(f"  Unique tag names in token parser:    {len(token_tag_names)}")
    print(f"  Tag names matched:                   {matched}")
    print(f"  Tag names missed by parser:          {missed}")
    print(f"  Extra tag names in parser:           {extra}")
    if boundary_deltas:
        avg_delta = sum(boundary_deltas) / len(boundary_deltas)
        exact_matches = sum(1 for d in boundary_deltas if d == 0)
        close_matches = sum(1 for d in boundary_deltas if d <= 2)
        print(f"  Boundary comparisons:                {len(boundary_deltas)}")
        print(f"  Exact boundary matches (delta=0):    {exact_matches}/{len(boundary_deltas)}")
        print(f"  Close boundary matches (delta<=2):   {close_matches}/{len(boundary_deltas)}")
        print(f"  Average boundary delta:              {avg_delta:.1f} tokens")

    # 10. BS4 hierarchy overview
    print(f"\n{'='*80}")
    print("BEAUTIFULSOUP HIERARCHY (top 3 levels)")
    print(f"{'='*80}")
    _print_bs4_tree(bs4_nodes, max_depth=3)

    return {
        "regex_tag_count": len(opening_tags),
        "token_parser_html_tag_count": len(token_html_tags),
        "matched_tag_names": matched,
        "missed_tag_names": missed,
        "extra_tag_names": extra,
        "boundary_deltas": boundary_deltas,
        "regex_tag_names": dict(regex_tag_names),
        "token_tag_names": dict(token_tag_names),
    }


def _print_bs4_tree(nodes, indent=0, max_depth=3):
    """Print the BS4 tree structure."""
    if indent > max_depth:
        return
    for node in nodes:
        prefix = "  " * (indent + 1)
        child_count = len(node["children"])
        text_len = node.get("text_length", 0)
        extra = f" ({child_count} children, ~{text_len} chars)" if child_count else f" (~{text_len} chars)"
        print(f"{prefix}<{node['tag']}>{extra}")
        _print_bs4_tree(node["children"], indent + 1, max_depth)


# ============================================================
# Runtime trace comparison
# ============================================================

def compare_with_trace(trace_path, token_tree, chunks, prompt_len=None, offline_token_texts=None, offline_tree_dict=None):
    """Compare the offline-parsed tree against a runtime trace JSON.

    prompt_len: number of tokens in the prompt (offline). Used to distinguish
    'prompt-internal' tree diffs from 'boundary' diffs caused by assistant generation.
    offline_token_texts: per-token decoded texts from the offline verify script.
    offline_tree_dict: pre-computed tree dict snapshot (before extract_leaf_chunks mutation).
        If None, falls back to tree_to_dict_full(token_tree).
    """
    print(f"\n{'='*80}")
    print(f"RUNTIME TRACE COMPARISON: {os.path.basename(trace_path)}")
    print(f"{'='*80}")

    with open(trace_path, "r") as f:
        trace = json.load(f)

    # Compare token texts if available in both
    runtime_token_texts = trace.get("token_texts")
    if runtime_token_texts and offline_token_texts:
        n_compare = min(len(offline_token_texts), len(runtime_token_texts))
        mismatches = []
        for i in range(n_compare):
            if offline_token_texts[i] != runtime_token_texts[i]:
                mismatches.append(i)
        print(f"\n  Token text comparison: {n_compare} tokens compared, {len(mismatches)} mismatches")
        if mismatches:
            print(f"\n  ** TOKEN TEXT MISMATCHES (first 20):")
            print(f"  {'Index':>6}  {'Offline':>30}  {'Runtime':>30}")
            print("  " + "-" * 70)
            for idx in mismatches[:20]:
                off = repr(offline_token_texts[idx])[:30]
                run = repr(runtime_token_texts[idx])[:30]
                print(f"  {idx:>6}  {off:>30}  {run:>30}")
            if len(mismatches) > 20:
                print(f"  ... ({len(mismatches) - 20} more)")
        else:
            print(f"  Token texts are IDENTICAL between offline and runtime")
    elif runtime_token_texts is None:
        print(f"\n  (No token_texts in trace JSON — re-run with updated backend to enable token text comparison)")

    # Compare basic metadata
    trace_chunks = trace.get("chunks", [])
    print(f"\n  {'Metric':<30} {'Offline':>10} {'Runtime':>10} {'Match':>6}")
    print("  " + "-" * 60)

    runtime_seq_len = trace.get("seq_len", "?")
    print(f"  {'seq_len':<30} {'---':>10} {runtime_seq_len:>10}")
    print(f"  {'num_chunks':<30} {len(chunks):>10} {len(trace_chunks):>10} {'OK' if len(chunks) == len(trace_chunks) else 'DIFF':>6}")

    # Compare chunk boundaries
    if trace_chunks:
        print(f"\n  {'Chunk':<10} {'Offline Start':>14} {'Runtime Start':>14} {'Offline End':>12} {'Runtime End':>12} {'Match':>6}")
        print("  " + "-" * 72)

        n_match = 0
        n_total = min(len(chunks), len(trace_chunks))
        for i in range(n_total):
            oc = chunks[i]
            rc = trace_chunks[i]
            start_ok = oc.start_idx == rc["start_idx"]
            end_ok = oc.end_idx == rc["end_idx"]
            both_ok = start_ok and end_ok
            if both_ok:
                n_match += 1
            status = "OK" if both_ok else "DIFF"
            print(
                f"  {i:<10} {oc.start_idx:>14} {rc['start_idx']:>14} "
                f"{oc.end_idx:>12} {rc['end_idx']:>12} {status:>6}"
            )
            if i >= 20 and not both_ok:
                remaining = n_total - i - 1
                if remaining > 0:
                    print(f"  ... ({remaining} more chunks)")
                break

        print(f"\n  Chunk boundary matches: {n_match}/{n_total}")

    # Compare tree structure if available
    trace_tree = trace.get("tree")
    if trace_tree:
        offline_tree_dict = offline_tree_dict or tree_to_dict_full(token_tree)
        diffs = []
        trees_match = _compare_tree_dicts(
            offline_tree_dict, trace_tree, diffs=diffs, prompt_len=prompt_len
        )
        # Separate diffs by kind
        prompt_diffs = [d for d in diffs if d["kind"] == "prompt_internal"]
        boundary_diffs = [d for d in diffs if d["kind"] == "boundary"]

        print(f"\n  Tree structure match (full):         {'YES' if trees_match else 'NO'}")
        print(f"  Prompt-internal diffs:               {len(prompt_diffs)}")
        print(f"  Boundary diffs (assistant expected):  {len(boundary_diffs)}")

        prompt_only_match = len(prompt_diffs) == 0
        print(f"  Tree structure match (prompt only):  {'YES' if prompt_only_match else 'NO'}")

        if prompt_diffs:
            print(f"\n  ** PROMPT-INTERNAL tree differences ({len(prompt_diffs)}):")
            print(f"     (These are REAL structural diffs within the prompt tokens)")
            for d in prompt_diffs[:20]:
                print(d["msg"])
            if len(prompt_diffs) > 20:
                print(f"  ... ({len(prompt_diffs) - 20} more)")

        if boundary_diffs:
            print(f"\n  Boundary tree differences ({len(boundary_diffs)}):")
            print(f"     (Expected: offline end_idx is at prompt boundary, runtime extends with assistant tokens)")
            for d in boundary_diffs[:5]:
                print(d["msg"])
            if len(boundary_diffs) > 5:
                print(f"  ... ({len(boundary_diffs) - 5} more)")

    # Show decode step summary if available
    decode_steps = trace.get("decode_steps", [])
    if decode_steps:
        print(f"\n  Decode steps recorded: {len(decode_steps)}")
        sparsities = [s.get("sparsity", 0) for s in decode_steps]
        if sparsities:
            print(f"  Sparsity range: {min(sparsities):.4f} - {max(sparsities):.4f}")
            print(f"  Average sparsity: {sum(sparsities)/len(sparsities):.4f}")


def _compare_tree_dicts(a, b, path="root", diffs=None, prompt_len=None):
    """Recursively compare two tree dicts. Returns True if structurally equal.

    Collects differences in `diffs` list. Each diff is a dict with:
      - 'msg': human-readable description
      - 'kind': 'prompt_internal' or 'boundary' (only for end_idx diffs)
    If prompt_len is given, end_idx diffs where BOTH values >= prompt_len-1
    are classified as 'boundary' (expected from assistant token generation).
    """
    if diffs is None:
        diffs = []
    match = True

    if a.get("node_type") != b.get("node_type"):
        diffs.append({
            "msg": f"  {path}: node_type differs: offline={a.get('node_type')} vs runtime={b.get('node_type')}",
            "kind": "prompt_internal",
        })
        match = False

    if a.get("start_idx") != b.get("start_idx"):
        diffs.append({
            "msg": f"  {path}: start_idx differs: offline={a.get('start_idx')} vs runtime={b.get('start_idx')}",
            "kind": "prompt_internal",
        })
        match = False

    a_end = a.get("end_idx")
    b_end = b.get("end_idx")
    if a_end != b_end:
        # Classify: if both ends are at or near the prompt boundary, it's
        # expected because the runtime tree grows with assistant tokens.
        if prompt_len is not None and a_end is not None and b_end is not None:
            # "boundary" = the offline node reaches the end of the prompt
            # (the runtime node would extend further with assistant tokens)
            is_boundary = (a_end >= prompt_len - 2)
        else:
            is_boundary = False
        kind = "boundary" if is_boundary else "prompt_internal"
        diffs.append({
            "msg": f"  {path}: end_idx differs: offline={a_end} vs runtime={b_end}",
            "kind": kind,
        })
        match = False

    a_children = a.get("children", [])
    b_children = b.get("children", [])
    if len(a_children) != len(b_children):
        diffs.append({
            "msg": (
                f"  {path}: children count differs: offline={len(a_children)} vs runtime={len(b_children)}"
                f" (offline labels: {[c.get('label','?')[:30] for c in a_children[-3:]]})"
                f" (runtime labels: {[c.get('label','?')[:30] for c in b_children[-3:]]})"
            ),
            "kind": "prompt_internal",
        })
        match = False
    for i, (ac, bc) in enumerate(zip(a_children, b_children)):
        child_label = ac.get("label", "?")[:20]
        child_path = f"{path}/{ac.get('node_type','?')}[{i}]({child_label})"
        if not _compare_tree_dicts(ac, bc, child_path, diffs, prompt_len):
            match = False
        if len(diffs) > 50:
            diffs.append({"msg": "  ... (truncated, more diffs exist)", "kind": "prompt_internal"})
            return False
    return match


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Verify tree parser on a request JSON file"
    )
    parser.add_argument(
        "--request-json",
        required=True,
        help="Path to the request JSON file (same format as /v1/chat/completions)",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to model (for loading tokenizer), e.g. $HF_MODELS/Qwen/Qwen3-VL-8B-Instruct",
    )
    parser.add_argument(
        "--min-chunk-size",
        type=int,
        default=16,
        help="Minimum tokens per chunk (default: 16)",
    )
    parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=256,
        help="Maximum tokens per chunk (default: 256)",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Path to save output JSON (default: <request_json>.tree_verify.json)",
    )
    parser.add_argument(
        "--show-tokens",
        action="store_true",
        help="Also print per-token texts (can be very long)",
    )
    parser.add_argument(
        "--show-token-range",
        type=str,
        default=None,
        help="Show tokens in a range, e.g. '100-200'",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run BeautifulSoup ground truth comparison",
    )
    parser.add_argument(
        "--trace-json",
        default=None,
        help="Path to a runtime trace JSON from tree_sparse_traces/ for comparison",
    )
    args = parser.parse_args()

    # Start capturing stdout to a buffer (for saving to text file later)
    tee = TeeWriter(sys.stdout)
    sys.stdout = tee

    # 1. Load request JSON
    print(f"Loading request from: {args.request_json}")
    request = load_request_json(args.request_json)
    messages = request["messages"]
    print(f"  Found {len(messages)} messages: {[m['role'] for m in messages]}")

    # 2. Load tokenizer
    print(f"\nLoading tokenizer from: {args.model_path}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # 3. Apply chat template and tokenize
    print("\nApplying chat template and tokenizing...")
    token_ids, prompt_text = apply_chat_template_and_tokenize(messages, tokenizer)
    print(f"  Prompt length: {len(prompt_text)} chars")
    print(f"  Token count: {len(token_ids)} tokens")

    # 4. Decode per-token
    print("\nDecoding per-token texts...")
    token_texts = decode_per_token(token_ids, tokenizer)

    if args.show_tokens:
        print("\n--- Per-Token Texts ---")
        for i, (tid, txt) in enumerate(zip(token_ids, token_texts)):
            print(f"  [{i:5d}] id={tid:6d}  {repr(txt)}")
        print("--- End Per-Token Texts ---\n")

    if args.show_token_range:
        start, end = map(int, args.show_token_range.split("-"))
        print(f"\n--- Tokens [{start}-{end}] ---")
        for i in range(start, min(end + 1, len(token_ids))):
            print(f"  [{i:5d}] id={token_ids[i]:6d}  {repr(token_texts[i])}")
        print(f"--- End Tokens [{start}-{end}] ---\n")

    # 5. Parse tree
    print("\nParsing ChatML/HTML tree...")
    tree = parse_chatml_tree(token_texts)
    print(f"  Root children: {len(tree.children)}")

    # 6. Print full tree
    print("\n" + "=" * 80)
    print("FULL TREE STRUCTURE")
    print("=" * 80)
    print_tree_full(tree)

    # Capture tree dict BEFORE extract_leaf_chunks, which mutates end_idx
    # via _fix_parent_ranges(). This matches the runtime backend order
    # (tree_sparse_backend.py calls tree.to_dict() before extract_leaf_chunks).
    tree_dict_snapshot = tree_to_dict_full(tree)

    # 7. Extract chunks
    print("\n" + "=" * 80)
    print(f"LEAF CHUNKS (min={args.min_chunk_size}, max={args.max_chunk_size})")
    print("=" * 80)
    chunks = extract_leaf_chunks(tree, args.min_chunk_size, args.max_chunk_size)
    print(f"  Total chunks: {len(chunks)}")
    total_covered = sum(c.token_count for c in chunks)
    print(f"  Tokens covered: {total_covered}/{len(token_ids)}")
    print()
    print_all_chunks(chunks)

    # 8. Show chunk token content for verification
    print("\n" + "=" * 80)
    print("CHUNK CONTENT PREVIEW (first 5 tokens of each chunk)")
    print("=" * 80)
    for c in chunks:
        start = c.start_idx
        end = min(c.start_idx + 5, c.end_idx + 1)
        preview_tokens = token_texts[start:end]
        preview = "".join(preview_tokens)
        if len(preview) > 100:
            preview = preview[:97] + "..."
        suffix = "..." if c.token_count > 5 else ""
        print(f"  chunk_{c.chunk_id:3d}: {repr(preview)}{suffix}")

    # 9. BeautifulSoup comparison
    comparison_result = None
    if args.compare:
        try:
            comparison_result = run_comparison(
                prompt_text, token_ids, token_texts, tokenizer, tree, chunks
            )
        except ImportError:
            print("\nWARNING: beautifulsoup4 not installed. Install with: pip install beautifulsoup4")
        except Exception as e:
            print(f"\nWARNING: BS4 comparison failed: {e}")
            import traceback
            traceback.print_exc()

    # 10. Runtime trace comparison
    if args.trace_json:
        if os.path.exists(args.trace_json):
            compare_with_trace(args.trace_json, tree, chunks, prompt_len=len(token_ids), offline_token_texts=token_texts, offline_tree_dict=tree_dict_snapshot)
        else:
            print(f"\nWARNING: Trace file not found: {args.trace_json}")

    # 11. Save to JSON
    output_path = args.output_json or args.request_json.replace(
        ".json", ".tree_verify.json"
    )
    result = {
        "request_json": os.path.abspath(args.request_json),
        "model_path": args.model_path,
        "num_messages": len(messages),
        "message_roles": [m["role"] for m in messages],
        "num_tokens": len(token_ids),
        "prompt_char_length": len(prompt_text),
        "config": {
            "min_chunk_size": args.min_chunk_size,
            "max_chunk_size": args.max_chunk_size,
        },
        "tree": tree_dict_snapshot,
        "chunks": [c.to_dict() for c in chunks],
        "chunk_content_previews": {},
    }
    # Add chunk content previews (first 10 tokens of each chunk)
    for c in chunks:
        start = c.start_idx
        end = min(c.start_idx + 10, c.end_idx + 1)
        preview_tokens = token_texts[start:end]
        result["chunk_content_previews"][str(c.chunk_id)] = {
            "first_tokens": preview_tokens,
            "first_token_ids": token_ids[start:end],
            "text_preview": "".join(preview_tokens),
        }
    if comparison_result:
        result["bs4_comparison"] = comparison_result

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")

    # 12. Summary stats
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Total tokens:    {len(token_ids)}")
    print(f"  Tree children:   {len(tree.children)}")
    print(f"  Total chunks:    {len(chunks)}")
    print(f"  Tokens covered:  {total_covered}/{len(token_ids)}")
    coverage = total_covered / len(token_ids) * 100 if token_ids else 0
    print(f"  Coverage:        {coverage:.1f}%")
    if chunks:
        sizes = [c.token_count for c in chunks]
        print(f"  Chunk sizes:     min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)/len(sizes):.1f}")

    # 13. Save terminal output as text file
    sys.stdout = tee._stdout  # restore original stdout
    txt_path = output_path.replace(".json", ".txt")
    with open(txt_path, "w") as f:
        f.write(tee.getvalue())
    print(f"Terminal output saved to: {txt_path}")


if __name__ == "__main__":
    main()
