# Tree Structure Visualization Guide

You now have **TWO types** of tree visualizations available:

## 1. Hierarchical Weight-Based Tree (`visualize_attention_hierarchical.py`)

**What it shows:** Groups tokens by attention weight ranges (tiers)

**Structure:**
```
Root (Query Token)
├── Very High (>0.1)
│   ├── token1
│   ├── token2
├── High (0.01-0.1)
│   ├── token10
│   └── token11
└── Medium (0.001-0.01)
    └── ...
```

**Use when:** You want to see which tokens get the most/least attention

**Command:**
```bash
python visualize_attention_hierarchical.py \
    --csv attention_weights/archive_20251207_230914/attn_weights_request_000001_layer_00_head_00.csv \
    --tokens attention_weights/archive_20251207_230914/attn_weights_request_000001_layer_00_head_00_tokens.csv \
    --layer 0 --head 0
```

**Output:** `attn_weights_request_000001_layer_00_head_00_focus8641_viz.html`

---

## 2. ChatML/HTML Structure Tree (`visualize_attention_tree_structure.py`) ⭐ NEW!

**What it shows:** The actual conversation structure (ChatML) and HTML DOM tree with attention overlaid

**Structure:**
```
Conversation (Root)
├── <system>
│   └── "You are an AI..."
├── <user>
│   ├── <div>
│   │   ├── "Hello,"
│   │   └── <ul>
│   │       ├── <li>
│   │       │   └── "Earth"
│   │       └── <li>
│   │           └── "Moon"
│   └── "Please help"
└── <assistant>
    └── "I'll help you..."
```

**Use when:** You want to see how attention flows through the conversation structure and HTML elements

**Command:**
```bash
python visualize_attention_tree_structure.py \
    --csv attention_weights/archive_20251207_230914/attn_weights_request_000001_layer_00_head_00.csv \
    --tokens attention_weights/archive_20251207_230914/attn_weights_request_000001_layer_00_head_00_tokens.csv \
    --layer 0 --head 0
```

**Output:** `attn_weights_request_000001_layer_00_head_00_focus8641_tree.html`

---

## Key Differences

| Feature | Weight-Based (`_viz.html`) | Structure-Based (`_tree.html`) |
|---------|---------------------------|-------------------------------|
| **Grouping** | By attention weight tiers | By ChatML/HTML structure |
| **Node types** | Weight ranges | ChatML tags, HTML tags, text |
| **Color coding** | By weight magnitude | By node type (ChatML/HTML/text) |
| **Best for** | Finding important tokens | Understanding structure |
| **Layout** | Partition/flame graph | Tree diagram |

---

## Usage Examples

### Example 1: Analyze conversation structure

```bash
# See how attention flows through the ChatML conversation
python visualize_attention_tree_structure.py \
    --csv attention_weights/archive_20251207_230914/attn_weights_request_000001_layer_00_head_00.csv \
    --tokens attention_weights/archive_20251207_230914/attn_weights_request_000001_layer_00_head_00_tokens.csv \
    --layer 0 --head 0
```

This will show:
- ChatML tags: `<system>`, `<user>`, `<assistant>`
- HTML elements: `<div>`, `<ul>`, `<li>`, etc.
- Text content within each element
- Attention weight for each node (opacity/color)

### Example 2: Compare both views

Generate both visualizations for the same data:

```bash
# Weight-based view
python visualize_attention_hierarchical.py \
    --csv attn_weights.csv \
    --tokens attn_weights_tokens.csv \
    --layer 0 --head 0

# Structure-based view
python visualize_attention_tree_structure.py \
    --csv attn_weights.csv \
    --tokens attn_weights_tokens.csv \
    --layer 0 --head 0
```

Then open both HTML files in different browser tabs to compare!

### Example 3: Focus on specific token in structure view

```bash
# See structure from token 1000's perspective
python visualize_attention_tree_structure.py \
    --csv attn_weights.csv \
    --tokens attn_weights_tokens.csv \
    --focus-token 1000
```

---

## Visual Guide

### Weight-Based Tree (Hierarchical)
```
┌─────────────────────────────────────┐
│ Query: "browser"                    │
├─────────────────────────────────────┤
│ Very High (>0.1) [5 tokens]         │ ← Grouped by weight
│  ├─ "You" (0.2145)                  │
│  ├─ "are" (0.1456)                  │
│  └─ ...                             │
├─────────────────────────────────────┤
│ High (0.01-0.1) [23 tokens]         │
│  └─ ...                             │
└─────────────────────────────────────┘
```

### Structure-Based Tree (ChatML/HTML)
```
┌─────────────────────────────────────┐
│ Conversation                        │
├─ <system>                           │ ← ChatML structure
│  └─ "You are an AI..."              │
├─ <user>                             │
│  ├─ <div>                           │ ← HTML structure
│  │  ├─ "Hello,"                     │
│  │  └─ <ul>                         │
│  │     ├─ <li> → "Earth"            │
│  │     └─ <li> → "Moon"             │
│  └─ "Please help"                   │
└─ <assistant>                        │
   └─ "I'll help you..."              │
└─────────────────────────────────────┘
```

---

## Interpreting the Tree Structure View

### Node Colors:
- 🟢 **Green:** ChatML tags (`<system>`, `<user>`, `<assistant>`)
- 🔵 **Blue:** HTML tags (`<div>`, `<ul>`, `<li>`, etc.)
- 🟠 **Orange:** Text content

### Node Opacity:
- **Bright/Opaque:** High attention weight
- **Dim/Transparent:** Low attention weight

### Hover Information:
- Node name (tag or text)
- Node type (chatml/html_tag/text)
- Token range (start-end indices)
- Average attention weight

---

## Tips

1. **Use structure view** to understand:
   - How the model navigates the conversation
   - Which HTML elements receive attention
   - Context flow through the document

2. **Use weight view** to understand:
   - Which specific tokens are most important
   - Overall attention distribution
   - Filtering by importance

3. **Compare both views** to get complete picture:
   - Structure view shows WHERE attention goes (in context)
   - Weight view shows HOW MUCH attention (magnitude)

---

## File Naming Convention

Both scripts follow the same naming pattern:

**Weight-based:**
- Default: `{csv_name}_focus{N}_viz.html`
- Show all: `{csv_name}_focus{N}_all_tokens_viz.html`

**Structure-based:**
- Always: `{csv_name}_focus{N}_tree.html`

Example:
```
attn_weights_request_000001_layer_00_head_00_focus8641_viz.html      # Weight view
attn_weights_request_000001_layer_00_head_00_focus8641_tree.html     # Structure view
```

---

## Troubleshooting

**Q: The tree structure looks empty or wrong**

A: Make sure your tokens contain ChatML markers (`<|im_start|>`, `<|im_end|>`) or HTML tags. The parser looks for these specific patterns.

**Q: Some HTML tags are not parsed correctly**

A: The parser uses a simple regex-based approach. Complex or malformed HTML might not parse perfectly. This is expected for visualization purposes.

**Q: Attention weights seem off in structure view**

A: Structure view shows **average attention** across all tokens in each node's range. This is different from individual token weights in the weight-based view.

---

Enjoy exploring your attention patterns in both views! 🎨
