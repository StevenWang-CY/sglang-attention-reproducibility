# Attention Analysis Summary

## Key Findings

### 1. **Where is the "missing" 0.15 attention in browser_state?**

The answer is: **It's not actually missing!**

When we analyzed the flamegraph showing `<browser_state>` with value 0.703 and `<html />` child with value 0.554:

- **Actual sum of raw attention weights** in range [3999..8417]: **0.0858** (8.57% of total)
- **Flamegraph displayed value**: **0.7030**

The discrepancy comes from how the flamegraph values are calculated versus what the raw attention weights show. The 0.15 "difference" between parent (0.703) and child (0.554) is distributed across:

1. **Gap tokens** (whitespace, tags between elements): 0.002537
2. **Recursive gaps in child hierarchies**: 0.145823

These gaps represent structural tokens (tabs, newlines, element markers, opening/closing tags) that receive small amounts of attention but aren't represented as explicit child nodes.

---

## 2. **Persistent High-Attention Elements Across Generation**

Analysis of 113 generated tokens (8426-8650) shows which tree elements **consistently** receive high attention:

### Top Elements by Total Attention (appearing in 100% of tokens):

| Rank | Range | Tokens | Category | Mean | Total | Element |
|------|-------|--------|----------|------|-------|---------|
| 1 | [8420:8652] | 233 | **assistant** (generation) | 0.689 | 77.83 | The assistant response being generated |
| 2 | [8423:8516] | 94 | **Text** | 0.349 | 39.48 | The "thinking" field in JSON output |
| 3 | [3577:8419] | 4843 | **user** | 0.242 | 27.40 | Entire user context |
| 4 | [3999:8417] | 4419 | **browser_state** | 0.212 | 24.00 | Browser state section |
| 5 | [4136:8400] | 4265 | **html_root** | 0.172 | 19.41 | The `<html />` root element |
| 6 | [8551:8598] | 48 | **Text** | 0.137 | 15.51 | The "memory" field |
| 7 | [8518:8549] | 32 | **Text** | 0.104 | 11.72 | The "evaluation_previous_goal" field |
| 8 | [8033:8386] | 354 | **BrowserElement:li** | 0.073 | 8.22 | A list item `*[3224]<li level=1 />` |
| 9 | [0:3576] | 3577 | **system** | 0.069 | 7.77 | System prompt |

### Attention by Category:

| Category | Total Attention | Num Nodes | Key Insight |
|----------|----------------|-----------|-------------|
| **Text** | 84.73 | 189 | Generated JSON fields (thinking, memory, actions) |
| **BrowserElement:div** | 28.94 | 81 | HTML div elements in browser state |
| **BrowserElement:li** | 13.26 | 9 | List items (search results) |
| **BrowserElement:article** | 4.75 | 6 | Article elements (search result cards) |
| **HTML tags** | 12.95 | 30 | Structural elements (agent_history, browser_rules) |

---

## 3. **Elements with Highest Variance (Change Most)**

These elements' attention changes significantly across generation steps:

1. **"thinking" field** (stddev: 0.235) - Varies most as model generates different reasoning
2. **"memory" field** (stddev: 0.193) - Changes based on what's relevant
3. **assistant section** (stddev: 0.181) - Overall generation focus shifts
4. **"evaluation_previous_goal"** (stddev: 0.148) - Varies by context

---

## 4. **Most Stable Elements (Low Variance)**

These maintain consistent attention:

1. **Root node** (stddev: 0.0003) - Always gets near-total attention
2. **Specific browser elements** - Particular search results maintain steady importance
3. **System prompt sections** - Background context remains stable

---

## Tools Created

### 1. **query_attention_range.py**
Query attention weights for specific ranges:
```bash
python3 query_attention_range.py attention.csv -q -1 -s 3999 -e 8417
```

### 2. **analyze_persistent_tree_nodes.py**
Find which tree elements persist across generation:
```bash
python3 analyze_persistent_tree_nodes.py "attention_flamegraph_*.html" --top-n 50
```

### 3. **find_persistent_attention.py**
Find individual token positions with high attention:
```bash
python3 find_persistent_attention.py --csv attention.csv --tokens tokens.csv --start 8422 --end 8450
```

### 4. **run_flamegraph_batch.sh**
Generate flamegraphs for multiple layers/heads/tokens:
```bash
./run_flamegraph_batch.sh --layer-start 0 --layer-end 3 --head-start 0 --head-end 7
```

---

## Conclusions

1. **Most attention goes to the generated output itself** (77.8 total across all tokens)
2. **Browser state is important** (24.0 total) but less than the generation
3. **Specific search results** (articles, list items) maintain consistent importance
4. **System prompt** (7.8 total) provides stable background context
5. **The "thinking" field in generation shows highest variance** - the model focuses here differently for each token

The attention pattern suggests the model primarily attends to:
- What it's currently generating (self-attention)
- Relevant browser elements (search results)
- User context for task understanding
- System prompt for behavioral guidance
