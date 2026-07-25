# Tree-Sparse Attention: Full Implementation Details

## 1. Overview

Tree-sparse attention is a custom attention backend for SGLang that exploits the structural hierarchy (ChatML roles, HTML tags, browser DOM elements) in the input to partition the KV cache into semantically meaningful **chunks**, then selects only the most relevant chunks per query via **centroid similarity**. This makes both prefill and decode sub-quadratic.

### Architecture

```
                        ┌──────────────────────────────────────────────────────────┐
                        │                  tree_sparse_backend.py                  │
                        │                  (TreeSparseAttnBackend)                 │
                        │                                                          │
                        │  ┌─────────────┐  ┌─────────────┐  ┌────────────────┐   │
                        │  │ tree_parser  │  │  centroid    │  │   sparse       │   │
                        │  │             │  │  manager     │  │   selector     │   │
                        │  │ - parse tree │  │ - store/get  │  │ - top-k select │   │
                        │  │ - extract   │  │ - incremental│  │ - build KV idx │   │
                        │  │   chunks    │  │   update     │  │ - batch meta   │   │
                        │  └─────────────┘  └─────────────┘  └────────────────┘   │
                        │                                                          │
                        │         FlashInfer wrappers (ragged, paged, decode)      │
                        └──────────────────────────────────────────────────────────┘
                                                   │
                                                   │  forward(q, k, v, ..., hidden_states, W_k)
                                                   │
                        ┌──────────────────────────────────────────────────────────┐
                        │               RadixAttention (base_attn_backend.py)      │
                        │               dispatches forward_extend / forward_decode  │
                        │               via **kwargs pass-through                  │
                        └──────────────────────────────────────────────────────────┘
                                                   │
                        ┌──────────────────────────────────────────────────────────┐
                        │                     qwen3.py (Qwen3Attention)            │
                        │               extracts W_k, passes hidden_states        │
                        └──────────────────────────────────────────────────────────┘
```

### Files

| File | Role |
|------|------|
| `tree_sparse_backend.py` | Main backend class. Handles registration, prefill, decode, logging. |
| `tree_sparse/tree_parser.py` | Parses token sequences into a tree of ChatML/HTML nodes, extracts leaf chunks. |
| `tree_sparse/centroid_manager.py` | Stores per-request, per-layer centroids. Computes from hidden states or K vectors. Incremental update during decode. |
| `tree_sparse/sparse_selector.py` | Top-k chunk selection via dot-product similarity. Builds sparse KV indices for FlashInfer. |
| `qwen3.py` | Model layer. Extracts `W_k` from `qkv_proj.weight`, passes `hidden_states` and `W_k` to backend. |
| `radix_attention.py` | Base attention dispatch. Forwards `**kwargs` (including `hidden_states`, `W_k`) to the backend. |
| `server_args.py` | CLI arguments: `--enable-tree-sparse`, `--tree-sparse-top-k`, etc. |

---

## 2. Component Details

### 2.1 Tree Parser (`tree_parser.py`)

**Purpose**: Turn a flat token sequence into a hierarchical tree, then flatten the tree's leaves into non-overlapping chunks.

**Step 1: Parse tokens into a tree** (`parse_chatml_tree`)

The parser recognizes these structural markers in the token stream:

| Pattern | Node Type | Example |
|---------|-----------|---------|
| `<\|im_start\|>role ... <\|im_end\|>` | `chatml` | ChatML message boundaries |
| `<\|vision_start\|> ... <\|vision_end\|>` | `vision` | Image token blocks (Qwen VL) |
| `<tag> ... </tag>` | `html_tag` | HTML elements like `<agent_history>`, `<step>` |
| `[1234]<div />` with tab indentation | `browser_element` | Browser DOM accessibility tree elements |
| Plain text between tags | `text` | Content text nodes |

The parser maintains a stack to track nesting. Browser state (tab-indented DOM elements) is handled specially using tab depth for parent-child relationships.

**Result**: A `TreeNode` tree like:
```
[root] all [0-5000]
  [chatml] system [0-50]
    [text] "You are a helpful..." [3-48]
  [chatml] user [51-4900]
    [html_tag] <agent_history> [55-4800]
      [html_tag] <step> [60-2000]
        [html_tag] <browser_state> [100-1900]
          [browser_element] [123]<div class="main"> [105-1895]
            [browser_element] [456]<a href="..."> [200-250]
            [browser_element] [789]<p /> Some text [251-300]
            ...
      [html_tag] <step> [2001-4800]
        ...
  [chatml] assistant [4901-5000]
```

**Step 2: Extract leaf chunks** (`extract_leaf_chunks`)

Walk the tree, collect all leaf nodes, then:
1. **Fill gaps**: If there are token ranges not covered by any leaf, create `gap` nodes
2. **Merge small leaves**: If a leaf has fewer than `min_chunk_size` (default 16) tokens, merge with the next sibling until the combined size reaches the minimum
3. **Split large leaves**: If a leaf has more than `max_chunk_size` (default 256) tokens, split into sub-chunks

**Result**: A list of `FlatChunk` objects, each with `(start_idx, end_idx, chunk_id, label)`. Chunks are non-overlapping and collectively cover the full sequence.

**Fallback**: If tree parsing fails or produces no structure, falls back to fixed-size chunks (`make_fixed_chunks`).

---

### 2.2 Centroid Manager (`centroid_manager.py`)

**Purpose**: Store and compute the **centroid** (mean key vector) for each chunk at each layer.

**Data structure**:
```python
CentroidEntry:
    chunks: List[FlatChunk]                              # chunk definitions
    centroids: Dict[int, Tensor]  # layer_id -> [num_chunks, num_kv_heads, head_dim]
    counts: Dict[int, Tensor]     # layer_id -> [num_chunks] (token counts for running mean)
```

**Three ways to compute centroids**:

**(a) From KV cache** (`update_centroids_for_layer`) -- original method, not currently used:
```
For each chunk [start, end]:
    physical_locs = kv_indices[start:end]
    chunk_keys = key_buffer[physical_locs]     # [chunk_len, num_kv_heads, head_dim]
    centroid = chunk_keys.float().mean(dim=0)   # [num_kv_heads, head_dim]
```
Cost: reads all seq_len key vectors from KV cache.

**(b) From hidden states** (`update_centroids_from_hidden_states`) -- current method:
```
For each chunk [start, end]:
    chunk_hidden = hidden_states[start:end]     # [chunk_len, hidden_dim]
    x_centroid = chunk_hidden.float().mean(dim=0)  # [hidden_dim]     -- average FIRST
    k_centroid = x_centroid @ W_k                  # [kv_dim]         -- project ONCE
    centroid = k_centroid.view(num_kv_heads, head_dim)
```
Cost: num_chunks matrix-vector multiplies instead of seq_len.

Mathematical justification:
```
mean({x_i @ W_k}) = (1/n) * sum(x_i @ W_k)
                   = ((1/n) * sum(x_i)) @ W_k
                   = mean({x_i}) @ W_k
```

**(c) Incremental update during decode** (`update_centroid_incremental`):
```
new_centroid = (old_centroid * n + new_key) / (n + 1)
```
Each new decode token updates the last chunk's centroid with a running mean. Cost: O(1) per token per layer.

---

### 2.3 Sparse Selector (`sparse_selector.py`)

**Purpose**: Given query vectors and chunk centroids, select the most relevant chunks and build FlashInfer-compatible sparse KV indices.

**(a) Top-k chunk selection** (`select_top_k_chunks`):

```python
# Handle GQA: average query heads that share the same KV head
# query: [num_q, num_qo_heads, head_dim]
# --> q_grouped: [num_q, num_kv_heads, head_dim]
q_grouped = query.view(num_q, num_kv_heads, group_size, head_dim).mean(dim=2)

# Dot-product similarity: [num_q, num_chunks]
scores = einsum("qkd,ckd->qc", q_grouped, centroids) * scaling

# Decode (single query): take top-k directly
# Prefill (multiple queries): take max score across all query tokens, then top-k
max_scores = scores.max(dim=0).values
_, top_indices = max_scores.topk(top_k)
```

**(b) Sparse KV index construction** (`build_sparse_kv_indices`):

From the selected chunk IDs, build the set of logical token positions to attend to:
```
positions = {first 4 tokens}                    # attention sinks (always)
           | {last 128 tokens}                  # sliding window (always)
           | {all tokens in selected chunks}    # sparse selection
```

Then map logical positions to physical KV pool locations:
```
physical_indices = req_to_token[req_pool_idx, sorted_positions]
```

**(c) Batch metadata** (`build_batch_sparse_metadata`):

Concatenate per-request sparse indices into FlashInfer's `(kv_indptr, kv_indices)` format:
```
kv_indptr = [0, len(req0_indices), len(req0_indices) + len(req1_indices), ...]
kv_indices = cat(req0_indices, req1_indices, ...)
```

---

### 2.4 Backend (`tree_sparse_backend.py`)

**Purpose**: Orchestrate the full flow. Plugs into SGLang's attention backend system.

#### Initialization

```python
TreeSparseAttnBackend.__init__(model_runner):
    # Read config from server_args
    self.top_k_chunks = server_args.tree_sparse_top_k          # default 8
    self.min_seq_len_for_sparse = server_args.tree_sparse_min_seq_len  # default 512
    self.min_chunk_size = server_args.tree_sparse_min_chunk_size       # default 16
    self.max_chunk_size = server_args.tree_sparse_max_chunk_size       # default 256
    self.always_include_recent = server_args.tree_sparse_recent_tokens # default 128
    self.always_include_first = 4                                      # attention sinks
    self.enable_sparse_prefill = not server_args.tree_sparse_full_prefill  # default True

    # Create FlashInfer wrappers
    self.prefill_wrapper_ragged = BatchPrefillWithRaggedKVCacheWrapper(...)
    self.prefill_wrapper_paged = BatchPrefillWithPagedKVCacheWrapper(...)
    self.decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(...)

    # Create centroid manager
    self.centroid_manager = CentroidManager(num_layers, num_kv_heads, head_dim, ...)
```

#### Request Registration (during prefill)

When a new request arrives in `_init_extend_metadata`, the backend:

1. Gets the request's token IDs
2. Decodes each token to text using the tokenizer
3. Calls `parse_chatml_tree(token_texts)` to build the tree
4. Calls `extract_leaf_chunks(tree)` to get chunks
5. Calls `centroid_manager.register_request(req_pool_idx, chunks)` to store chunk structure
6. Saves tree + chunks to JSON for debugging

#### Prefill Flow

**`forward_extend(q, k, v, layer, forward_batch, hidden_states, W_k)`**

Dispatch based on `enable_sparse_prefill`:

**Sparse prefill** (default) -- `_forward_extend_sparse`:
```
1. Compute centroids: centroid_manager.update_centroids_from_hidden_states(hidden_states, W_k)
2. Save K, V to cache: token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)
3. For each request:
   a. Get centroids for this layer
   b. select_top_k_chunks(q_i, centroids, chunks, top_k)
   c. build_sparse_kv_indices(selected_ids, chunks, req_to_token, ...)
4. build_batch_sparse_metadata(batch_kv_indices)
5. prefill_wrapper_paged.begin_forward(qo_indptr, kv_indptr, kv_indices, ...)
6. o = prefill_wrapper_paged.forward(q, kv_buffer, causal=True, ...)
```

**Full prefill** (with `--tree-sparse-full-prefill`) -- standard FlashInfer:
```
1. Save K, V to cache
2. Full attention via ragged and/or paged wrappers (same as flashinfer_backend)
3. Compute centroids: centroid_manager.update_centroids_from_hidden_states(hidden_states, W_k)
```

The only difference: in sparse prefill, centroids are computed BEFORE attention (to enable chunk selection), and attention uses sparse indices. In full prefill, centroids are computed AFTER attention (just for later decode use).

#### Decode Flow

**`forward_decode(q, k, v, layer, forward_batch)`**

Every layer, every decode step:

```
1. Save new K, V to cache

2. Check if any request needs sparse attention:
   - seq_len >= min_seq_len_for_sparse?
   - request is registered?
   - centroids exist for this layer?

3a. If no request needs sparse: use pre-planned full attention wrapper (fast path)

3b. If sparse needed:
    For each request:
        centroids, chunks = centroid_manager.get_centroids(req_pool_idx, layer_id)
        selected_ids = select_top_k_chunks(q_i, centroids, chunks, top_k)
        indices = build_sparse_kv_indices(selected_ids, chunks, ...)

    Re-plan decode wrapper with sparse indices:
        decode_wrapper.begin_forward(kv_indptr, kv_indices, ...)

    Run sparse attention:
        o = decode_wrapper.forward(q, kv_buffer, ...)

4. Incrementally update centroids:
   centroid_manager.update_centroid_incremental(req_pool_idx, layer_id, new_key=k[i])
   # Running mean: new_centroid = (old_centroid * n + new_key) / (n + 1)
```

Key point: chunk selection happens **independently per layer**. Different layers can select different chunks because each layer has its own centroids.

---

### 2.5 Model Layer Integration (`qwen3.py`)

The model layer does two extra things for the tree_sparse backend:

```python
class Qwen3Attention:
    def forward(self, positions, hidden_states, forward_batch):

        # 1. Extract W_k (K projection weight) from the fused QKV weight matrix
        #    qkv_proj.weight shape: [q_size + 2*kv_size, hidden_size]
        #    W_k = qkv_proj.weight[q_size : q_size+kv_size, :].t()
        #    W_k shape: [hidden_size, kv_size]  (kv_size = num_kv_heads * head_dim)
        W_k = self.qkv_proj.weight[k_start:k_end, :].t()

        # 2. Standard QKV projection (unchanged)
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([q_size, kv_size, kv_size])
        q, k = apply_qk_norm(q, k)
        q, k = rotary_emb(positions, q, k)

        # 3. Pass hidden_states and W_k through to backend via **kwargs
        attn_output = self.attn(q, k, v, forward_batch,
                                hidden_states=hidden_states, W_k=W_k)
```

The `**kwargs` pass through `RadixAttention.forward` -> `base_attn_backend.forward` -> `tree_sparse_backend.forward_extend`.

Note: `hidden_states` here is the **pre-projection** hidden state (before QKV). This is what enables the `mean(x) @ W_k = mean(x @ W_k)` optimization.

Note: The centroid computation ignores QK norm and RoPE -- it uses `x @ W_k` directly, not the full `RoPE(norm(x @ W_k))`. This is an approximation that works in practice because the centroid is used for relative ranking (which chunk is most similar), not for exact attention computation.

---

## 3. Data Flow Summary

### Prefill (first forward pass for a request)

```
hidden_states                              [seq_len, hidden_dim]
    │
    ├─→ qkv_proj(hidden_states) → q, k, v   [seq_len, q/kv_size]
    │       │
    │       ├─→ apply_qk_norm, rotary_emb
    │       │
    │       └─→ passed to backend as q, k, v
    │
    ├─→ W_k = qkv_proj.weight[k_start:k_end, :].t()
    │
    └─→ Both passed to tree_sparse_backend.forward_extend(q, k, v, ..., hidden_states, W_k)
            │
            ├─→ centroid_manager.update_centroids_from_hidden_states(hidden_states, W_k)
            │       For each chunk: centroid = mean(hidden_states[start:end]) @ W_k
            │
            ├─→ set_kv_buffer(k, v)  -- save all K, V to cache
            │
            ├─→ select_top_k_chunks(q, centroids)  -- pick chunks
            │
            ├─→ build_sparse_kv_indices(selected_chunks)  -- get KV pool positions
            │
            └─→ FlashInfer paged attention with sparse indices → output
```

### Decode (each subsequent token)

```
hidden_states                              [1, hidden_dim]  (single new token)
    │
    ├─→ qkv_proj → q, k, v                 [1, q/kv_size]
    │
    └─→ tree_sparse_backend.forward_decode(q, k, v, ...)
            │
            ├─→ set_kv_buffer(k, v)  -- append to cache
            │
            ├─→ For each layer independently:
            │       centroids = centroid_manager.get_centroids(req, layer)
            │       selected = select_top_k_chunks(q, centroids)
            │       indices = build_sparse_kv_indices(selected)
            │       decode_wrapper.begin_forward(sparse indices)
            │       o = decode_wrapper.forward(q, kv_buffer)
            │
            └─→ centroid_manager.update_centroid_incremental(req, layer, new_key=k)
                    new_mean = (old_mean * n + k) / (n + 1)   -- running mean
```

---

## 4. What Gets Approximated

| Component | Exact or Approximate? | Details |
|-----------|----------------------|---------|
| KV cache | Exact | All K, V are written to cache during prefill |
| Centroid computation | Approximate | Ignores QK norm and RoPE. Uses `mean(x) @ W_k` instead of `mean(RoPE(norm(x @ W_k)))` |
| Chunk selection | Approximate | Only top-k chunks selected per query. Remaining chunks are dropped |
| Attention within selected chunks | Exact | FlashInfer runs exact attention over the selected KV subset |
| Attention sinks + sliding window | Exact | First 4 tokens and last 128 tokens are always included |
| Decode centroid update | Approximate | Running mean: each new token updates the last chunk's centroid. Does not recompute from scratch |

---

## 5. Configuration

| Server Argument | Default | Description |
|----------------|---------|-------------|
| `--enable-tree-sparse` | `False` | Use tree_sparse as the attention backend |
| `--tree-sparse-full-prefill` | `False` | Disable sparse prefill, use full dense attention during prefill |
| `--tree-sparse-top-k` | `8` | Number of chunks to select. Higher = less sparse, better quality |
| `--tree-sparse-min-seq-len` | `512` | Sequences shorter than this always use full attention |
| `--tree-sparse-min-chunk-size` | `16` | Minimum tokens per chunk (small leaves get merged) |
| `--tree-sparse-max-chunk-size` | `256` | Maximum tokens per chunk (large leaves get split) |
| `--tree-sparse-recent-tokens` | `128` | Sliding window size, always included regardless of selection |

Hard-coded:
- `always_include_first = 4` -- attention sinks (BOS + first few tokens)
- Decode step logging interval: every 40 steps

---

## 6. Complexity

For a sequence of length `n` with `C` chunks, selecting `k` chunks:

| Operation | Full Attention | Tree-Sparse |
|-----------|---------------|-------------|
| Prefill attention | O(n^2) | O(n * (k * chunk_size + sink + window)) |
| Decode attention per token | O(n) | O(k * chunk_size + sink + window) |
| Centroid computation (prefill) | N/A | O(C) matrix-vector multiplies |
| Chunk selection (decode) | N/A | O(C * num_kv_heads * head_dim) dot products |
| Centroid update (decode) | N/A | O(1) running mean |

Typical numbers: n=10K, C=100, k=8, chunk_size~100, sink=4, window=128 -> attend to ~932 tokens instead of 10,000.

---

## 7. Logging and Debugging

### Server logs

```
[TreeSparse] Registered req 42: seq_len=5000, num_chunks=85, parse_mode=tree
[TreeSparse] Using optimized centroid computation: mean(x) @ W_k. Sparse prefill: ENABLED
[TreeSparse] Decode layer=0 req=42: selected 8/85 chunks, 650+128(recent)+4(sink) of 5000 tokens
[TreeSparse] Stats (last 40 steps): sparse_layers=120, avg_sparsity=84.3%
```

### JSON trace files

Saved to `$TREE_SPARSE_LOG_DIR` (default: `/vast/.../tree_sparse_traces/`):

```json
{
  "req_pool_idx": 42,
  "seq_len": 5000,
  "parse_mode": "tree",
  "tree": { "node_type": "root", "children": [...] },
  "chunks": [ {"chunk_id": 0, "label": "system", "start_idx": 0, "end_idx": 48}, ... ],
  "decode_steps": [
    {
      "decode_step": 0,
      "selected_chunk_ids": [3, 12, 45, 67, 71, 78, 82, 84],
      "sparsity": 0.843
    },
    ...
  ]
}
```

---

## 8. Limitations

1. **QKV projection is still full during sparse prefill**: All tokens get projected to Q, K, V. Only the attention computation is sparse. Future work: project K, V only for selected chunks.

2. **Centroids ignore QK norm and RoPE**: The centroid `mean(x) @ W_k` does not apply QK normalization or rotary position embeddings. This is an approximation that works for ranking but not exact attention.

3. **Fixed chunk boundaries**: Once chunks are defined during registration, they don't change. New decode tokens go into the last chunk's centroid via running mean.

4. **No CUDA graph support**: Tree-sparse uses dynamic index construction per layer, which is incompatible with CUDA graphs. Must use `--disable-cuda-graph`.

5. **Tokenizer required**: Tree parsing needs to decode token IDs to text, so a tokenizer is lazily loaded. Falls back to fixed-size chunks if unavailable.

6. **Per-layer re-planning**: During decode, FlashInfer's `begin_forward` is called per layer (to set sparse indices). This adds overhead but is necessary because different layers select different chunks.
