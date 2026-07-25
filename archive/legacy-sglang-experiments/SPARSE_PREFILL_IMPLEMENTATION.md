# Tree-Sparse Attention: Full Implementation Details

## 1. Architecture Overview

Tree-sparse attention is a custom attention backend for SGLang that replaces full O(n²) attention with sparse O(n*k) attention by exploiting the structural hierarchy of the input (ChatML messages, HTML tags, browser DOM elements).

```
┌─────────────────────────────────────────────────────────────────┐
│                        Request Lifecycle                        │
│                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │ Prefill  │───>│  Centroids   │───>│  Decode (per token)  │  │
│  │          │    │  Computed     │    │  Sparse Attention    │  │
│  └──────────┘    └──────────────┘    └──────────────────────┘  │
│       │                                        │               │
│  Parse tree          mean(x)@W_k          top-k chunks         │
│  Extract chunks      per chunk            per query per layer  │
│  Register request                         incremental update   │
└─────────────────────────────────────────────────────────────────┘
```

### Components

| Component | File | Role |
|-----------|------|------|
| **Backend** | `tree_sparse_backend.py` | Orchestrator: registration, prefill dispatch, decode with per-layer sparse selection |
| **Tree Parser** | `tree_sparse/tree_parser.py` | Parses token sequences into a ChatML/HTML tree, extracts leaf chunks |
| **Centroid Manager** | `tree_sparse/centroid_manager.py` | Stores per-request per-layer centroids (mean key vectors), supports incremental updates |
| **Sparse Selector** | `tree_sparse/sparse_selector.py` | Top-k chunk selection by query-centroid dot product, KV index construction |
| **Model Layer** | `models/qwen3.py` | Extracts W_k weight matrix, passes hidden_states to backend |
| **RadixAttention** | `layers/radix_attention.py` | Forwards `**kwargs` (hidden_states, W_k) from model layer to backend |
| **Server Args** | `server_args.py` | Configuration flags |

---

## 2. Tree Parser (`tree_parser.py`)

### Purpose

Converts a flat token sequence into a hierarchical tree, then extracts leaf nodes as "chunks" — contiguous token ranges that form semantic units.

### Parsing

`parse_chatml_tree(token_texts) -> TreeNode`

The parser recognizes three levels of structure:

1. **ChatML markers**: `<|im_start|>role ... <|im_end|>` — top-level message boundaries (system, user, assistant)
2. **HTML tags**: `<tag>...</tag>` — nested structure within messages (e.g., `<agent_history>`, `<step>`, `<browser_state>`)
3. **Browser DOM elements**: Tab-indented `[1501]<div />` patterns — browser accessibility tree nodes, nested by indentation depth

Each recognized pattern becomes a `TreeNode`:

```python
@dataclass
class TreeNode:
    node_type: str   # 'root', 'chatml', 'html_tag', 'browser_element', 'text', 'vision'
    label: str       # e.g. "system", "<agent_history>", "[1501]<div />"
    start_idx: int   # inclusive token index
    end_idx: int     # inclusive token index
    children: List[TreeNode]
```

Example tree for a typical agentic prompt:

```
[root] all [0-8191]
  [chatml] system [0-42]
    [text] "You are a helpful..." [3-41]
  [chatml] user [43-8100]
    [html_tag] <agent_history> [50-8000]
      [html_tag] <step> [55-2000]
        [html_tag] <browser_state> [60-1900]
          [browser_element] [1]<div /> "Search" [70-85]
          [browser_element] [2]<a /> "Home" [86-95]
          ...
      [html_tag] <step> [2001-4000]
        ...
    [text] "Please click..." [8001-8100]
  [chatml] assistant [8101-8191]
```

### Chunk Extraction

`extract_leaf_chunks(root, min_chunk_size=16, max_chunk_size=256) -> List[FlatChunk]`

Walks the tree and collects leaf nodes. Then:

1. **Gap filling**: Any token positions not covered by a leaf get a synthetic "gap" node, so every token belongs to exactly one chunk
2. **Merging small leaves**: Leaves smaller than `min_chunk_size` (default 16) are merged with the next sibling
3. **Splitting large leaves**: Leaves larger than `max_chunk_size` (default 256) are split into sub-chunks

Result is a list of non-overlapping `FlatChunk` objects covering the full sequence:

```python
@dataclass
class FlatChunk:
    start_idx: int   # inclusive
    end_idx: int     # inclusive
    chunk_id: int    # sequential ID
    label: str       # inherited from tree node
```

### Fallback

If tree parsing fails or produces no structure, `make_fixed_chunks(seq_len, chunk_size=64)` creates uniform fixed-size chunks.

---

## 3. Centroid Manager (`centroid_manager.py`)

### Purpose

Stores and maintains per-chunk mean key vectors ("centroids") for every request, at every layer. These centroids are used during decode to quickly determine which chunks are most relevant to the current query.

### Data Structure

```python
class CentroidManager:
    _entries: Dict[int, CentroidEntry]   # req_pool_idx -> entry

@dataclass
class CentroidEntry:
    chunks: List[FlatChunk]
    centroids: Dict[int, torch.Tensor]   # layer_id -> [num_chunks, num_kv_heads, head_dim]
    counts: Dict[int, torch.Tensor]      # layer_id -> [num_chunks] (token count per chunk)
```

### Lifecycle

1. **`register_request(req_pool_idx, chunks)`** — Called once during prefill. Stores the chunk structure.

2. **`update_centroids_from_hidden_states(req_pool_idx, layer_id, hidden_states, W_k, seq_len)`** — Called per layer during prefill. Computes centroids using the optimized method:

   ```
   For each chunk with tokens {x_1, ..., x_m}:
       centroid = mean({x_i}) @ W_k        # 1 projection instead of m
   ```

   This exploits the linearity of matrix multiplication:
   ```
   mean(x_i @ W_k) = (1/m) Σ(x_i @ W_k)
                    = ((1/m) Σ x_i) @ W_k
                    = mean(x_i) @ W_k
   ```

   Cost: O(num_chunks) projections instead of O(seq_len).

3. **`get_centroids(req_pool_idx, layer_id)`** — Called per layer during decode. Returns `(centroids, chunks)` for top-k selection.

4. **`update_centroid_incremental(req_pool_idx, layer_id, new_key, chunk_id=-1)`** — Called per layer during decode. Updates the last chunk's centroid with the running mean formula:
   ```
   new_mean = (old_mean * n + new_key) / (n + 1)
   ```

5. **`remove_request(req_pool_idx)`** / **`cleanup_stale(active_set)`** — Called when requests finish.

### Precision

Centroids are computed in float32 for numerical stability, then stored in model dtype (float16/bfloat16) for memory efficiency.

---

## 4. Sparse Selector (`sparse_selector.py`)

### Top-K Chunk Selection

`select_top_k_chunks(query, centroids, chunks, top_k, ...) -> (selected_ids, scores)`

```
query:     [num_q_tokens, num_qo_heads, head_dim]
centroids: [num_chunks, num_kv_heads, head_dim]
```

Steps:

1. **GQA handling**: Average query heads within each KV head group:
   ```python
   # num_qo_heads = 32, num_kv_heads = 8 => group_size = 4
   q_grouped = query.view(num_q, num_kv_heads, group_size, head_dim).mean(dim=2)
   # Result: [num_q, num_kv_heads, head_dim]
   ```

2. **Dot-product similarity**:
   ```python
   scores = einsum("qkd,ckd->qc", q_grouped, centroids) * scaling
   # Result: [num_q, num_chunks]
   ```

3. **Selection**:
   - **Decode** (num_q=1): Take top-k of the single row
   - **Prefill** (num_q>1): Take max score across all query tokens per chunk, then top-k of that. This selects chunks that are most relevant to *any* query token in the sequence.

### Sparse KV Index Construction

`build_sparse_kv_indices(selected_chunk_ids, chunks, req_to_token, ...) -> Tensor`

Builds the set of physical KV pool indices that FlashInfer should attend to:

```
final_positions = attention_sinks ∪ sliding_window ∪ selected_chunk_tokens
```

Specifically:
- **Attention sinks**: First 4 tokens (always included, critical for attention stability)
- **Sliding window**: Last 128 tokens (always included, local context)
- **Selected chunks**: All tokens from the top-k chunks

These logical positions are then mapped to physical KV pool locations via `req_to_token[req_pool_idx, positions]`.

### Batch Metadata

`build_batch_sparse_metadata(batch_kv_indices, bs, device) -> (kv_indptr, kv_indices)`

Packs per-request sparse indices into the CSR-like format FlashInfer expects:
- `kv_indptr[i]` = start offset for request i in `kv_indices`
- `kv_indices` = concatenated physical KV pool locations for all requests

---

## 5. Backend (`tree_sparse_backend.py`)

### Initialization

`TreeSparseAttnBackend.__init__(model_runner)` sets up:

- Model dimensions: `num_qo_heads`, `num_kv_heads`, `head_dim`, `num_layers`
- Tree-sparse config from server args: `top_k_chunks`, `min_seq_len_for_sparse`, `min/max_chunk_size`, `always_include_recent/first`
- `enable_sparse_prefill`: True by default, disabled with `--tree-sparse-full-prefill`
- `CentroidManager` instance
- Three FlashInfer wrappers:
  - `prefill_wrapper_ragged` (`BatchPrefillWithRaggedKVCacheWrapper`) — new-to-new attention during prefill
  - `prefill_wrapper_paged` (`BatchPrefillWithPagedKVCacheWrapper`) — new-to-prefix attention during prefill, and sparse prefill
  - `decode_wrapper` (`BatchDecodeWithPagedKVCacheWrapper`) — decode attention (replanned per-layer with sparse indices)

### Request Registration (`_register_request_tree`)

Called once per new request during `_init_extend_metadata`. Flow:

1. Get full token IDs from `forward_batch.tree_sparse_token_ids`
2. Decode each token ID to text via tokenizer (lazy-loaded)
3. `parse_chatml_tree(token_texts)` → tree
4. `extract_leaf_chunks(tree)` → chunks
5. `centroid_manager.register_request(req_pool_idx, chunks)`
6. Save tree + chunks to JSON log for debugging

### Prefill (`forward_extend`)

The dispatch logic:

```python
use_sparse = (
    self.enable_sparse_prefill       # flag is on (default)
    and hidden_states is not None    # model layer passed hidden states
    and W_k is not None              # model layer passed K weight
    and save_kv_cache                # not a speculative pass
)

if use_sparse:
    o = self._forward_extend_sparse(q, k, v, ...)
else:
    # standard full FlashInfer attention (3 cases: no-prefix, ragged+paged, paged-only)
```

#### Full Prefill Path (with `--tree-sparse-full-prefill`)

Same as the standard FlashInfer backend. Three cases based on prefix:

1. **No prefix** (`extend_no_prefix`): Pure ragged causal attention — `prefill_wrapper_ragged.forward(q, k, v, causal=True)`
2. **Has prefix, use ragged** (`use_ragged`): Two-part attention merged:
   - Ragged: new tokens attend to new tokens (causal)
   - Paged: new tokens attend to prefix KV cache (non-causal)
   - `merge_state(o1, s1, o2, s2)` combines via log-sum-exp
3. **Paged only**: `prefill_wrapper_paged.forward(q, kv_buffer, causal=True)`

After attention, centroids are computed from hidden_states for future decode.

#### Sparse Prefill Path (default: `_forward_extend_sparse`)

```
Step 1: Compute centroids from hidden_states FIRST
        centroid_manager.update_centroids_from_hidden_states(...)
        → For each chunk: mean(x_i) @ W_k

Step 2: Save ALL k, v to KV cache
        (decode will need full cache)

Step 3: For each request in batch:
        if seq_len >= min_seq_len and has centroids:
            select_top_k_chunks(q_i, centroids, chunks, top_k)
            build_sparse_kv_indices(selected + sinks + window)
        else:
            use full indices (short sequence fallback)

Step 4: Pack into batched CSR format
        build_batch_sparse_metadata(batch_kv_indices)

Step 5: Re-plan FlashInfer paged wrapper with sparse indices
        prefill_wrapper_paged.begin_forward(qo_indptr, kv_indptr, kv_indices, ...)

Step 6: Run sparse attention
        prefill_wrapper_paged.forward(q, kv_buffer, causal=True)
```

**Key difference**: In full prefill, every query token attends to all previous KV tokens. In sparse prefill, each query token only attends to KV tokens from: selected top-k chunks + first 4 tokens + last 128 tokens.

### Decode Metadata Init (`_init_decode_metadata`)

Called once per decode step (before any layer runs). Sets up the starting state:

1. **Cleanup stale requests**: Removes centroid entries and registration for requests no longer in the batch
2. **Plan decode wrapper with full indices**: Builds full KV indices for all requests using the Triton kernel `create_flashinfer_kv_indices_triton`, and calls `decode_wrapper.begin_forward(...)`. This gives a working default — if no layer needs sparse attention, this pre-planned wrapper is used as-is.

```python
# CSR-format indices for FlashInfer
kv_indptr[i] = cumulative sum of seq_lens       # where each request's KV starts
kv_indices   = physical KV pool locations        # built by Triton kernel from req_to_token

decode_wrapper.begin_forward(kv_indptr, kv_indices, kv_last_page_len,
                              num_qo_heads, num_kv_heads, head_dim, page_size=1)
```

### Decode (`forward_decode`)

Called once **per layer** per decode step. Each layer independently selects its own top-k chunks based on that layer's centroids and the current query token. This is the core sparse attention loop.

```
Step 1: Save new k, v to KV cache
        token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)

Step 2: Check if ANY request in the batch needs sparse attention
        needs_sparse = any(
            seq_len >= min_seq_len
            and req in registered_reqs
            and centroids exist for this layer
        )

Step 3a: If no request needs sparse:
         Use the pre-planned full wrapper directly (no re-planning overhead):
           decode_wrapper.forward(q, kv_buffer)

Step 3b: If any request needs sparse:
         For EACH request in the batch:
           if sparse eligible:
             q_i = q[i:i+1]  (single query token, shape [1, num_qo_heads, head_dim])

             # Select top-k chunks for THIS layer
             selected_ids, scores = select_top_k_chunks(
                 q_i, centroids[layer_id], chunks, top_k)

             # Build sparse KV indices (selected + sinks + window)
             indices = build_sparse_kv_indices(
                 selected_ids, chunks, req_to_token, req_pool_idx, seq_len,
                 always_include_recent=128, always_include_first=4)

           else:
             # Full indices for short/unregistered requests
             indices = req_to_token[req_pool_idx, :seq_len]

         # Pack per-request indices into batched CSR format
         kv_indptr, kv_indices = build_batch_sparse_metadata(sparse_list, bs)

         # Re-plan the decode wrapper with sparse indices
         decode_wrapper.begin_forward(kv_indptr, kv_indices, ...)

         # Run attention with sparse KV set
         o = decode_wrapper.forward(q, kv_buffer)

Step 4: Incrementally update centroids with the new key token
        For each request:
          centroid_manager.update_centroid_incremental(
              req_pool_idx, layer_id, new_key=k[i:i+1], chunk_id=-1)

          # Running mean formula:
          #   new_centroid = (old_centroid * n + new_key) / (n + 1)
          # Always updates the LAST chunk (decode tokens are appended)

Step 5: Periodic stats + JSON logging (every 40 steps, on last layer only)
        - Logs: sparse_layers, full_layers, avg_sparsity, avg_tokens_attended
        - Flushes per-request JSON trace files with decode step selections
```

**Key design decisions**:

- **Per-layer selection**: The decode wrapper is re-planned per-layer because different layers attend to different chunks. Layer 0 might focus on system prompt chunks while layer 30 focuses on recent user input. Each layer's centroids capture that layer's own attention pattern.

- **Lazy sparse check**: If no request in the batch needs sparse attention (all short sequences, or first decode step before centroids exist), the pre-planned full wrapper is reused with zero overhead — no re-planning, no chunk selection.

- **Mixed batch**: Within a single batch, some requests can use sparse attention while others use full attention. The CSR format handles variable-length KV sets per request naturally.

- **JSON trace logging**: At layer 0, every decode step records which chunks were selected, how many tokens were attended, and the sparsity ratio. This is written to `$TREE_SPARSE_LOG_DIR/req_{idx}_{timestamp}.json` for offline analysis.

- **Incremental centroid update**: New decode tokens are always assigned to the last chunk. The centroid is updated with the running mean formula so future decode steps reflect the growing context.

---

## 6. Model Layer Integration (`qwen3.py`)

The model's attention layer (`Qwen3Attention.forward`) was modified to:

1. **Extract W_k** — the K portion of the fused QKV weight matrix:
   ```python
   # qkv_proj.weight shape: [(q_size + 2*kv_size), hidden_size]
   # K portion: rows [q_size : q_size + kv_size]
   qkv_weight = self.qkv_proj.weight
   W_k = qkv_weight[q_size : q_size + kv_size, :].t()
   # W_k shape: [hidden_size, kv_size]
   ```

2. **Pass hidden_states and W_k** through RadixAttention to the backend:
   ```python
   attn_output = self.attn(q, k, v, forward_batch,
                           hidden_states=hidden_states, W_k=W_k)
   ```

3. **RadixAttention.forward** accepts `**kwargs` and forwards them to the backend's `forward_extend` or `forward_decode`:
   ```python
   # In radix_attention.py
   def forward(self, q, k, v, forward_batch, save_kv_cache=True, **kwargs):
       return forward_batch.attn_backend.forward(q, k, v, self, forward_batch,
                                                  save_kv_cache, **kwargs)
   ```

**Note**: W_k extraction only happens when the backend has a `centroid_manager` attribute (i.e., tree_sparse backend). For other backends, W_k stays None and the kwargs are harmless.

---

## 7. What Is Sparse vs What Is Not

| Operation | Status | Notes |
|-----------|--------|-------|
| Tree parsing + chunking | One-time per request | O(n) string matching |
| QKV projection | **Full** (all n tokens) | Done in model layer before backend |
| Centroid computation | **Optimized**: O(C) projections | mean(x) @ W_k instead of mean(x @ W_k) |
| KV cache write | **Full** (all n tokens) | Decode needs full cache |
| **Prefill attention** | **Sparse**: O(n*k) | Only selected chunks + sinks + window |
| **Decode attention** | **Sparse**: O(k) per token | Per-layer top-k selection |
| Centroid update (decode) | O(1) per token per layer | Running mean of last chunk |

---

## 8. Configuration

Sparse prefill is **ON by default** when using `--attention-backend tree_sparse`.

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-7B-Instruct \
    --attention-backend tree_sparse \
    --disable-cuda-graph \
    --tree-sparse-top-k 8 \
    --tree-sparse-min-seq-len 512 \
    --tree-sparse-min-chunk-size 16 \
    --tree-sparse-max-chunk-size 256 \
    --tree-sparse-recent-tokens 128
```

| Flag | Default | Effect |
|------|---------|--------|
| `--attention-backend tree_sparse` | — | Enables the tree-sparse backend |
| `--tree-sparse-full-prefill` | off | Falls back to full dense prefill (sparse is default) |
| `--tree-sparse-top-k` | 8 | Number of chunks selected per query per layer |
| `--tree-sparse-min-seq-len` | 512 | Sequences shorter than this use full attention |
| `--tree-sparse-min-chunk-size` | 16 | Tree leaves smaller than this are merged |
| `--tree-sparse-max-chunk-size` | 256 | Tree leaves larger than this are split |
| `--tree-sparse-recent-tokens` | 128 | Sliding window size (always attended to) |

Hardcoded: `always_include_first = 4` (attention sinks).

---

## 9. Data Flow Diagram

```
                          qwen3.py Qwen3Attention.forward()
                         ┌─────────────────────────────────┐
                         │                                 │
  hidden_states ────────►│  W_k = qkv_weight[k_start:     │
  [n, hidden_dim]        │         k_end, :].t()           │
                         │                                 │
                         │  q, k, v = qkv_proj(hidden)     │
                         │  q, k = qk_norm(q, k)           │
                         │  q, k = rotary_emb(pos, q, k)   │
                         │                                 │
                         │  attn(q, k, v, forward_batch,   │
                         │       hidden_states=hidden,     │
                         │       W_k=W_k)                  │
                         └──────────┬──────────────────────┘
                                    │
                                    ▼
                     radix_attention.py RadixAttention.forward()
                         ┌─────────────────────────────────┐
                         │  forward_batch.attn_backend      │
                         │    .forward(q,k,v, **kwargs)     │
                         └──────────┬──────────────────────┘
                                    │
                         ┌──────────▼──────────────────────┐
                         │  base_attn_backend.forward()     │
                         │    is_decode? → forward_decode() │
                         │    else      → forward_extend()  │
                         └──────────┬──────────────────────┘
                                    │
              ┌─────────────────────┴─────────────────────┐
              │                                           │
    ┌─────────▼──────────┐                    ┌───────────▼─────────┐
    │  forward_extend()  │                    │  forward_decode()   │
    │                    │                    │                     │
    │  sparse prefill?   │                    │  1. Save k,v cache  │
    │  ┌──yes──┐  ┌─no─┐ │                    │  2. Per-layer:      │
    │  │sparse │  │full│ │                    │     select_top_k    │
    │  │path   │  │path│ │                    │     build_sparse    │
    │  └───────┘  └────┘ │                    │     re-plan wrapper │
    │                    │                    │     run attention   │
    │  compute centroids │                    │  3. Update centroid │
    │  save k,v cache    │                    │     incrementally   │
    └────────────────────┘                    └─────────────────────┘
```

---

## 10. JSON Logging

Every request gets a JSON trace file at `$TREE_SPARSE_LOG_DIR/req_{idx}_{timestamp}.json`:

```json
{
  "req_pool_idx": 42,
  "seq_len": 8192,
  "parse_mode": "tree",
  "tree": { ... },               // full TreeNode hierarchy
  "chunks": [                    // flat chunk list
    {"chunk_id": 0, "label": "system", "start_idx": 0, "end_idx": 41, "token_count": 42},
    ...
  ],
  "config": { "top_k": 8, ... },
  "decode_steps": [              // one entry per decode step (layer 0 only)
    {
      "decode_step": 0,
      "selected_chunk_ids": [3, 7, 12, 15, 20, 25, 30, 35],
      "selected_chunks": [ ... ],
      "total_tokens_attended": 1432,
      "sparsity": 0.825
    },
    ...
  ]
}
```

---

## 11. Limitations

1. **QKV projection is still full in prefill**: All n tokens get projected to Q, K, V. Only the attention computation is sparse. A future optimization could project K, V only for selected tokens.

2. **No CUDA graph support**: The backend requires `--disable-cuda-graph` because the decode wrapper is re-planned per-layer with different sparse indices.

3. **Centroid quality depends on tree parsing**: If chunks don't align with semantic boundaries, centroids may misrepresent chunk content.

4. **Single chunk set per prefill**: During sparse prefill, one set of top-k chunks is selected for all query tokens (using max score across queries). Per-token selection would be more precise but complicates the FlashInfer index structure.

5. **Incremental centroid update is approximate**: During decode, new tokens are always added to the last chunk. If the model generates content that semantically belongs to an earlier chunk, the centroid update may be inaccurate. In practice this is fine because decode tokens are always appended.
