# Component Breakdown Reference

## Plot 1: Simple Breakdown (3 components)

**High-level view of major blocks:**

1. 🟢 **Attention Block** - All attention-related operations
2. 🔵 **MLP Block** - Feed-forward network
3. ⚫ **Others** - Embedding, logits, overhead

---

## Plot 2: Detailed Breakdown (9 components)

**All individual components from SGLang timing logs:**

### Attention Block Components (6):
1. 🟣 **input_ln** - Input layer normalization
2. 🔴 **qkv_proj** - Query, Key, Value projection
3. 🟠 **kv_cache_save** - Save to KV cache
4. 🟡 **kv_cache_load** - Load from KV cache
5. 🟢 **attention** - Attention computation
6. 🔵 **o_proj** - Output projection

### MLP Block Components (2):
7. 💙 **post_attn_ln** - Post-attention layer normalization
8. 🔵 **mlp** - MLP/FFN computation

### Overhead (1):
9. ⚫ **others** - Embedding, logits, sampling, scheduler overhead

---

## Component Mapping

From the log format:
```
Layers 1-35 (avg per layer x 35):
  [Attention block]    0.227 x 35 =    7.960 ms
    input_ln           0.009 x 35 =    0.332 ms  ← Extracted
    qkv_proj           0.096 x 35 =    3.372 ms  ← Extracted
    kv_cache_save      0.029 x 35 =    0.999 ms  ← Extracted
    kv_cache_load      0.008 x 35 =    0.274 ms  ← Extracted
    attention          0.051 x 35 =    1.782 ms  ← Extracted
    o_proj             0.034 x 35 =    1.201 ms  ← Extracted
  [MLP block]          0.116 x 35 =    4.074 ms
    post_attn_ln       0.019 x 35 =    0.670 ms  ← Extracted
    mlp                0.097 x 35 =    3.404 ms  ← Extracted
```

**"others"** is calculated as:
```
others = total_decode_latency - sum(all_8_components)
```

This captures:
- Embedding lookups
- Logits computation
- Sampling overhead
- Scheduler overhead
- Memory transfer overhead

---

## Example Output

After running the scripts, you'll get:

### Simple Plot
```
┌─────────────────────────────┐
│  Decode Latency Breakdown   │
├─────────────────────────────┤
│  ████████ MLP (40%)         │
│  ████████████ Attention (55%)│
│  ██ Others (5%)             │
└─────────────────────────────┘
```

### Detailed Plot (All 9 Components)
```
┌─────────────────────────────┐
│  Detailed Breakdown         │
├─────────────────────────────┤
│  ███ mlp (25%)              │
│  █ post_attn_ln (5%)        │
│  ██ o_proj (10%)            │
│  ████ attention (15%)       │
│  █ kv_cache_load (3%)       │
│  ██ kv_cache_save (7%)      │
│  ██████ qkv_proj (25%)      │
│  █ input_ln (5%)            │
│  ██ others (5%)             │
└─────────────────────────────┘
```

---

## Notes

- **Percentages** are automatically calculated and displayed on the plot
- **Colors** are distinct for each component for easy identification
- **Small components** (< 2%) may not show percentage labels to avoid clutter
- **Legend** shows all components in reverse order (top of bar = top of legend)
- **Batch sizes** on X-axis, **latency (ms)** on Y-axis
