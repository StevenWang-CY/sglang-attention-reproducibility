# QKV Instrumentation Findings

## Summary

We instrumented SGLang to add **direct wall-clock timing** AND compare it with **CUDA event timing**.

## Key Data: Batch 16 vs Batch 32

### Batch 16 - Layer 0 (First Decode Step)
```
Direct wall-clock: 25.607 ms
CUDA timer:        30.990 ms
```

### Batch 32 - Layer 0 (First Decode Step)
```
Direct wall-clock:  0.960 ms
CUDA timer:         9.732 ms
```

## Critical Discovery

**The wall-clock timing is MUCH shorter than CUDA timing because:**

1. **CUDA kernels launch asynchronously**
   - `qkv_proj(hidden_states)` returns immediately after launching kernel
   - Wall-clock timing captures just the Python/launch overhead
   - Actual GPU execution continues asynchronously

2. **CUDA Event timing captures actual GPU work**
   - `start_ev.elapsed_time(end_ev)` waits for GPU to finish
   - Shows real kernel execution time on GPU

3. **Batch 32 launches faster (0.960 ms) vs Batch 16 (25.607 ms)**
   - Batch 16 first iteration includes initialization overhead
   - Batch 32 kernel launch is optimized/cached

## The Real Question

The CUDA timer shows:
- **Batch 16: 30.990 ms**
- **Batch 32: 9.732 ms**

But the original tree-sparse timing report showed:
- **Batch 16: 1.60 ms**
- **Batch 32: 0.59 ms**

**Why the huge difference?**

### Possibility 1: First iteration vs steady-state

The instrumented data shows the FIRST decode step. The original timing was likely:
- Averaged over multiple decode steps
- Excluded warmup/first iteration
- Measured only steady-state performance

### Possibility 2: Different modes

The benchmark might be running in:
- **Prefill mode** (first iteration): Processes all tokens at once (large batch)
- **Decode mode** (subsequent): Generates one token at a time (small effective batch)

The "batch" in the original report might mean something different.

### Possibility 3: Multi-token generation

The original benchmark generates 10 tokens per request. The timing might be:
- **Per-token** average: Total time / 10 tokens
- **Per-layer-per-token**: Even more averaged

## Next Steps

1. **Check if there are multiple decode iterations in the log**
   - Look beyond the first iteration
   - See if timing stabilizes

2. **Understand what the original "batch" means**
   - Is it the number of requests?
   - Or the number of tokens processed simultaneously?

3. **Compare total QKV time across all layers**
   - Sum up all layers for both batches
   - See if it matches the original 1.60 ms / 0.59 ms pattern

## Current Status

The instrumentation is working correctly! We're capturing:
- ✅ Wall-clock timing (Python overhead)
- ✅ CUDA event timing (GPU execution)
- ✅ Per-layer breakdowns
- ✅ Batch size identification

But we need to understand:
- ❓ Why instrumented timing doesn't match original report
- ❓ What "batch" means in the original context
- ❓ Whether we're measuring prefill or decode mode
