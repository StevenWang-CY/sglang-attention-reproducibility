# Batch Size Tuning Guide for SGLang

## Understanding the Parameters

### `--max-prefill-tokens`
- **What**: Maximum total tokens that can be in prefill phase at once
- **Formula**: `batch_size × avg_tokens_per_request × safety_margin`
- **Impact**: Too small → requests queued; Too large → OOM

### `--chunked-prefill-size`
- **What**: Size of prefill chunks for interleaving with decode
- **Formula**: Typically `8K - 16K` for good balance
- **Impact**:
  - Smaller chunks → Better decode latency, more overhead
  - Larger chunks → Higher throughput, blocked decode

## Your Workload Analysis

From your logs (`runtime_qwen3vl_20260329_145205.log`):
- **Tokens per request**: ~9,660 tokens
- **Batch size tested**: 10
- **Total tokens (batch 10)**: 96,600 tokens ✅ fits in current 100K limit

For **batch size 64**:
- **Total tokens**: 64 × 9,660 = **618,240 tokens**
- **Current limit**: 100,000 tokens ❌ **TOO SMALL**

## Recommended Configurations

### For Batch Size 1-16 (Small Batches)
```bash
--max-prefill-tokens 200000      # 200K
--chunked-prefill-size 8192      # 8K chunks
```
- ✅ Low latency
- ✅ Quick request admission
- ⚠️  Lower throughput

### For Batch Size 32 (Medium Batches)
```bash
--max-prefill-tokens 400000      # 400K
--chunked-prefill-size 12288     # 12K chunks
```
- ✅ Balanced latency/throughput
- ✅ Good for mixed workloads

### For Batch Size 64 (Large Batches) ⭐ RECOMMENDED
```bash
--max-prefill-tokens 800000      # 800K
--chunked-prefill-size 12288     # 12K chunks
```
- ✅ Handles 64 requests (618K tokens + 30% headroom)
- ✅ 12K chunks = good latency/throughput balance
- ✅ Memory efficient

### For Batch Size 128+ (Very Large Batches)
```bash
--max-prefill-tokens 1500000     # 1.5M
--chunked-prefill-size 16384     # 16K chunks
```
- ✅ Maximum throughput
- ⚠️  Higher memory usage
- ⚠️  Higher per-request latency

## Chunked Prefill Size Guidelines

| Chunk Size | Latency | Throughput | Use Case |
|------------|---------|------------|----------|
| 4,096 | ⭐⭐⭐ Best | ⚠️ Lower | Interactive, real-time |
| 8,192 | ⭐⭐ Good | ✅ Good | Balanced, recommended |
| 12,288 | ⭐ OK | ✅ Better | Batch processing |
| 16,384 | ⚠️ Higher | ⭐⭐⭐ Best | Throughput-focused |
| 32,768+ | ❌ Poor | ⭐⭐⭐ Best | Offline batch jobs |

## Memory Calculations

### Your Model: Qwen3-VL-8B-Instruct
- **Model size**: ~16 GB (FP16)
- **KV cache per token**: ~0.5 MB (depends on num_layers, hidden_size)
- **Context length**: 40,960 tokens

### Memory for KV Cache (Batch 64)
```
KV cache memory = batch_size × context_length × kv_cache_per_token
                ≈ 64 × 40,960 × 0.5 MB
                ≈ 1.3 GB per GPU (with tensor parallelism)
```

### Total GPU Memory (Estimated)
- Model weights: ~16 GB
- KV cache (batch 64): ~1.3 GB per GPU
- Activations: ~2-4 GB
- **Total**: ~20-22 GB per GPU

✅ Should fit on A100 (40GB/80GB) or H100 with multi-GPU TP

## How to Choose

### Priority: Latency
```bash
--max-prefill-tokens 640000
--chunked-prefill-size 8192    # Smaller chunks → faster decode
```

### Priority: Throughput
```bash
--max-prefill-tokens 1000000
--chunked-prefill-size 16384   # Larger chunks → less overhead
```

### Priority: Balanced (Recommended)
```bash
--max-prefill-tokens 800000
--chunked-prefill-size 12288   # Sweet spot
```

## Testing Your Configuration

### Step 1: Start server with new settings
```bash
./start_sglang_qwen3vl_batch64.sh
```

### Step 2: Run batch experiments
```bash
./run_batch_experiments.sh
```

### Step 3: Monitor for issues

**Watch for OOM (Out of Memory):**
```bash
grep -i "out of memory\|OOM" qwen3vl-log/*.log
```

**Watch for queue buildup:**
```bash
grep "#queue-req:" qwen3vl-log/*.log | grep -v "#queue-req: 0"
```

**Check token usage:**
```bash
grep "token usage:" qwen3vl-log/*.log | tail -20
```

### Step 4: Analyze results
```bash
python parse_timing_logs.py batch_experiment_logs/
python plot_latency_breakdown.py batch_experiment_logs/
```

## Troubleshooting

### Problem: OOM Errors
**Solution**: Reduce `max-prefill-tokens` or batch size

```bash
--max-prefill-tokens 640000  # Reduce from 800K
```

### Problem: High Queue Length (#queue-req > 0)
**Solution**: Increase `max-prefill-tokens`

```bash
--max-prefill-tokens 1000000  # Increase from 800K
```

### Problem: High Decode Latency
**Solution**: Reduce `chunked-prefill-size`

```bash
--chunked-prefill-size 8192  # Reduce from 12288
```

### Problem: Low Throughput
**Solution**: Increase `chunked-prefill-size`

```bash
--chunked-prefill-size 16384  # Increase from 12288
```

## Quick Reference Table

| Batch Size | max-prefill-tokens | chunked-prefill-size | Memory Est. |
|------------|-------------------|---------------------|-------------|
| 1-8        | 200,000           | 8,192               | ~18 GB      |
| 16         | 200,000           | 8,192               | ~19 GB      |
| 32         | 400,000           | 12,288              | ~20 GB      |
| 64         | **800,000** ⭐     | **12,288** ⭐        | ~22 GB      |
| 128        | 1,500,000         | 16,384              | ~26 GB      |

## Final Recommendation for Batch 64

Use the new script I created:
```bash
./start_sglang_qwen3vl_batch64.sh
```

Settings:
- ✅ `--max-prefill-tokens 800000`
- ✅ `--chunked-prefill-size 12288`

This provides the best balance for your workload!
