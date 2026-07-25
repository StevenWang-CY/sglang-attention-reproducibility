# Source Code Instrumentation Guide

Since all profiling tools are blocked, the only way to get timing data without admin permissions is to instrument the SGLang source code directly.

## Step 1: Locate the QKV Projection Code

The QKV projection happens in SGLang's attention implementation. Likely locations:

```bash
# Find SGLang installation
python -c "import sglang; print(sglang.__file__)"

# Look for attention/QKV code
cd $(dirname $(python -c "import sglang; print(sglang.__file__)"))
find . -name "*.py" | xargs grep -l "qkv_proj\|query_key_value"
```

Typical locations:
- `sglang/srt/layers/attention.py`
- `sglang/srt/models/<model_name>/modeling.py`

## Step 2: Add CUDA Event Timing

Add this timing code around the QKV projection:

```python
import torch

class YourAttentionLayer:
    def forward(self, hidden_states):
        # ADD: Create CUDA events
        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

        # EXISTING CODE: QKV projection
        qkv = self.qkv_proj(hidden_states)  # or whatever the actual call is

        # ADD: Measure timing
        if torch.cuda.is_available():
            end.record()
            torch.cuda.synchronize()
            elapsed_ms = start.elapsed_time(end)

            # Log to file for analysis
            import os
            log_file = os.environ.get('QKV_TIMING_LOG', '/tmp/qkv_timing.log')
            with open(log_file, 'a') as f:
                batch_size = hidden_states.shape[0]
                f.write(f"QKV GEMM: {elapsed_ms:.6f} ms, batch={batch_size}\n")

        # Continue with rest of forward pass
        return qkv  # or continue processing
```

## Step 3: Run Benchmarks

```bash
export QKV_TIMING_LOG=/vast/projects/liuv/pennnetworks/jiaheng/sglang_log/qkv_direct_timing.log

# Batch 16
python measure_batch_latency_offline.py ... --batch-sizes 16

# Batch 32
python measure_batch_latency_offline.py ... --batch-sizes 32

# Analyze results
cat qkv_direct_timing.log | grep "batch=16"
cat qkv_direct_timing.log | grep "batch=32"
```

## Step 4: Compare Results

This will give you:
- ✅ Actual QKV GEMM execution time
- ✅ Per-batch measurement
- ✅ Bypasses tree-sparse timing framework
- ✅ Direct CUDA event measurement (microsecond precision)
- ✅ No admin permissions needed

## Expected Results

If the hypothesis is correct:
- Batch 16: ~1.60 ms (matches tree-sparse timing)
- Batch 32: ~2.61+ ms (NOT 0.59 ms - this is the missing time!)

This would **definitively prove** that:
1. The actual GEMM work is happening
2. The tree-sparse timing is missing it at batch 32
3. The anomaly is an instrumentation blind spot, not a real performance gain

---

## Alternative: Intercept at PyTorch Level

If you can't modify SGLang, you can monkey-patch PyTorch's linear layer:

```python
import torch
import torch.nn as nn

# Save original forward
_orig_linear_forward = nn.Linear.forward

def timed_linear_forward(self, input):
    if torch.cuda.is_available() and input.shape[0] in [16, 32]:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

    result = _orig_linear_forward(self, input)

    if torch.cuda.is_available() and input.shape[0] in [16, 32]:
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end)

        # Check if this looks like QKV projection (output dim is 3x hidden)
        if result.shape[-1] == 3 * input.shape[-1]:
            print(f"QKV Linear: {elapsed:.3f} ms, batch={input.shape[0]}")

    return result

# Monkey patch
nn.Linear.forward = timed_linear_forward

# Now run your benchmark
import sglang as sgl
# ... rest of your code
```

This requires no SGLang source modification!

---

## Conclusion

Source code instrumentation is:
- ✅ Guaranteed to work (no permission issues)
- ✅ Most accurate (direct CUDA event timing)
- ✅ Can target exact operation (QKV projection)
- ✅ Bypasses all instrumentation frameworks

The only downside is you need to:
- Find the right code location
- Modify source (or monkey-patch)
- Reinstall/restart

But it will give you **definitive proof** of where the missing 2.61 ms is going.
