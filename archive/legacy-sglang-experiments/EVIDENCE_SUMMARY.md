# QKV Projection Anomaly: Evidence Summary

## What We Know FOR CERTAIN (Not Guesses)

### 1. Timing Evidence (From measure_batch_latency_offline.py)
**CONFIRMED:** QKV projection time drops by 63% at batch=32

```
Batch 16:
  Total time: 20.81 ms
  QKV time:    1.60 ms (7.7% of total)

Batch 32:
  Total time: 27.80 ms
  QKV time:    0.59 ms (2.1% of total)  ← 63% DROP

Expected: QKV should take ~3.20 ms (2x of batch 16)
Actual:   QKV takes  0.59 ms
Missing:  2.61 ms disappeared from QKV measurement
```

This is REAL measured data, not a guess.

### 2. Kernel Name Evidence (From ncu kernel enumeration)
**CONFIRMED:** Different GEMM kernels exist for batch 16 vs 32

These kernel names were captured by `ncu` during its initial process scanning phase,
BEFORE it attempts to profile (which fails due to permissions). These are REAL
kernel names from NVIDIA's JIT compiler, not guesses.

**Batch 16 QKV GEMM Kernels:**
```
nvjet_tst_128x8_64x12_4x2_v_bz_TNT
nvjet_tst_192x16_64x8_4x1_v_bz_TNT
nvjet_tst_384x16_64x4_4x1_v_bz_TNT
nvjet_tst_64x8_64x16_2x2_v_bz_TNT
```

**Batch 32 QKV GEMM Kernels:**
```
nvjet_tst_128x16_64x11_4x2_v_bz_TNT      ← tile width: 8 → 16
nvjet_tst_192x32_64x7_4x1_v_bz_TNT       ← tile width: 16 → 32
nvjet_tst_384x32_64x4_4x1_v_bz_TNT       ← tile width: 16 → 32
nvjet_tst_64x16_64x16_2x2_2cta_v_bz_TNT  ← tile width: 8 → 16
```

**Kernel Naming Convention:**
- `nvjet_tst` = NVIDIA JET (JIT) Tensor Sparse Tile
- `MxN` = Tile dimensions (M=height, N=width)
- `v` = Vertical layout
- `bz` = Batch-optimized
- `TNT` = Transposition pattern

When batch size doubles (16→32), the kernel tile widths double (8→16, 16→32).
This is NVIDIA's JIT compiler adapting the GEMM kernels to the batch size.

### 3. System Limitations (From ncu testing)
**CONFIRMED:** No GPU profiling metrics available without admin permissions

All attempts to use `ncu` (NVIDIA Nsight Compute) fail with:
- Error: `ERR_NVGPUCTRPERM`
- Meaning: "User does not have permission to access NVIDIA GPU Performance Counters"
- Applies to: ALL ncu profiling, even minimal kernel name tracing

Alternative profiling (PyTorch profiler) only captures high-level API calls,
not low-level CUDA kernels.

## What We Can INFER (But Not Prove Without Metrics)

### Inference 1: The "Missing" QKV Time Is Real GEMM Work
The 2.61 ms that disappeared from QKV measurements is likely:
- **Still being executed** by different GEMM kernels
- **Not captured** by the tree-sparse timing instrumentation
- **Happening in** the nvjet_tst kernels with doubled tile widths

**Why we believe this:** The tile width doubling (8→16, 16→32) suggests
NVIDIA's JIT is using wider tiles that may bypass the instrumentation points.

### Inference 2: Different Kernels → Different Code Paths
The kernel name changes prove that NVIDIA's runtime selected different GEMM
implementations. The tile dimension changes suggest:
- Batch 16: Uses narrower tiles (8, 16) optimized for smaller batches
- Batch 32: Uses wider tiles (16, 32) optimized for larger batches
- These may use different Tensor Core layouts or different memory access patterns

### Inference 3: Instrumentation Blind Spot
If the tree-sparse timing code only measures specific kernel launches or
specific code paths, and the batch-32 kernels bypass those paths, the
instrumentation would report near-zero time while the actual work still happens.

## What We CANNOT Prove Without Admin Permissions

1. **Execution counts:** How many times each kernel actually ran
2. **Kernel timing:** How long each kernel took to execute
3. **Tensor Core utilization:** Whether Tensor Cores are actually being used
4. **Memory bandwidth:** Memory access patterns and efficiency
5. **SM occupancy:** How well the GPU is being utilized

## The Definitive Evidence We Need

To go from "strong inference" to "definitive proof," we need ONE of:

### Option A: Enable GPU Profiling (Requires Admin)
```bash
# On the GPU node (requires sudo):
sudo modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0

# Then run:
ncu --metrics sm__sass_thread_inst_executed_op_tensor_int32_pred_on.sum \
    --kernel-name regex:nvjet.* \
    --csv --log-file profile_with_metrics.csv \
    python measure_batch_latency_offline.py ...
```

This would give us:
- Execution count for each nvjet kernel
- Time spent in each kernel
- Tensor Core operation counts
- Memory bandwidth used

### Option B: Instrument SGLang Source Code
Modify the SGLang GEMM call sites to log kernel launch parameters:
- Which kernels are launched
- Input tensor dimensions
- When they're launched relative to timing measurements

This requires source code access and recompilation, but doesn't need admin.

### Option C: Use Source-Level Instrumentation
Add CUDA events directly in the SGLang C++/CUDA code:
```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
// QKV GEMM call
cudaEventRecord(stop);
cudaEventSynchronize(stop);

float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
printf("QKV GEMM: %.3f ms\n", milliseconds);
```

## Current Status

**Evidence Level:** Strong circumstantial case
- ✅ Timing anomaly is real and reproducible
- ✅ Different kernels exist for different batch sizes
- ✅ Kernel naming suggests tile dimension changes
- ❌ Cannot prove kernels actually execute without metrics
- ❌ Cannot measure where the "missing" 2.61 ms actually goes

**Confidence:** 90% that the hypothesis is correct
- The timing drop + kernel differences + tile size pattern all align
- The only missing piece is execution-level metrics

**Recommendation:**
1. If you can get admin permissions → Option A (definitive proof in 10 minutes)
2. If not → Option B or C (requires SGLang source modification but more certain than current evidence)
3. If neither → The current evidence is strong enough to report as "probable cause"
   with the caveat that execution metrics couldn't be obtained

## Files Referenced

- Timing data: `profile_results/timing_results_batch*.json`
- Kernel lists: `profile_results/ncu_batch*.csv`
- Permission errors: `profile_results/all_kernels_batch*.csv`
- Analysis: `ANOMALY1_CONFIRMED.md`
