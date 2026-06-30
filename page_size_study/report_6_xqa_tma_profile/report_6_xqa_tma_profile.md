# Report 6 — Profiling TMA on consumer Blackwell (XQA, and the sm120 kernel stack)

> **PI steer (2026-06-21):** *"你可以选择去 profile 使用了 tma 的 kernel，比如说 xqa"* — profile a **TMA**-using
> kernel, e.g. **XQA**, to ground the page_size→decode-latency mechanism (reports 3/5, a *software*
> gather-coalescing argument) in **hardware counters**.

## Headline
Two findings, each verified two independent ways (ncu counters **and** `cuobjdump` SASS scan for the
sm120 TMA opcode `UTMALDG`), on an **RTX 5060 Ti (consumer Blackwell, sm120)**:

1. **XQA — and *every* attention kernel — uses NO TMA on sm120.** `kernel_mha` streams KV via **cp.async
   (LDGSTS)**, identical to FlashInfer. Decode is **DRAM-bandwidth-bound (~95 % of peak)**, so `page_size`
   and kernel choice are second-order: latency is flat across pages (≤1.5 %) and XQA ≈ FlashInfer at scale.
2. **On sm120, TMA is emitted *only* by CUTLASS-Blackwell GEMM kernels.** FlashInfer's cutlass fp8 GEMM
   uses it (134 MB TMA loads dynamically; 848 `UTMALDG` statically) and so does the cutlass fused-MoE
   (208 `UTMALDG` in its sm120 grouped-GEMM). Everything else — cuBLAS/cuDNN GEMM and all attention —
   falls back to pre-Blackwell kernels (Ampere `cutlass_80` / Ada `sm89_xmma`) or cp.async, with TMA = 0.

So TMA on consumer Blackwell is **real and reachable, but dormant in the default attention/GEMM paths**;
XQA's TMA path is a **Hopper (sm90) / datacenter-Blackwell (sm100, B200)** thing, not the sm120 build.

| | |
|---|---|
| **Primary kernel** | XQA `kernel_mha` (`flashinfer.decode.trtllm_batch_decode_with_kv_cache(backend="xqa")`), JIT for `sm_120a` |
| **Compared to** | FlashInfer fused decode; + a stack-wide survey (GEMM/attention/MoE) |
| **Model dims** | Qwen3-VL-2B text: 16 Q / 8 KV heads (GQA-2), head_dim 128, bf16 |
| **Regime** | TRUE batch — B independent sequences, each its own paged KV (no shared prefix) |
| **Counters** | `ncu` **2025.3.1** as root on **phastform** (RTX 5060 Ti, sm120, drv 580.95.05 / CUDA 13.0) |
| **Static scan** | `cuobjdump --dump-sass` of the JIT-compiled `.so`/`.cuda.o`, grep `UTMALDG` (sm120 TMA opcode) |
| **Latency** | RTX 5060 Ti (gray), CUDA-event timing (no profiler) |
| **page_size** | XQA: **{16,32,64,128}** only (ps<16 rejected). FlashInfer: {1,16,32,64,128} |
| **Data** | `offline_batch_results/xqa_profile/` — `*.csv` (108 XQA/FI counter cells), `survey*.csv` (stack survey) |
| **Figures** | [fig_xqa_tma.png](fig_xqa_tma.png) (XQA vs FI counters) · [fig_xqa_latency.png](fig_xqa_latency.png) (latency) |
| **Audit** | [SUDO_CHANGES.md](SUDO_CHANGES.md) — every privileged action + the one persistent change (matched ncu install) |

---

# Part A — XQA decode characterized

## A1. XQA structurally requires `page_size ∈ {16, 32, 64, 128}`
XQA rejects `page_size < 16` outright (`ValueError: Invalid page_size: 1, only 16,32,64,128 are
supported`), every cell. So the `ps1`-vs-large-page question from reports 3–5 is **undefined** for XQA —
it only lives at large pages (a TMA-style kernel is built around contiguous ≥16-token runs).

## A2. Correctness
XQA vs FlashInfer reference (matched canonical KV): `cos = 0.99999`, `max|Δ| ≈ 0.002–0.004` (bf16), all
supported pages. We profiled a numerically correct kernel.

## A3. XQA uses no TMA; decode is DRAM-bandwidth-bound
Across **all 48 XQA cells (and 60 FlashInfer cells)** the TMA counters are exactly **0** — both use
cp.async. Decode reads the full distinct KV once (e.g. bs8/kv4096 = 8·4096·8·128·2·2 ≈ **134 MB**, measured
134 MB at **~95 % peak DRAM**), so `page_size` shifts the *access pattern* but not the bytes moved:

XQA, bs8 / kv4096, vs page_size (ncu):

| page | dur (µs) | DRAM read | DRAM %peak | global-ld | L2 hit | L1 hit |
|---:|---:|---:|---:|---:|---:|---:|
| 16  | 325 | 134 MB | 94.8 % | **49,664** | **24 %** | **37 %** |
| 32  | 328 | 134 MB | 94.4 % | 25,088 | 12.5 % | 21.6 % |
| 64  | 327 | 134 MB | 94.5 % | 20,992 | 0.16 % | 0.49 % |
| 128 | 326 | 134 MB | 94.5 % | **12,800** | **0.16 %** | **0.30 %** |

Smaller pages → **more, smaller global loads** (`global_ld ∝ 1/page`, ~4× ps128→ps16) with a much **higher
cache-hit rate** (L2 24 % vs 0.16 %): the extra index/gather traffic is served from L2/L1, costing
**neither DRAM bandwidth nor latency**. Hence **duration is flat (≤1.5 %)**, matching the CUDA-event sweep.

## A4. XQA vs FlashInfer
- **Latency** (counter duration ratio): **1.00–1.03** for bs ≥ 8 (≈equal); XQA **+24–32 %** only at
  bs1/kv1024, where work is tiny and the kernel is *overhead*-bound (XQA DRAM 52 % vs FI 67 % there). Matches
  the CUDA-event sweep (+66 % bs1/kv1024 → +0.2 % bs64/kv16384) — see [fig_xqa_latency.png](fig_xqa_latency.png).
- Both bandwidth-bound at scale (~95–97 % peak), both **no TMA**, both cp.async.
- **Occupancy:** XQA ~16 % vs FlashInfer ~63 %; SM throughput ~8–10 % vs ~19 % — XQA saturates HBM with far
  fewer active warps, same bandwidth ceiling.

---

# Part B — Which sm120 kernels actually use TMA? (extensive survey)

I profiled a broad set of ops two ways: **dynamic** (ncu TMA counters) and **static** (`cuobjdump` SASS,
counting the sm120 TMA opcode `UTMALDG`). Both agree.

## B1. Dynamic — ncu TMA counters

| op (bf16 unless noted) | kernel dispatched on sm120 | TMA load | uses TMA? |
|---|---|---:|:--:|
| Triton matmul w/ `make_tensor_descriptor` (control) | `matmul_tma` | 2.15 GB | ✅ |
| **FlashInfer fp8 GEMM `bmm_fp8(backend="cutlass")`** | `cutlass::device_kernel<GemmUniversal…>` | **134 MB** | ✅ |
| FlashInfer fp8 GEMM `bmm_fp8(backend="cudnn")` | `cudnn…sm80_matMul` (Ampere) | 0 | ❌ |
| FlashInfer fp8 GEMM `bmm_fp8(backend="cublas")` | `sm89_xmma…` (Ada) | 0 | ❌ |
| cuBLAS GEMM bf16 / fp16 (`torch.matmul`) | `cutlass_80_tensorop…` (Ampere) | 0 | ❌ |
| fp8 GEMM (`torch._scaled_mm`) | `sm89_xmma…` (Ada) | 0 | ❌ |
| FlashInfer GEMM `mm_bf16` / `bmm_bf16` | *unsupported on cc 12.0* | — | — |
| **XQA decode** `kernel_mha` | cp.async | 0 | ❌ |
| FlashInfer decode `BatchDecodeWithPagedKVCache` | cp.async | 0 | ❌ |
| FlashInfer prefill `SinglePrefillWithKVCacheKernel` | cp.async | 0 | ❌ |
| PyTorch SDPA | `pytorch_flash::flash_fwd_kernel` | 0 | ❌ |
| RMSNorm (neg-control) | `norm::RMSNormKernel` | 0 | ❌ |

## B2. Static — `UTMALDG` count per compiled kernel (`cuobjdump`)
Corroborates the counters and reaches kernels ncu couldn't (the MoE, whose dynamic run compiles for >40 min):

| compiled object (sm120) | `UTMALDG`+UTMA instrs | uses TMA? |
|---|---:|:--:|
| `gemm_sm120.so` (FlashInfer cutlass fp8 GEMM) | **848** | ✅ |
| `fused_moe_120/…gemm_grouped_**sm120**…cuda.o` (cutlass MoE, sm120 path) | **208** | ✅ |
| `fused_moe_120/…gemm_grouped_**sm80**…cuda.o` (MoE Ampere fallback) | 0 | ❌ |
| `xqa_…sm120.so` (XQA decode) | **0** | ❌ |
| `batch_decode_…sm120.so` (FlashInfer decode) | **0** | ❌ |
| Triton `matmul_tma.cubin` (control) | UTMALDG×6, UTMASTG, UTMACCTL | ✅ |

So the cutlass **MoE** *does* use TMA on sm120 (via its sm120 grouped-GEMM, 208 UTMA) — confirmed
statically without waiting for its multi-minute JIT/run; its sm80 fallback variant has none.

## B3. Positive control — sm120 can run TMA and the counter fires
A Triton matmul written to *explicitly* use TMA (`tl.make_tensor_descriptor` → `cp.async.bulk.tensor`;
`scripts/tma_positive_control.py`) issues **2.15 GB TMA loads + 17.8 M TMA-pipe cycles, 0 cp.async**. So the
counter reports billions when TMA is used → the `0`s for XQA/cuBLAS are **genuine**, not a GeForce cap.
(Cross-check: a bf16 GEMM shows tensor-pipe 268 M / DRAM 1.34 GB — counters are live; that GEMM just runs an
Ampere `cutlass_80` kernel with TMA = 0.)

## B4. Takeaways
1. **TMA is reachable on sm120, but only via CUTLASS-Blackwell GEMM** (fp8 GEMM, MoE grouped-GEMM, or a
   hand-written Triton/CUTLASS kernel).
2. **The backend decides.** Same fp8 GEMM: `cutlass`→TMA, `cudnn`→Ampere, `cublas`→Ada. cuBLAS bf16/fp16 →
   Ampere `cutlass_80`.
3. **No attention kernel uses TMA on sm120** — XQA, FlashInfer decode/prefill, PyTorch FlashAttention all
   cp.async. For the PI's target (XQA / attention), TMA is simply not on the sm120 path.

---

## Interpretation — how this grounds the page_size study
Reports 3/5 argued *in software* that small pages hurt only when decode is access-pattern-bound (shared
prefix), not bandwidth-bound (true batch). Report 6 confirms it *in hardware*: in a true batch the XQA/FI
decode kernels are **HBM-bandwidth-bound at ~95 %**, the page-size access-pattern change is **absorbed by
L2/L1** (A3), and no exotic memory engine (TMA) is even in play — so `page_size` is free. The page penalty
of reports 3/5 lives in the *cache-resident* (shared-prefix) regime, which report 7 profiles directly.

## Honest scope / caveats
- **"Profile a TMA kernel" → XQA doesn't use TMA here.** That's the finding. XQA's TMA path needs **Hopper
  H100 (sm90)** or **B200 (sm100)** — not available. The Triton control + cutlass GEMM/MoE give real
  TMA-in-use data on sm120 instead.
- **`mm_fp8` (trtllm-gen, backend="trtllm_low_latency") not captured:** it launches via PDL/graph that ncu
  could not attach to ("No kernels were profiled"), and only its metadata (not the cubin) is cached, so it
  was excluded from both methods. It is NVIDIA's Blackwell trtllm-gen GEMM and very likely uses TMA, but is
  **unverified here**.
- **cutlass MoE:** confirmed via static UTMA scan (208); the *dynamic* counters were not collected because
  its JIT compile (dozens of cutlass kernels) exceeds ~40 min.
- **Hardware split:** latency on gray's 5060 Ti, counters on phastform's 5060 Ti (same sm120 class); XQA
  validation reproduced on both.
- Latency effects across pages are ≤1.5 % — real, bandwidth-explained, small. Practical takeaway:
  *page_size is ~free for true-batch decode on this hardware.*

## Verification (two methods agree)
- **TMA = 0** for XQA/FI in **all 108 ncu cells** (recomputed from raw CSV by `analyze_xqa_ncu.py`) **and**
  `UTMALDG = 0` in the XQA/decode `.so` (`cuobjdump`).
- **TMA > 0** for the cutlass GEMM/MoE both ways (134 MB / 848 UTMA; 208 UTMA) and the Triton control
  (2.15 GB / UTMALDG present).
- **DRAM bytes** match the analytic KV size (134 MB @ bs8/kv4096) at ~95 % peak.
- **Counter trustworthy:** positive control fires billions; tensor-pipe/DRAM return large values elsewhere.
- **ncu blocker root-caused:** ncu 2025.1.1 failed `LibraryNotLoaded` on the CUDA-13 driver → fixed by
  installing `nsight-compute-2025.3.1` (`cuda-nsight-compute-13-0`). See SUDO_CHANGES.md.

## Reproduce
```bash
# sudo-free (latency / correctness / page constraint):
python bench_xqa.py --validate
python bench_xqa.py --latency --backends xqa flashinfer \
    --page-sizes 1 16 32 64 128 --batch-sizes 1 8 32 64 --seq-lens 1024 4096 16384 \
    --output offline_batch_results/xqa_profile_5060ti/xqa_latency.json

# counters — needs a CUDA-13-matched ncu as root (logs every privileged action):
sudo -E bash profile_xqa_ncu.sh                 # XQA vs FlashInfer 108-cell sweep
sudo -E bash _survey_run.sh ; sudo -E bash _survey2b_run.sh   # stack-wide TMA survey
python tma_positive_control.py                  # Triton TMA control (+ ncu to confirm)
# static corroboration (no GPU/root): cuobjdump --dump-sass <kernel>.so | grep -c UTMALDG

# locally:
python3 page_size_study/scripts/analyze_xqa_ncu.py
python3 page_size_study/scripts/make_xqa_figure.py
```
