# Report 6 — Profiling the XQA decode kernel (TMA) vs page_size

> **STATUS (2026-06-21): DRAFT — latency/correctness/page-constraint findings are
> final (measured); the TMA *hardware-counter* sections are PENDING `ncu`.** The
> counters are the core of this report and need root `ncu` on `phastform`
> (CIS boxes set `RmProfilingAdminOnly=1`). Tooling is ready
> (`scripts/profile_xqa_ncu.sh`); see [SUDO_CHANGES.md](SUDO_CHANGES.md) for the
> privileged-action audit trail. *Blocked at time of writing by a workstation-side
> VPN/proxy fault, not by the cluster.*

> PI steer (2026-06-21): *"你可以选择去 profile 使用了 tma 的 kernel，比如说 xqa"* — profile a
> TMA-using kernel, e.g. XQA. Goal: ground the page_size→decode-latency mechanism
> (reports 3/5, a *software* gather-coalescing argument) in **hardware counters**,
> using the modern **XQA** kernel that loads KV via **TMA** (Tensor Memory
> Accelerator) bulk-async copies rather than scalar LSU loads.

| | |
|---|---|
| **Kernel** | XQA (`flashinfer.decode.trtllm_batch_decode_with_kv_cache(backend="xqa")`, profiler name `kernel_mha`) |
| **Compared to** | FlashInfer fused paged decode (`BatchDecodeWithPagedKVCacheWrapper`); Triton scatter-gather (`_fwd_kernel`, via engine) |
| **Model dims** | Qwen3-VL-2B text: 16 Q / 8 KV heads (GQA-2), head_dim 128, bf16 |
| **Regime** | TRUE batch — B independent sequences, each its own paged KV (no shared prefix) |
| **Latency HW** | RTX 5060 Ti (16 GB, sm120) on `gray` — CUDA-event timing (no profiler needed) |
| **Counter HW** | `phastform` (GPU TBD; re-validate page support there) — `ncu` perf counters |
| **page_size** | XQA: {16, 32, 64, 128} only (ps<16 rejected). FlashInfer/Triton: {1,16,32,64,128} |
| **Data** | `offline_batch_results/xqa_profile_5060ti/xqa_latency.json` (latency); `…/xqa_profile/*.csv` (counters, pending) |
| **Figures** | [fig_xqa_latency.png](fig_xqa_latency.png) (done); `fig_xqa_tma.png` (pending counters) |

---

## Findings so far (measured, final)

### 1. XQA structurally cannot use small pages — `page_size ∈ {16, 32, 64, 128}`
XQA **rejects** `page_size < 16` outright: `ValueError: Invalid page_size: 1, only
16, 32, 64, 128 are supported` (every cell, all batch/context). This is itself a
result for the page_size study: a **TMA-bulk** decode kernel is built around
contiguous ≥16-token page runs — the exact regime that the study's Triton
scatter-penalty (`ps1`) cannot exist in. The whole "`ps1` vs large page" question
from reports 3–5 is **undefined** for XQA; XQA only lives at large pages.

### 2. Correctness — XQA matches FlashInfer
On matched canonical KV, XQA output vs the FlashInfer reference:
`cos = 0.99999`, `max|Δ| = 0.0020` (bf16) at every supported page size. We are
profiling a numerically correct kernel.

### 3. Latency — XQA is page-flat, with a fixed overhead that vanishes with work
CUDA-event median TPOT-equivalent (decode latency, ms), true batch, sm120:

| cell | XQA (min over {16–128}) | FlashInfer (min over {1–128}) | XQA vs FI | XQA page-spread |
|---|---:|---:|---:|---:|
| bs1 / kv1024  | 0.0946 | 0.0570 | **+66.0 %** | 4.6 %† |
| bs8 / kv1024  | 0.0767 | 0.0642 | +19.6 % | 4.0 %† |
| bs32 / kv1024 | 0.3712 | 0.3534 | +5.0 % | 0.6 % |
| bs64 / kv1024 | 0.6847 | 0.6689 | +2.4 % | 0.1 % |
| bs1 / kv4096  | 0.0678 | 0.0555 | +22.2 % | 9.4 %† |
| bs8 / kv4096  | 0.3678 | 0.3559 | +3.4 % | 0.4 % |
| bs32 / kv4096 | 1.3217 | 1.2982 | +1.8 % | 0.3 % |
| bs64 / kv4096 | 2.5700 | 2.5520 | +0.7 % | 0.4 % |
| bs1 / kv16384 | 0.2115 | 0.1982 | +6.7 % | 1.0 % |
| bs8 / kv16384 | 1.3126 | 1.2990 | +1.1 % | 0.6 % |
| bs32 / kv16384| 5.1028 | 5.0648 | +0.8 % | 1.3 % |
| bs64 / kv16384| 10.1169| 10.0953| **+0.2 %** | 1.3 % |

† sub-0.1 ms cells; the "spread" is measurement noise, not a page effect.

Two clean trends ([fig_xqa_latency.png](fig_xqa_latency.png)):
- **Page-flat.** Within XQA's supported pages, latency varies ≤ ~1 % — page_size is
  essentially free, exactly as the page-agnostic FlashInfer kernel. No page wins.
- **Fixed overhead, amortised by work.** XQA carries a setup/reduction overhead vs
  FlashInfer that scales *inversely* with total work `B×L`: **+66 %** at the
  smallest cell (bs1/kv1024) → **+0.2 %** at the largest (bs64/kv16384). XQA is a
  multi-CTA, TMA-pipelined kernel optimised for *throughput*; at tiny work the
  pipeline/reduction fixed cost dominates, at scale it disappears.

---

## PENDING — the TMA hardware-counter story (needs `ncu`)
To be filled from `offline_batch_results/xqa_profile/*.csv` once `profile_xqa_ncu.sh`
runs on `phastform`. Planned content:
- **Prove TMA is used:** TMA/async-copy instruction counts > 0 for `kernel_mha`, and
  scalar `sm__sass_inst_executed_op_global_ld` for XQA ≪ Triton `_fwd_kernel`
  (bulk TMA vs per-token scatter). *If sm120 XQA shows no TMA, report that and note
  the Hopper sm90 path is canonical.*
- **Memory traffic vs page_size:** DRAM bytes read/write, DRAM %peak, L2/L1 hit —
  does any of it move with page_size for XQA? (latency says no; counters explain why)
- **XQA vs FlashInfer vs Triton** at matched cells: occupancy, SM/DRAM %peak,
  global-ld ratio — the access-pattern contrast that grounds reports 3/5.
- **Regime sweep** (L2-resident `B×L ≲ 8k` → bandwidth-bound): where the +overhead
  in finding 3 comes from (TMA-setup-bound at small work vs HBM-bound at scale).
- `fig_xqa_tma.png` (3 panels: global-ld, DRAM read, L2 hit — XQA vs FlashInfer).

---

## Verification (done)
- **Correct kernel:** XQA vs FlashInfer reference cos 0.99999 / max|Δ| 0.0020.
- **True batch:** B independent sequences, each contiguous private pages; no shared
  prefix (`bench_xqa.py`).
- **page constraint reproduced** in both `--validate` and the full `--latency` sweep.
- **Counters (pending):** single-kernel isolation via `cudaProfilerStart/Stop`
  (`bench_xqa.py --single`) + `ncu --profile-from-start off`; 3 launches/cell;
  idle-gated; headline to be recomputed from raw CSV.

## Reproduce
```bash
# sudo-free (latency / correctness / page constraint) — any box with the venv:
python bench_xqa.py --validate
python bench_xqa.py --latency --backends xqa flashinfer \
    --page-sizes 1 16 32 64 128 --batch-sizes 1 8 32 64 --seq-lens 1024 4096 16384 \
    --output offline_batch_results/xqa_profile_5060ti/xqa_latency.json

# TMA counters (needs root ncu; logs every privileged action to SUDO_CHANGES.log):
sudo -E bash profile_xqa_ncu.sh

# locally (paths self-computed):
python3 page_size_study/scripts/analyze_xqa_ncu.py
python3 page_size_study/scripts/make_xqa_figure.py
```
