# Report 16 — Switching decode to TensorRT-LLM (`trtllm_mha`): page behavior in both regimes (shared-prefix + true-batch), and what "TensorRT-LLM" actually is on sm120

**Prompted by the PI: "把 flashinfer 切换成 TensorRT-LLM 也试一下" (also try switching FlashInfer to
TensorRT-LLM).** Measured on phastform (RTX 5060 Ti, sm120, exclusive), same rebuilt SGLang
(`0eded9e208`, behaviorally-matched June commit) as report 15, so only the attention backend differs.
Metrics use the study's **correct** decode measurement (steady-state per-step from engine logs / bench_one_batch
`median_decode_latency`), never the admission-ramp-contaminated harness whole-batch TPOT.

Published evidence, script mapping, and retention qualifications are listed in
[the report provenance note](provenance.md).

## What "switching to TensorRT-LLM" resolves to on this hardware (established before any latency number)

| Fact | Evidence (all empirically verified on this stack — flashinfer 0.6.6) |
|---|---|
| **On sm120, the genuine TRT-LLM-GEN attention kernels don't run** — `backend="trtllm-gen"` decode and the trtllm prefill (`trtllm_batch_context_with_kv_cache`) both **fail with `RuntimeError: TllmGenFmhaRunner … Unsupported architecture`** when invoked *directly* (bypassing SGLang). Only `backend="xqa"` runs. | Decode support can be reprobed with `python scripts/bench_xqa.py --validate --backend trtllm-gen` (and `--backend xqa` as the positive control). The direct prefill probe was executed on the box, but its standalone script/log was not retained; the matching source architecture gate is recorded below. |
| **SGLang mirrors this**: `trtllm_mha` prefill is gated to sm100 (so on sm120 use split `decode=trtllm_mha, prefill=flashinfer`); decode is *dispatched* to XQA via `is_xqa_impl = is_sm90 or is_sm120`. | Engine boot `ValueError: TRTLLM MHA … prefill … only supported on … (SM100)` (`server_args.py:3132-3160`); `trtllm_mha_backend.py:158`. The gate is **not over-conservative** — it matches the kernel's own `Unsupported architecture` above. |
| **On sm120, trtllm_mha decode = the XQA kernel** (`kernel_mha`, report 6). | engine JIT log compiles `xqa_input_bf16_..._page_size_16_head_dim_128_head_group_ratio_2`; matches the direct test. |
| **`page_size=1` is impossible**; engine restricts to **{16,32,64}** (128 silently coerced to 64). | `server_args.py`: "TensorRT-LLM MHA only supports page_size of 16, 32 or 64, changing page_size…to 64"; XQA kernel rejects <16 (`bench_xqa --validate`: {16,32,64,128} OK, 1/2/4/8/256 REJECT). |
| **Decode output is correct** on sm120. | `cos(XQA, FlashInfer) = 0.99999`, max|Δ|=0.002, every accepted page; engine generate clean (no NaN). |

So the honest framing: on this GPU, "switching to TensorRT-LLM" means **running the XQA decode kernel with a
FlashInfer prefill** — the genuine trtllm-gen FMHA kernels return *Unsupported architecture* on sm120, so
they're not merely un-dispatched by SGLang, they don't run. And it **cannot express the `page_size=1`
pathology** reports 3/14/15 chased. The "Unsupported architecture" is a **source-level hard arch check**
(`mSM == kSM_100 || kSM_103`, datacenter Blackwell only) present in **both** flashinfer 0.6.6 and 0.6.15
(latest version tested on 2026-07-17) — see the version check below.

**Version check — SOURCE-VERIFIED across flashinfer 0.6.6 AND 0.6.15 (latest version tested on 2026-07-17), 2026-07-17.** I upgraded a
fresh venv to flashinfer 0.6.15 + torch `2.9.1+cu129` and installed a full **CUDA 12.9 toolkit (nvcc 12.9)** on
the box, then invoked the genuine trtllm-gen kernels directly. Result: the trtllm-gen FMHA runner **hard-gates
the architecture** —
`FLASHINFER_CHECK(mSM == kSM_100 || mSM == kSM_103, "Unsupported architecture")` (`fmhaRunner.cuh:37`) — i.e.
**only sm100/sm103 (datacenter Blackwell); there is no `kSM_120`/sm120 code path anywhere in the trtllm-gen
fmha sources** (verified by grep). So `backend="trtllm-gen"` decode **and** trtllm prefill raise *Unsupported
architecture* on sm120 in **both** the June-era stack (0.6.6) and flashinfer 0.6.15, the latest version tested on 2026-07-17 — even
with CUDA 12.9 + nvcc 12.9 present. (0.6.15 adds a *separate, earlier* gate "SM 12.x requires CUDA >= 12.9"
that governs its *other* kernels like XQA — orthogonal to the trtllm-gen arch check; with CUDA 12.9 XQA runs,
trtllm-gen still hits the sm100/sm103 hard check. An intermediate "Ninja build failed" seen before installing
nvcc-12.9 was a missing-toolchain red herring, not sm120 support.) **So on sm120, `trtllm_mha` decode = XQA,
full stop — in every flashinfer version tested.** *Caveat:* NVIDIA's **standalone** TensorRT-LLM library is a
different codebase and is **not** what SGLang's `trtllm_mha` uses (that path goes through flashinfer); whether
that separate library has sm120 attention is out of scope/untested. To run the genuine trtllm-gen kernel needs
an **SM100 B200/GB200**; H100/sm90 also dispatches to XQA.

## 1. Shared-prefix regime (Arm A1) — steady-state per-step decode, radix ON

Qwen3-VL-2B, shared prefix ≈6.1k tokens, radix ON, CUDA graph ON, steady-state ms/step (median over 765
full-concurrency steps; `decode=trtllm_mha/prefill=flashinfer` for the trtllm arm).

| backend | bs8 | bs16 | bs32 | page-flatness |
|---|---|---|---|---|
| **trtllm_mha (XQA)** | ps16 13.18 / ps32 12.94 / ps64 **12.83** | ps16 16.80 / ps32 16.44 / ps64 **16.20** | ps16 24.04 / ps32 23.35 / ps64 **22.93** | **NOT flat: bigger page → faster, spread 2.8 / 3.7 / 4.8%** |
| flashinfer | ps1 12.94 / ps16 12.95 / ps128 12.94 | ps1 17.51 / ps16 17.48 / ps128 17.58 | ps1 24.96 / ps16 24.98 / ps128 24.99 | flat (≤0.1 / 0.6 / 0.1%) |
| triton | ps1 15.55 / ps16 15.54 / ps128 15.57 | ps1 21.31 / ps16 21.29 / ps128 21.36 | ps1 33.42 / ps16 33.70 / ps128 33.90 | flat (ps1 −0.2 / −0.3 / −1.4%) |

**Findings:**
- **trtllm_mha (XQA) is the FIRST backend in the study whose decode is genuinely `page_size`-sensitive**, and
  the sign is **inverted** vs the (artifactual) ps1-penalty story: **bigger page = faster** (ps64 fastest,
  ps16 slowest), monotone across all three batch sizes, spread up to **4.8% (bs32)**. Mechanism: unlike
  FI/Triton (normal FI plans decode with page→1; Triton's kernel receives no engine-page argument), XQA
  reads KV in real page-granular chunks, so larger pages reduce the page-count-dependent load work. Arm C
  ncu measures ps16/ps64 global-load ratios of **2.33–2.39×**; the total count is not a pure mathematical
  `1/page` law because fixed and other page-independent instructions remain.
- **At its best page (ps64), XQA decode is *faster* than FlashInfer** in this shared/L2-reuse regime:
  bs32 **22.93 vs 24.96 ms (−8.1%)**, bs16 16.20 vs 17.48 (−7.3%), bs8 12.83 vs 12.94 (−0.9%) — and far
  under Triton (33.42). So "switching decode to TensorRT-LLM" is a **modest win at larger batch**, *if* you
  set page 64 and can live without ps1.
- **`page_size=1` is inexpressible for trtllm** (radix-off / XQA floor) — the pathological small page simply
  isn't on trtllm's menu; the smallest it offers (ps16) is its *slowest* page.
- FI and Triton reconfirm the study's page-invariance (≤1.5%). For Triton, ps16 is lower by only
  0.01–0.02 ms at bs8/16, while ps1 is 1.4% below ps128 at bs32; this does not define a meaningful monotone
  page trend and is consistent with reports 5/15.

## 2. True-batch regime (Arm A2) — `bench_one_batch`, distinct KV (reports 5 & 13 extended)

Decode TPOT = `median_decode_latency × 1000` (ms/token; median over 63 steps, first discarded — already the
correct per-step metric). One round; cells ~17–20 ms carry ~1–2% bench noise.

| model / cell (regime) | trtllm_mha (ps16/32/64) | flashinfer (ps1/16/128) | triton (ps1/16/128) | best-of-each |
|---|---|---|---|---|
| Qwen2.5-3B B2/L1024 (**off-wall**) | 17.22 / 17.85 / **17.15** | **17.43** / 17.74 / 18.23 | 17.48 / 17.72 / **17.38** | trtllm 17.15 ≈ FI 17.43 ≈ T 17.38 |
| Qwen3-VL-2B B8/L4096 (**on-wall**) | 19.85 / 20.44 / **19.77** | **19.77** / 20.09 / 20.55 | **19.64** / 19.96 / 20.47 | trtllm 19.77 ≈ FI 19.77 ≈ T 19.64 |

The evidence directory also retains an XQA-only B1/L512 page pilot (ps16/32/64). It is not included in
the cross-backend table and is not used to broaden the two-cell comparison.

**Findings:**
- **In the two representative true-batch cells measured here**, the best page of each backend converges
  within **~1%**. The ~8% XQA advantage seen in the measured shared-prefix cells is absent at these two
  points. This supports a regime-specific interpretation, but one round and two shapes do not establish a
  universal result for every independent-sequence workload.
- **trtllm's page trend does not cleanly reproduce in these true-batch cells** — spread ~3–4% but non-monotonic (ps32 is
  a consistent outlier-high; ps16 ≈ ps64). Because A2 has only one round, the statistical significance of
  that range is undetermined; unlike the shared-prefix result, it is not a clean monotone bigger-page-faster
  pattern. This is consistent with regime gating, but the one-round A2 cells are not by themselves a
  mechanism proof.
- **FlashInfer has ps1 fastest in both cells; Triton does not have one universal winner**: ps1 is fastest in
  B8/L4096, while ps128 is ~0.6% faster in B2/L1024. The common conclusion is only that neither backend shows
  a ≥5% ps1 penalty in these cells, consistent with reports 5/13.
- Both regimes are weight-DRAM-dominated at these cells (report 8/13), so all page effects are ≤~5%.

## 3. Kernel profiling (Arm C) — *why* XQA is page-sensitive, and only off the wall

Microbench (`bench_xqa.py`, CUDA-event median), XQA (=engine trtllm_mha decode) vs FlashInfer, bs32/kv8192:

| regime | backend | ps16→ps64 (page-sensitivity) | XQA ps64 vs FI ps64 | scatter (contig→shuffle) |
|---|---|---|---|---|
| **shared** (L2-reuse, off-wall) | XQA | **+9.7%** (bigger page faster, saturates ps64) | **−12.6% (XQA faster)** | +4.4–4.9% |
| shared | FlashInfer | +0.3% (flat) | — | +2.3% |
| **distinct** (DRAM-bound, true batch) | XQA | **−0.7% (flat)** | +2.0% (XQA ≈ FI) | — |
| distinct | FlashInfer | +0.0% (flat) | — | — |

**Mechanism (confirmed):**
- **XQA's decode reads KV in real page-granular loads**, so a smaller page = more global-load instructions
  (measured ps16/ps64 ratio **2.33–2.39×**; not an exact inverse-page formula for total instructions). In the
  studied SGLang paths, FlashInfer receives a per-token lowering and
  plans decode with page→1, while the Triton decode kernel is page-agnostic; neither kernel sees the engine
  page size as XQA does.
- **The XQA page cost is regime-gated in the isolated transition sweep**: off the DRAM wall
  (shared/L2-reuse), extra load instructions/traffic extend the critical path → +9.7% ps16→ps64; in the
  measured distinct sweep (~95–96% DRAM at ps64), duration is page-flat. This isolated-kernel result explains
  the clean A1 shared direction. A2 is a separate one-round engine test: its two cells show only that the
  shared monotone trend is not cleanly reproduced, not that every true-batch page effect is zero.
- **XQA beats FlashInfer only off the wall** (shared −12.6% at the kernel, matching A1's ~8% engine gain);
  in the measured DRAM-bound isolated cell they are close (XQA +2%). In A2, the best page of each backend
  converges within ~1% at two points, while each backend still has a one-round non-monotonic ~3–4% page range.
- **Scatter (fragmented pages) costs XQA a bounded +4–5%** (slightly more than FI's +2%), but engine
  shared-prefixes allocate contiguously (reports 14/15), so this is a "what-if-fragmented" bound, not the
  operating point.

**ncu counters (`kernel_mha`, bs32/kv8192) prove the cause:**

| regime | page | dur | DRAM% | L2-hit% | **global loads** |
|---|---|---|---|---|---|
| shared | ps16 | 750 µs | 22.1 | 94.9 | **10980** |
| shared | ps32 | 721 µs | 17.6 | 95.6 | 5518 |
| shared | ps64 | 684 µs | 13.1 | 96.5 | 4608 |
| shared | ps128 | 683 µs | 13.0 | 96.5 | 2788 |
| distinct | ps16 (bs8/kv4096) | 327 µs | 94.5 | 24.0 | 1380 |
| distinct | ps128 (bs8/kv4096) | 327 µs | 94.8 | 0.2 | 356 |

- **Global-load count has a strong page-count-dependent component in both regimes** (10980→2788 as page
  16→128; 1380→356 distinct; ps16/ps64 is 2.33–2.39× in the transition sweep). Smaller page ⇒
  more, smaller loads ⇒ more total sectors ⇒ DRAM% climbs (13→22%) and duration rises — **but only off the
  wall**. In the distinct/DRAM-bound cell, DRAM sits at ~95% at *every* page, so the extra loads are absorbed
  and duration is flat (327≈327 µs) despite the same page-dependent load increase. This is the counter-level proof of
  the regime-gating.
- **Scatter (shuffle) leaves global-load count and sectors/request unchanged** (10980, 13.62 at ps16, both
  contig and shuffle). This rules out more load instructions or more sectors per request as the source of
  the +4–5% scatter spread; the available counters do not uniquely separate address-dependency/cache
  latency from other scheduling-latency causes.

**Kernel identity / TMA:** the engine's `trtllm_mha` decode JIT-compiles
`xqa_input_bf16_..._page_size_16_head_dim_128_head_group_ratio_2` — the same `kernel_mha` XQA kernel report
6 profiled; **ncu shows TMA-load bytes = 0 at every cell (no TMA on sm120, cp.async only), confirming report 6.**

### 3.1 Deep profiling — *why* page matters in shared-prefix and not in true-batch (comprehensive)

A dedicated ncu study (`profile_report_16_mechanism.sh` + `profile_report_16_triton_flashinfer.sh`, sm120, exclusive) settles the
mechanism three independent ways. **Figure:** `fig_page_regime_mechanism.png`.

**(1) The page effect is gated by the memory regime — a transition sweep.** XQA, bs32, page
{16,32,64,128} × {shared,distinct} × seq-len {2k,4k,8k,16k}. The **page-dependent instruction ratio** is
the same in both regimes (ps16 issues about **2.38×** the global-load instructions of ps64); what differs is
whether those extra instructions/transactions extend the measured critical path:

| regime | DRAM%-of-peak (ps64) | SM% | **ps16→ps64 decode time** | ps16→ps64 global loads |
|---|---:|---:|---:|---:|
| distinct / true-batch (kv 2k–16k) | **95–96%** | 7.8 | **−0.9…+1.0% (FLAT)** | 2.33–2.39× |
| shared / off-wall (kv 2k–8k) | **10–13%** | 27–30 | **+9–10% (ps16 slower)** | 2.33–2.39× |
| shared @ kv16k (working set spills L2) | **82%** (nearing wall) | 29 | **+6.8% (effect shrinking)** | 2.39× |

The transition data support a regime-gated critical-path effect: in shared/L2-reuse cells the extra
page-dependent load instructions/traffic extend duration; in the measured distinct cells duration remains
flat while DRAM is already near peak. This does **not** mean extra bytes are generally free, nor does it
establish a universal theorem that bandwidth saturation hides every instruction difference. It is the
empirical result for these shapes and this XQA path. The shared kv16k point (DRAM 82%) lies between the two
measured regimes and retains a smaller +6.8% spread.

**(2) Warp-state character confirms the two regimes** (`--set full`, bs32/kv8192): distinct decode is
**`long_scoreboard`-dominated (13–16 warps/issue) = memory-latency/DRAM-bound**; shared decode has
**low `long_scoreboard` (~0.4), math-pipe/wait-dominated = L2-resident, off the DRAM wall.** (Note: the shared
page cost is *not* a rising-per-load-latency effect — it is simply ~2.38× more load instructions issued in a
regime where instruction/traffic throughput, not DRAM bandwidth, sets the time.)

**(3) Triton & FlashInfer are page-INVARIANT at the kernel — so their ps1 edge is engine-side, not the
kernel** (`bench_pagecost`, engine-faithful contig=ps1-layout vs block128=ps128-layout):

| backend | global loads (ps1-layout vs ps128-layout) | kernel time (contig vs block128) |
|---|---|---|
| **XQA (trtllm)** | **2.38× more at ps16 than ps64**; strong page-count component | +10% shared / flat distinct |
| **Triton** | **1.00× — identical (2,156,800 loads every cell)** | flat ≤1% |
| **FlashInfer** | identical | **−0.0%** |

XQA reads KV **page-granularly** (smaller pages produce more page-count-dependent load work); the studied
Triton/FlashInfer paths receive page-independent token-level work (FlashInfer's plan uses page→1; Triton's
kernel is page-agnostic), so their kernel load-count is page-independent. Their small engine-level page
ordering is therefore **not an attention-kernel effect**. It is source-consistent with the allocator path
(ps>1 runs a paged-allocation kernel/step; ps1 uses a bare slice — report 15), but the exact component
attribution remains **inferred** because no timed decomposition isolates that allocator offset. The clean
resolution is that only XQA has measured page-sensitive kernel work in this matrix, and even that changes
duration only in the measured off-wall regime.

The headline shared-XQA ps16/ps64 counters are replay-invariant: re-profiling reproduces their global-load
counts exactly and DRAM bytes within 0.2%. Other sweep cells were not all independently replayed. The
microbenches call the *identical* FlashInfer XQA / SGLang Triton kernels the engine uses (kernel identity
confirmed above), and the engine steady-state TPOT (A1/A2) matches these directions end-to-end.

## 4. Answer to the PI

**"把 flashinfer 切换成 TensorRT-LLM 也试一下" — done. Four things to report:**

1. **On our RTX 5060 Ti (sm120), "TensorRT-LLM" decode = the XQA kernel, not the genuine TRT-LLM-GEN kernel —
   source-verified across flashinfer 0.6.6 AND 0.6.15 (latest version tested on 2026-07-17).** Invoking the genuine `trtllm-gen` decode and
   trtllm prefill kernels *directly* raises `Unsupported architecture` on sm120; the runner hard-checks
   `mSM == kSM_100 || kSM_103` (datacenter Blackwell only — no sm120 path in the sources), and this holds even
   after installing CUDA 12.9 + nvcc 12.9. Only XQA runs; SGLang's split requirement + XQA dispatch match this
   exactly. Output correct (cos vs FlashInfer = 0.99999). To run the genuine kernel needs an SM100 B200
   (H100/sm90 also falls back to XQA). *(NVIDIA's standalone TensorRT-LLM library is a separate codebase SGLang
   doesn't use here; its sm120 status is out of scope.)*

2. **It genuinely helps in the shared-prefix regime — a modest, real win.** With page 64, XQA decode is
   **~7–8% faster than FlashInfer** at bs16/32 (steady-state: bs32 22.9 vs 25.0 ms), and far faster than
   Triton. This is the first backend in the study where switching *does* move decode latency.

3. **The shared-prefix advantage did not appear in the two representative true-batch cells tested** — the
   one-round best page of each backend converges within ~1%, while each backend still has a non-monotonic
   ~3–4% page range. This is narrow, regime-consistent evidence, not a universal claim about every
   independent-sequence workload.

4. **`page_size` matters for TensorRT-LLM in a way it never did for FlashInfer/Triton** — but *inverted* from
   the old ps1 worry: **bigger page is faster** (ps16 slowest, ps64 fastest; up to ~10% at the kernel), and
   `page_size=1` **can't be used at all** (XQA floor is 16; ps128 is coerced to 64). Cause (ncu-proven): XQA
   reads KV in page-granular loads; smaller pages add a strong page-count-dependent load component
   (ps16/ps64 **2.33–2.39×**), and that extra work extends duration in the measured off-wall cells.
   **Practical recommendation: if you switch decode to TensorRT-LLM on this hardware, use `--page-size 64`.**

Net: for the measured sm120 build and shared-prefix workload, switching this decode path to XQA is beneficial
at page 64; the experiment does not prove that shared-prefix is the only profitable workload. It gives XQA
(report 6), not the real trtllm-gen kernel — to test that, we'd need a supported SM100-class box (optional
follow-on).

## 5. Evidence classes

| Claim | Class | Basis |
|---|---|---|
| The genuine trtllm-gen decode+prefill kernels **don't run on sm120** ("Unsupported architecture") in flashinfer **0.6.6 AND 0.6.15 (latest version tested on 2026-07-17)** (even with CUDA 12.9 + nvcc 12.9); only XQA runs | **Measured (direct) + source; prefill raw-log gap** | `bench_xqa.py --validate --backend trtllm-gen` for decode plus the recorded `fmhaRunner.cuh:37` hard-check `mSM==kSM_100\|\|kSM_103` (no kSM_120 in sources). The direct prefill probe's standalone log was not retained. |
| sm120 `trtllm_mha` decode = XQA; prefill gated to sm100 (matches the kernel); page∈{16,32,64}; ps1 impossible | **Measured** | engine boot errors + `is_xqa_impl` source + JIT log + `bench_xqa.py --validate` |
| Decode output correct on sm120 | **Measured** | cos(XQA,FI)=0.99999, no NaN |
| Shared-prefix: XQA page-sensitive (bigger=faster ~5% engine / ~10% kernel); ps64 ~8% faster than FI | **Measured, reproduced** | A1 steady-state (765 steps, non-overlapping distributions) + C1 microbench **3 reps** (+9.6/9.7/9.4%, −13.2/−13.2/−12.8%, ~0.3% var) |
| Two representative true-batch cells: best-of-backend results converge (~1%) | **Measured with narrow scope** | A2 bench_one_batch `median_decode_latency`, one off-wall + one on-wall cell |
| Cause = smaller pages add page-count-dependent global-load work; duration response is regime-gated | **Measured, replay-invariant** | ncu: gld 10980→2788 (shared), ps16/ps64 2.33–2.39× in the transition sweep, flat duration at 95% DRAM (distinct); re-profile reproduces gld exactly + DRAM bytes <0.2% |
| XQA = report-6 kernel, no TMA on sm120 | **Measured** | ncu kernel name `kernel_mha`, TMA-load bytes = 0 |
| Scatter cost +4–5%; not explained by gld or sectors/request | **Measured; finer latency attribution unresolved** | ncu: gld/sec-req unchanged contig↔shuffle |
| Not a report-15-style artifact (ramp/contention/config/phantom) | **Measured (5 checks, §6)** | non-overlap + 3-rep + regime-specificity + replay-invariance + causal counter |
| Studied FI/Triton kernels are page-flat under their token-level lowering | **Established (reports 10/15) + reconfirmed** | A1/C1 flatness + prior source trace |

## 6. Is this a measurement artifact (à la report 15)? — no, verified five ways

The report-15 failure modes were: (a) the harness whole-batch TPOT folding a cold-prefill admission ramp into
"decode", (b) contention time-dilation on a shared box, (c) config-dependent non-reproducibility, (d) ncu
"phantom" (replay-varying counters). Each is explicitly ruled out for the report-16 findings:

1. **Not the harness-TPOT artifact.** The headline uses steady-state per-step decode (`#running-req==B`,
   `decode_log_interval=1`), never `(end−first_token)/(tok−1)`. The A1 per-step distributions are tight and
   **non-overlapping**: trtllm-ps64 median 22.93 (p90 23.99) vs FI-ps1 median 24.96 (p10 24.65) at bs32 — the
   ~8% gap sits entirely outside the per-step spread; no ramp tail inflates it.
2. **Reproducible, tight variance.** An independent one-process CUDA-event microbench (no engine, no radix, no
   admission, no flush_cache) run **3× back-to-back** gives XQA ps16→ps64 = **+9.6 / +9.7 / +9.4%** and XQA-ps64
   vs FI-ps64 = **−13.2 / −13.2 / −12.8%** — ~0.3% rep-to-rep. The engine (A1) and microbench (C1) are two
   independent measurement paths that agree.
3. **Not contention/clock drift.** Box exclusive (0 foreign procs, verified). Decisive discriminator:
   **the clean effect is regime-specific** — present in shared-prefix (off-wall), but page-flat in the
   isolated distinct transition sweep. A2 separately shows one-round best-of-backend convergence within
   ~1% at two cells, while the within-backend page ranges remain non-monotonic ~3–4%. A contention/clock
   bias would inflate trtllm consistently across these controls; it doesn't. In the
   microbench, XQA and FI are timed back-to-back in the same process, so no cross-run drift is possible.
4. **ncu counters replay-invariant (report-13's phantom test).** Re-profiling the same cell reproduces
   `global_ld` **exactly** (ps16 10979.56 = 10979.56; ps64 4608.0 = 4608.0) and DRAM-read bytes to <0.2%
   (17.92↔17.94 MB; 9.43↔9.43 MB). `global_ld` is a deterministic instruction count — it cannot be a phantom —
   and its measured ps16/ps64 increase is the causal work counter, so the mechanism is not an artifact of timing.
5. **Mechanism is causal and regime-consistent.** Smaller page ⇒ more global loads ⇒ more DRAM bytes
   (17.9 MB @ps16 vs 9.4 MB @ps64) ⇒ higher DRAM% (22% vs 13%) ⇒ slower — but only where DRAM isn't already
   saturated (distinct = 95% DRAM ⇒ flat). The number, direction, and regime-gating of the effect all follow
   from one measured counter.

**Also re-verified (never assumed) — and one intermediate error corrected in the open:** the genuine trtllm-gen
decode *and* trtllm prefill kernels were invoked *directly* and both return `Unsupported architecture` on
sm120, in flashinfer **0.6.6 AND 0.6.15 (latest version tested on 2026-07-17)**, even after installing CUDA 12.9 + nvcc 12.9 — the runner
hard-checks `mSM == kSM_100 || kSM_103` with no sm120 path in the sources. Mid-investigation I briefly
mis-concluded (from a "Ninja build failed" that was really just a missing nvcc) that newer flashinfer had
started targeting sm120; the full toolchain + source read overturned that. So "on sm120 you get XQA, not
trtllm-gen" is now measured *and* source-verified across versions (§What… above).

## Optional follow-on — genuine trtllm-gen on SM100 (B200)
Only SM100 yields the real TRT-LLM-GEN decode kernel (TMA-using). ~1 day on vast.ai: rebuild
sglang+flashinfer for sm100, rerun `bench_xqa.py --backend {xqa,trtllm-gen}` + engine split-backend
`trtllm_mha`. Decide after these sm120 numbers.

**Companions:** [report 6](../06-xqa-tma-profile/report.md) (the XQA kernel = what
trtllm_mha is here), [report 5](../05-true-batch/report.md) & [report 13](../13-true-batch-off-wall/report.md) (true-batch page-invariance extended),
[report 15](../15-log-forensics-and-upper-bound/report.md) (the artifact + steady-state method).
