# Report 15 — Gray-log forensics closes reports 3 & 4 (both the same measurement artifact, contention retracted); the off-wall regime reconciled with report 7; and the constructed upper bound on a real e2e `ps1` penalty

**Prompted by three challenges from Chuyue (2026-07-16): (1) "Are you sure report 4 isn't real? You claim
contention — but you also claimed report 3 was contention and that was false." (2) "Report 7 says shared
prefix does NOT reach the DRAM bandwidth wall; report 14 claims it does — which is it?" (3) "Construct a
scenario where bandwidth is not saturated and the attention-kernel `ps1` penalty genuinely shows up
end-to-end, like the +8.6% FlashInfer one." All three were justified: the report-4 contention claim was
unproven speculation, the report-14 "at the wall" claim was wrong, and the constructive question deserves
a measured answer, not an assertion.**

| | |
|---|---|
| **Headline 1** | **Gray's own original logs settle reports 3 AND 4 without any re-run.** Both logs contain per-step decode lines. Gray's *steady-state* decode was **page-invariant all along** — report 3 (Triton, bs8–48): ps1-vs-ps128 = **−0.3/−0.5/−0.4/−1.2%**; report 4 (FlashInfer 39k, 3 interleaved pairs): **−0.7/−0.2/−0.7%** — while the *reported* TPOT showed +11–25% and +8.2/+8.4/+9.2%. Both headline numbers were the **admission-ramp/TPOT-formula artifact, measured on gray itself**. |
| **Headline 2** | **Contention is retracted entirely** — for report 4 as well. Gray's decode steps were never inflated; nothing about either result needs a co-tenant. (Report 14's earlier "report 4 = contention" claim was an unproven inference from one non-reproducing re-run on a different build — the same mistake pattern as the original report-3 "contention" call.) |
| **Headline 3** | **Report 7 was right; report 14's "DRAM wall" wording was wrong.** ncu of the *exact* report-3 cell (Triton stage1, shared, B32/L9663): **DRAM 7.1–7.3% of peak, L2-hit 96.6% (= (32−1)/32, report-12's reuse law), SM 32%** — deeply off the wall. `t_attn(ps1)=t_attn(ps128)` not because a wall hides a penalty, but because in the real engine the `kv_indices` are **contiguous** and the kernel is **page-agnostic** → identical work. Even *synthetic* worst-case token-scatter (ps1-frag layout vs ps128's block layout) costs the kernel only **+0.5% (bs32/9.7k) → +6% (bs8)** — batch-dependent, `sectors/request` flat (8.55), and it never arises in a real fresh-prefix decode. |
| **Headline 4** | **Constructed bound for this hardware/build/tested construction**: stacking the tested amplifiers (Triton × shared prefix × bs32 × a ps1 pool churned until the prefix is **verified fragmented to `frac_contig=0.027`** — 97% token-scattered), the end-to-end steady-state `ps1` penalty is **−1.25% (ps1 is *faster*)**. This experiment did not construct a +8.6%-style decode penalty. The explanation that ps128's extra allocator kernel supplies ps1's head start is source-consistent but remains **inferred**, not a timed component decomposition. |
| **Run** | 2026-07-16/17, phastform (RTX 5060 Ti, exclusive). Gray-log forensics from the published `data/raw/report-15-gray-log-forensics/report-03-dense-sweep/hb_tri_g_ps*.{log,json}` and `report-04-flashinfer-graph/fgi_p*_ps*.log` subsets (byte-identical copies of the gray originals, June 18–19). ncu on `bench_pagecost.py` shared cells; constructed arms via `drive_fragmentation_experiment.py` (churn→fragment→shared-prefix decode) with kvidx-contiguity verification. The constructed arm's independent raw-run directory is not retained. |
| **Provenance** | [Evidence map and retention gaps](provenance.md) |

## 1. Gray-log forensics — the decisive evidence was in the original logs all along

Both of gray's original runs were logged with `decode_log_interval=1`, so every decode step's latency is
recoverable from the logs themselves (`Decode batch … gen throughput`), independent of the harness's
summary formula. Extracting steady-state at full concurrency:

**Report 3 (Triton, shared ~9.7k prefix, graph ON, gray, 2026-06-18)** — reported (`hb_tri_g_ps*.json`) and
steady-state (same `.log`, 635 steps/cell) from one experiment:

| bs | gray *reported* ps1/ps128 | reported Δ | gray *steady* ps1/ps128 | steady Δ |
|---:|---:|---:|---:|---:|
| 8  | 26.52 / 23.78 | **+11.5%** | 20.18 / 20.25 | **−0.3%** |
| 16 | 38.01 / 32.05 | **+18.6%** | 27.70 / 27.84 | **−0.5%** |
| 32 | 63.45 / 50.90 | **+24.7%** | 44.81 / 44.98 | **−0.4%** |
| 48 | 96.77 / 78.20 | **+23.7%** | 69.88 / 70.76 | **−1.2%** |

(The reported column matches the study's original report-3 anchor to the decimal — this *is* gray's
report 3, now decomposed.)

**Report 4 (FlashInfer, ~39k shared prefix, bs8, graph ON, interleaved paired, gray, 2026-06-19), 635 steps/run:**

| pair | reported ps1/ps128 | steady ps1/ps128 |
|---|---|---|
| p1 | 37.67 / 34.82 (**+8.2%**) | 32.21 / 32.44 (**−0.7%**) |
| p2 | 38.10 / 35.14 (**+8.4%**) | 32.54 / 32.59 (**−0.2%**) |
| p3 | 38.72 / 35.47 (**+9.2%**) | 32.74 / 32.98 (**−0.7%**) |

Three consequences:

1. **Both anchor results are the same artifact**, proven from gray's own data: the harness's
   `(end − first_token)/(tokens−1)` folds the cold-shared-prefix **admission ramp** into "decode," and the
   ramp is larger at ps1 (report 4: ps1 charged +5.5–6.0 ms/tok of ramp vs ps128's +2.4–2.6; the
   *difference* is the +8.6%). Gray's prefill→full-batch-decode wall: 8–9 s (ps1) vs 7–8 s (ps128),
   all repeats.
2. **Contention is retracted for report 4 too.** Gray's steady-state was never inflated. Note the paired
   protocol reproduced +8.6% consistently precisely *because* the artifact is deterministic — pairing
   defeats contention, not formula artifacts. (My intermediate claim "report 4 = contention," based on one
   flat re-run at a different build/prompt, is withdrawn; the re-run merely failed to reproduce the
   *artifact*, which is config/build-sensitive, and steady-state was flat in both.)
3. **Commit-identity correction:** report 14 called the phastform rebuild "gray's *exact* commit
   `0eded9e208`." Gray's own ServerArgs use the *old* `cuda_graph_max_bs` field; `0eded9e208` uses the
   renamed `cuda_graph_max_bs_decode`. So the rebuild is a **behaviorally-matched June-era commit, not
   gray's literal checkout** (it reproduces gray's reported artifact *and* gray's steady-state within ~2%).
   With gray's own logs now carrying the proof, the rebuild is corroborative, not load-bearing.

## 2. Report 7 reconciled — the shared-prefix kernel is OFF the wall (report 14's wording corrected)

ncu of the **exact** report-3 operating point (Triton `_fwd_grouped_kernel_stage1`, shared KV, B32,
L=9663, 16q/8kv, cold `--cache-control all`), plus the pre-existing kv6144 cells:

`contig` = a ps1 fresh (fully contiguous) layout, `block128` = ps128's 128-run layout, `scatter` = a ps1
maximally-fragmented (randperm) layout. `ps1-vs-ps128` under scatter = **scatter − block128** (single-shot
ncu after warmup; µs cells carry ~1% noise):

| cell (shared, Triton) | contig | block128 (≈ps128) | scatter (ps1-frag) | **ps1-frag vs ps128** | DRAM% | L2-hit% | sec/req |
|---|---:|---:|---:|---:|---:|---:|---:|
| B32/L9663 | 1.368 ms | 1.386 ms | 1.393 ms | **+0.5%** | 7.1–7.3 | 96.6 | 8.55 |
| B32/L6144 | 963 µs | 962 µs | 984 µs | **+2.3%** | 6.4 | 96.6 | 8.39 |
| B8/L6144 | 254 µs | 245 µs | 259 µs | **+6.0%** | 23.0 | 87.3 | 8.39 |

- **Report 7 is confirmed at the exact cell**: shared-prefix decode runs at **7% of DRAM peak** — nowhere
  near the bandwidth wall. L2-hit follows report 12's reuse law exactly ((R−1)/R: 31/32 = 96.9% at B32,
  7/8 = 87.5% at B8).
- **Report 14's "the kernel is at the DRAM wall here" claim is corrected**: that arithmetic divided
  demanded bytes by DRAM bandwidth while ignoring the ~97% L2 reuse. The correct — and simpler — reason
  `t_attn(ps1) = t_attn(ps128)` (in-engine timing hook, 1.574 vs 1.586 ms/layer) is **identical work**: the
  engine's shared-prefix `kv_indices` are contiguous at both page sizes and the kernel is page-agnostic. No
  "hiding" mechanism is needed.
- **Even forced synthetic scatter is bounded and batch-dependent** (+0.5% at bs32/9.7k → +6% at bs8),
  `sectors/request` unchanged (latency, not coalescing). These are single-shot microbench values with
  visible noise (bs8 `block128 < contig` is itself sub-noise) — treat them as an order-of-magnitude *ceiling*,
  not precise; and note this scatter layout does not occur for a real fresh prefix (§3).

## 3. The constructed break-attempt — how big can a *real* e2e `ps1` penalty get?

Chuyue's constructive question: bandwidth is *not* saturated off the wall — so build the best case for
`p_e = f_attn · p_k` and measure it honestly. Design (all amplifiers stacked):

- **Triton** (the scatter-sensitive backend), **shared prefix** ~6.1k tokens (L2-reuse regime, f_attn
  large), **bs32**, CUDA graph ON.
- **Genuinely fragmented ps1 pool** — the one lever that changes the kernel's work. Construction mirrors
  real serving: waves of 16 *concurrent* decodes interleave their KV token-by-token (round-robin per-step
  allocation); the waves **exceed the pool** so LRU eviction frees token-strided regions; the shared prefix
  then allocates from that fragmented free list.
- **Fragmentation is *verified at the exact measured config*, not assumed** — dumping the phase-B decode
  `kv_indices` via the `.pth` hook in an eager twin arm (mem 0.85, churn 3×16×2048): the bs32 shared prefix
  comes out **`frac_contig = 0.027`, 5928 runs, max_run 86** (97% token-scattered). A fresh control is
  `frac_contig ≈ 1.0` (1 run). *(Important negative sub-result: fragmentation requires pool overcommit — a
  lighter churn that fit the pool without eviction left the prefix `frac_contig = 0.998`, i.e. the allocator
  hands a fresh prefix a contiguous run unless eviction forces holes. So a scattered shared prefix is
  reachable but not the default.)*
- **Metric: steady-state per-step decode from the engine's own logs** (the whole point of this saga; graph
  ON). Controls: fresh-pool arms; FlashInfer arms (scatter-robust).

**Results** (Qwen3-VL-2B, Triton/FI, shared ~6.1k prefix, bs32, graph ON, 765 steps/arm):

| arm | prefix `frac_contig` | steady ps1 | steady ps128 | **ps1 − ps128** |
|---|---:|---:|---:|---:|
| Triton **fresh** | ≈1.0 (contiguous) | 33.45 | 33.89 | **−1.28%** |
| Triton **fragmented** | **0.027** (verified) | 33.57 | 33.99 | **−1.25%** |
| FlashInfer fragmented | — | 25.12 | 25.13 | **−0.04%** |

- **Even with the prefix verified 97%-scattered, ps1 is −1.25% (faster) than ps128.** Fragmenting the prefix
  adds only **+0.35%** to ps1's own step (33.45 → 33.57) — the real-layout scatter cost, consistent with the
  ≤+2% microbench at bs32. It does not overcome ps1's intrinsic **−1.28%** head start.
- **Source-consistent explanation for why ps1 starts ahead:** ps128 runs an extra per-step paged-allocator
  kernel (`alloc_decode_kernel`, grid=bs) that ps1 (a bare `free_pages[:n]` slice) skips in the traced
  `0eded9e208` source. This is consistent with an order-1% offset, but no timed component decomposition
  isolates it. The −1.25% e2e result and 0.027 contiguity are measured; the allocator attribution is inferred.
- **FlashInfer is flat** (−0.04%), as expected (its plan hard-codes page_size=1; scatter-robust, reports
  10/11) — confirming the whole effect is at most a small Triton-specific kernel term, offset e2e.

## 4. Answers

- **"Is report 4 real?"** The +8.6% number is real and reproducible *as a number*, but gray's own logs show
  it was never a decode effect: steady-state was page-invariant (−0.2…−0.7%) in all three of gray's paired
  runs. It is the admission-ramp/TPOT-formula artifact — same family as report 3. My earlier "contention"
  explanation is retracted; no contention is needed (or supported) by the data.
- **"Report 7 vs report 14 — is the shared-prefix kernel at the bandwidth wall?"** No. Report 7 was right:
  measured at the exact report-3 cell, DRAM is at 7% with 96.6% L2-hit. Report 14's "wall" wording is
  corrected; page-invariance follows from identical work, not from a wall hiding a penalty.
- **"Did the tested construction produce a real off-wall scenario where ps1 loses e2e like +8.6%?"**
  **No.** With the tested amplifiers stacked and the ps1 shared prefix *verified fragmented* to
  `frac_contig = 0.027`, steady-state
  ps1 is **−1.25%** (faster than ps128). The only page-dependent kernel term tested — synthetic
  scattered-index latency —
  is ≤+6% (bs8) / ≤+2% (bs32) even under synthetic worst case, and in the real engine it is (a) mostly
  absent (a fresh prefix allocates contiguously) and (b) when forced, offset by ps1's cheaper per-step
  allocator. FlashInfer is flat. So the +8.6%-style number has **no decode-side realization**; it was the
  admission-ramp measurement artifact all along. For this hardware, build, and tested construction, the
  measured end-to-end result is **ps1 ≤ ps128**; this is a bound on the experiment performed, not a universal
  ceiling over all allocator states or workloads. The largest measured *kernel-only* term in this matrix is
  the bounded ≤6% synthetic scattered-index latency, which did not reach TPOT here.

## 5. Evidence classes (per the rigor complaint — what is measured vs inferred)

| Claim | Class | Basis |
|---|---|---|
| Gray reports 3/4 steady-state page-invariant | **Measured** | gray's own per-step logs (635 steps/cell, both reports) |
| The +25%/+8.6% = admission-ramp artifact | **Measured** | reported−steady decomposition + ramp walls in the same logs |
| No contention in either result | **Measured (negative)** | steady-state uninflated in every gray run |
| Rebuild = gray's exact commit | **Retracted** | ServerArgs field mismatch (old vs new graph-field name) |
| Shared 9.7k cell off-wall (DRAM 7%) | **Measured** | ncu, exact cell, cold |
| `t_attn` page-invariant; identical-work explanation | **Measured timing + source-supported inference** | kv_indices contiguity dump + page-agnostic kernel source + timing hook |
| Kernel scatter ceiling (ps1-frag vs ps128) = +0.5% (bs32) … +6% (bs8) | **Measured (single-shot, ~1% noise)** | ncu duration, synthetic worst-case scatter, 3 cells |
| Real fresh prefix allocates contiguously; fragmentable only under pool overcommit | **Measured** | kvidx contiguity dump: fresh/light-churn ≈1.0, heavy-churn 0.027 |
| e2e: even a verified-fragmented (0.027) prefix → ps1 −1.25% (no penalty) | **Measured** | §3, 765 steps/arm, graph ON |
| ps1's head start = ps128's extra per-step alloc kernel | **Inferred (source)** | `alloc_decode_kernel` in commit `0eded9e208`; consistent with measured e2e |

**Companion:** [report 14](../14-shared-prefix-benchmark-artifact/report.md) (the
mechanism; its "wall" wording and report-4-contention claim are corrected by this report),
[report 7](../07-shared-prefix-mechanism/report.md) (vindicated),
[report 12](../12-reuse-not-residency/report.md) (reuse law, confirmed at
the exact cell), [report 13](../13-true-batch-off-wall/report.md) (the distinct
off-wall kernel latency, unchanged).
