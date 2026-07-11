# Report 13 — Constructing a TRUE batch that is **off the DRAM wall** (bs2×1k, DRAM < 50%), and what `page_size` costs there

> Ask (Chuyue, 2026-07-11): *"Construct a case in `bench_one_batch` (individual-request mode) where DRAM is
> **not** the bottleneck — as small as batch 2 × length 1k, as long as DRAM is under 50% (it is ~95% today in
> individual requests). Then comprehensively test large vs small page sizes in that constructed scenario."*
> Reports 7/8/12 established that a true batch is DRAM-bandwidth-bound (~95% of peak, L2-hit ≈ 0) and that
> *shrinking the KV footprint* does not lift it. This report constructs the case that **does** exist — small
> **work** (batch × context × KV-heads) leaves the wall via *under-parallelism*, not caching — demonstrates it
> **inside `bench_one_batch` itself** with ncu, and then runs the full page-size battery in that regime.

| | |
|---|---|
| **Question** | (1) Can individual-request decode sit at **DRAM < 50%**, at sizes as small as B2/L1k? (2) In that regime, do small pages finally lose? |
| **Answer** | (1) **Yes — constructed and verified in-engine.** Qwen2.5-3B (GQA-2) at **B2/L1024**: decode-attn kernel = **40.5–40.8%** DRAM in the microbench and **in-engine (ncu inside `bench_one_batch`, §1 table)**. The governing variable is total decode work **B·L·kv_heads ≲ ~4k head-tokens** (≈2 MB/layer), not footprint per se: the kernel runs out of CTAs (waves/SM ≤ 0.11, occupancy 8%), so DRAM idles — **L2-hit stays ≤ ~8–17% (cold)**, i.e. off-wall ≠ cache-resident. (2) **Page size still never costs ≥5% where it can be measured reliably**, *but* the off-wall regime exposes a real, reproducible **kernel-level** `ps1` cost that the wall used to hide: **+5.3%/+5.5% (two independent ncu rounds) at the smallest cells**, +1–3% through the transition, ~0 at the wall — with DRAM-read bytes identical (ratio 1.001) → it is per-token **index-walk latency**, the GQA analog of report 11's MLA mechanism. At wall-clock it is invisible (a ~0.13 ms two-kernel launch floor swamps 8–25 µs kernels), and in engine TPOT it is weight-bound-invisible — the 105-run graph-ON TPOT sweep has **`ps1` fastest in every cell** (−1…−3.6% vs ps128, both backends; no page ever ≥5% slower). Worst-case `ps1` fragmentation (random token order): **≤ +3.1%** kernel time off-wall, `sectors/request` identical → DRAM-access latency, not coalescing. |
| **Model shapes** | GQA-8 = Qwen3-VL-2B (16q/8kv/128d, 4096 B/tok/layer) · **GQA-2 = Qwen2.5-3B (16q/2kv/128d, 1024 B/tok/layer)** — the report-5 "most page-vulnerable" real model, whose shape is what makes bs2×1k off-wall |
| **HW / method** | RTX 5060 Ti (sm120, 16 GB, 32 MB L2, ~448 GB/s) on **phastform** · ncu 2025.3.1 as root, `--cache-control all` (cold = faithful, report 12), `--single`-launch isolation; metric set = report-12 + `launch__grid_size/waves` + sectors/request · CUDA-event ladders ×3 rounds · engine = `sglang.bench_one_batch` (true batch by construction), TPOT graph-ON + **ncu attached to eager decode steps** (new instrument) |
| **Data** | `offline_batch_results/offwall_profile/` (123 ncu cells + `rep2/` repro pass + ladders) · `offline_batch_results/bench_one_batch_offwall_5060ti/` (TPOT) · [SUDO_CHANGES.md](SUDO_CHANGES.md) |
| **Figure** | [fig_offwall.png](fig_offwall.png) |

---

## 1. Constructing the case (Phase 1) — small **work**, not small footprint, leaves the wall

Reports 8/12 killed the footprint route: a small *distinct* KV is still cold-streamed at 79–96% DRAM
(8 MB → 537 MB), because saturation needs only enough in-flight loads. What they did not map is the corner
where the kernel **cannot issue enough parallel work to saturate HBM**. The (B, L) × {GQA-8, GQA-2} ncu grid
(cold, distinct, ps128; 123 cells) maps it:

**DRAM % of peak (cold, ps128) — GQA-8 (Qwen3-VL-2B shape):**

| B \ L | 128 | 256 | 512 | 1024 | 2048 | 4096 | 6144 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **1** | **15.5** | **28.4** | 53.3 | 66.9 | 78.9 | 86.6 | — |
| **2** | **29.9** | 51.9 | 67.4 | 79.1 | 86.2 | 93.1 | 86†|
| **4** | 50.7 | 68.3 | 79.5 | 86.5 | 93.4 | 93.8 | — |
| **8** | 68.4 | 78.9 | 86.9 | 92.7 | 94.2 | **95.1** | **95.5** |
| **16/32** | 79/87 | 87/93 | 93/— | — | — | — | — |

**GQA-2 (Qwen2.5-3B shape):**

| B \ L | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---:|---:|---:|---:|---:|
| **1** | **11.5** | **22.5** | **41.0** | 59.6 | — |
| **2** | **22.2** | **40.8 ← the constructed case** | 62.0 | 76.2 | 83.7 |
| **4** | **41.9** | 60.2 | 74.6 | 82.5 | — |
| **8** | 62.2 | 75.7 | 83.3 | 90.1 | 91.7 |

(† B2/L6144 hits FlashInfer's split cap differently; the row trend is unaffected.)

- **The two arms collapse onto one curve in total work `B·L·kv_heads`** (fig panel B): the 50% crossing sits
  at ≈ **4k head-tokens ≈ 2 MB of KV per layer** on this GPU, wherever it comes from. GQA-8 needs
  B·L ≤ ~512 (B2 → L ≤ 256); GQA-2 buys 4× → **B2/L1024 = 40.5–40.8%**, B1/L2048 = 41%, B4/L512 = 42%.
  18 cold cells land under the 50% bar (min 11.3% at GQA-2 B1/L512).
- **Mechanism = under-parallelism (latency-bound), not caching.** At B2/L1024-GQA-2 the plan launches
  **32 CTAs on 36 SMs** (waves/SM 0.11, occupancy 8.3%, SM 27%); the kernel takes ~12 µs to stream 2.1 MB
  (178 GB/s of 448). Cold **L2-hit stays 0.1–17%** across every cell (the higher single digits at tiny GQA-2
  cells are index/q/output overhead traffic, not KV reuse — report 12's law: distinct ⇒ R=1 ⇒ ~0), so this is
  **not** the shared-prefix cache regime of report 7: DRAM is quiet because the SMs can't demand bytes fast
  enough, not because the bytes come from L2. FlashInfer's split-KV *does* fight back (grid = kv_heads ×
  ⌈L/128⌉-ish chunks; it filled the GPU well enough that GQA-8 B2/L1024 still reaches 79%) — which is exactly
  why the report-8 footprint ladder never fell below 79%: at B8×8 KV heads there is always a wave of CTAs.
  You must starve **B × kv_heads × split** below ~1 wave to get under 50%.
- **Warm control** (`--cache-control none`, the report-8/12 residency artifact): the flagship cells stay
  off-wall warm too (GQA-2 B2/L1024: 30.7% DRAM / 37% L2-hit) — the classification is robust to cache state.
- The B2/L1024 **GQA-8** cell the ask literally named lands at **79%** — off the 95% ceiling but not under 50:
  with 8 KV heads there is too much head-parallelism. The user's size is achievable **with the real model that
  has the right shape** (Qwen2.5-3B, GQA-2) — or at GQA-8 by dropping to B2/L≤256/B1/L≤384.

### Verified **inside `bench_one_batch`** (the literal ask) — and a kernel-dispatch surprise

ncu attached to real engine decode steps (eager, `-k` on the decode-attention kernels, two full mid-run
steps, per-launch median over 56–72 launches — a new instrument for this study; §4 method notes). Finding
the right `-k` exposed a fact that *changes the constructed cell's regime for the better*: sglang's
`should_use_tensor_core()` dispatches GQA **group_size ≥ 4** models (Qwen2.5-3B = 16q/2kv = group 8) to
FlashInfer's **tensor-core (prefill-template) kernel** (`BatchPrefillWithPagedKVCacheKernel`), not the
CUDA-core decode kernel the microbench grid above profiles:

| engine cell (true batch) | kernel | dur µs | **DRAM %** | L2-hit % | GB/s | grid | waves | reading |
|---|---|---:|---:|---:|---:|---:|---:|---|
| **Qwen2.5-3B B2/L1024, ps128** | tensor-core plan (2 launches/layer, grids 40+72) | 36.7 med | **8.1** | 59.9 | 35 | 40/72 | 0.78 | **off-wall in the engine**; step accounting closes: Σread 92.5 MB ≈ 36×(2.1 KV+ovh) |
| **Qwen2.5-3B B2/L1024, ps1** | tensor-core plan | 36.8 med | **8.0** | 59.9 | 35 | 40/72 | 0.78 | byte-identical to ps128 (Σread 92.5 = 92.5 MB) |
| Qwen3-VL-2B B2/L1024, ps128 | CUDA-core | 26.5 | 72.6 | 2.1 | 320 | 144 | 0.44 | GQA-8 contrast: wall's shoulder; Σread 474.7 MB = 2 steps × 28 × 8.4 MB ✓ |
| Qwen3-VL-2B B2/L1024, ps1 | CUDA-core | 26.5 | 72.7 | 2.1 | 320 | 144 | 0.44 | counters byte-identical vs ps128 (Σread 474.7 = 474.7) |
| Qwen2.5-3B B8/L4096, ps128 | tensor-core plan | 38.6 med | 7.7 | 59.9 | 33 | 40/72 | 0.78 | every captured launch off-wall, but Σread (92.6 MB) covers only ~8% of the step's 1.2 GB KV — the plan chunks/persists work across launches, so **per-step attribution is unresolved**; the wall-side tc reference is carried by the microbench tc arm instead (§2a-tc) |

**The constructed scenario holds — emphatically — in the literal `bench_one_batch`:** a stock SGLang decode
of Qwen2.5-3B at batch 2 × 1k context runs its attention kernel at **8% of peak DRAM** (35 GB/s of 448).
Three additional facts fall out:

1. **In-engine, `--page-size` changes nothing at the kernel** (Qwen2.5-3B: 8.0 ≈ 8.1%, Σread 92.5 = 92.5 MB;
   Qwen3-VL-2B: 72.6 ≈ 72.7%, Σread 474.7 = 474.7 MB — counters byte-identical) — the counter-level
   confirmation of report 10's source reading (sglang always plans FlashInfer decode with page_size=1 and a
   per-token index; `--page-size` only moves allocator contiguity).
2. **The engine's tensor-core decode plan is expensive at this work size.** It issues **two** launches per
   layer (grids 40 + 72; the split-merge kernel is separate and excluded), each ~30–44 µs — ~6× the attention
   time of *either* single-launch microbench wrapper on the identical cell (CUDA-core 11.9 µs @ 40.8% DRAM;
   `use_tensor_cores=True` wrapper 11.2 µs @ 43.3% DRAM, one launch, grid 32). The dispatch rule
   (`group_size ≥ 4 → tensor cores`) has no work/batch term, and the engine-side plan it triggers is tuned
   for throughput cells — at agentic operating points it multiplies attention-kernel time several-fold
   (still only ~74 µs/layer against a weight-bound ~ms step, so TPOT barely notices; §2c). Same genre as
   vLLM's no-cache-term `use_cascade_attention` heuristic (proposal A's impact demo).
3. **Its L2-hit ≈ 60% (vs 7% for the microbench tc wrapper and 2% for CUDA-core) is intra-launch tile
   re-reading of the KV through L2**, not cross-request caching: per launch the kernel demands ~3× the cell's
   distinct KV (per-launch DRAM read 1.22–1.35 MB at 60% hit ⇒ ~3.2 MB demanded vs 2.1 MB KV) — the engine
   plan's Q/KV tiling re-traverses KV where the microbench plans hold their reuse in SMEM/registers. Report
   12 §7's absorbed-pre-L2 rule, made visible by a *plan* change on the same workload. Accounting check: the
   72 captured launches = one full 36-layer step; their summed DRAM read ≈ 92 MB ≈ 36 × (2.1 MB KV +
   overheads) — the whole step's attention traffic at 8% of peak.

## 2. What page size costs there (Phase 2) — the comprehensive battery

Four instruments, per study convention: the **native page ladder** (FlashInfer `plan(page_size)` — what vLLM/
TRT-LLM-style engines exercise), **worst-case fragmentation** (sglang-faithful per-token index, contig/
block128/scatter values — report 10's real lever), **ncu attribution** for both, and the **engine tiers**.

### 2a. Native `page_size` ladder — a real, hidden-until-now kernel effect; still < 5% wall-clock everywhere

- **Kernel level (ncu duration, cold; deterministic):** `ps1` costs **+1–5%** vs `ps128` through the
  off-wall/transition region on GQA-8 — **reproducible**: B1/L512 = **+5.3% / +5.5%** in two fully independent
  rounds (`rep2/`), B2/L128 +4.7/+9.6%, B2/L1024 +2.3/+1.1%, decaying to **~0 at the wall** (B8/L4096 +0.3/0.0%).
  `dram__bytes_op_read` ps1/ps128 = **1.001–1.005** everywhere → *the bytes are identical*; the cost is the
  **128× longer per-token `kv_indices` walk**, pure extra latency — visible only when the kernel is not
  bandwidth-saturated. This is the **GQA analog of report 11's MLA result** (scatter cost exposed at
  15–38% DRAM), now shown for the page-size knob itself: **the wall was what made `ps1` free in reports 5/8/10;
  take the wall away and the index walk has a small real price.**
  Caveats stated plainly: these are 8–25 µs kernels, so ±0.5 µs launch quanta matter — GQA-2's off-wall cells
  (11–16 µs, even fewer CTAs) show sign-mixed ±3–6% ratios rather than a clean trend; and at big-L cells the
  ps1 plan sometimes *wins* ~1–2% by splitting finer (grid 288–320 vs 256 — more CTAs = better waves). Both
  effects are bounded single-digit.
- **Wall-clock level (CUDA-event, 3 rounds × 8 pages × 45 cells):** the off-wall cells all sit on a
  **~0.128 ms two-kernel launch floor** (decode + split-merge kernel + dispatch ≈ 5–10× the kernel itself), so
  the ladder reads **±4% sign-unstable noise** — median `ps1`-vs-`ps128` spans −3.96%…+3.89% with the best page
  jumping randomly between ps1…ps128 across cells and rounds; at the DRAM-bound anchors it tightens to the
  familiar ≤1.4% flat. **No page choice, large or small, costs ≥5% wall-clock in the constructed regime** —
  the kernel-level +5% is hidden a second time, now by launch overhead instead of bandwidth.
- **XQA** (page ≥16 only): flat, 0.4–4.9% spreads at the same floor — page-insensitive off-wall too.

**§2a-tc — the tensor-core arm (the kernel sglang actually dispatches for GQA-2 models).** Re-running the
GQA-2 grid + ladder with `--tensor-cores` (added flag; kernel = `BatchPrefillWithPagedKVCacheKernel`, one
launch + merge):
- Same off-wall boundary: **B2/L1024-tc = 43.1–43.3% DRAM** (11.4 µs, grid 32, waves 0.44), B1/L2048 = 43.5%,
  qualifying cells identical to the CUDA-core arm — and the tc kernel **does reach the wall at big work**
  (B8/L8192 = 91.6%, 171 µs, 394 GB/s), so sglang's `group≥4` dispatch is sound at throughput sizes and
  suboptimal only in the off-wall corner (§1 point 2).
- Same page law: tc `ps1` reads **+1.1…+6.9%** kernel time vs `ps128` in the off-wall/transition cells
  (B1/L1024 +6.6%, B2/L2048 +4.1%; DRAM-read ratio 1.004 → still pure index-walk latency), ~0 at the wall —
  **with a contention caveat**: the tc-grid window partially overlapped engine TPOT runs (the §2b phantom's
  window), so treat these ratios as indicative; the *clean* quantitative anchor for the off-wall `ps1` kernel
  cost remains the CUDA-core arm's independently-reproduced +5.3/+5.5% (B1/L512, `rep2/`). Wall-clock: the
  same ~0.135–0.14 ms floor; min-across-rounds `ps1`-vs-`ps128` within ±4.2% (T2c; min = the
  contention-robust estimator — report-5 protocol).

### 2b. Worst-case `ps1` fragmentation — bounded, and *smaller* than MLA's, for the predicted reason

`kv_indices` values contig vs block128 vs **fully scattered** (maximal ps1 fragmentation, report-10 tool):

| cell (cold ncu) | DRAM% | scatter vs contig (kernel) | sectors/request |
|---|---:|---:|---|
| GQA-8 B1/L512 (off-wall) | 49 | **+9.3%**‡ | 12.70 = 12.70 |
| GQA-8 B2/L1024 | 78 | +1.6% | 12.66 = 12.66 |
| GQA-2 B2/L1024 (**the case**) | 41 | **+2.5%** | 12.66 = 12.66 |
| GQA-2 B8/L1024 | 74 | +3.1% | 12.65 = 12.65 |
| GQA-2 B8/L8192 | 93 | +1.4% | 14.68 = 14.68 |
| GQA-8 B8/L4096 (wall) | 95 | +1.1% | 15.09 = 15.09 |

Wall-clock (3 rounds): −1.1%…+1.8%, i.e. floor-noise off-wall, +1.8% consistent at the wall-side anchor.
`sectors/request` **identical to the second decimal** in every pair → not coalescing; random token order costs
**DRAM-access latency** (row-buffer/TLB), exposed off the wall but blunted by GQA's **4 KB-per-token burst**
(vs MLA's 1152 B → +4%; ‡ the +9.3% sits on a 10 µs kernel = one ~1 µs quantum, treat as ≤O(10%) bound, not a
point estimate). `block128` ≈ contig (±0.5%) everywhere — **realistic paged layouts are free off-wall too.**
The tensor-core kernel behaves the same: scatter ≈ contig at the constructed cell (11.3 vs 11.6 µs, clean
repro pass; wall-clock ±0.5%), ≤ +1.6% consistent at its B8/L8192 wall anchor.

> **A phantom worth recording (proposal-B exhibit).** The *first* pagecost-tc ncu window returned
> `scatter = 24.1 µs vs contig 11.4 µs` — an apparent **2.11×**, which would have been the study's first ≥5%
> distinct-KV page effect. It was a **measurement artifact**: that window overlapped a concurrently running
> engine TPOT process (an idle-gate race let two runners share the GPU), and GPU sharing time-dilates the
> profiled kernel — signature: duration ×2.1 with *identical* bytes and sectors/request and DRAM% halved
> (42.8 → 21.8). An independent clean pass (`rep2/`) refuted it (scatter 11.3 vs contig 11.6 µs), and the
> wall-clock rounds agree (±0.5%). Same lesson as report 9's preemption phantom: **verify exclusivity before
> believing a page effect** — contention manufactured a 2× "penalty" from nothing.

### 2c. Engine tiers — TPOT can't see any of it (and why that's the right answer)

`bench_one_batch`, graph ON, 3 rounds paired (105 runs, 0 errors), FlashInfer {1,2,4,8,16,32,128} +
Triton {1,8,32,128} — decode TPOT ms/token, median across rounds:

| cell | backend | ps1 | ps8 | ps32 | ps128 | **ps1 vs ps128 (med / min)** |
|---|---|---:|---:|---:|---:|---:|
| **Qwen2.5-3B B2/L1024** (the case) | FlashInfer | **17.54*** | 18.23 | 18.02 | 18.13 | **−3.3% / −3.1%** |
| | Triton | **17.43*** | 17.74 | 17.80 | 17.73 | −1.7% / −2.1% |
| Qwen2.5-3B B1/L512 | FlashInfer | **17.60*** | 18.26 | 17.90 | 17.82 | −1.3% / −1.8% |
| | Triton | **17.21*** | 17.92 | 17.68 | 17.75 | −3.0% / −2.1% |
| Qwen2.5-3B B8/L4096 (wall ref) | FlashInfer | **20.60*** | 20.95 | 21.00 | 21.28 | −3.2% / −3.1% |
| | Triton | **20.31*** | 20.58 | 20.56 | 20.53 | −1.1% / −1.5% |
| Qwen3-VL-2B B2/L1024 | FlashInfer | **11.46*** | 11.60 | 11.60 | 11.88 | −3.6% / −3.1% |
| | Triton | **11.34*** | 11.46 | 11.68 | 11.47 | −1.1% / −0.6% |

(Full FlashInfer fine ladder at the flagship cell: 17.54–18.23 ms across all seven pages, spread ~4%.)

- **`ps1` is the fastest page in every cell, on both backends** — and **no page, large or small, is ever ≥5%
  slower**. The −1…−3.6% `ps1` lead is the report-5/8 engine-level sliver (allocator/metadata path), *not*
  the attention kernel: T7 shows the in-engine kernel counters are byte-identical across `--page-size`.
- Why flat-with-a-tiny-ps1-lead is the *correct* outcome, not a null: per-step DRAM traffic here is
  **weights 6.2 GB vs KV 0.07 GB** (KV ≈ 1% of traffic) — the report-8 roofline; on top of that sits the §2a
  launch floor, ~10× the attention kernel; and in eager terms the whole attention kernel is 12–37 µs of a
  ~18 ms step (~0.2%). **In a true batch the page knob remains free at the engine level even when the
  attention kernel is off the DRAM wall.**
- Note: sglang's decode path always builds the **per-token (ps1-shaped) index** and plans with page_size=1
  regardless of `--page-size` (report 10 ground truth; confirmed here by identical in-engine counters across
  `--page-size`) — so the native-ladder effect of §2a is a statement about engines that pass real page sizes
  to the kernel plan (vLLM-FlashInfer, TRT-LLM), and an (unrealizable ~1–5%-of-25µs) upper bound on what
  sglang could shave off-wall by page-quantizing its decode index.

## 3. What this adds to the study (and to proposal A)

1. **R2 is not MLA-exclusive.** The study's regime taxonomy had [R2] "R=1, latency-bound" occupied only by
   MLA (report 11). This report shows **standard GQA enters R2 at small work** — `B·L·kv_heads ≲ 4k
   head-tokens` on a 36-SM part — i.e. exactly the agentic-browser operating point (1–4 concurrent requests,
   ~1k contexts, GQA-2-class small models). The off-wall boundary is a *parallelism* threshold, not a
   footprint threshold — reconciling with report 8 (footprint ladders at B8×8 heads never left the wall
   because 64+ CTAs always saturate).
2. **The page pattern's cost is regime-gated in *both* directions, and now measured on the knob itself.**
   On-wall: hidden by bandwidth (reports 5/8/10, ≤1.3%). Off-wall: **exposed** — +5.3/+5.5% reproducible
   kernel cost for the per-token index at B1/L512, +2.5% worst-case scatter at the constructed bs2×1k cell —
   but **bounded single-digit** and then re-hidden by the launch floor (µs-kernels) and the weight stream
   (TPOT). The ≥5% *practical* penalty remains exclusive to the shared-prefix cache-bound regime + a
   gather-sensitive kernel (reports 3/7). The final rule sharpens to: **`ps1`'s decode cost is only ever
   *measurable* off the wall, and only ever *matters* under reuse.**
3. **Methodologically:** ncu-inside-`bench_one_batch` (eager, `-k` filter, mid-run launch windows) is now a
   working instrument (T7), closing the gap between weightless microbench and engine TPOT; and the wall-clock
   floor finding (0.128 ms ≈ 5–10× kernel time at small batch) quantifies why sub-50µs kernel effects are
   unmeasurable by CUDA-event ladders — both feed proposal B's artifact taxonomy (a sixth class: *launch-floor
   masking*).

## 4. Method / controls / honesty

- **Grid**: `bench_xqa.py --kv-mode distinct` (contiguous pages per seq), cold ncu per study convention;
  `--num-q-heads/--num-kv-heads` flags added (defaults preserve every prior invocation byte-for-byte; this is
  report 12 §9's named future work). GQA-2 arm = real Qwen2.5-3B attention shape (config verified: 36 L,
  16q/2kv/128d).
- **Reproducibility**: the near-threshold ps1 ratios re-measured in an independent second ncu pass (`rep2/`,
  fresh processes): B1/L512 1.053→1.055, B8/L4096 1.003→1.000. Sub-16 µs cells carry ±1 quantum noise and are
  reported as bounds, not points.
- **Engine ncu**: eager mode (`--disable-cuda-graph`) so launches are visible; skip = 20×n_layers lands past
  the warmup run's 15 decode steps (bench_one_batch warms with `min(32, output_len)`); count = 2×n_layers = two
  full steps; per-launch median over 56–72 launches. Cold `--cache-control all` — in a real multi-layer step
  the inter-layer traffic evicts L2 anyway (report 8). Graph-ON TPOT tier is measured separately (§2c) — the
  page comparison itself is graph-ON, per report-5 convention.
- **True batch verified**: `bench_one_batch` synthesizes B independent random sequences (report 5); engine
  logs show `Capture cuda graph bs [B]` (TPOT tier) and per-cell pool fit; no radix path exists in this tool.
- **Idle-gated, skip-guarded, audit-logged** runners; password never on disk; all root-created files chowned
  back ([SUDO_CHANGES.md](SUDO_CHANGES.md)).
- **TPOT tier execution notes (full disclosure):** the first launch of `bob_offwall.sh` died post-graph-capture
  on every FlashInfer cell (`FileNotFoundError: 'ninja'` — the venv bin was missing from PATH, so sglang's
  runtime JIT failed; the phastform gotcha) and its window overlapped the ncu chain via an idle-gate race
  (the §2b phantom's cause). The runner was fixed (PATH + mem-fraction floor 0.55) and fully re-run; the final
  data set is 105/105 runs with pages back-to-back per round, of which only 4 first-launch Triton JSONLs
  predate the fix (kept — min-across-rounds absorbs any inflation, and their cells were re-measured in 2 clean
  rounds). All ncu numbers quoted in §1–§2 come from windows with no concurrent process (verified; the one
  contaminated window is documented and its cells re-measured in `rep2/`).
- **Scope**: one consumer GPU (36 SMs, 448 GB/s, 32 MB L2). The 4k-head-token boundary scales with SM count ×
  per-SM bandwidth demand — on an H100 (132 SMs, 3.35 TB/s) the off-wall region is *larger* in B·L (more SMs to
  starve, more bandwidth to saturate); mapping it there is proposal A's roadmap item. Triton engine-kernel
  counters off-wall not measured (same scope note as reports 7/8: its token-level gather needs the engine
  allocator; TPOT tier covers it and is flat).

## 5. Reproduce

```bash
# phastform (RTX 5060 Ti, sm120). Latency + fragmentation ladders (user, ~15 min):
bash page_size_study/scripts/bench_offwall_lat.sh
# ncu grid + patterns + warm + GQA-2 arm (root, ~35 min, 123 cells; resumable):
echo PW | sudo -S -E bash ~/sglang_log/run_offwall_ncu_all.sh
# repro pass for the near-5% cells:
echo PW | sudo -S -E OUTDIR=.../offwall_profile/rep2 CACHE=all TOOL=xqa QH=16 KH=8 \
  CELLS="1 512|2 128|2 1024|1 1024|4 256|8 4096" PAGES="1 128" bash ~/sglang_log/profile_offwall_ncu.sh
# engine: ncu inside bench_one_batch (root, ~15 min) + TPOT page sweep (user, ~3 h):
echo PW | sudo -S -E bash ~/sglang_log/profile_bob_ncu.sh
bash ~/sglang_log/bob_offwall.sh
# locally:
python3 page_size_study/scripts/analyze_offwall.py       # tables T1–T7
python3 page_size_study/scripts/make_offwall_figure.py   # fig_offwall.png
```

Companion reports: [7](../report_7_sharedprefix_mechanism/report_7_sharedprefix_mechanism.md) (the wall and
the cache flip) · [8](../report_8_smallkv_truebatch/report_8_smallkv_truebatch.md) (footprint doesn't lift it)
· [10](../report_10_flashinfer_truebatch_pagecost/report_10_flashinfer_truebatch_pagecost.md) /
[11](../report_11_mla_scatter_pagecost/report_11_mla_scatter_pagecost.md) (scatter at/off the wall, MLA) ·
[12](../report_12_reuse_not_residency/report_12_reuse_not_residency.md) (reuse-not-residency; cold=faithful).
