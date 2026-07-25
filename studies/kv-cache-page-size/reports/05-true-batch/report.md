# Report 5 — `page_size` in the TRUE-batch regime (`bench_one_batch`)

> **Scope correction (2026-07-17; reports 15–16).** The measured FlashInfer/Triton true-batch cells remain
> valid. The earlier report-3/4 whole-call values were not shared-prefix decode penalties; their steady
> decode was page-flat. This report therefore supports only the tested FlashInfer/Triton true-batch scope,
> not a universal rule. Report 16 shows a page-aware XQA kernel can have a page effect in an off-wall shared
> regime.

> PI steer (2026-06-20): *"你可以跑 `bench_one_batch` mode … 这个是真正的batch"* — the earlier ≥5 % numbers
> came from a **shared-prefix (radix-ON) whole-call harness**, where B identical requests share one KV copy.
> Reports 14–15 later showed that those numbers were admission-ramp artifacts rather than steady-decode
> penalties. This report independently measures steady decode with **B genuinely independent sequences**.

| | |
|---|---|
| **Tool** | `python -m sglang.bench_one_batch` (SGLang's canonical decode microbenchmark) |
| **Model** | Qwen3-VL-2B-Instruct |
| **Hardware** | RTX 5060 Ti (16 GB, sm120) on `gray.cis.upenn.edu` |
| **Regime** | **TRUE batch** — B independent random sequences, each its own KV; **no shared prefix / radix** |
| **CUDA graph** | **ON** (captured at exactly the batch, `--cuda-graph-bs B`) |
| **Backends** | Triton, FlashInfer |
| **page_size** | {1, 8, 32, 128} |
| **Metric** | decode **TPOT = `median_decode_latency` × 1000** ms/token (median over `output_len−1`=63 steps) |
| **Data** | `studies/kv-cache-page-size/data/raw/bench_one_batch_5060ti/bob_*.jsonl` |
| **Figures** | [fig_bob_truebatch.png](fig_bob_truebatch.png) (main sweep) · [fig_bob_adversarial.png](fig_bob_adversarial.png) (adversarial break-search) |

---

## Headline

**In the measured FlashInfer/Triton true-batch cells, `page_size=1` is the *fastest-or-tied* page — at every
batch and context length.** The earlier whole-call ≥5% values do not represent a steady decode penalty.
A subsequent **adversarial
search** (next-but-one section) actively tried to break this on FlashInfer+graph — high batch to 512,
L2-resident KV, long single sequences, the most page-vulnerable GQA-2 model — and **found no
counter-example** (max `ps1` penalty +0.2 % across 19 cells).

- **Triton:** true-batch `ps1` vs `ps128` = **−0.2 % to −1.0 %** (ps1 fastest in these samples). Reports 3–4
  had reported **+11.5 % (bs8)** and **+24.6 % (bs32)** using the now-superseded whole-call metric.
- **FlashInfer:** true-batch `ps1` vs `ps128` = **−0.7 % to −1.2 %** (ps1 fastest in these samples). Report 4
  had reported **+8.6 %** in a long-context cell using that same non-equivalent whole-call metric.
- The true-batch page-size spread is tiny (≤ ~1.2 %). Small pages happen to be fastest in these runs, but
  later kernel checks found no page-padding over-read and put the difference at the noise/metadata floor.

---

## Original mechanism hypothesis and current interpretation

The original report attributed the contrast to a Triton gather/coalescing penalty. Reports 14–16 supersede
that attribution for the measured engine path: the studied FlashInfer/Triton steady work is page-invariant,
and reports 3/4's large contrast came from the whole-call metric. The valid result here is narrower:
independent-sequence FlashInfer/Triton cells were near-flat and did not show a ≥5% `ps1` penalty.

- **Shared prefix (radix ON):** B requests share one small KV copy, but each still *reads* the whole context
  every step → huge read count over a **tiny** footprint (e.g. bs32 @ 10k ≈ 320 k reads over a 10 k cache).
  The old whole-call metric amplified the cold shared-prefix admission ramp; same-run steady decode is flat.
- **True batch (this report):** B independent sequences = B× the **distinct** KV bytes. What fits in 16 GB is
  ≤ ~49 k *distinct* tokens, and decode is **bandwidth-bound over distinct memory**. No page-size-specific
  coalescing penalty was demonstrated; the measured ordering stays within roughly 1%.

For this SGLang build, FlashInfer's decode lowering uses a page-independent per-token index list. That
source fact is consistent with the measured flatness; it does not generalize to page-aware backends.

---

## Results — TRUE batch, CUDA graph ON (decode TPOT, ms/token)

Robust estimator: **minimum across the 2 rounds** (decode-latency contention can only *inflate* a sample, so
the minimum is the cleanest; this neutralises one B8/L4096 round-1 contention blip without hand-dropping data).
Full per-round values are in the JSONL; the paired-mean±std agrees (see "Robustness" below).

### Triton
| batch / input_len | KV (B×L) | ps1 | ps8 | ps32 | ps128 | **ps1 vs ps128** | best |
|---|---:|---:|---:|---:|---:|---:|:--:|
| B8 / L4096  | 32 k | 19.03 | 19.19 | 19.18 | 19.21 | **−1.0 %** | ps1 |
| B16 / L2048 | 32 k | 19.30 | 19.46 | 19.48 | 19.48 | **−1.0 %** | ps1 |
| B32 / L1024 | 32 k | 20.08 | 20.28 | 20.27 | 20.28 | **−1.0 %** | ps1 |
| B16 / L3072 | 49 k | 23.73 | 23.83 | 23.81 | 23.78 | **−0.2 %** | ps1 |
| B8 / L6144  | 49 k | 23.38 | 23.57 | 23.51 | 23.57 | **−0.8 %** | ps1 |
| B2 / L24576 | 49 k | 23.08 | 23.28 | 23.27 | 23.28 | **−0.8 %** | ps1 |

### FlashInfer
| batch / input_len | KV (B×L) | ps1 | ps8 | ps32 | ps128 | **ps1 vs ps128** | best |
|---|---:|---:|---:|---:|---:|---:|:--:|
| B8 / L4096  | 32 k | 19.11 | 19.25 | 19.26 | 19.26 | **−0.8 %** | ps1 |
| B16 / L2048 | 32 k | 19.41 | 19.58 | 19.57 | 19.64 | **−1.2 %** | ps1 |
| B32 / L1024 | 32 k | 20.07 | 20.37 | 20.28 | 20.28 | **−1.1 %** | ps1 |
| B16 / L3072 | 49 k | 23.73 | 23.94 | 23.85 | n/a § | — | ps1 of {1,8,32} |
| B8 / L6144  | 49 k | 23.41 | 23.57 | 23.54 | 23.57 | **−0.7 %** | ps1 |
| B2 / L24576 | 49 k | 23.22 | 23.36 | 23.41 | 23.35 | **−0.6 %** | ps1 |

**Every cell: `ps1` is the fastest-or-tied page (within ≤ ~1 %).** No backend, batch, or context length
produces a `ps1` penalty (the 12-round paired CIs in "Verification" put `ps1` at −0.3 % to −1.1 % vs the best
page — consistently fastest by a small margin, never ≥5 % slower).

§ FlashInfer `ps128` at exactly B16/L3072 hits a **deterministic build bug** in this SGLang checkout —
`AttributeError: 'TreeCacheNamespace' object has no attribute 'evict'` — on all 4 attempts (it is not a
memory/measurement issue; `ps1/8/32` run fine, and `ps1` is the fastest of the three). The long-context
FlashInfer conclusion is fully covered by the two other 49 k cells (B8/L6144, B2/L24576), both `ps1`-fastest.

---

## Historical whole-call metric beside true-batch steady decode

The values below use the same backend and roughly matched cells, but the columns are **not equivalent
metrics**: the first is the superseded whole-call result from reports 3–4, while the second is steady decode
from this report. The numerical contrast is historical context, not a causal estimate of prefix sharing.

| scenario | historical shared-prefix whole-call metric (superseded) | **TRUE-batch steady decode** (this report) | numerical contrast |
|---|---:|---:|---:|
| **Triton, bs8** | **+11.5 %** (≈10 k, report 3) | **−1.0 %** (B8/L4096) | ~12 pts |
| **Triton, bs32** | **+24.6 %** (≈10 k, report 3) | **−1.0 %** (B32/L1024) | ~26 pts |
| **FlashInfer, long ctx** | **+8.6 %** (~39 k/bs8 paired, report 4) | **−0.7 %** (B8/L6144, 49 k) | ~9 pts |

Later same-log analysis shows that the historical bars were admission-ramp artifacts. Independently, the
true-batch steady-decode cells here are flat at the study's practical ±5 % threshold.

---

## Adversarial search — actively trying to make `ps1` lose (FlashInfer + CUDA graph ON)

The main sweep above only covered one model, total KV ≥ 32 k, batch ≤ 32, pages {1,8,32,128}. To stress-test
the "ps1 always fastest" claim, I then **hunted for a counter-example** in the corners that sweep never
touched. Current source tracing shows that this FlashInfer lowering uses a page-independent per-token index
list, so the search should be read as an empirical stress test across batch, context, KV footprint, model,
and graph settings—not as a test of a 128× page-table-walk mechanism:

| corner attacked | configs (FlashInfer, graph ON, true batch) | result |
|---|---|---|
| **High batch × short ctx** | Qwen3-VL-2B B64–256 × L64–128, fine grid {1,2,4,8,16,32,64,128} | `ps1` fastest/tied (max +0.2 %) |
| **L2-resident small KV** (≤32 MB → B×L ≲ 8 k cached) | Qwen3-VL-2B, total KV 2–16 k, B8–32 | `ps1` fastest (−0.5 % to −2.2 % vs ps128) |
| **Long single sequence** | Qwen3-VL-2B B1 × L32768 / L49152 | `ps1` fastest (−0.7 % to −0.9 %) |
| **Most-vulnerable model** (GQA-2, 4× less KV/layer), pushed to **batch 512** | Qwen2.5-3B B128–512 × L64–128 | `ps1` fastest/tied (≤0.3 %) |
| **Adversarial graph knob** (decode grown to 320 tok, `output-len` 192) | Qwen3-VL-2B B128/L128 | `ps1` +0.2 % (no break) |

**Verdict: NO BREAK.** Across **all 19 FlashInfer cells** the maximum `ps1` penalty vs the fastest page is
**+0.2 %** — i.e. `ps1` is the fastest page (or tied within noise) in *every* cleanly-measured cell, on both
backends and all three models. See [fig_bob_adversarial.png](fig_bob_adversarial.png).

- **The one apparent candidate was noise.** A single-shot smoke at B256/L128 showed `ps1` 40.4 ms vs `ps16`
  38.1 ms (+6 %). A **12-round paired** re-measure (`bobs_*` data) refuted it: `ps1` is the stable, fastest
  page (mean 38.86 ms, cross-launch CV 0.66 %), while the larger pages carry the noise (ps8 mean 39.24 ms,
  CV 2.0 %). The single-shot +6 % caught `ps1` high and another page low by chance — the classic
  single-pass-noise trap, which is why every headline here uses paired / min-across-rounds.
- **What the stress test establishes.** Across the tested FlashInfer/Triton true-batch cells, distinct-KV
  traffic is large and the measured page ordering stays flat. This is an empirical statement about these
  backends, models, and cells—not a proof that true batching can never expose a page effect. Report 16's
  page-aware XQA result is the concrete counterexample to such a universal claim.
- **Coverage gaps (logged, not hidden):** configs above ~49 k tokens don't fit 16 GB; the FlashInfer
  `TreeCacheNamespace.evict` build bug blocks the larger-page cells **at B≥128 (the largest page that runs is
  ps32 for Qwen2.5-3B / ps16 for Qwen3-VL-2B)** — so the high-batch verdict is strictly `ps1`-vs-ps≤32 (tied);
  no `ps1`-vs-`ps128` claim is made there. Qwen3-VL-4B FlashInfer OOMs at high batch (8.4 GB VL weights +
  workspace); every page that does complete is reported, but the missing cells remain unmeasured.

Data tags `bobx_*` (2B ladder + L2/long-ctx), `bobm_*` (cross-model), `bobg_*` (graph knobs), `bobs_*`
(12-round paired noise-floor/CI confirmation). Runners
`bob_break.sh`, `bob_xmodel.sh`, `bob_cd_knobs.sh`, `bob_4bfix.sh`, `bob_final.sh`, `bob_stats.sh`; analysis
`analyze_bob_break.py` + `make_bob_break_figure.py`.

---

## Honest scope / caveats

- **The exact shared-prefix regime is unreachable as a true batch on 16 GB.** Shared prefix lets bs8 each
  "see" 39 k of context while storing **one** 39 k KV copy (~4.4 GB). The independent equivalent is
  8 × 39 k = 312 k distinct tokens (~33 GB) — impossible here. So the true-batch sweep probes the *reachable*
  space: 32–49 k distinct KV, bs2–32, both "many short" and "few long" sequences. Across **all** of it `ps1`
  is fastest or tied. No claim is made for the unreachable matched-footprint cells or for page-aware kernels.
- **`ps128` baseline drift** (the report-4 lesson) is controlled by measuring all four pages back-to-back per
  round and taking the per-round paired/min comparison, not cross-launch averages.
- Effects here are ≤ ~1.2 % and consistently ordered in these samples, but they sit at the measured
  noise/engine-metadata floor. The practical takeaway is "page_size is ~free in these true-batch cells,"
  not "pick ps1 for a 1 % win."

---

## Verification

- **CUDA graph ON, captured at the batch:** every cell logs `Capture cuda graph bs [B]` (e.g. Triton
  B8/L4096 and FlashInfer B8/L6144). No `--disable-cuda-graph`.
- **True batch confirmed:** `bench_one_batch` builds B independent random sequences with a dummy tree-cache —
  no radix/shared prefix by construction (`bench_one_batch.py` `prepare_synthetic_inputs_for_latency_test`).
- **Metric checked** against the printed `Decode.  median latency:` line == JSONL `median_decode_latency`.
- **Per-cell fit:** `max_total_num_tokens` ≥ B×(L+OL) verified from each log; mem-fraction tuned per cell
  (0.60–0.72 short, 0.63–0.64 long) so the static pool covers the KV need with runtime headroom.
- **Robustness:** 2 rounds, all pages back-to-back per round; min-across-rounds and paired-mean±std agree on
  sign for the main-sweep cells **except one near-zero tie** — Triton B16/L3072, where min gives `ps1` −0.2 %
  but the per-round paired mean is +0.15 % (a coin-flip at the noise floor; `ps1` is rank-1 in one round,
  rank-4 in the other, all within ~0.4 %). No cell is anywhere near ±5 % by either estimator.
- **High-rep statistical confirmation (`bobs_*`, 12 paired rounds each).** To pin the noise floor and the
  effect size, four decisive cells were measured 12× with pages back-to-back. Cross-launch CV is **0.1–0.7 %**
  (the larger pages at high batch up to ~2 %). Paired `ps1`-vs-best 95 % CIs: **B8/L6144 −0.81 % ± 0.34 %**,
  **B16/L2048 −1.09 % ± 0.49 %**, **B256/L128 `ps1` fastest (CV 0.66 %)**, **Qwen2.5-3B B512/L64 −0.32 % ± 0.19 %**.
  So `ps1` is within ≤ ~1 % of the best page, CIs excluding +5 % by a wide margin. **Is the ≤1 % a real `ps1`
  advantage? No** — report 7's ncu over-read test shows `ps128` reads the *same* DRAM bytes as `ps1` (ratio
  0.998–0.999; no partial-page over-read). The source-traced index is page-independent per token, and a clean
  single-kernel microbench is tied to ±0.8 % with kernel latency tracking KV bytes rather than a page-table
  length effect. The honest takeaway: **page_size is
  ~free in a true batch — `ps1` is tied with large pages (the ≤1 % engine "lead" is at the noise / engine-
  metadata floor, not a kernel advantage), and never ≥5 % slower.** Not "ps1 fastest."

## Reproduce

Runners (in `studies/kv-cache-page-size/scripts/`) are `scp`'d to gray's `~/sglang_log/` and run there in tmux;
they write to `studies/kv-cache-page-size/data/raw/bench_one_batch_5060ti/` (rsync'd back). Each is idle-gated and uses a
skip-guard so re-launches only fill missing cells.

```bash
# on gray (RTX 5060 Ti):
bash bob_sweep.sh      # main 96-cell sweep (page{1,8,32,128} × {triton,flashinfer} × 6 (B,L) × 2 rounds)
bash bob_rerun.sh      # long-ctx cells that OOM'd at high mem-fraction, re-run at lower mf (0.64)
bash bob_fixup.sh      # B2/L24576 pool fit (mf 0.63) + the flaky-crash ps128 re-run
# adversarial break-search (fine page grid, high batch, L2-resident, long-seq, cross-model, graph knobs):
bash bob_break.sh      # high-batch × short-ctx fine grid (Qwen3-VL-2B)
bash bob_xmodel.sh     # cross-model: Qwen2.5-3B (GQA-2, batch→512) + Qwen3-VL-4B
bash bob_cd_knobs.sh   # L2-resident small KV + long single-seq + adversarial graph knobs
bash bob_4bfix.sh ; bash bob_final.sh   # 4B fit retries + corrected phase-G knobs

# locally (paths self-computed; run from anywhere):
python3 studies/kv-cache-page-size/scripts/analyze_bench_one_batch.py        # main-sweep tables -> fig_bob_truebatch.png
python3 studies/kv-cache-page-size/scripts/make_bench_one_batch_figure.py
python3 studies/kv-cache-page-size/scripts/analyze_bob_break.py              # adversarial tables (bobx/bobm)
python3 studies/kv-cache-page-size/scripts/make_bob_break_figure.py          # fig_bob_adversarial.png
```
