# Report 5 — `page_size` in the TRUE-batch regime (`bench_one_batch`)

> PI steer (2026-06-20): *"你可以跑 `bench_one_batch` mode … 这个是真正的batch"* — the earlier ≥5 % numbers
> (Triton +11–25 %, FlashInfer +8.6 %) used the **shared-prefix (radix-ON)** trick, where B identical
> requests share **one** KV copy. That is not a real batch. This report re-measures the page_size decode
> effect with **B genuinely independent sequences** and quantifies how much of the earlier effect was
> shared-prefix-specific.

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
| **Data** | `offline_batch_results/bench_one_batch_5060ti/bob_*.jsonl` |
| **Figures** | [fig_bob_truebatch.png](fig_bob_truebatch.png) (main sweep) · [fig_bob_adversarial.png](fig_bob_adversarial.png) (adversarial break-search) |

---

## Headline

**In a true batch, `page_size=1` is the *fastest* page — at every batch, every context length, on *both*
backends.** The ≥5 % `ps1` penalty from the earlier reports was **entirely a shared-prefix artifact**; it
does not exist when each request holds its own KV. The PI's instinct was correct. A subsequent **adversarial
search** (next-but-one section) actively tried to break this on FlashInfer+graph — high batch to 512,
L2-resident KV, long single sequences, the most page-vulnerable GQA-2 model — and **found no
counter-example** (max `ps1` penalty +0.2 % across 19 cells).

- **Triton:** true-batch `ps1` vs `ps128` = **−0.2 % to −1.0 %** (ps1 fastest) — versus **+11.5 % (bs8)** and
  **+24.6 % (bs32)** in the shared-prefix regime (report 3).
- **FlashInfer:** true-batch `ps1` vs `ps128` = **−0.7 % to −1.2 %** (ps1 fastest) — versus **+8.6 %** in the
  shared-prefix long-context corner (report 4).
- The page-size *spread* in a true batch is tiny (≤ ~1.2 %) and ordered the **opposite** way: small pages are
  marginally **faster** (no page-padding over-read), `ps128` marginally slowest. Nowhere near ±5 %.

---

## Why a "true batch" changes the answer (mechanism)

The Triton `ps1` penalty is a **gather-coalescing** cost: `ps1` scatters one KV index per token, large pages
give long contiguous physical runs that the kernel reads coalesced (report 3 §mechanism). That cost only
becomes first-order when decode is dominated by the *access pattern* rather than raw KV *volume*.

- **Shared prefix (radix ON):** B requests share one small KV copy, but each still *reads* the whole context
  every step → huge read count over a **tiny** footprint (e.g. bs32 @ 10k ≈ 320 k reads over a 10 k cache).
  Access pattern dominates → the `ps1` scatter penalty is amplified to +20–25 %.
- **True batch (this report):** B independent sequences = B× the **distinct** KV bytes. What fits in 16 GB is
  ≤ ~49 k *distinct* tokens, and decode is **bandwidth-bound over distinct memory**. The coalescing penalty
  is a rounding error; the only residual page effect is `ps128`'s slight padding over-read, so **`ps1` wins**.

FlashInfer's decode kernel is page-agnostic; its sole page-dependent cost is the per-step `begin_forward`
page-table build (size `B × ctx/page`), which under CUDA graph is captured once. In a true batch that cost is
likewise negligible → flat, `ps1`-fastest.

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

**Every cell: `ps1` is the fastest page.** No backend, batch, or context length produces a `ps1` penalty.

§ FlashInfer `ps128` at exactly B16/L3072 hits a **deterministic build bug** in this SGLang checkout —
`AttributeError: 'TreeCacheNamespace' object has no attribute 'evict'` — on all 4 attempts (it is not a
memory/measurement issue; `ps1/8/32` run fine, and `ps1` is the fastest of the three). The long-context
FlashInfer conclusion is fully covered by the two other 49 k cells (B8/L6144, B2/L24576), both `ps1`-fastest.

---

## The key comparison — how much was the shared prefix?

Same backend, matched as closely as 16 GB allows. The shared-prefix figures are from this study's earlier
reports (graph-ON, paired where noted).

| scenario | shared-prefix (radix ON) | **TRUE batch** (this report) | swing |
|---|---:|---:|---:|
| **Triton, bs8** | **+11.5 %** (≈10 k, report 3) | **−1.0 %** (B8/L4096) | ~12 pts |
| **Triton, bs32** | **+24.6 %** (≈10 k, report 3) | **−1.0 %** (B32/L1024) | ~26 pts |
| **FlashInfer, long ctx** | **+8.6 %** (~39 k/bs8 paired, report 4) | **−0.7 %** (B8/L6144, 49 k) | ~9 pts |

The entire ≥5 % effect collapses. In a real batch `page_size=1` is never the wrong choice on this workload.

---

## Adversarial search — actively trying to make `ps1` lose (FlashInfer + CUDA graph ON)

The main sweep above only covered one model, total KV ≥ 32 k, batch ≤ 32, pages {1,8,32,128}. To stress-test
the "ps1 always fastest" claim, I then **hunted for a counter-example** in the corners that sweep never
touched, guided by the mechanism: FlashInfer's *only* page-dependent cost is walking `paged_kv_indices`
(128× longer at `ps1`, `flashinfer_backend.py:514-560`), and that walk only surfaces when it is **not hidden
by KV-read bandwidth**. So I attacked every regime that could starve the KV reads or inflate the index:

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
  38.1 ms (+6 %). A **paired ×5** re-measure refuted it: `ps1` is stable (~38.8 ms) while `ps16` is the noisy
  one (38.9–41.8 ms); min-paired gives `ps1` −2 %. Classic single-pass-noise trap — exactly why every claim
  here uses paired/min-across-rounds.
- **Why a true batch can't break `ps1` (mechanism).** The penalty needs the index walk (∝ `B×ctx/page`
  entries, latency-bound) to be visible against the KV-read cost. A true batch reads `B ×` *distinct* KV bytes
  at full HBM bandwidth, which dwarfs the walk. To make KV cheap you must cache it (tiny footprint) — that
  *is* the shared-prefix regime, which 16 GB cannot reproduce as a true batch at the required `B×ctx`. Even
  GQA-2 (4× less KV/layer) at batch 512 doesn't tip the balance. So the shared-prefix `+8.6 %` has **no
  true-batch analog** on this hardware.
- **Coverage gaps (logged, not hidden):** configs above ~49 k tokens don't fit 16 GB; the FlashInfer
  `TreeCacheNamespace.evict` build bug blocks some larger-page cells (e.g. ps64/128 at B256/L128); and
  Qwen3-VL-4B FlashInfer OOMs at high batch (8.4 GB VL weights + FlashInfer workspace) — but **every page that
  *does* measure shows `ps1` fastest**, 4B *Triton* shows `ps1` fastest, and 4B (GQA-8, 36 layers) is
  bracketed by the exhaustively-tested 2B and predicted *flatter* (more layers amortise the index walk).

Data tags `bobx_*` (2B ladder + L2/long-ctx), `bobm_*` (cross-model), `bobg_*` (graph knobs). Runners
`bob_break.sh`, `bob_xmodel.sh`, `bob_cd_knobs.sh`, `bob_4bfix.sh`, `bob_final.sh`; analysis
`analyze_bob_break.py` + `make_bob_break_figure.py`.

---

## Honest scope / caveats

- **The exact shared-prefix regime is unreachable as a true batch on 16 GB.** Shared prefix lets bs8 each
  "see" 39 k of context while storing **one** 39 k KV copy (~4.4 GB). The independent equivalent is
  8 × 39 k = 312 k distinct tokens (~33 GB) — impossible here. So the true-batch sweep probes the *reachable*
  space: 32–49 k distinct KV, bs2–32, both "many short" and "few long" sequences. Across **all** of it `ps1`
  is fastest. The mechanism (above) explains why extrapolation holds: more distinct KV makes decode *more*
  bandwidth-bound, i.e. *less* page-sensitive, not more.
- **`ps128` baseline drift** (the report-4 lesson) is controlled by measuring all four pages back-to-back per
  round and taking the per-round paired/min comparison, not cross-launch averages.
- Effects here are ≤ ~1.2 % — real and consistent in sign (ps1 fastest, ps128 slowest) but small; the
  practical takeaway is "page_size is ~free in a true batch," not "pick ps1 for a 1 % win."

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
  sign and magnitude for every cell.

## Reproduce

Runners (in `page_size_study/scripts/`) are `scp`'d to gray's `~/sglang_log/` and run there in tmux;
they write to `offline_batch_results/bench_one_batch_5060ti/` (rsync'd back). Each is idle-gated and uses a
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
python3 page_size_study/scripts/analyze_bench_one_batch.py        # main-sweep tables -> fig_bob_truebatch.png
python3 page_size_study/scripts/make_bench_one_batch_figure.py
python3 page_size_study/scripts/analyze_bob_break.py              # adversarial tables (bobx/bobm)
python3 page_size_study/scripts/make_bob_break_figure.py          # fig_bob_adversarial.png
```
