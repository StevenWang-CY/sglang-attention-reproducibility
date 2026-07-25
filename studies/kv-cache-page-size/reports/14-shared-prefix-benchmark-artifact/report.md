# Report 14 — The shared-prefix `page_size=1` "+11–25%" (report 3) is a **benchmark-measurement artifact**, not a decode-time page penalty

**The PI (Jiaheng Lu) asked us to *profile* the shared-prefix case: is the earlier Triton `ps1` "+11–25%"
(report 3, shared prefix, high batch) reproducible, and — if neither shared-prefix nor a true batch
saturates DRAM bandwidth — why does the shared-prefix case "change a lot"? After a behaviorally matched
June-era rebuild and profiling on an *exclusive* GPU, the answer is:**

> **The "+25%" reproduces (it is real and robust), but it is NOT a decode/attention/page-size cost.**
> **The steady-state per-decode-step latency is byte-for-byte page-invariant (ps1 43.80 ms vs ps128 43.83 ms,
> 475 steps). The reported "+25%" lives entirely in the offline benchmark's TPOT *formula*, which folds a
> cold-prefill *straggler* phase into "decode." That prefill phase is larger at `page_size=1`. Measured
> correctly, page size is free in decode.**

| | |
|---|---|
| **Headline** | At the configuration matched to gray's run (prompt ≈9.7k, ctx 12288, bs32, Triton, radix ON, graph ON), on an **exclusive** RTX 5060 Ti, the harness reports `ps1`/`ps128` TPOT = **67.41 / 51.27 ms (+31%)** — reproducing the original whole-call behavior. But the engine's own per-step decode logs show steady-state = **43.80 / 43.83 ms (±0.1%)**. The +31% is the *difference between the reported number and the true decode step*, and it is an **admission-ramp/metric artifact**, not decode. |
| **Proof it's not contention** | It reproduces on an **exclusive** box (0 co-tenants). My earlier "contention" call was wrong — it came from testing the *wrong config* (a longer 16.4k prompt / ctx 20480, which happens not to trigger the artifact → looked flat). Gray's real config triggers it every time. |
| **Proof it's not the kernel** | Steady-state decode identical; the decode `kv_indices` are contiguous at both page sizes; the Triton decode kernel is **page-agnostic** (`grep -c page_size = 0`); a full source trace finds **no** `O(num_pages)` per-decode-step op (and where the allocator *does* branch, `ps1` is the *cheaper* path). Every attention-kernel microbench ≤6%. |
| **FlashInfer report 4** | The separate long-context FI "+8.6%" (report 4, 39k, bs8) is the **same admission-ramp artifact** — settled by report 15's forensics on gray's own paired logs: gray's steady-state was **32.21/32.44, 32.54/32.59, 32.74/32.98 ms (−0.2…−0.7%)** while its reported showed +8.2/+8.4/+9.2%. (An earlier draft here called it "contention" from one flat exclusive re-run — retracted; see [report 15](../15-log-forensics-and-upper-bound/report.md).) |
| **Run** | RTX 5060 Ti (sm120), `phastform`, 2026-07-16, exclusive. gray's stack rebuilt: sglang `0eded9e208` + torch 2.9.1+cu128 + sgl-kernel 0.3.21 + flashinfer 0.6.6 + `num_kv_splits=8`. Model Qwen3-VL-2B. |

## TL;DR

1. **Reproducible — yes.** A behaviorally-matched June commit (`0eded9e208`; see report 15 — not gray's
   literal checkout) + the matched gray configuration on an exclusive box reproduces the harness's
   "+11–25%" (bs-dependent), with tiny cross-repeat std. It is a robust, real *number*. (Chuyue's
   experiment was not sloppy and not un-reproducible.)
2. **But it is not a decode penalty.** The engine's per-step decode logs show **identical** steady-state
   TPOT at ps1 and ps128 — 43.80 vs 43.83 ms over 475 steps at gray's config, and 68.34 vs 68.20 ms at a
   second (longer-prefix) config. Page size never changes the decode step.
3. **The "+25%" is a benchmark-formula artifact.** `measure_batch_latency_offline.py` computes
   `TPOT = (end_time − first_token_time)/(tokens − 1)` and calls `engine.flush_cache()` **before every
   repeat**. With B concurrent requests over a *cold* shared prefix, requests finish prefill at staggered
   times; the interval between the *first* request's first token and the *whole batch* reaching steady
   decode is counted as "decode." That cold-prefill **straggler gap** is ~3× larger at `page_size=1`
   (≈2.2 s vs ≈0.7 s at bs32), and dividing it by (tokens−1) produces the fake "+16 ms/token."
4. **It's config-dependent, which is the tell.** A real decode page-penalty would not vanish by changing
   the prompt length. This one does: gray's config (9.7k / ctx 12288) → +31%; a 16.4k / ctx 20480 config
   → flat (both at *identical* steady-state). A prefill/scheduling origin explains that; a decode-kernel
   origin cannot.
5. **`f_attn` hypothesis:** the *framework* `p_e = f_attn · p_k` is correct, but here `p_k ≈ 0` — the
   attention kernel does byte-identical work at both page sizes (token-flat pool, contiguous indices,
   page-agnostic kernel). There is no kernel penalty to amplify. The observation that motivated it (the
   benchmark "+25%") is real but originates upstream of attention, in the prefill/measurement path.
6. **Report 4 (FI long-context "+8.6%") is the same admission-ramp artifact, not contention** — corrected
   by report 15: gray's own interleaved-paired logs show steady-state page-invariant (−0.2…−0.7%) in all
   three pairs while reported showed +8.2…+9.2%. Same family as report 3, FlashInfer flavor.

## The decisive measurement: reported TPOT vs. steady-state decode

Same runs, two ways of reading them. `steady-state` = per-step decode latency taken directly from the
engine's `Decode batch … gen throughput` logs (`decode_log_interval=1`), at full `#running-req = B`:

**gray's config — prompt ≈9.7k (9,663 tok), ctx 12288, bs32, Triton, radix ON, graph ON, ×5:**

| | reported TPOT `(end−first_tok)/(tok−1)` | **steady-state decode step** (n steps) |
|---|---|---|
| ps1 | 67.41 ms | **43.80 ms** (475) |
| ps128 | 51.27 ms | **43.83 ms** (475) |
| ps1 − ps128 | **+31.5%** | **−0.07%** |

The reported +31.5% is entirely the gap between the reported number and the true step: ps1 carries a
+23.6 ms/token prefill-straggler inflation, ps128 only +7.4 ms → their *difference* is the "+31%."
Steady-state decode is flat.

**A second config — prompt 16.4k (16,420 tok), ctx 20480, bs32 (reproduction-note run):**

| | reported TPOT | steady-state decode step (n) |
|---|---|---|
| ps1 | 68.44 ms | **68.34 ms** (635) |
| ps128 | 70.32 ms | **68.20 ms** (634) |
| ps1 − ps128 | **−2.7%** (flat) | **+0.2%** |

Here reported ≈ steady-state (the prefill artifact is small for both), so the run looks flat — **and the
steady-state is still page-invariant.** Steady-state decode is page-invariant in *both* configs; only the
*reported* number carries the config-dependent prefill artifact. (This is exactly why an earlier draft of
this report, run only at this second config, wrongly concluded "not reproducible → contention.")

### Across batch sizes — the reported artifact reproduces gray's +11→+24%, steady-state stays flat

prompt9k, ctx 12288, Triton, radix ON, graph ON, ×5, exclusive. `reported` = harness TPOT; `steady` = per-step
decode from the engine's logs at `#running-req = B`:

| bs | reported ps1 / ps128 | **reported ps1−ps128** | steady ps1 / ps128 | **steady ps1−ps128** | gray's logged |
|---:|---:|---:|---:|---:|---:|
| 8  | 28.13 / 24.57 | **+14.5%** | 20.00 / 19.98 | **+0.1%** | +11.5% |
| 16 | 40.61 / 32.67 | **+24.3%** | 27.27 / 27.29 | **−0.0%** | +18.6% |
| 32 | 67.53 / 51.27 | **+31.7%** | 43.82 / 43.84 | **−0.0%** | +24.6% |
| 48 | 90.99 / 73.64 | **+23.6%** | 68.11 / 68.57 | **−0.7%** | +23.7% |

The reported column reproduces gray's +11→+24% shape; the steady-state column is flat (±0.7%) at every batch
size. Note *both* page sizes carry a prefill-ramp inflation (reported > steady: ps1 by +41/49/54/34%, ps128 by
+23/20/17/7%) — the cold-prefill straggler affects both, but ps1's is ~2–3× larger, and their **difference**
is the reported "ps1 penalty." Page size never touches the decode step.

### The amortization signature: a fixed prefill overhead, not a per-token cost

Same bs32 cell, generating 96 vs 512 tokens. A *fixed* cold-prefill ramp divided by (tokens−1) must shrink
as tokens grow; a genuine per-token decode cost would not:

| max-tokens | reported ps1 / ps128 | **reported ps1−ps128** | steady ps1 / ps128 |
|---:|---:|---:|---:|
| 96  | 67.53 / 51.27 | **+31.7%** | 43.82 / 43.84 |
| 512 | 49.58 / 46.66 | **+6.3%**  | 45.09 / 45.10 |

The reported "ps1 penalty" collapses **+31.7% → +6.3%** when the decode is 5× longer, while steady-state
stays flat. Quantitatively: the ps1−ps128 cold-prefill ramp is a fixed ≈1.3 s → +14 ms/token over 95 tokens
(mt96) but only +2.5 ms/token over 511 (mt512). This is the definitive signature of a fixed prefill overhead
folded into a per-token metric — not a decode/page cost.

## Directly testing the `f_attn` theory (`p_e = f_attn · p_k`)

The hypothesis is that `ps1` makes the *attention kernel* slower (`p_k > 0`) and that a large attention
step-fraction `f_attn` (∝ B·L) amplifies it to TPOT. The framework is correct; it makes 4 predictions, all
falsified:

1. **Premise `p_k > 0` — measured directly, it's 0.** Timing the decode attention kernel (`forward_decode`,
   CUDA events, eager, bs32, 9.7k prefix — Chuyue's exact large-`f_attn` regime): **per-layer attention =
   1.574 ms (ps1) vs 1.586 ms (ps128)** — identical (ps1 marginally *faster*). `f_attn` is indeed large here
   (per-step attention ≈ 44 ms of the eager step; **f_attn ≈ 44% eager**, larger under graph) — so the
   premise about `f_attn` is right, but there is **no kernel penalty to amplify**: `f_attn · p_k = 0.44 × 0 = 0`.
   *(Note: attention is a large fraction of the step but is NOT bandwidth-bound here — report 15 ncu shows
   this cell runs at 7% DRAM / 97% L2-hit, off the wall. The 1.57 ms/layer is L2-/latency-bound, not a DRAM
   stream; an earlier draft's "the step is the KV read" implied DRAM-bound and is corrected. Page-invariance
   of `t_attn` follows from identical work, not from a wall.)*
2. **A per-step penalty is token-count-independent — but it amortizes.** `f_attn·p_k` would slow *every*
   decode step, so the TPOT penalty must not depend on how many tokens you generate. It collapses
   **+31.7% → +6.3%** from 96→512 tokens (§ amortization). A per-step attention penalty cannot do that; a
   fixed prefill overhead ÷ tokens does.
3. **The step that *contains* attention doesn't slow down.** Steady-state decode step (attention included)
   = 43.80 (ps1) vs 43.83 (ps128). Because `f_attn≈1` in graph, this step is a *sensitive probe* of the
   attention kernel — and it reads identical. If `p_k` were even 5%, at `f_attn≈1` we'd see ~5% here.
4. **Wrong sign in L.** `f_attn ∝ L`, so a longer prefix should give a *bigger* penalty. It gives a
   *smaller* one: 9.7k → +31%, 16.4k → flat. The effect is anti-correlated with `f_attn`.

The surface fact Chuyue cites — "only large batch + context shows it" — is reproduced by the prefill-ramp
mechanism too (high batch = more requests to admit = longer cold-prefix ramp; shared prefix = the straggler),
so it fits *both* theories. The 4 tests above are what separate them, and they all point away from the kernel.

## Why the reported number differs (mechanism of the artifact)

The harness sends B identical prompts in one `engine.generate(stream=True)` call, with the KV cache flushed
first, so the shared prefix is prefilled **cold** each repeat:

- `start_time` — before `generate`.
- `first_token_time` — the **first** streamed chunk: emitted as soon as the *earliest* request clears its
  (chunked) prefill and produces a token, while other requests are still prefilling/attaching.
- `end_time` — when **all** B requests finish.
- `tokens_generated` — request[0]'s output length (e.g. 96).
- `TPOT = (end_time − first_token_time)/(tokens_generated − 1)`.

Decomposition at gray's config (bs32): TTFT is **identical** (1425 vs 1429 ms) and steady decode is
**identical** (43.8 ms × 96 ≈ 4.2 s), but the whole call takes 7.84 s (ps1) vs 6.30 s (ps128). The extra
≈1.5 s is the **mixed prefill+decode ramp**: after the first request's first token, requests 2…32 attach
to the cold shared prefix and join the batch. That ramp is ≈2.2 s at ps1 vs ≈0.7 s at ps128. Divided by
95 tokens, the ramp *difference* is the fake +16 ms/token. It is a **prefill-side / cold-cache scheduling**
difference, entirely outside the (page-invariant) decode step. In real serving — warm radix cache,
continuous batching, per-request inter-token latency — this term does not appear as a decode penalty.

**Where the ramp difference is localized — shared-prefix admission (measured); exact internal component
still inferred.** The scheduler's own throughput logs localize the difference to the admission phase. When
the (B−1) requests that share the cold prefix are admitted:

| | admission of the 29 cache-sharing requests (bs32) |
|---|---|
| ps1 | one batch, 270,536 cached tokens, **~357 tok/s → ~1.85 s** |
| ps128 | batches of 16+13, **~11,000–15,000 tok/s → ~0.28 s** |

ps128 processes *more* new tokens (3,712 vs 662) yet finishes ~6× faster, so the observed difference is not
explained by new-token compute alone. Source inspection shows page-granular radix keys and complete-page
sharing (`child_key(self.page_size)`, `match(key, page_size=…)`, `radix_cache.py:531`), which makes
token-granular bookkeeping a plausible contributor at ps1. However, the experiment did **not** isolate the
duration of each matching, wiring, allocation or scheduling function. The counts \(B\times L\) versus
\(B\times\lceil L/P\rceil\) are a source-derived work model, not a measured operation count or timing
decomposition. The directly supported conclusion is therefore: **the excess interval is in cold
shared-prefix admission; its exact internal cost split remains pending.** Report 3's smaller radix-OFF
whole-call differences are only corroborating context and do not by themselves identify the function.

## Reconciliation with report 13 (there *is* a kernel `ps1` cost — a different, bounded one)

Report 13 contains a native FlashInfer microbenchmark and an SGLang engine measurement; they must be kept
separate:

- The native FlashInfer ladder showed repeatable **microsecond-level** differences in some distinct/off-wall
  cells. But native `plan(P)` changes page-table length, KV layout and planning metadata together. It is not
  SGLang's per-token lowering, and identical DRAM bytes do not uniquely identify an “index-walk latency”
  mechanism. The internal cause is **underdetermined** by those counters. It is also system-invisible in the
  measured engine cells.
- In the SGLang shared-prefix
  bs32/9.7k regime the `kv_indices` are **contiguous at both page sizes** (`frac_contig≈1.0`) and the kernel
  is page-agnostic, so the two page sizes execute **identical work** — hence the direct measurement
  `t_attn(ps1)=1.574` vs `t_attn(ps128)=1.586 ms`. *(Correction, report 15: an earlier draft added "the
  kernel is at the DRAM wall here," inferring it from bytes÷bandwidth arithmetic that ignored L2 reuse.
  ncu of the exact cell shows the opposite — DRAM 7.3%, L2-hit 96.6% ((R−1)/R at R=32), i.e. deeply
  off-wall, exactly report 7's regime. The page-invariance needs no wall: identical work suffices. Forced
  worst-case scatter in this regime costs only +1.8%, `sectors/request` flat.)*
- Thus the native microsecond spread cannot explain the +25% whole-call result. The defensible partition is:
  **Report 13 constructs an under-parallelized off-wall regime and observes a small native-wrapper spread
  whose internal cause is unresolved; Report 14/15 locate the old shared-prefix +25% in admission and show
  the SGLang steady decode is page-flat.**

## Source-level confirmation (why steady-state *must* be flat)

A full trace of the per-decode-step path at commit `0eded9e208` (allocator → `prepare_for_decode` →
attention metadata → decode kernel → KV write) finds **no operation whose cost scales with `num_pages`
(= pool/page_size)**:

- Allocator is the only page-branched per-step op, and `ps1` is the **cheaper** branch: `alloc_token_slots`
  is a bare `free_pages[:n]` **view** (O(1)); `ps128` runs an extra `alloc_decode_kernel` (grid=B). The one
  genuinely O(pool)-at-ps1 op (`torch.cat` in `free`, 128× larger at ps1) runs **only on request free**,
  never per decode step, and never fires in a fixed-batch decode.
- Attention metadata (`_fill_kv_indptr_and_indices` → `create_flashinfer_kv_indices_triton`) is **per-token
  O(B·L) with no `page_size` argument** — identical at both page sizes.
- The decode kernel (`triton_ops/decode_attention`) takes only `kv_indptr, kv_indices, Lk` — **page-agnostic**.
- KV pool is token-flat (`[size+1,1,H,D]` at ps1 vs `[(size+128)/128,128,H,D]` at ps128 → identical flat
  addresses); `set_kv_buffer` writes by token `loc`.

So the decode step is page-agnostic end to end — consistent with the identical steady-state measurement.

## Kernel-level profiling (bounds the largest *possible* page effect)

On `bench_pagecost.py` (engine-faithful token-flat pool + per-token `kv_indices`), ncu cold,
`sectors/request` as the coalescing-vs-latency discriminator:

- **Triton scatter vs block128** = **+2–6%**; **FlashInfer** +0.1–1.3%. `sectors/request` **identical**
  contig→scatter (Triton 8.39→8.39, FI 15.28→15.28), which rules out a change in transactions per request
  as the explanation; these counters alone do **not** uniquely distinguish address-dependency latency,
  instruction scheduling, cache timing or another latency-side cause.
- **4-D paged pool + page-aware kernel**: ps1-vs-ps128 = **±4%** across `num_kv_splits ∈ {4,8,16,32}`.
- Largest attention-kernel page effect anywhere ≈ 6%, in the *distinct* (independent) regime, and it is
  bounded latency. `p_e = f_attn·p_k ≤ p_k` ⇒ no kernel path to +25% under any `f_attn`.

## Answering the PI, with the real data

- **"Is the earlier Triton +25% reproducible?"** **Yes — the whole-call behavior reproduces on an exclusive box at the matched gray
  config**, robustly. But profiling shows it is **not a decode/page effect**: the true per-decode-step
  latency is identical at ps1 and ps128 (43.80 vs 43.83 ms). The "+25%" is produced by the offline
  benchmark's TPOT formula folding a **cold-prefill straggler phase** (which is longer at `page_size=1`)
  into "decode." It vanishes when measured per-step, and it is config-dependent (a real decode penalty
  would not be).
- **"When neither saturates BW, why does shared-prefix change a lot?"** The *decode* doesn't change — it's
  page-invariant off the wall (L2-resident, contiguous, page-agnostic kernel). What changes with page size
  is the **cold-prefill/radix-attach ramp** of a large shared prefix, and the benchmark's TPOT definition
  charges that ramp to decode. Off the memory-BW wall, the reused prefix is L2-bound (report 7), and there
  the page-agnostic kernel makes page size free; the only nonzero kernel effect is the bounded +2–6%
  independent-batch scatter *latency*, which a shared prefix *suppresses*.
- **`f_attn`:** the framework is right and worth keeping; it just has nothing to amplify here because the
  per-kernel page penalty `p_k ≈ 0` on this build (byte-identical attention work at both page sizes). The
  reproducible "+25%" that motivated it is real but lives in the prefill/measurement path, which profiling
  was exactly the right tool to surface.

## Corrections to the study

- **Report 3 §C.6 & report 4** — the "+11–25% (Triton) / +8.6% (FI) shared-prefix `ps1` penalty" is **not a
  decode page-size penalty**. Both are the **benchmark-formula (cold-prefill admission-ramp) artifact** —
  proven for *both* from gray's own per-step logs in report 15 (gray steady-state: report 3 −0.3…−1.2%,
  report 4 −0.2…−0.7%). §C.6's coalescing mechanism does not exist (kernel page-agnostic). *(An earlier
  draft called report 4's number a contention artifact — retracted in report 15.)*
- **Reports 5/7/10/11/13 stand** — true-batch `ps1`≈`ps128`, the DRAM↔L2 flip, the FI/MLA scatter=latency
  bound, and the off-wall kernel study are unaffected (and correctly never found a decode kernel mechanism
  for the +25% — because there isn't one).
- **The corrected rule for the studied SGLang FlashInfer/Triton build:** there is no reuse-gated ≥5%
  `ps1` steady-decode penalty. Those paths were page-invariant in the measured engine regimes; the old
  whole-call gap is a cold-admission artifact, and synthetic scatter produces only a bounded kernel-side
  spread. This is **not universal across kernels**: report 16's XQA path really performs page-dependent
  global loads and is the explicit counterexample.

## Method / honesty

- Two prior conclusions in this investigation were **wrong and are retracted**: (i) "the +25% is GPU
  contention" (it reproduces on an exclusive box), and (ii) "the +25% is a real engine per-step decode
  cost" (steady-state decode is identical). Both fell to the direct per-step measurement.
- The rebuild (`0eded9e208`) was originally described as "gray's exact commit" — **corrected in report 15**:
  gray's ServerArgs use the pre-rename `cuda_graph_max_bs` field, so gray ran a slightly *earlier* June
  commit. The rebuild is **behaviorally matched** (reproduces gray's reported artifact and gray's
  steady-state within ~2%), and gray's own logs now carry the artifact proof directly, so nothing rests on
  commit identity. gray itself is offline (powered down 2026-07-15); the reproduction is on phastform.
- Steady-state numbers come from the engine's own per-step throughput logs (475/635 steps), not the
  harness summary — the whole point is that the summary metric is the thing under scrutiny.

## Reproduce and inspect

The exact exclusive-run logs underlying some summary rows were not retained; see
[reproduction.md](reproduction.md) for the evidence boundary. The maintained analysis paths are:

```bash
# Run from studies/kv-cache-page-size/.
python scripts/analyze_report_14_shared_prefix.py
python scripts/make_report_14_figure.py
```

## Figure
[fig_reconcile.png](fig_reconcile.png) (regenerate: `python scripts/make_report_14_figure.py`): **(A)** across bs
8/16/32/48, the harness's *reported* TPOT shows +14/+24/+32/+24% while the engine's *steady-state* decode is
page-invariant (±0.7%); **(B)** amortization — at bs32 the reported penalty collapses +31.7%→+6.3% when the
decode grows 96→512 tokens (fixed prefill overhead ÷ more tokens), steady-state flat; **(C)** two configs —
the reported artifact is config-dependent (prompt9k/ctx12288 → +31%, prompt16k/ctx20480 → flat) but the
steady-state decode is page-invariant in both (a real decode cost would not be config-dependent).

**Companion reports:** [report 3](../03-dense-sweep/report.md) (the +25%
claim, now a benchmark-measurement artifact), [report 5](../05-true-batch/report.md)
(true-batch flat), [report 7](../07-shared-prefix-mechanism/report.md)
(DRAM↔L2 flip), [report 13](../13-true-batch-off-wall/report.md) (off-wall kernel).
