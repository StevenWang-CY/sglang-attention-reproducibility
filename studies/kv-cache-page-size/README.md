# KV-cache page size and decode latency

This study asks when a smaller KV-cache `page_size` (or vLLM `block_size`) increases decode latency, with a
target effect of at least 5%. The experiments cover SGLang and a vLLM cross-check, primarily on an RTX 5060
Ti 16 GB (sm120), with reference measurements where identified in individual reports.

## Current synthesis

The study's current answer is conditional on the executed backend:

1. The earlier `page_size=1` slowdowns of +11–25% (Triton) and +8.2–9.2% (FlashInfer) were present in a
   whole-call benchmark metric, but the same runs' per-step logs show page-flat steady decode. Reports 14
   and 15 identify the discrepancy as a cold shared-prefix admission ramp folded into the old TPOT formula.
2. In the measured SGLang FlashInfer and Triton paths, decode is lowered to page-independent per-token
   indices. Page size does not change steady kernel work, so off-wall operation alone is not sufficient to
   create a page effect.
3. On sm120, SGLang's `trtllm_mha` decode path selects XQA. XQA accepts pages 16, 32, and 64—not page 1—and
   performs page-dependent load work. Report 16 measures a 9–10% isolated ps16-to-ps64 difference in
   shared/L2-reuse cells and up to 4.8% at steady engine level; the effect washes out near the DRAM wall.
4. These findings are not a universal law for every attention backend, GPU, allocator state, or software
   revision. Each report records its own scope and method.

## Report index

Status terms:

- **Current**: retained in the final synthesis.
- **Current, scoped**: measurement remains valid, but a broader causal interpretation was narrowed.
- **Historical**: original observation is preserved, while a later report supersedes its mechanism.
- **Superseded**: useful methodology/history only; do not use as the current answer.

| # | Report | Status | Contribution |
|---:|---|---|---|
| 01 | [Kernel microbenchmark](reports/01-kernel-microbenchmark/report.md) | Superseded | Early isolated-kernel and end-to-end baselines; documents cold-start sensitivity. |
| 02 | [Low-batch baseline](reports/02-low-batch-baseline/report.md) | Superseded | Clean low-batch remeasurement and early “no ≥5%” result. |
| 03 | [Dense sweep](reports/03-dense-sweep/report.md) | Historical | Preserves the original shared-prefix whole-call +11–25% observation; report 15 shows its steady decode was page-flat. |
| 04 | [FlashInfer + CUDA graph](reports/04-flashinfer-cuda-graph/report.md) | Historical | Preserves the paired +8.2–9.2% whole-call result; report 15 closes the same-run steady decode as page-flat. |
| 05 | [True-batch study](reports/05-true-batch/report.md) | Current, scoped | FlashInfer/Triton true-batch cells show no ≥5% small-page penalty; not generalized to page-aware kernels. |
| 06 | [XQA and TMA profile](reports/06-xqa-tma-profile/report.md) | Current | Establishes XQA identity, supported pages, latency baseline, and absence of TMA on sm120. |
| 07 | [Shared-prefix mechanism](reports/07-shared-prefix-mechanism/report.md) | Current, scoped | Establishes the shared-versus-distinct L2/DRAM regimes; retracts the former claim that off-wall operation alone causes a page penalty. |
| 08 | [Small-KV true batch](reports/08-small-kv-true-batch/report.md) | Current, scoped | Shrinking distinct KV does not create a practical FlashInfer/Triton page penalty in the measured cells. |
| 09 | [vLLM block-size true batch](reports/09-vllm-block-size-true-batch/report.md) | Current | Clean vLLM FlashInfer true batches are flat; the apparent 15% cell was KV-pool preemption. |
| 10 | [FlashInfer page cost](reports/10-flashinfer-page-cost/report.md) | Current, scoped | Source trace and weightless tests show page-independent index length and ≤1.3% worst-case scatter cost for the measured MHA path. |
| 11 | [MLA scatter page cost](reports/11-mla-scatter-page-cost/report.md) | Current, scoped | Measures a bounded worst-case scatter effect up to 4.07% in the tested MLA kernel; realistic block-64 layout is flat. |
| 12 | [Reuse, not residency](reports/12-reuse-not-residency/report.md) | Current | Explains L2-hit behavior through intra-kernel reuse rather than footprint residency. |
| 13 | [True batch off the DRAM wall](reports/13-true-batch-off-wall/report.md) | Current, scoped | Constructs under-parallelized off-wall cells and measures the launch/host floor; native FlashInfer microbenchmark attribution remains underdetermined. |
| 14 | [Shared-prefix benchmark artifact](reports/14-shared-prefix-benchmark-artifact/report.md) | Current | Behaviorally matched rebuild separates the old whole-call metric from page-flat steady decode. |
| 15 | [Log forensics and constructed bound](reports/15-log-forensics-and-upper-bound/report.md) | Current | Uses the original report-3/4 logs for same-run closure, retracts contention, reconciles report 7, and records the constructed fragmentation bound. |
| 16 | [TensorRT-LLM MHA / XQA](reports/16-trtllm-mha-xqa/report.md) | Current | Profiles the SGLang `trtllm_mha` decode path on sm120 and identifies a genuine page-aware kernel effect by memory regime. |

## Evidence and reproducibility

- [`DATA_MANIFEST.md`](DATA_MANIFEST.md) distinguishes published evidence from local-only collections and
  records known gaps.
- [`data/README.md`](data/README.md) describes the data layout and immutability rules.
- [`data/SHA256SUMS`](data/SHA256SUMS) covers every other published file below `data/` (the manifest
  cannot checksum itself).
- [`scripts/`](scripts/) contains benchmark, profiling, parsing, and figure-generation code.
- Each report directory contains its report, figures, and any report-specific provenance or reproduction
  note.

Only named report-14/15/16 raw evidence sets are versioned. The broader available local raw collection for
earlier reports remains at `data/raw/` and is intentionally ignored. Binary Nsight captures and derived profiler
databases are excluded when text/CSV evidence is sufficient.

## Known provenance limits

- The report-14 rebuild used SGLang commit `0eded9e208` as a behaviorally matched June-era build; it was not
  the original gray experiment's literal checkout.
- Report 15's original report-3/4 logs are published, but the constructed fragmentation arm's independent
  raw-run directory has not been recovered. The report's table is derived evidence, not a replacement for
  those absent logs.
- Report 16 retains engine logs, JSONL results, profiler CSVs, and replay summaries. The direct
  `trtllm-gen` prefill probe cited in the narrative was not retained as a standalone raw log; the maintained
  `bench_xqa.py` validation path covers decode backend support and correctness.
- Machine paths inside raw logs and provenance records are intentionally preserved.
