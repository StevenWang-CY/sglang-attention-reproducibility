# Report 14 reproduction and provenance

## What was rebuilt

Report 14 used SGLang commit `0eded9e208` as a behaviorally matched June-era build on an exclusive RTX 5060
Ti (sm120). It was selected to reproduce the relevant scheduler and attention behavior; it is **not**
documented as the literal checkout used by the original gray runs.

The matched configuration used Qwen3-VL-2B, a shared prompt of approximately 9.7k tokens, context 12,288,
Triton attention, radix cache enabled, and CUDA graph enabled. A longer-prompt control used the same build.

## Result that must be preserved

The whole-call harness metric reproduced a large ps1-versus-ps128 difference at the matched configuration,
while the engine's steady per-step decode logs were page-flat. Original report-3 and report-4 logs,
published under `data/raw/report-15-gray-log-forensics/`, independently show the same separation between the
old whole-call metric and steady decode.

This result does **not** support the earlier contention explanation. Report 15 retracts contention and
closes the original runs directly from their own per-step logs. The cold shared-prefix admission ramp is
included in the old TPOT interval; the precise internal phase decomposition was inferred rather than
separately instrumented.

## Retained evidence

| Path | Role |
|---|---|
| `data/raw/report-14-shared-prefix-page-kernel-rtx5060ti/` | Related rebuild summaries, prompt input, KV-index contiguity dumps, shared/distinct latency runs, profiler CSV controls, and machine provenance |
| `data/raw/report-15-gray-log-forensics/report-03-dense-sweep/` | Original report-3 JSON/log pairs used for same-run whole-call versus steady-decode closure |
| `data/raw/report-15-gray-log-forensics/report-04-flashinfer-graph/` | Original report-4 paired logs used for same-run closure |
| `data/curated/report-14-figure-values.csv` | Derived values used by the report-14 reconciliation figure, including row-level provenance limitations |

The separate exclusive-run per-step log set underlying some report-14 summary rows was not retained. The
report and derived CSV preserve those values, but neither is a substitute for the missing raw logs. This is
an explicit archive gap, not a claim of complete recovery.

## Maintained analysis commands

Run from the study root:

```bash
python scripts/analyze_report_14_shared_prefix.py
python scripts/make_report_14_figure.py
```

The figure generator reads `data/curated/report-14-figure-values.csv` and writes
`reports/14-shared-prefix-benchmark-artifact/fig_reconcile.png`, independent of the caller's working
directory.

The benchmark harness is `scripts/measure_batch_latency_offline.py`. The retained shared-prompt input is
`data/raw/report-14-shared-prefix-page-kernel-rtx5060ti/prompt10k.json`. Reconstructing the original launch
also requires the recorded SGLang/model environment; machine-specific paths in the raw provenance note are
historical values, not portable defaults.

## Interpretation boundary

- Reproduced: the separation between a large whole-call metric and page-flat steady decode in the matched
  shared-prefix construction.
- Independently closed: the original report-3/4 runs' steady decode, from their retained logs.
- Not reproduced as a steady penalty: a page-size-dependent FlashInfer/Triton decode cost.
- Not claimed: literal checkout identity, a separately timed admission-phase component breakdown, or a
  complete raw archive for every report-14 table.
