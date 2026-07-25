# Report 16 provenance

## Published evidence map

| Report section | Evidence directory | Scope |
|---|---|---|
| A1 shared-prefix engine | `../../data/raw/report-16-shared-prefix-engine/` | FlashInfer, Triton, and `trtllm_mha`/XQA logs plus parsed summary; one recorded round |
| A2 true batch | `../../data/raw/report-16-true-batch-engine/` | Two representative cross-backend cells plus an XQA-only B1/L512 pilot, one round each |
| C1 primary profiles | `../../data/raw/report-16-ncu-primary/` | XQA/FlashInfer CSV exports and replay controls |
| §3.1 mechanism sweep | `../../data/raw/report-16-mechanism-sweep/` | Page, regime, length, layout, and backend controls |
| §6 verification | `../../data/raw/report-16-verification/` | Three correctness/latency replay summaries |

Maintained runners, parsers, and figure generators are catalogued in
[`../../scripts/README.md`](../../scripts/README.md). Binary Nsight captures are intentionally excluded;
the committed profiler evidence consists of portable CSV/text exports.

## Qualifications

- The true-batch comparison is narrow evidence: one round for one off-wall and one on-wall cross-backend
  cell. The extra B1/L512 records are an XQA-only page pilot, not a backend comparison. None establishes
  convergence across rounds or universal backend ordering.
- The direct `trtllm-gen` prefill architecture probe was not retained as a standalone log. The report marks
  that gap and bases the retained decode check on `bench_xqa.py --validate --backend trtllm-gen` plus the
  documented source guard.
- XQA results are for the measured sm120 FlashInfer/SGLang path; they are not measurements of NVIDIA's
  standalone TensorRT-LLM library.

All published files are covered by [`../../data/SHA256SUMS`](../../data/SHA256SUMS).
