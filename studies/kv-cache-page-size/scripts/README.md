# Experiment and analysis scripts

This directory contains the maintained harnesses, profilers, parsers, and figure generators for the
KV-cache page-size study. Scripts resolve the study root from their own location; machine-specific model,
Python, profiler, and output paths can be supplied through the environment variables documented in each
entry point.

## Current evidence paths

| Reports | Primary scripts | Role |
|---|---|---|
| 06 | `bench_xqa.py`, `profile_xqa_ncu.sh`, `profile_tma_survey_ncu.sh`, `analyze_xqa_ncu.py` | XQA support, latency, counters, and TMA controls |
| 14–15 | `drive_shared_prefix_generation.py`, `profile_report_14_shared_prefix_ncu.sh`, `analyze_report_14_shared_prefix.py`, `make_report_14_figure.py` | Shared-prefix engine/kernel controls and report-14 figure generation |
| 15 | `drive_fragmentation_experiment.py`, `run_fragmentation_experiment.sh`, `parse_fragmentation_logs.py` | Constructed fragmentation experiment and parsing |
| 16 A1 | `run_report_16_fragmentation.sh`, `parse_fragmentation_logs.py` | Shared-prefix backend comparison |
| 16 A2 | `run_report_16_true_batch.sh`, `parse_true_batch_logs.py` | True-batch engine comparison |
| 16 C1/§3 | `profile_report_16_ncu.sh`, `profile_report_16_mechanism.sh`, `profile_report_16_triton_flashinfer.sh`, `parse_report_16_ncu.py`, `make_report_16_mechanism_figure.py` | XQA mechanism and profiler controls |
| 16 verification | `run_report_16_verification.sh`, `bench_xqa.py` | Correctness and repeatability checks |

The published evidence corresponding to these scripts is allow-listed under `../data/raw/`; see
[`../DATA_MANIFEST.md`](../DATA_MANIFEST.md).

## Earlier report support

The remaining `make_*`, `analyze_*`, `bench_*`, `profile_*`, `bob_*`, vLLM, TMA survey, reuse, and off-wall
scripts support reports 01–13. Their larger raw collections remain local and are explicitly catalogued in
the data manifest. The original all-purpose `run_engine_experiments.sh` is retained in
[`../../../archive/legacy-sglang-experiments/`](../../../archive/legacy-sglang-experiments/) because several
of its modes require machine-specific inputs or helpers that were not retained.

Figure generators for historical reports 03–04 reproduce their original whole-call metrics but label
those values as superseded steady-decode evidence. `vast_nsys_setup.sh` is a frozen provisioning recipe for
a rented `/workspace` container; `vast_nsys_run.sh` retains container-oriented model/output defaults while
accepting `MODEL_PATH`, `OUTPUT_DIR`, `PYTHON_BIN`, `BENCH_SCRIPT`, and `NSYS_BIN` overrides.

## Execution conventions

- Shell entry points are executable and pass `bash -n`.
- Generated raw output belongs below `../data/raw/`; unapproved directories remain ignored by default.
- Binary Nsight captures (`*.ncu-rep`, `*.nsys-rep`, SQLite exports) are local-only. Published profiler
  evidence uses text/CSV exports.
- Historical paths embedded in recorded logs are preserved. Maintained scripts use repository-relative
  defaults and explicit environment overrides.
