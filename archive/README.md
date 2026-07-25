# Legacy SGLang experiments

`legacy-sglang-experiments/` preserves the repository's pre-study technical material: attention-weight
captures, sparse-attention experiments, profiler exports, runtime logs, analysis scripts, and older
descriptive reports.

This is a provenance archive, not the current narrative. Filenames and internal layouts are largely
preserved so historical references remain recognizable. Some archived documents contain provisional
mechanism claims, machine-specific paths, or references to environments that are no longer present.
Nothing here should override a newer report in `studies/`.

The H100 input-length reference data formerly located at the repository root is retained under
`legacy-sglang-experiments/offline_batch_results/H100-80GB-HBM3/`.

`legacy-sglang-experiments/run_engine_experiments.sh` is the original multi-mode study runner. It is
archived rather than presented as maintained code because several modes refer to environment-specific
inputs or helpers that were not retained.
