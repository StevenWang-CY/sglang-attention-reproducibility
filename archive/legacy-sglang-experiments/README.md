# Legacy SGLang experiments

This directory preserves the repository's pre-study technical material: attention-weight captures,
sparse-attention experiments, profiler exports, runtime logs, analysis scripts, and older descriptive
reports.

It is a provenance archive, not the current narrative. Filenames and internal layouts are largely
preserved so historical references remain recognizable. Documents and scripts may contain provisional
mechanism claims, machine-specific paths, or dependencies on environments that are no longer present.
Nothing here overrides a newer report under [`../../studies/`](../../studies/).

The historical H100 input-length reference data is under
`offline_batch_results/H100-80GB-HBM3/`. The original `run_engine_experiments.sh` multi-mode runner is
retained here because several modes require inputs or helpers that were not retained.

Exact duplicate top-level renders in `offline_batch_results_flashinfer_tree_sparse/` were deduplicated in
favor of the canonical copies inside their timestamped run directories.
