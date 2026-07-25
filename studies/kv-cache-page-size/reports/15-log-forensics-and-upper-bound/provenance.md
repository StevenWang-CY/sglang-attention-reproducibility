# Report 15 provenance

## Published evidence

- `../../data/raw/report-15-gray-log-forensics/report-03-dense-sweep/` contains the four original
  `hb_tri_g_ps{1,8,32,128}` JSON/log pairs used for the whole-call-versus-steady-decode reanalysis.
- `../../data/raw/report-15-gray-log-forensics/report-04-flashinfer-graph/` contains the three original
  paired `fgi_p{1,2,3}_ps{1,128}.log` sets.
- `../../data/raw/report-14-shared-prefix-page-kernel-rtx5060ti/` contains the rebuild, index-contiguity,
  and kernel controls reused by this report.
- The current parsers and experiment drivers are in `../../scripts/`, principally
  `parse_fragmentation_logs.py`, `drive_fragmentation_experiment.py`, and
  `run_fragmentation_experiment.sh`.

The published gray subset is byte-for-byte copied from the retained original raw collection; recorded
machine paths and timestamps were not rewritten.

## Retention gaps

The raw-run directory for the constructed fragmentation experiment was not independently retained. Its
derived measurements and configuration are recorded in the report, but the repository does not present
that section as fully rerunnable evidence. Some report-14 exclusive-run per-step logs are likewise absent;
the retained report-14 controls are a related, not complete, archive of every summarized value.

These gaps are also listed centrally in [`../../DATA_MANIFEST.md`](../../DATA_MANIFEST.md).
