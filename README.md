# SGLang Attention and Serving: Reproducibility Artifacts

This repository collects technical reports, measurement logs, profiler exports, figures, and analysis
scripts from experiments on SGLang attention and serving behavior. It is organized as an academic artifact
repository: current claims live with their supporting evidence, while older experiments are retained in a
clearly marked archive.

## Repository map

| Path | Purpose |
|---|---|
| [`studies/kv-cache-page-size/`](studies/kv-cache-page-size/) | Active, report-oriented study of KV-cache page size and decode latency |
| [`archive/legacy-sglang-experiments/`](archive/legacy-sglang-experiments/) | Historical attention, sparse-attention, profiling, and latency material preserved for provenance |
| [`ARTIFACT_POLICY.md`](ARTIFACT_POLICY.md) | Inclusion, exclusion, provenance, and data-integrity rules |

The active page-size study contains 16 sequential reports. Reports 14–16 are the current reconciliation:

- earlier `page_size=1` whole-call slowdowns in the shared-prefix benchmark were admission-ramp/metric
  artifacts, not steady decode penalties;
- the measured SGLang FlashInfer and Triton paths perform page-invariant steady work in the tested cells;
- the XQA kernel selected by `trtllm_mha` on the tested sm120 stack performs page-dependent work and shows
  a measurable page-size effect off the DRAM wall.

These statements are scoped to the recorded hardware, software builds, and experiment constructions. Start
with the [study index](studies/kv-cache-page-size/README.md), then use its
[data manifest](studies/kv-cache-page-size/DATA_MANIFEST.md) to trace each claim to published or local-only
evidence.

## Published versus local material

The Git history contains technical reports, scripts, figures, selected raw text/JSON/CSV evidence, and
checksums. Research-direction notes, literature ideation, presentation builds, the generated research
portal, editor state, duplicate drafts, and profiler binaries remain available in the working copy under
`_local/` but are intentionally ignored.

The page-size study's broader available local raw collection also remains in
`studies/kv-cache-page-size/data/raw/`. Only the evidence sets explicitly listed as **published** in the
data manifest are versioned. This keeps the repository reviewable without silently implying that an absent
raw directory was recovered.

## Reproducibility

1. Read the status and limitations in the [study index](studies/kv-cache-page-size/README.md).
2. Locate the evidence set in the [data manifest](studies/kv-cache-page-size/DATA_MANIFEST.md).
3. Verify published data with [`data/SHA256SUMS`](studies/kv-cache-page-size/data/SHA256SUMS).
4. Run the associated analysis or figure script from
   [`studies/kv-cache-page-size/scripts/`](studies/kv-cache-page-size/scripts/).

Raw logs and profiler CSV exports are preserved as recorded, including machine-specific absolute paths.
Those paths document the original environment; the maintained scripts resolve repository paths relative to
their own location or expose machine-specific settings through environment variables.

## Citation and license status

No software or data license has been asserted in this repository, and no author list beyond the Git record
has been inferred. Cite the repository URL and the exact Git commit used. Add a project-approved license or
formal citation file only after the relevant owners confirm those terms.
