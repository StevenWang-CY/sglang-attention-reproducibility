# Artifact policy

This policy defines what belongs in the public Git repository and what remains local.

## Included

- Technical reports with an explicit current, historical, or superseded status.
- Figures used by those reports.
- Analysis, benchmark, parsing, and profiling scripts needed to interpret or reproduce the findings.
- Small curated datasets and experiment inputs.
- Selected raw text, JSON, JSONL, and CSV evidence that directly supports current reports.
- Provenance notes and SHA-256 checksums.
- Historical technical material under `archive/` when removing it would erase useful provenance.

## Kept local and ignored

- Research directions, literature ideation, novelty verdicts, and proposal drafts.
- Presentation source/build directories, rendered slide inspections, and temporary office files.
- Generated websites and their dependency/build trees.
- Editor or agent state, runtime caches, and duplicate drafts.
- Binary profiler captures and derived SQLite databases when a text/CSV export is retained.
- Large raw experiment collections that are not part of the explicitly published evidence boundary.

Local-only material is organized under `_local/` by artifact type, research direction, and ISO date. The
page-size study's unpublished raw data stays under `studies/kv-cache-page-size/data/raw/` because the
analysis scripts use that canonical location; `.gitignore` admits only named published evidence sets.

## Data integrity and provenance

- Raw evidence is not rewritten merely to remove machine paths or normalize formatting.
- Renames and directory moves may change paths but not measurement contents.
- Published study data is covered by `studies/kv-cache-page-size/data/SHA256SUMS`.
- Missing source logs are recorded as provenance gaps; derived tables and figures are not described as a
  substitute for absent raw runs.
- A behaviorally matched rebuild is not described as the original experiment's literal checkout.
- Historical reports remain available, but the study index and report banners identify claims superseded
  by later evidence.

## Commit hygiene

Before publication, validate relative links, Python and shell syntax, staged-file scope, ignored local
material, checksums, large files, and likely credentials. Commits should identify only confirmed authors
and should not add inferred co-author trailers.
