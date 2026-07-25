# Study data

| Directory | Role |
|---|---|
| `curated/` | Small normalized or derived artifacts used by reports and figures |
| `inputs/` | Technical workload inputs |
| `raw/` | Recorded logs, JSON/JSONL outputs, and profiler CSV exports |
| `SHA256SUMS` | Integrity manifest for every other file published below `data/` |

The working copy may contain more raw directories than Git. Publication is allow-listed in the repository
`.gitignore`; consult [`../DATA_MANIFEST.md`](../DATA_MANIFEST.md) before treating a directory as available
to another clone.

Raw evidence is immutable during organization. Machine-specific paths, timestamps, and tool output are
preserved. Maintained scripts should use repository-relative paths and environment variables rather than
copying those historical paths as defaults.
