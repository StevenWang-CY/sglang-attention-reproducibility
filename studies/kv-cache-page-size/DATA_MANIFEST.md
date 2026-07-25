# Data manifest

All study data is rooted at [`data/`](data/). This manifest distinguishes files published in Git from
larger experiment collections that remain in the working copy only.

## Published evidence

| Path | Supports | Contents and scope |
|---|---|---|
| `data/curated/` | Reports 01 and 03; report-14 figure | Small benchmark JSON, order controls, datacenter reference outputs, consistency tables, and a clearly labeled derived table for the report-14 figure. |
| `data/inputs/browser-request-samples/` | Historical workload construction | Request/tree samples retained as technical inputs, not benchmark results. |
| `data/raw/report-14-shared-prefix-page-kernel-rtx5060ti/` | Report 14 and report 15 controls | Rebuild summaries, engine index-contiguity dumps, shared/distinct kernel CSVs, prompt input, and run provenance. The separate exclusive-run per-step logs underlying some report-14 summary values were not retained. |
| `data/raw/report-15-gray-log-forensics/report-03-dense-sweep/` | Report 15 §1 | Exact original `hb_tri_g_ps{1,8,32,128}` JSON/log pairs used to compare the old whole-call metric with same-run steady decode. |
| `data/raw/report-15-gray-log-forensics/report-04-flashinfer-graph/` | Report 15 §1 | Exact original three paired `fgi_p{1,2,3}_ps{1,128}.log` runs used for same-run steady-decode closure. |
| `data/raw/report-16-shared-prefix-engine/` | Report 16 A1 | Shared-prefix engine logs for FlashInfer, Triton, and `trtllm_mha`/XQA, plus parsed summaries. |
| `data/raw/report-16-true-batch-engine/` | Report 16 A2 | One round each for the documented off-wall B2/L1024 and on-wall B8/L4096 cells, plus an XQA-only B1/L512 pilot across pages 16/32/64. |
| `data/raw/report-16-ncu-primary/` | Report 16 C1 | Primary XQA/FlashInfer profiler CSVs, JSON controls, and replay exports. Binary profiler captures are excluded. |
| `data/raw/report-16-mechanism-sweep/` | Report 16 §3.1 | XQA page/regime/length sweep, full-profile cells, and FlashInfer/Triton controls. |
| `data/raw/report-16-verification/` | Report 16 §6 | Three replay summaries for latency/correctness verification. |
| `data/SHA256SUMS` | All other published files below `data/` | SHA-256 digest manifest using paths relative to `data/`; the manifest does not checksum itself. |

The Git checkout contains the published paths above. `.gitignore` rejects every other direct child of
`data/raw/` unless it is explicitly admitted.

## Local-only raw collections

The following directories remain locally under `data/raw/` for ongoing analysis but are not part of the
public Git artifact:

| Collection | Primary reports | Summary |
|---|---:|---|
| `dense_5060ti/`, `gen_models_5060ti/` | 03–04 | Dense, high-batch, cross-model, and FlashInfer graph sweeps. The exact report-15 subset is separately published. |
| `bench_one_batch_5060ti/` | 05 | SGLang true-batch main and adversarial sweeps. |
| `xqa_profile/`, `xqa_profile_5060ti/` | 06 | XQA/FlashInfer profiler and latency sweeps, including TMA survey material. |
| `sharedprefix_profile/` | 07 | Shared-versus-distinct profiler and engine controls. |
| `bench_one_batch_smallkv_5060ti/`, `smallkv_profile/` | 08 | Small-KV engine and isolated-kernel ladders. |
| `vllm_smallkv_5060ti/` | 09 | vLLM FlashInfer block-size and preemption controls. |
| `pagecost_profile/` | 10–11 | FlashInfer MHA and MLA contiguity/scatter tests. |
| `reuse_profile/` | 12 | L2 reuse/residency grid and cold/warm controls. |
| `offwall_profile/`, `bench_one_batch_offwall_5060ti/` | 13 | Off-wall profiler, engine, and timeline experiments. |
| `triton_sweep_5060ti/`, `triton_clean_5060ti/`, `fi_clean_5060ti/`, `ps_e2e_5060ti/`, `ps_nocudagraph_5060ti/`, `vllm_blocksize_5060ti/` | 01–03 | Early baseline and cross-check collections. |

The H100 input-length dataset is outside this study and now lives at
`archive/legacy-sglang-experiments/offline_batch_results/H100-80GB-HBM3/`.

## Explicit gaps and qualifications

1. Report 15's constructed fragmentation experiment does not have an independently retained raw-run
   directory. The published report records the derived result and marks this gap.
2. Some report-14 summary values are retained in the report and derived figure table, while the matching
   exclusive-run per-step log set is absent. The available report-14 raw directory contains related rebuild,
   contiguity, and kernel-control evidence and must not be described as a complete archive of every table.
3. Report 16's direct `trtllm-gen` prefill architecture probe was not saved as a standalone raw log.
4. `report-16-true-batch-engine/` contains one round for the two cross-backend cells and an XQA-only
   B1/L512 pilot; it does not establish multi-round convergence.
5. Raw logs may contain historical absolute machine paths. They are provenance, not portable defaults.

## Naming and immutability

- Report evidence directories use `report-NN-<descriptive-slug>`.
- Raw measurement contents are not edited during curation.
- Derived summaries and figures identify their source or provenance limitation.
- Additions to a published evidence set require regenerating `data/SHA256SUMS`.
