# KV-cache `page_size` (block_size) vs decode latency — study index

**Research question:** find a scenario where KV-cache `page_size=1` is **not** the lowest-latency choice, by ≥5% (PI: Jiaheng Lu). Engine: SGLang (+ a vLLM cross-check). Hardware: RTX 5060 Ti (16 GB, sm120) on `gray.cis.upenn.edu`, with H100/B200 reference data where noted.

Everything new from this study lives under `page_size_study/`. **Each report sits in its own directory together with the PNG figures it uses.** The large raw result dirs stay in their canonical (gitignored) location `offline_batch_results/`, catalogued in **[DATA_INDEX.md](DATA_INDEX.md)**.

---

## Reports — each folder = one report + its figures

| # | report (folder/file) | status | figures in folder | one-line |
|---|---|---|---|---|
| 3 | **[report_3_dense_sweep_MAIN/](report_3_dense_sweep_MAIN/report_3_dense_sweep_MAIN.md)** | ✅ **CURRENT — start here** | 13 (`fig_dense_*`, `fig_models_*`, `fig_vllm_blocksize`) | The ≥5% answer: Triton `ps1` +15–40% (broad); FlashInfer corners; cross-model; vLLM. |
| 4 | **[report_4_flashinfer_graph_case/](report_4_flashinfer_graph_case/report_4_flashinfer_graph_case.md)** | ✅ **CURRENT** | 2 (`fig_flashinfer_graph_fullsweep`, `_longctx`†) | PI's hard case: FlashInfer + CUDA graph ON + no OOM → `ps1` +8.6% at ~39k/bs8 (paired). |
| 5 | **[report_5_bench_one_batch/](report_5_bench_one_batch/report_5_bench_one_batch.md)** | ✅ **CURRENT** | 2 (`fig_bob_truebatch`, `fig_bob_adversarial`) | PI's "真正的batch": in a **true batch** (`bench_one_batch`, independent seqs) `ps1` is the **fastest** page everywhere → the ≥5% effect was **shared-prefix-specific**. Incl. an **adversarial break-search** (high batch→512, L2-resident, long-seq, GQA-2) that found **no** `ps1` penalty (max +0.2%). |
| 1 | [report_1_kernel_microbench/](report_1_kernel_microbench/report_1_kernel_microbench.md) | 🗄 superseded | 5 (`fig1`–`fig5`) | Isolated-kernel + early e2e microbench; cold-start-artifact lessons. |
| 2 | [report_2_lowbatch_clean/](report_2_lowbatch_clean/report_2_lowbatch_clean.md) | 🗄 superseded | 6 (`fig_v2_*`, `fig6`–`fig9`‡) | Low-batch clean re-measure; "no ≥5%" verdict (overturned by report 3). |

† superseded by `fig_flashinfer_graph_fullsweep`. ‡ early V2 draft, never embedded.

**Headline finding** (report 3): on Triton, `page_size=1` loses **+15–40%** to `page_size=128` at high batch + shared prefix — every model tested, even with CUDA graph ON; magnitude tracks **layer count**, not KV/token. FlashInfer is far more robust but **not immune** (report 4: a narrow long-context corner). vLLM can't even express the pathology (it disallows `block_size < 16`). **Crucial scope (report 5):** this whole effect is **shared-prefix-specific** — in a *true* batch (independent sequences, `bench_one_batch`) `ps1` is the **fastest** page at every batch and context, both backends; the ≥5% penalty disappears. An **adversarial break-search** (FlashInfer+graph: batch→512, L2-resident KV, long single-seq, GQA-2 model) confirms it: **no** true-batch `ps1` penalty exists (max +0.2% across 19 cells).

---

## Layout

```
page_size_study/
  README.md                          ← you are here (master index)
  DATA_INDEX.md                      ← raw result dirs in offline_batch_results/ → experiment → report
  REPO_GUIDE.html                    ← standalone HTML guide to the sglang_log repo
  report_1_kernel_microbench/        report_1*.md  +  fig1–fig5
  report_2_lowbatch_clean/           report_2*.md  +  fig_v2_*, fig6–fig9
  report_3_dense_sweep_MAIN/         report_3*.md  +  fig_dense_*, fig_models_*, fig_vllm_blocksize   (CURRENT)
  report_4_flashinfer_graph_case/    report_4*.md  +  fig_flashinfer_graph_fullsweep, _longctx        (CURRENT)
  report_5_bench_one_batch/          report_5*.md  +  fig_bob_truebatch, fig_bob_adversarial            (CURRENT)
  scripts/                           figure generators + the vLLM measurement harness
  data/                              small local data (kernel-bench JSON, noise-control, consistency note)
```

Within a report folder, the report links its figures by bare filename (same directory). The main measurement harness, `measure_batch_latency_offline.py`, stays at the **repo root** (a tracked repo file, not new to this study).

---

## Generators (which script writes into which report folder)

Each generator reads raw data from `offline_batch_results/` (see DATA_INDEX.md) and writes its PNGs **into the report folder it belongs to**. Run from anywhere:

| script | writes into | figures | status |
|---|---|---|---|
| [scripts/make_dense_figures.py](scripts/make_dense_figures.py) | `report_3_dense_sweep_MAIN/` | `fig_dense_*` (10) + `data/dense_consistency.md` | ✅ current |
| [scripts/make_models_figures.py](scripts/make_models_figures.py) | `report_3_dense_sweep_MAIN/` | `fig_models_generalization/_driver`, `fig_vllm_blocksize` | ✅ current |
| [scripts/make_flashinfer_fullsweep_figure.py](scripts/make_flashinfer_fullsweep_figure.py) | `report_4_flashinfer_graph_case/` | `fig_flashinfer_graph_fullsweep` | ✅ current |
| [scripts/make_bench_one_batch_figure.py](scripts/make_bench_one_batch_figure.py) | `report_5_bench_one_batch/` | `fig_bob_truebatch` (+ `analyze_bench_one_batch.py` prints tables) | ✅ current |
| [scripts/make_bob_break_figure.py](scripts/make_bob_break_figure.py) | `report_5_bench_one_batch/` | `fig_bob_adversarial` (+ `analyze_bob_break.py` prints tables) | ✅ current |
| [scripts/make_flashinfer_graph_figure.py](scripts/make_flashinfer_graph_figure.py) | `report_4_flashinfer_graph_case/` | `fig_flashinfer_graph_longctx` | 🗄 superseded |
| [scripts/make_v1_figures.py](scripts/make_v1_figures.py) | `report_1_kernel_microbench/` | `fig1`–`fig5` | 🗄 superseded |
| [scripts/make_v2_figures.py](scripts/make_v2_figures.py) | `report_2_lowbatch_clean/` | `fig_v2_summary/_contention` | 🗄 superseded |
| [scripts/make_v2_extra_figures.py](scripts/make_v2_extra_figures.py) | `report_2_lowbatch_clean/` | `fig6`–`fig9` (draft) | 🗄 superseded |

```bash
python3 page_size_study/scripts/make_dense_figures.py               # report 3 figures
python3 page_size_study/scripts/make_models_figures.py              # cross-model + vLLM (report 3)
python3 page_size_study/scripts/make_flashinfer_fullsweep_figure.py # report 4 figure
python3 page_size_study/scripts/make_bench_one_batch_figure.py      # report 5 figure (true batch)
```

The vLLM harness [scripts/measure_vllm_block_size.py](scripts/measure_vllm_block_size.py) (re)generates `offline_batch_results/vllm_blocksize_5060ti/`. The true-batch runners [scripts/bob_sweep.sh](scripts/bob_sweep.sh), [scripts/bob_rerun.sh](scripts/bob_rerun.sh), [scripts/bob_fixup.sh](scripts/bob_fixup.sh) (main sweep) and the adversarial-search runners [scripts/bob_break.sh](scripts/bob_break.sh), [scripts/bob_xmodel.sh](scripts/bob_xmodel.sh), [scripts/bob_cd_knobs.sh](scripts/bob_cd_knobs.sh), [scripts/bob_4bfix.sh](scripts/bob_4bfix.sh), [scripts/bob_final.sh](scripts/bob_final.sh) (run on gray, idle-gated, skip-guarded) generate `offline_batch_results/bench_one_batch_5060ti/` for report 5.
