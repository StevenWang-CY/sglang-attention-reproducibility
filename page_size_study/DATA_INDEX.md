# Data index — raw result dirs

The large raw measurement outputs are **not** moved into `page_size_study/`: they stay in the repo's canonical results location `offline_batch_results/`, which is **gitignored** (`offline_batch_results/*`, except the committed `H100-80GB-HBM3/`). That dir is also the rsync target the gray runs write to and the path every figure generator reads. This file catalogues what each dir is.

Path: all dirs below are under `offline_batch_results/`.

| dir | size | experiment | read by (script → report) |
|---|---|---|---|
| **`dense_5060ti/`** | 74 M | Dense `batch × page_size` re-run (radix ON/OFF), Qwen3-VL-2B — the MAIN sweep | `make_dense_figures.py`, `make_models_figures.py` → report 3 |
| **`gen_models_5060ti/`** | 17 M | Cross-model sweep (Qwen2.5-3B, Qwen3-VL-2B/4B/8B-FP8) + FlashInfer graph-ON long-ctx (`fgf_*`, `fgr_*`, `fgi_*`) | `make_models_figures.py`, `make_flashinfer_fullsweep_figure.py`, `make_flashinfer_graph_figure.py` → reports 3 & 4 |
| **`bench_one_batch_5060ti/`** | few M | **TRUE-batch** `bench_one_batch` data, CUDA graph ON, independent seqs → report 5. Main sweep: Qwen3-VL-2B page{1,8,32,128}×{triton,flashinfer}×6 (B,L)×2 rounds (`bob_r*_*.jsonl`). **Adversarial break-search:** `bobx_*` (2B fine-grid high-batch/L2-resident/long-seq), `bobm_<model>_*` (Qwen2.5-3B GQA-2 to batch 512, Qwen3-VL-4B), `bobg_*` (graph knobs). | `analyze_bench_one_batch.py`+`make_bench_one_batch_figure.py` (main); `analyze_bob_break.py`+`make_bob_break_figure.py` (adversarial) → report 5 |
| `triton_sweep_5060ti/` | 72 K | Triton page sweep, 10k ctx, CUDA graph ON (`tri_10k_g_*`) | `make_v2_figures.py`, `make_dense_figures.py`, `make_v2_extra_figures.py` → reports 2, 3 |
| `triton_clean_5060ti/` | 32 K | Triton, no-graph, clean (`tri_clean_ng_*`) | `make_v2_figures.py`, `make_dense_figures.py` → reports 2, 3 |
| `fi_clean_5060ti/` | 32 K | FlashInfer clean, graph & no-graph (`fi_clean_{g,ng}_*`) | `make_v2_figures.py`, `make_dense_figures.py` → reports 2, 3 |
| `ps_e2e_5060ti/` | 16 K | End-to-end engine TPOT baseline, FlashInfer graph ON (`results_ps*`) | `make_v1_figures.py`, `make_v2*` , `make_dense_figures.py` → reports 1, 2, 3 |
| `ps_nocudagraph_5060ti/` | 20 K | FlashInfer graph ON/OFF, long ctx, bs1 (`ctx*_ps*`) | `make_v2_figures.py`, `make_v2_extra_figures.py` → report 2 |
| `vllm_blocksize_5060ti/` | 76 K | vLLM `block_size` sweep (sm120, FLASH_ATTN backend) | `make_models_figures.py`, `make_v2_extra_figures.py` → report 3 |
| `H100-80GB-HBM3/` | 1.2 M | **Committed** reference data (Qwen3-VL-32B input-length sweep) — *not part of this study's new data* | — |

## Small local data (in `page_size_study/data/`)

| file | what |
|---|---|
| `page_size_attn_bench_5060ti.json` | Isolated FlashInfer decode-attention kernel microbench, RTX 5060 Ti (report 1) |
| `confirm_A_5060ti.json` | Warm re-measurement of the bs1/kv1024 cell (corrects a cold-start artifact; report 1) |
| `noise_test/{fwd,rev}_*.json` | Forward/reverse page-order noise control (report 1, Fig 5) |
| `dense_consistency.md` | Self-consistency table emitted by `make_dense_figures.py` |

(The datacenter kernel bench `page_size_attn_bench.json` referenced by report 1 is a pre-existing repo-root file, not part of this study.)
