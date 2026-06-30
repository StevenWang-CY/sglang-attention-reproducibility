# Report 10 — The decisive test: FlashInfer + CUDA graph + distinct KV has *no* `page_size=1` decode cost (host page-independent + kernel scatter-robust)

> **PI ask (Jiaheng Lu, relayed):** for empirical closure on whether `page_size=1` can *ever* be notably worse than `page_size=128` under {CUDA graph ON, FlashInfer backend, **no shared prefix**}, run the one experiment that could still surface a penalty: push the per-step paged-index work to its extreme (B8 × L131072 = ~1M page entries), isolate the suspected per-step host cost (`init_forward_metadata_replay_cuda_graph` → index build + `plan`), and confirm with ncu whether the captured decode kernel stays DRAM-bound. *"Either outcome closes the question."*

| | |
|---|---|
| **Date** | 2026-06-26 |
| **Machine** | phastform.cis.upenn.edu — RTX 5060 Ti (sm120, 16 GB, L2 = 32 MB, ~448 GB/s HBM) |
| **Build** | `sglang_CW` editable checkout (`/home/wangcy07/latency_bench/sglang_CW`), flashinfer **0.6.6**, torch 2.9.1+cu128 |
| **Model dims** | Qwen3-VL-2B text: 16 Q / 8 KV heads (GQA-2), head_dim 128, KV = 4096 B/token/layer |
| **Regime** | distinct KV (true batch, **no shared prefix**), `page_size`→FlashInfer kernel shaped exactly as the engine, CUDA-graph-shaped |
| **Scripts** | [bench_pagecost.py](../scripts/bench_pagecost.py), [profile_pagecost_ncu.sh](../scripts/profile_pagecost_ncu.sh), [make_pagecost_figure.py](../scripts/make_pagecost_figure.py) |
| **Data** | `offline_batch_results/pagecost_profile/` (`pagecost_gather.json`, `pagecost_plan.json`, `flashinfer_{contig,scatter}_all_bs{B}_kv{L}.csv`) |
| **Figure** | [fig_pagecost.png](fig_pagecost.png) · **Audit:** [SUDO_CHANGES.md](SUDO_CHANGES.md) |

## Headline

**In this build, FlashInfer decode pays no `page_size=1` penalty in a true batch — by *either* possible mechanism — confirmed at the exact extreme cell (B8 × L131072 = 1,048,576 page entries).**

1. **The host path is page-independent *and* length-insensitive.** Reading the actual source settled it: the SGLang FlashInfer backend builds a **per-token** `kv_indices` list of length `seq_lens_sum = B·ctx` for **every** page size (`paged_kernel_lens = seq_lens`, no division), and hard-codes `page_size = 1` into the FlashInfer `plan`. So ps1 and ps128 do **identical** host work. And measured: `plan()` host cost is **~0.26 ms flat from 64 → 2,097,152 entries** — it does not even scale with index length. The "128× longer index list at ps1 → ≥5% host cost" hypothesis (which earlier reasoning and the literature-construct pass had assumed) is **false for this build**.
2. **The kernel path is gather-robust.** The *only* thing `--page-size` actually changes here is the allocator granularity → the **values (contiguity)** of `kv_indices`. Worst-case **fully-scattered** indices (random permutation = maximal ps1 fragmentation) cost the FlashInfer decode kernel only **+0.8 % to +1.3 %** vs contiguous — across the whole footprint ladder (0.03 → 8.6 GB), including B8×131072. `block128` (ps128-like contiguous runs) is identical to contiguous (±0.1 %).
3. **ncu says why:** contig and scatter both run at **96 % DRAM, 0.7 % L2-hit, and an *identical* 15.6 sectors/request**. Each token's KV is a contiguous **4 KB burst** (≥ the cache line), so scattering *which* chunks are read does not change *how efficiently* each is read; the kernel stays DRAM-bandwidth-bound either way.

**Verdict:** there is no constructible {graph ON + FlashInfer + distinct} cell where ps1 is notably (≥5 %) worse. The historical FlashInfer "+8.6 %" (report 4) is **not** a per-step decode host/index/plan/gather cost in this build. Prior microbenches (reports 7/8) used contiguous `arange` indices for both page sizes and so never tested scatter — this report closes that gap and finds scatter negligible anyway.

---

## 1. The code correction that reframed the experiment

The originally-specified experiment ("isolate the per-step `init_forward_metadata_replay_cuda_graph` cost; ps1 builds a 128× longer index list") rested on a mechanism that **does not exist in this build.** Read directly from `sglang_CW`:

- **KV pool is token-flat** — `memory_pool.py:841` allocates `(size + page_size, head_num, head_dim)`, i.e. `[num_tokens, n_kv, head_dim]`, *not* `[num_pages, page_size, …]`. `--page-size` does not reshape the KV tensor.
- **Decode index list is per-token, length `B·ctx`, page-independent** — in `flashinfer_backend.py` `call_begin_forward`, `paged_kernel_lens = seq_lens` (line 1051; no `//page_size` anywhere on the decode path), so `create_flashinfer_kv_indices_triton` emits `paged_kernel_lens_sum = seq_lens_sum = B·ctx` entries regardless of page size.
- **`page_size` is hard-coded to `1`** in both `begin_forward`/`fast_decode_plan` call sites. `self.page_size` is read only on the **speculative-draft** path (`generate_draft_decode_kv_indices`), not normal decode.

**Consequence:** for FlashInfer decode, `--page-size` changes *only* the allocator's granularity → which physical token slots a sequence occupies → the **values** of `kv_indices` (contiguous vs scattered). It does **not** change the index length, the plan, or the KV tensor shape. So the real (and only) question is: **does a scattered `kv_indices` (what a fragmented ps1 pool produces) slow the FlashInfer decode kernel?** Reports 7/8 could not answer it — their microbench used `kv_indices = arange(...)` (perfectly contiguous) for both ps1 and ps128.

## 2. Method

[bench_pagecost.py](../scripts/bench_pagecost.py) is weightless (no model), so the KV pool for B8×131072 (~4.3 GB) fits 16 GB and we reach the user's exact extreme cell. It mirrors the engine: token-flat KV `[N, 2, 1, n_kv, hd]`, per-token `kv_indices` of length `B·ctx`, `page_size=1` into `plan`.

- **`--mode plan`** — times `plan()` (the `fast_decode_plan` analogue, the dominant per-step host cost) as a function of index length, 64 → 2.1M entries. Bounds the host hypothesis.
- **`--mode gather`** — times the decode kernel (`wrapper.run`) at fixed distinct footprints with `kv_indices` = **contig** (fresh ps1/ps128), **block128** (ps128 contiguous runs, shuffled), **scatter** (random permutation = worst-case fragmented ps1). The 8 cells are 8 independent random permutations → 8 independent scatter realizations.
- **`--single` + ncu** ([profile_pagecost_ncu.sh](../scripts/profile_pagecost_ncu.sh), root, `--cache-control all` = cold = faithful to real multi-layer decode) — DRAM%, L2-hit%, sectors/request, SM% for contig vs scatter at B8×32768 and B8×131072.

## 3. Results

**Host `plan()` cost — flat, length-insensitive (per decode step):**

| index length (entries) | 64 | 512 | 8,192 | 65,536 | 262,144 | 1,048,576 | 2,097,152 |
|---|---|---|---|---|---|---|---|
| host ms | 0.257 | 0.267 | 0.258 | 0.270 | 0.269 | **0.258** | **0.257** |

ps1 (1.05M entries) vs ps128 (8.2k entries) at B8×131072: **0.2584 vs 0.2577 ms** — identical. Even a hypothetical 128× index would not cost more.

**Kernel gather — scatter ≤ +1.3 % everywhere:**

| B | L | tokens | footprint | contig (ms) | block128 | scatter | scatter vs contig |
|---|---|---|---|---|---|---|---|
| 8 | 1,024 | 8,192 | 0.03 GB | 0.1340 | 0.1318 | 0.1326 | −1.03 % |
| 8 | 8,192 | 65,536 | 0.27 GB | 0.7163 | 0.7159 | 0.7255 | +1.29 % |
| 8 | 32,768 | 262,144 | 1.07 GB | 2.6029 | 2.6016 | 2.6238 | +0.80 % |
| 8 | 65,536 | 524,288 | 2.15 GB | 5.1135 | 5.1135 | 5.1628 | +0.96 % |
| **8** | **131,072** | **1,048,576** | **4.29 GB** | **10.1418** | **10.1397** | **10.2409** | **+0.98 %** |
| 8 | 262,144 | 2,097,152 | 8.59 GB | 20.2042 | 20.2015 | 20.3777 | +0.86 % |
| 1 | 131,072 | 131,072 | 0.54 GB | 1.3451 | 1.3436 | 1.3562 | +0.83 % |
| 1 | 524,288 | 524,288 | 2.15 GB | 5.1181 | 5.1155 | 5.1736 | +1.09 % |

**ncu (cold), captured decode kernel, contig vs scatter:**

| cell | pattern | DRAM % peak | L2 hit % | sectors/request | SM % |
|---|---|---|---|---|---|
| B8 L32768 | contig | 96.6 | 0.7 | 15.58 | 19.4 |
| B8 L32768 | scatter | 95.7 | 0.7 | 15.58 | 19.2 |
| B8 L131072 | contig | 97.0 | 0.7 | 15.64 | 19.4 |
| B8 L131072 | scatter | 95.9 | 0.7 | 15.64 | 19.2 |

## 4. Mechanism — why scatter is free

The decode kernel is **DRAM-bandwidth-bound** (96 % of peak, 0.7 % L2-hit: distinct KV is cold-streamed every step, consistent with reports 7/8). The page-index scatter changes *which* token-KV chunks are fetched but not the **memory-transaction efficiency of each fetch**: one token's KV = `2 × n_kv × head_dim × 2 B = 4096 B`, a single contiguous burst that is already ≥ the cache line / well above the 32-B sector. So `sectors/request` is **identical** (15.6) for contiguous and scattered indices — the gather coalesces to the same number of transactions regardless of ordering, and total DRAM bytes are unchanged. The residual +1 % is sub-noise (minor page-table/TLB locality). This is exactly *not* the Triton mechanism (report 3): Triton's `_fwd_kernel` walks per-token indirect indices inside a `BLOCK_N=64` tile where contiguity changes coalescing; FlashInfer's fused paged kernel fetches each token-page as one burst.

## 5. Reconciliation with the rest of the study

- **Reports 5/8 (true batch flat):** confirmed and explained on this build — a fresh `bench_one_batch` allocation is contiguous at both page sizes, *and* even maximal scatter would only cost ≤1.3 %. Doubly flat.
- **Reports 7/8 microbench ("FI kernel page-flat"):** correct conclusion, but they used contiguous `arange` for both page sizes, so they tested only the easy case. This report tests the hard case (scatter) — still flat.
- **Report 4 ("FlashInfer ps1 +8.6 %", shared prefix, 39k, bs8, graph ON):** whatever it measured, it is **not** a per-step decode-path host/index/plan/gather cost in this build — all of those are page-independent (host) or scatter-robust (kernel). **Build verified (2026-06-27):** report 4 (gray) and this report (phastform) ran on a **byte-identical** `flashinfer_backend.py` (md5 `c169393e…`, mtime **2026-03-30**, unchanged on both boxes; `sglang_CW` is not even a git repo) — so it is *not* a "different/earlier build" with a page-divided index: the decode index was already token-flat / page-independent / `page_size`-hardcoded-to-1 at report-4 time, and grep finds **no** `//page_size`/`cdiv(...page)` anywhere. The remaining explanations are therefore (a) **measurement drift** (the documented ±4 % baseline wander in that exact 39k/bs8/shared/graph regime, plus the paired-vs-unpaired saga), or (b) a **non-decode-index engine effect** (radix-cache / scheduler / allocation) — the one untested sliver being **scattered `kv_indices` in the L2-resident (shared) regime** (report 10 tested scatter only DRAM-bound). It has **no true-batch analog**.
- **The construct workflow's "resolved mechanism" (per-step `create_flashinfer_kv_indices` + `fast_decode_plan`, 128× at ps1):** the *locus* (replay-side hook) was right, but the *scaling claim* (128× longer at ps1) is **wrong for this build** — the list is `B·ctx` at both page sizes, and `plan()` is length-insensitive anyway. Corrected here from source + measurement.

## 6. Verdict & limits

**There is no constructible {CUDA graph ON + FlashInfer + distinct KV} cell where `page_size=1` is notably worse — neither host nor kernel, neither at the extreme index length (2.1M entries) nor under worst-case fragmentation, on sm120.** This is the empirical closure the PI asked for. It strengthens the study's unifying rule: the `ps1` decode penalty is **reuse-gated (shared prefix), not footprint- or fragmentation-gated**.

Limits: single consumer GPU (sm120, 32 MB L2, ~448 GB/s) — the scatter-robustness rests on the 4 KB/token burst exceeding the cache line, which holds for any modern GQA/MHA layout but should be re-checked for MLA and on datacenter GPUs with different L2/sector behavior. The historical report-4 +8.6 % remains formally unexplained but is now bounded to *not* be a decode-path page cost on this build.

**Caveat closed — see [report 11](../report_11_mla_scatter_pagecost/report_11_mla_scatter_pagecost.md):** the MLA case (1152 B/token latent, decode *off the DRAM wall* via cross-head reuse) was tested. Worst-case scattered `ps1` costs more than MHA (~+4 % vs ~+1 %, because MLA is latency-bound not bandwidth-saturated, so scattered-page DRAM-access latency is exposed) but **saturates at +4.07 % and never reaches 5 %**, and realistic `block≥64` paging is free. So the conclusion holds for MLA too.
