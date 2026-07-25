# Report 11 — Closing the MLA caveat: a small latent + latency-bound decode makes worst-case `ps1` scatter cost ~4 %, but it saturates **below** the 5 % threshold

> **Scope note (2026-07-17).** This report's measured BF16 MLA microbenchmark bound remains valid for its
> tested shapes and build. It does not establish a universal shared-reuse rule or cover FP8 MLA,
> full-engine end-to-end effects, datacenter GPUs, or other kernel families. Those remain unmeasured here.

> **PI ask (Jiaheng Lu, relayed):** report 10 closed FlashInfer+graph+distinct for MHA/GQA but flagged one untested case — **MLA** (DeepSeek-V2/V3), where the per-token KV is a single ~1152 B latent (vs MHA's 4096 B) and decode is high-arithmetic-intensity (the latent is reused across all query heads), i.e. *off the DRAM wall even with distinct KV*. That is exactly the regime where the page pattern is **not** bandwidth-hidden — so scatter could finally matter. Test it comprehensively: does worst-case `ps1` fragmentation cross 5 % for MLA?

| | |
|---|---|
| **Date** | 2026-06-26 |
| **Machine** | phastform.cis.upenn.edu — RTX 5060 Ti (sm120, 16 GB, L2 = 32 MB, ~448 GB/s HBM) |
| **Build** | flashinfer **0.6.6** `BatchMLAPagedAttentionWrapper` (backend=auto), torch 2.9.1+cu128 |
| **MLA dims** | `head_dim_ckv=512`, `head_dim_kpe=64` → **1152 B/token** (bf16); num query heads **16** (V2-Lite-like, memory-bound) and **128** (V2-like, higher-intensity) |
| **Regime** | distinct KV (true batch, **no shared prefix**); `kv_indices` values **contig / block64 / scatter** (= worst-case fragmented `ps1`) |
| **Scripts** | [bench_mla_pagecost.py](../../scripts/bench_mla_pagecost.py), [profile_mla_pagecost_ncu.sh](../../scripts/profile_mla_pagecost_ncu.sh), [make_mla_pagecost_figure.py](../../scripts/make_mla_pagecost_figure.py) |
| **Data** | `studies/kv-cache-page-size/data/raw/pagecost_profile/` (`mla_pagecost.json`, `mla_worstcase.json`, `mla_h{H}_{contig,scatter}_all_bs{B}_kv{L}.csv`) |
| **Figure** | [fig_mla_pagecost.png](fig_mla_pagecost.png) · **Audit:** [provenance.md](provenance.md) · **Predecessor:** [report 10](../10-flashinfer-page-cost/report.md) |

## Headline

**MLA is the most page-sensitive distinct-KV case in the whole study — and it still does not reach 5 %.** Worst-case fully-scattered `kv_indices` (maximal `ps1` fragmentation) cost the FlashInfer MLA decode kernel:

- **16 heads (V2-Lite, memory-bound): +2.8 % to +4.0 %**; pushed to the worst case (B1, single long scattered sequence) it **saturates at +4.07 %** and stays there from 0.3 GB → 2.4 GB (it does **not** grow into 5 %).
- **128 heads (V2, higher-intensity): +1.7 % to +2.7 %**.
- **`block64` (realistic paged layout, contiguous 64-token runs) ≈ contiguous everywhere (±0.3 %)** — only *full* token-level fragmentation costs anything.

**Mechanism (ncu, cold):** the cost is **not** coalescing — `sectors/request` is **identical** contig vs scatter (10.9 / 11.9 / 13.2; the 1152 B latent is an aligned multi-cache-line burst regardless of order). It is **DRAM-access latency** (row-buffer + TLB misses from random page order), and it shows **only because MLA decode is latency-bound, not bandwidth-saturated**: DRAM is at **15–38 % of peak** here (vs MHA's **96 %** in report 10). MHA saturates bandwidth, so scattered-but-coalesced reads still run at the wall and scatter is hidden (≤1.3 %); MLA leaves the wall (latent reuse across heads), so the extra per-access latency is exposed — but bounded, hence the ~4 % plateau.

**Verdict for this measured MLA path:** worst-case token-scatter remains **under 5%** (~4.07%), and the
tested block-64 layout is flat. This is a scoped kernel bound, not a universal rule that ≥5% requires
cross-sequence reuse; report 16 demonstrates that page-aware kernel work is the other necessary dimension.

---

## 1. Why MLA is the one case that could have broken it

Report 10 showed MHA/GQA `ps1` scatter is ≤1.3 % because (a) the host path is page-independent and (b) the kernel is **bandwidth-saturated** (96 % DRAM), so a scattered-but-internally-coalesced gather still runs at the wall. MLA breaks assumption (b): the single compressed latent (`kv_lora_rank=512` + rope `64` = 1152 B/token, bf16) is read **once and reused across all query heads** inside the kernel, so MLA decode has high arithmetic intensity and is **not** bandwidth-bound. If "scatter is cheap only because it's hidden behind the bandwidth wall," MLA — off the wall — should expose it. (It also has a 3.5× smaller per-token burst, the closest thing to the "sub-cache-line" worry; though 1152 B is still 9 aligned cache lines.)

## 2. Method

[bench_mla_pagecost.py](../../scripts/bench_mla_pagecost.py) (weightless: B8×L131072 MLA-KV ≈ 1.2 GB) drives `flashinfer.mla.BatchMLAPagedAttentionWrapper` with `page_size=1`, one query token per request (decode), and `kv_indices` set to **contig** (fresh ps1/ps128), **block64** (ps64-like contiguous runs, shuffled), or **scatter** (random permutation = worst-case fragmented ps1). Footprint ladder B8×{4k…262k} and B1×{131k…2.1M}; head counts 16 and 128; CUDA-event median (20 warmup / 100 iters). ncu ([profile_mla_pagecost_ncu.sh](../../scripts/profile_mla_pagecost_ncu.sh), root, cold) at B8×131072 (h16, h128) and B1×524288 (h16). The full DeepSeek-V2-Lite *engine* doesn't fit 16 GB (≈16 B params), so this is the isolated-kernel test — the decisive level, as in reports 7/8/10.

## 3. Results

**Scatter & block64 vs contiguous (kernel latency):**

| heads | B | L | footprint | contig (ms) | block64 | **scatter** |
|---|---|---|---|---|---|---|
| 16 | 8 | 8,192 | 0.08 GB | 0.486 | −0.1 % | **+3.3 %** |
| 16 | 8 | 32,768 | 0.30 GB | 2.280 | −0.0 % | **+2.8 %** |
| 16 | 8 | 131,072 | 1.21 GB | 9.324 | +0.0 % | **+3.1 %** |
| 16 | 8 | 262,144 | 2.42 GB | 18.602 | +0.0 % | **+3.1 %** |
| 16 | 1 | 524,288 | 0.60 GB | 3.182 | +0.0 % | **+4.0 %** |
| 128 | 8 | 131,072 | 1.21 GB | 15.596 | +0.0 % | **+2.2 %** |
| 128 | 8 | 262,144 | 2.42 GB | 31.117 | +0.1 % | **+2.4 %** |
| 128 | 1 | 524,288 | 0.60 GB | 6.297 | +0.1 % | **+2.7 %** |

**Worst-case bound (B1, h16, single long scattered sequence) — does it cross 5 %?**

| L | footprint | contig (ms) | scatter |
|---|---|---|---|
| 262,144 | 0.30 GB | 1.642 | +3.98 % |
| 524,288 | 0.60 GB | 3.181 | +4.08 % |
| 1,048,576 | 1.21 GB | 6.306 | +4.07 % |
| 2,097,152 | 2.42 GB | 12.561 | **+4.07 %** |

It **saturates at ~4.07 %** — flat across an 8× footprint range; it does not trend toward 5 %.

**ncu (cold), contig vs scatter:**

| cell | pattern | DRAM % | L2 hit % | **sectors/request** | SM % |
|---|---|---|---|---|---|
| h16 B8 L131072 | contig | 26.0 | 1.8 | **10.91** | 23.9 |
| h16 B8 L131072 | scatter | 25.4 | 1.8 | **10.89** | 23.5 |
| h128 B8 L131072 | contig | 15.6 | 50.2 | **11.95** | 28.6 |
| h128 B8 L131072 | scatter | 15.4 | 50.3 | **11.92** | 28.2 |
| h16 B1 L524288 | contig | 38.4 | 0.3 | **13.19** | 35.5 |
| h16 B1 L524288 | scatter | 37.5 | 0.3 | **13.19** | 34.6 |

## 4. Mechanism

Two ncu facts pin it:
1. **`sectors/request` is identical** (contig ≈ scatter to ±0.02). The 1152 B latent (ckv 1024 B = 8 cache lines + kpe 128 B = 1 cache line, both 128-B-aligned) is fetched as the same number of memory transactions regardless of ordering — so this is **not** a coalescing/over-fetch effect. (Same finding as MHA in report 10.)
2. **DRAM is only 15–38 % of peak** (not MHA's 96 %). MLA decode at small batch is **latency-/occupancy-bound** (one query token × few heads → low parallelism even with split-KV). Scatter raises the *average* DRAM-access latency (random 4 KB-page order → DRAM row-buffer misses + TLB misses); because the kernel is latency-bound rather than bandwidth-saturated, that extra latency is **not** hidden and shows directly as +3–4 %.

This sharpens report 10's rule: scatter is cheap when the kernel is **bandwidth-saturated** (MHA, 96 % → scatter hidden, ≤1.3 %) and becomes visible (but bounded) when the kernel is **latency-bound** (MLA, ≤38 % → +~4 %). The more memory-bound head count (16, DRAM 26–38 %, L2-hit ~0) is more scatter-sensitive than the higher-intensity one (128, DRAM 15 %, L2-hit 50 %) — consistent: more DRAM streaming → more latency exposure.

## 5. Implications for real systems

- **Realistic paging is free.** `block64` ≈ contiguous (±0.3 %). SGLang and vLLM MLA backends use `page_size ≥ 64`, so the latent of a sequence is stored in contiguous 64-token runs — the page-table indices point to contiguous blocks, not random tokens. The +4 % requires **token-level (`ps1`) full fragmentation**, which paged allocators specifically avoid.
- **Even the pathological case is < 5 %.** A maximally fragmented `ps1` MLA pool tops out at ~4.07 %; it never reaches the PI's 5 % bar.
- So MLA does **not** provide the distinct-KV `ps1` penalty the study has been hunting — it is the closest (≈4 % vs MHA ≈1 %), confirming the gradient (smaller burst + off-the-wall → more scatter latency) while still landing below threshold.

## 6. Verdict & limits

**The report-10 caveat is closed for the measured FlashInfer distinct-KV paths: across MHA/GQA and MLA,
there is no ≥5% `ps1` decode penalty**—MHA ≤1.3% (bandwidth-saturated, scatter hidden), MLA ≤~4.07%
(latency-bound, scatter partly exposed but bounded), and realistic `block≥64` paging is flat in both.
Reports 15–16 later refine the cross-study conclusion: sharing/off-wall operation can expose extra work,
but a page effect occurs only when the chosen backend actually performs page-dependent work.

Limits: single consumer GPU (sm120; 32 MB L2; ~448 GB/s; specific DRAM/TLB behavior). **fp8 MLA KV** (576 B/token — the genuinely smallest burst, ~4.5 cache lines) is untested (sm120 fp8-MLA kernel availability unverified) and is the one remaining knob that could push the latency-bound scatter cost higher; the saturating ~4 % bf16 result and the 128-B-aligned burst structure suggest it would likely still stay sub-5 %, but that is an inference, not a measurement. The DeepSeek-V2-Lite *engine* (true end-to-end TPOT) doesn't fit 16 GB; this is the isolated-kernel level, consistent with the rest of the study.
