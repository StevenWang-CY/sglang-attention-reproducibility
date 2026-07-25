# Report 12 — Why is L2-hit ≈ 0% for distinct (true-batch) decode? **Reuse, not residency.**

> **PI ask (Jiaheng Lu, relayed by Chuyue):** he is uncertain why report 7 finds an **almost-0% L2 cache hit-rate in the distinct (true-batch) case** — intuition says "the L2 is 32 MB, the KV should give *some* hits." Investigate the original setting, reproduce across **more batch×context combinations**, and **fully determine the cause** with controlled experiments.

| | |
|---|---|
| **Question** | Is distinct-KV decode L2-hit ≈0% because the footprint exceeds L2, or because each KV byte is read *once* (no reuse)? |
| **Answer** | **Reuse, not residency.** L2-hit ≈ **(R−1)/R**, where R = each KV byte's intra-kernel reuse factor. Distinct decode = **R=1** (every KV byte read exactly once; GQA's 2× head-share is resolved in registers/SMEM, never re-traversing L2) → ≈0% **even when the footprint fits in L2**. Reuse is the *only* thing that makes L2 hits: shared prefix (R=B) or MLA (R≈heads). |
| **Method** | `bench_xqa.py` FlashInfer decode kernel, ncu `--single` (one launch), **cold `--cache-control all`** = faithful to real per-step decode. Reuse knob = `--kv-mode shared` with batch B (B sequences re-read one KV copy → R=B; B=1 shared ≡ distinct). + raw L2 hit/miss **sector counts** to decompose the rate. |
| **Model / HW** | Qwen3-VL-2B (16 Q / 8 KV heads, 4096 B/token/layer) · RTX 5060 Ti (16 GB, sm120, **L2 = 32 MB**) · **phastform** (ncu 2025.3.1 as root) |
| **Figures / data** | [fig_reuse_law.png](fig_reuse_law.png) (headline) · [fig_reuse_grid.png](fig_reuse_grid.png) · `studies/kv-cache-page-size/data/raw/reuse_profile/` (58 cells) · [provenance.md](provenance.md) |

## Headline

**The 0% is correct, reproducible across the whole (batch, context) plane, and has a clean mechanism: L2-hit tracks the intra-kernel reuse factor R, with footprint essentially irrelevant.**

- **Reuse-factor sweep at a *pinned* 4 MB footprint (1/8 of L2):** L2-hit climbs **2.6 → 50 → 75 → 87 → 94 → 97%** for R=B=1→32, matching **(R−1)/R** to ≤0.5 pp for R≥2 (R=1 floors at ~2.5%, the small incidental non-KV/boundary-read hits — the same ≤2% floor seen in the distinct grid; R=1 is "no reuse" and is ≈0% in the reuse sense). Only R changes; the footprint never moves.
- **Reproducibility grid (5 batch × 4 context):** distinct L2-hit is **0.01–2.0% in *every* cell, including sub-L2 ones** (B1/L1024 = 4 MB → 2.0%); shared is **banded by B (=R) and flat in L** (B8 = 87.0/87.3/87.4/87.4% across L=1k…16k).
- **Cold vs warm (the artifact behind the doubt):** the only way distinct shows L2 hits is **warm** (no cache flush) at sub-L2 footprints — **73.5% at 16.8 MB, decaying to 0.1% above 32 MB**. That is **cross-launch residency** from the 100-iter loop re-running the same kernel on resident KV; it is *not* intra-kernel reuse and does not occur in real decode (28 layers + weights evict everything between a layer's consecutive steps). **Cold ≈ 0% is the faithful number.**
- **Raw counts kill the "rate-metric quirk":** distinct L2 misses are **99.4–99.8% of all sector lookups** (compulsory first-touch misses) and **miss×32 B ≈ DRAM bytes read** (33.8 ≈ 33.6 MB; 268.7 ≈ 268.5 MB) — every KV byte goes to HBM exactly once. Shared misses ≈ 1/8 (hit ≈ 7/8). The hit-rate equals hit/(hit+miss) to <0.1 pp.

**Bottom line for the PI:** a cache *hit* requires re-reading a line while it is still resident. Distinct true-batch decode reads each KV byte once per kernel, so there is nothing to re-hit — 0% is exactly what a no-reuse stream gives, regardless of whether 4 MB or 512 MB of KV "fits" in the 32 MB L2.

---

## 1. The premise the doubt rests on

The intuition "32 MB L2 ⇒ some hits" conflates **residency** (does the data fit?) with **reuse** (do we read the same line twice while it's resident?). Cache *hit-rate measures reuse*. A pure streaming read — each line touched once — has 0% hit-rate **by definition**, even if the whole stream fits in cache. The original report-7 number was measured **cold** (`--cache-control all`, flushing between ncu replay passes), which is faithful to real decode: by the time a layer's attention runs again next step, 27 other layers + 4 GB of weights have thrashed the L2. So the only question is whether distinct decode has intra-kernel reuse. It does not.

## 2. The model

For a KV byte with intra-kernel reuse factor R: the 1st touch is a compulsory miss; the other R−1 touches hit L2 iff still resident → **L2-hit ≈ (R−1)/R** when the re-read working set ≤ L2, and **≈0% when R=1, independent of footprint.** Distinct MHA decode has R=1 (one query token reads each KV byte once; GQA's 2× is absorbed in SMEM). Shared prefix gives R=B (B sequences re-read one copy). MLA gives R≈heads (one latent re-read across query heads).

## 3. Reproducibility grid (Exp A) — the PI's "more sets/combinations"

FlashInfer, cold, ps128, B∈{1,4,8,16,32} × L∈{1024,4096,8192,16384}. **L2-hit %:**

| B \ L | 1k (d/s) | 4k (d/s) | 8k (d/s) | 16k (d/s) |
|---|---|---|---|---|
| **1** | 2.0 / 2.5 | 1.2 / 1.2 | 0.7 / 0.7 | 0.4 / 0.3 |
| **4** | 1.2 / 74.8 | 0.3 / 74.8 | 0.2 / 75.0 | 0.1 / 74.9 |
| **8** | 0.6 / 87.0 | 0.2 / 87.3 | 0.1 / 87.4 | 0.1 / 87.4 |
| **16** | 0.3 / 93.5 | 0.1 / 93.7 | 0.0 / 93.7 | 0.0 / 93.8 |
| **32** | 0.1 / 96.6 | 0.0 / 96.8 | 0.0 / 96.7 | 0.0 / 96.8 |

(d = distinct, s = shared.) **Distinct ≤2% everywhere** — including sub-L2 cells (B1/L1024 = 4 MB, B4/L1024 = 16 MB). **Shared is banded by B (=R) and flat across L** — footprint (which grows 16× along each row) does not move the hit-rate; R does. **B1-shared ≈ B1-distinct** (2.5 ≈ 2.0%): with no batch there is no reuse, so "shared" with B=1 is operationally identical to distinct (built-in consistency check C3).

## 4. Decisive #1 (Exp B) — reuse at a FIXED footprint

Shared, cold, ps128, **L=1024 → footprint pinned at 4 MB (1/8 L2)**, vary B=R:

| R=B | 1 | 2 | 4 | 6 | 8 | 12 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|---|---|---|
| **measured L2-hit %** | 2.6 | 50.2 | 74.8 | 83.0 | 87.0 | 91.3 | 93.5 | 95.6 | 96.6 |
| **(R−1)/R model %** | 0 | 50.0 | 75.0 | 83.3 | 87.5 | 91.7 | 93.8 | 95.8 | 96.9 |
| DRAM % peak | 66.7 | 66.0 | 50.1 | 36.9 | 30.2 | 20.6 | 15.8 | 10.9 | 8.6 |

The hit-rate traces **(R−1)/R** to ≤0.5 pp for R≥2 (R=1 sits ~2.5 pp above the model's 0% — the small incidental-hit floor, not KV reuse) **with the footprint constant at 4 MB** — so the climb cannot be a residency effect; it is purely reuse. DRAM% falls in lockstep (more reuse → fewer bytes streamed). **Dual control** (fix R=B8, sweep footprint 1 MB→160 MB by varying L): L2-hit stays **flat ~87%** across a 160× footprint range (incl. >L2) — the level is set by R, not footprint. The orthogonal cross (Fig A panels A+B) is the proof.

## 5. Decisive #2 (Exp C) — cold vs warm distinct: the residency artifact, isolated

Distinct, B8, ps128, footprint = B·L·4096:

| L (footprint) | **cold** L2-hit % | **warm** L2-hit % |
|---|---|---|
| 512 (16.8 MB) | 1.3 | **73.5** |
| 1024 (33.6 MB) | 0.6 | 20.4 |
| 2048 (67 MB) | 0.3 | 5.0 |
| 4096 (134 MB) | 0.2 | 1.3 |
| 8192 (268 MB) | 0.1 | 0.4 |
| 16384 (537 MB) | 0.1 | 0.1 |

**Cold ≈0% everywhere.** **Warm** peaks at **73.5% just below L2** and decays to ~0 above 32 MB — the classic signature of **cross-launch residency**: the previous loop iteration left the KV in L2 and this iteration's (single, R=1) touch hits it. The warm hit is footprint-shaped and fragile; it is *not* intra-kernel reuse and does not happen in real multi-layer decode. This is exactly the warm intuition behind the PI's doubt — and it is the artifact, not the truth.

## 6. Count decomposition (Exp E) — the 0% survives in raw sectors

Cold, ps128 (sector counts, not just the rate):

| cell | rate % | hit/(hit+miss) % | miss / total | miss×32 B | DRAM read |
|---|---|---|---|---|---|
| distinct B8/L1024 | 0.56 | 0.56 | **0.994** | 33.8 MB | 33.6 MB |
| distinct B8/L8192 | 0.09 | 0.09 | **0.998** | 268.7 MB | 268.5 MB |
| shared B8/L1024 | 87.03 | 87.10 | 0.129 | 4.4 MB | 4.2 MB |
| shared B8/L8192 | 87.39 | 87.44 | 0.126 | 33.8 MB | 33.6 MB |

(1) The rate **equals** hit/(hit+miss) to <0.1 pp → it is an honest ratio, not a metric quirk. (2) Distinct misses are **99.4–99.8% of all lookups** → compulsory first-touch misses. (3) **miss×32 B ≈ DRAM bytes read** → every missed sector goes to HBM and accounts for the full KV stream. (4) Shared reads **~1/8** the bytes (hit ≈ 7/8) — the bytes that don't miss are exactly the reused ones. The 0% is real at the count level (hit.sum ≈ 6 k of ~1.06 M lookups).

## 7. Corroboration (Exp D) — MLA cross-head reuse (report 11 data)

A *second, independent* reuse axis. From report 11 (cold, **distinct** KV): MLA **h128 → 50%** L2-hit, **h16 → 1.8%**, vs MHA-GQA2 → 0.6%. Even distinct KV gets L2 hits when one latent is re-read across query heads. **Honest caveat (it strengthens the thesis):** the MLA points sit *below* the cross-sequence (R−1)/R curve (Fig grid panel D) because head-reuse, like GQA's 2×, is **largely absorbed in registers/SMEM** — it only spills back to L2 when the head count is large enough (h128 → 50%; h16 stays ~2%). This is the *same* mechanism that makes GQA-2's 2× reuse invisible to L2 in distinct MHA. Unifying statement: **L2 sees only the reuse that actually re-traverses L2** — cross-sequence (shared prefix) or head-reuse too large to hold on-chip; small head-group reuse is resolved before L2.

## 8. Confounds & controls

- **Warm-residency masquerading as reuse** → primary regime cold; warm shown only in §5 as the named artifact (C1).
- **Reuse entangled with footprint** → §4 pins footprint at 4 MB for all R + the dual fixed-R footprint sweep is flat (C2).
- **B1-shared ≟ distinct** → coincide at ~2% (§3) (C3).
- **ps1 vs ps128 sliver** → distinct ps1 = 1.6/0.8% vs ps128 = 0.6/0.1% (both <2%, ≈0). This SGLang lowering uses the same per-token B×L index length at both pages (report 10), so the sliver is not evidence of page-table-length traffic or KV reuse; the total DRAM-read ratio is effectively unity (0.998–0.999) (C4).
- **Counter validity** → the same `lts__t_sector_hit_rate.pct` reads 75–97% for shared on the identical kernel — it is not stuck at 0 (C9).
- **Wrong kernel row / warmup** → `--single` + `cudaProfilerStart/Stop`, `load_ncu` picks the max-duration row, `--warmup 8` (C5).

## 9. Caveats / honesty

- **Cold is a faithful proxy** for real 28-layer inter-step eviction (we cannot run 28 real layers in one isolated ncu launch); the cold/warm contrast in §5 justifies it.
- MLA is a *corroborating* axis, not a clean quantitative (R−1)/R point (head-reuse is partly SMEM-absorbed — §7).
- Single consumer GPU (sm120, 32 MB L2); absolute % is hardware-specific, but the **law** (L2-hit ↔ R, flat-in-footprint) is architecture-independent.
- A direct **GQA-ratio sweep** (vary `NUM_KV_HEADS` to dial the head-share) would test the "absorbed pre-L2" claim head-on; it needs a small `bench_xqa.py` edit and is left as optional future work — the MLA-h16 point already demonstrates the absorption.

## 10. Verify / reproduce

```bash
# phastform, root for ncu. cold = faithful. (idle-gated and audit-logged.)
export PATH=~/venvs/bench_sglang/bin:/usr/local/cuda-12.8/bin:$PATH
sudo -E bash studies/kv-cache-page-size/scripts/run_reuse_all.sh   # grid + reuse sweep + cold/warm + ps1
# locally:
python3 studies/kv-cache-page-size/scripts/make_reuse_figure.py        # fig_reuse_law.png (headline)
python3 studies/kv-cache-page-size/scripts/make_reuse_grid_figure.py   # fig_reuse_grid.png
```
Scripts: [profile_reuse_ncu.sh](../../scripts/profile_reuse_ncu.sh), [run_reuse_all.sh](../../scripts/run_reuse_all.sh), [make_reuse_figure.py](../../scripts/make_reuse_figure.py), [make_reuse_grid_figure.py](../../scripts/make_reuse_grid_figure.py).

**Companion reports:** [report 7](../07-shared-prefix-mechanism/report.md) (the DRAM↔cache flip this explains), [report 8](../08-small-kv-true-batch/report.md) (the cold/warm footprint ladder), [report 11](../11-mla-scatter-page-cost/report.md) (the MLA cross-head-reuse data).
