#!/usr/bin/env python3
"""Report 9 figure (CORRECTED after audit): vLLM TRUE-batch block_size vs decode TPOT
(FlashInfer, CUDA graph ON, no prefix sharing).

AUDIT FINDING: the apparent ≥5% small-block penalty was a KV-pool-OVERFLOW / PREEMPTION artifact.
On 16 GB the KV pool holds ~57k (blk128) – 65k (blk16) tokens; a cell is a CLEAN true batch only if
B·(L+gen) fits (max concurrency ≥ B). In CLEAN cells block_size is FLAT (≤0.2%) even when KV-dominant;
the ±6–21% effects appear ONLY in cells whose batch can't fit (vLLM wave-schedules/preempts).

Reads vllm_b{B}_l{L}_blk{blk}.json (original sweep) + fit_b{B}_l{L}_blk{blk}.json (clean re-measure).
Writes ../report_9_vllm_blocksize_truebatch/fig_vllm_smallkv.png:
  (A) blk16-vs-blk128 (%) vs pool headroom (pool / B·(L+gen)) — flat when it fits (≥1), artifact when not.
  (B) the clean fitting cells: block_size flat even in the KV-dominant regime.
"""
import json, glob, re
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "offline_batch_results" / "vllm_smallkv_5060ti"
OUT = Path(__file__).resolve().parents[1] / "report_9_vllm_blocksize_truebatch"
OUT.mkdir(exist_ok=True)
POOL16, POOL128 = 64816, 56672   # measured GPU KV cache (tokens) at util 0.90, blk16 / blk128
GEN = 128                         # max_tokens (context grows by GEN during the TPOT measure)
BLK = [16, 32, 64, 128]

def load(pat):
    c = defaultdict(dict)
    for f in glob.glob(str(D / pat)):
        m = re.search(r"_b(\d+)_l(\d+)_blk(\d+)\.json$", Path(f).name)
        if not m: continue
        d = json.loads(Path(f).read_text())
        if d.get("tpot_median_ms", 0) > 0:
            c[(int(m[1]), int(m[2]))][int(m[3])] = d["tpot_median_ms"]
    return c
orig = load("vllm_b*_l*_blk*.json"); fit = load("fit_b*_l*_blk*.json")

fig = plt.figure(figsize=(15, 5.8))
gs = fig.add_gridspec(1, 2, wspace=0.24, left=0.07, right=0.985, top=0.82, bottom=0.13)
axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

# Panel A: blk16/blk128 (%) vs pool headroom = POOL128 / (B*(L+GEN)); >=1 => all-block clean
allcells = {}
for src in (orig, fit):
    for (B, L), d in src.items():
        if 16 in d and 128 in d:
            allcells[(B, L)] = d
for (B, L), d in sorted(allcells.items()):
    head = POOL128 / (B * (L + GEN))                 # >=1: fits even at blk128 (fully clean)
    pen = (d[16] / d[128] - 1) * 100
    clean = head >= 1.0
    axA.scatter(head, pen, s=60, color=("#2e8b57" if clean else "#c0392b"),
                edgecolor="k", lw=0.4, zorder=3)
    if abs(pen) >= 3 or not clean:
        axA.annotate(f"B{B}/L{L}", (head, pen), fontsize=6.3, xytext=(3, 2), textcoords="offset points")
axA.axhline(0, color="k", lw=1)
axA.axhline(5, color="crimson", ls="--", lw=1.2, label="±5%"); axA.axhline(-5, color="crimson", ls="--", lw=1.2)
axA.axvline(1.0, color="gray", ls=":", lw=1.5)
axA.text(1.05, axA.get_ylim()[1]*0.86, "batch FITS →\n(clean true batch)", fontsize=7.5, color="#2e8b57")
axA.text(0.95, axA.get_ylim()[1]*0.86, "← batch OVERFLOWS\n(preempted)", fontsize=7.5, color="#c0392b", ha="right")
axA.set_xscale("log"); axA.set_xlabel("KV-pool headroom = pool ÷ B·(L+gen)   [≥1 ⇒ all sequences fit]")
axA.set_ylabel("blk16 TPOT vs blk128 (%)")
axA.set_title("(A) The ±5–21% effects appear ONLY when the batch can't fit the KV pool\n"
              "(preempted, red). Where it fits (green) block_size is flat.", fontweight="bold", fontsize=10)
axA.legend(fontsize=8, loc="lower left"); axA.grid(alpha=0.3, which="both")

# Panel B: clean fitting cells — block_size flat even when KV-dominant
cells = sorted(fit.keys(), key=lambda k: k[0]*k[1])
cmap = plt.cm.viridis(np.linspace(0.15, 0.85, max(1, len(cells))))
for i, (B, L) in enumerate(cells):
    d = fit[(B, L)]
    if 128 not in d: continue
    ys = [(d[b]/d[128]-1)*100 if b in d else np.nan for b in BLK]
    kvgb = B*L*114688/1e9
    axB.plot(range(len(BLK)), ys, "o-", color=cmap[i], lw=2, ms=6,
             label=f"B{B}/L{L} ({kvgb:.1f}GB KV)")
axB.axhline(0, color="k", lw=1); axB.axhline(5, color="crimson", ls="--", lw=1.2, label="+5% threshold")
axB.set_xticks(range(len(BLK))); axB.set_xticklabels([f"blk{b}" for b in BLK])
axB.set_ylim(-3, 7); axB.set_xlabel("vLLM block_size (page_size)")
axB.set_ylabel("decode TPOT vs blk128 (%)")
axB.set_title("(B) CLEAN true batches (concurrency ≥ B), KV-dominant (KV>4.1GB weights):\n"
              "block_size FLAT (≤0.2%) — no ≥5% small-block penalty", fontweight="bold", fontsize=10)
axB.legend(fontsize=7.2, ncol=2, loc="upper center"); axB.grid(alpha=0.3)

fig.suptitle("Report 9 — vLLM block_size in a TRUE batch (Qwen3-VL-2B, RTX 5060 Ti sm120, FlashInfer, CUDA graph ON): "
             "no clean ≥5% penalty; apparent break = KV-pool-overflow preemption",
             fontsize=11.5, fontweight="bold", y=0.97)
out = OUT / "fig_vllm_smallkv.png"
fig.savefig(out, dpi=130, bbox_inches="tight"); print("wrote", out)

print("\n=== clean (fit_*) cells: blk16/blk128 ===")
for (B, L) in cells:
    d = fit[(B, L)]
    if 16 in d and 128 in d:
        print(f"  B{B}/L{L} KV={B*L*114688/1e9:.1f}GB head={POOL128/(B*(L+GEN)):.2f}  blk16/blk128={(d[16]/d[128]-1)*100:+.1f}%")
print("=== preempted (orig) break cells ===")
for (B, L) in sorted(orig):
    d = orig[(B, L)]
    if 16 in d and 128 in d and POOL128/(B*(L+GEN)) < 1:
        print(f"  B{B}/L{L} head={POOL128/(B*(L+GEN)):.2f}  blk16/blk128={(d[16]/d[128]-1)*100:+.1f}% (PREEMPTED)")
