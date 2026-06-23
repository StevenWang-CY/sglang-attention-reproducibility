#!/usr/bin/env python3
"""Figure for report 5: page_size in the TRUE-batch regime (bench_one_batch).

Reads offline_batch_results/bench_one_batch_5060ti/bob_*.jsonl
  tag: bob_r{round}_b{B}_l{L}_{backend}_ps{ps};  TPOT(ms) = median_decode_latency*1000.
Robust estimator: MIN across rounds (decode-latency contention only inflates, never
deflates → the minimum is the cleanest sample; this neutralises the one B8/L4096
round-1 contention blip without hand-dropping data).

Writes into ../report_5_bench_one_batch/ :
  fig_bob_truebatch.png  — (A) Triton TPOT vs page, (B) FlashInfer TPOT vs page,
                            (C) true-batch vs shared-prefix ps1-penalty bars.
"""
import json, glob, re
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "offline_batch_results" / "bench_one_batch_5060ti"
OUT = Path(__file__).resolve().parents[1] / "report_5_bench_one_batch"
OUT.mkdir(exist_ok=True)
PAGES = [1, 8, 32, 128]
TAG = re.compile(r"bob_r(\d+)_b(\d+)_l(\d+)_(triton|flashinfer)_ps(\d+)")

# rows[(be,B,L)][ps] = [tpot per round]
rows = defaultdict(lambda: defaultdict(list))
for f in sorted(glob.glob(str(DATA / "bob_*.jsonl"))):
    m = TAG.search(Path(f).name)
    if not m:
        continue
    _, B, L, be, ps = int(m[1]), int(m[2]), int(m[3]), m[4], int(m[5])
    try:
        rec = json.loads(Path(f).read_text().strip().splitlines()[-1])
    except Exception:
        continue
    rows[(be, B, L)][ps].append(rec["median_decode_latency"] * 1000.0)

def tpot_min(be, B, L, ps):
    v = rows.get((be, B, L), {}).get(ps, [])
    return min(v) if v else None

# configs in KV order
CONFIGS = [(8, 4096), (16, 2048), (32, 1024), (16, 3072), (8, 6144), (2, 24576)]
def label(B, L):
    return f"B{B}/L{L}\n({B*L//1000}k KV)"

fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.34, wspace=0.22)
axT = fig.add_subplot(gs[0, 0])
axF = fig.add_subplot(gs[0, 1])
axB = fig.add_subplot(gs[1, :])

cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(CONFIGS)))
for ax, be, title in [(axT, "triton", "Triton"), (axF, "flashinfer", "FlashInfer")]:
    for ci, (B, L) in enumerate(CONFIGS):
        ys = [tpot_min(be, B, L, p) for p in PAGES]
        base = tpot_min(be, B, L, 128)
        if base is None or any(y is None for y in ys):
            # still plot whatever pages exist, normalised to ps128 if available
            if base is None:
                continue
        pct = [(y / base - 1) * 100 if y is not None else np.nan for y in ys]
        ax.plot(range(len(PAGES)), pct, "o-", color=cmap[ci], lw=2, ms=7,
                label=label(B, L).replace("\n", " "))
    ax.axhline(0, color="k", lw=1)
    ax.axhline(5, color="crimson", ls="--", lw=1.2, label="+5% threshold")
    ax.axhline(-5, color="seagreen", ls=":", lw=1)
    ax.set_xticks(range(len(PAGES)))
    ax.set_xticklabels([f"ps{p}" for p in PAGES])
    ax.set_ylim(-8, 14)
    ax.set_xlabel("page_size")
    ax.set_ylabel("decode TPOT vs ps128 (%)")
    ax.set_title(f"{title} — TRUE batch (independent seqs, CUDA graph ON)", fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, ncol=2, loc="upper left")

# ---- Panel C: true-batch vs shared-prefix ps1-penalty (matched points) ----
# shared-prefix reference values (this study's earlier reports):
#   Triton bs8  @~10k shared prefix, graph-ON: +11.5%   (report_3 §C / table)
#   Triton bs32 @~10k shared prefix, graph-ON: +24.6%   (report_3)
#   FlashInfer ~39k bs8 shared prefix, graph-ON, PAIRED: +8.6%  (report_4)
def true_pen(be, B, L):
    p1, p128 = tpot_min(be, B, L, 1), tpot_min(be, B, L, 128)
    return None if (p1 is None or p128 is None) else (p1 / p128 - 1) * 100

comps = [
    ("Triton bs8\n(true: B8/L4096)", 11.5, true_pen("triton", 8, 4096)),
    ("Triton bs32\n(true: B32/L1024)", 24.6, true_pen("triton", 32, 1024)),
    ("FlashInfer long-ctx\n(true: B2/L24576)", 8.6, true_pen("flashinfer", 2, 24576)),
]
comps = [(lbl, sp, tb) for (lbl, sp, tb) in comps if tb is not None]
x = np.arange(len(comps)); w = 0.38
sp_vals = [c[1] for c in comps]; tb_vals = [c[2] for c in comps]
b1 = axB.bar(x - w/2, sp_vals, w, label="shared-prefix (radix ON) — earlier reports",
             color="indianred")
b2 = axB.bar(x + w/2, tb_vals, w, label="TRUE batch (bench_one_batch)", color="steelblue")
axB.axhline(0, color="k", lw=1)
axB.axhline(5, color="crimson", ls="--", lw=1.2, label="+5% threshold")
axB.set_xticks(x); axB.set_xticklabels([c[0] for c in comps], fontsize=9)
axB.set_ylabel("ps1 decode TPOT vs ps128 (%)")
axB.set_title("ps1 penalty COLLAPSES in a true batch: the ≥5% effect was shared-prefix-specific",
              fontweight="bold")
axB.legend(fontsize=9, loc="upper right")
axB.grid(axis="y", alpha=0.3)
for bars in (b1, b2):
    for r in bars:
        h = r.get_height()
        axB.annotate(f"{h:+.1f}%", (r.get_x() + r.get_width()/2, h),
                     ha="center", va="bottom" if h >= 0 else "top", fontsize=8.5, fontweight="bold")

fig.suptitle("KV-cache page_size in the TRUE-batch regime (Qwen3-VL-2B, RTX 5060 Ti, CUDA graph ON)\n"
             "bench_one_batch: B independent sequences, no shared prefix",
             fontsize=13, fontweight="bold")
out = OUT / "fig_bob_truebatch.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)

# print the numbers the figure uses (for the report)
print("\n=== ps1 vs ps128 (min-across-rounds, %) ===")
for be in ("triton", "flashinfer"):
    for (B, L) in CONFIGS:
        v = true_pen(be, B, L)
        print(f"  {be:<10} B{B}/L{L:<6} KV={B*L//1000}k : "
              + (f"{v:+.1f}%" if v is not None else "MISSING"))
