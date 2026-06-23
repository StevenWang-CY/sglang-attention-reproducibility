#!/usr/bin/env python3
"""Figure for report 5's adversarial-search section: across every corner attacked (high batch, L2-resident
small KV, long single-seq, GQA-2 / extra-layers models), FlashInfer + CUDA graph ON + TRUE batch keeps
page_size=1 the fastest page. No break.

Reads bobx_* (Qwen3-VL-2B break ladder + L2/long-ctx) and bobm_<model>_* (cross-model). Min-across-rounds.
Writes ../report_5_bench_one_batch/fig_bob_adversarial.png:
  (A) forest plot: ps1 vs the BEST page (%) for every FlashInfer cell, by model -> all hug 0, none >= +5%.
  (B) page-curves (TPOT normalised to ps1) for the most extreme cells -> ps1 lowest / flat.
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
TAGX = re.compile(r"bobx_r(\d+)_b(\d+)_l(\d+)_(triton|flashinfer)_ps(\d+)")
TAGM = re.compile(r"bobm_([a-z0-9_]+?)_r(\d+)_b(\d+)_l(\d+)_(triton|flashinfer)_ps(\d+)")
MODEL_LABEL = {"q3vl_2b": "Qwen3-VL-2B (GQA-8)", "q25_3b": "Qwen2.5-3B (GQA-2)", "q3vl_4b": "Qwen3-VL-4B (GQA-8, 36L)"}
MODEL_COLOR = {"q3vl_2b": "#1f77b4", "q25_3b": "#d62728", "q3vl_4b": "#2ca02c"}

rows = defaultdict(lambda: defaultdict(list))  # (model,be,B,L) -> ps -> [tpot]
for f in sorted(glob.glob(str(DATA / "bobx_*.jsonl")) + glob.glob(str(DATA / "bobm_*.jsonl"))):
    name = Path(f).name
    mm = TAGM.search(name)
    if mm:
        mdl, B, L, be, ps = mm[1], int(mm[3]), int(mm[4]), mm[5], int(mm[6])
    else:
        m = TAGX.search(name)
        if not m:
            continue
        mdl, B, L, be, ps = "q3vl_2b", int(m[2]), int(m[3]), m[4], int(m[5])
    try:
        rec = json.loads(Path(f).read_text().strip().splitlines()[-1])
    except Exception:
        continue
    rows[(mdl, be, B, L)][ps].append(rec["median_decode_latency"] * 1000.0)

def mn(d, p):
    v = d.get(p, [])
    return min(v) if v else None

# ---- collect FlashInfer cells: ps1 vs best-page % ----
pts = []  # (model, B, L, ps1_vs_best_pct, KV)
for (mdl, be, B, L), d in rows.items():
    if be != "flashinfer":
        continue
    avail = {p: mn(d, p) for p in d if mn(d, p) is not None}
    if 1 not in avail or len(avail) < 2:
        continue
    best = min(avail.values())
    pts.append((mdl, B, L, (avail[1] / best - 1) * 100, B * L))
pts.sort(key=lambda r: (list(MODEL_LABEL).index(r[0]) if r[0] in MODEL_LABEL else 9, r[4]))

fig = plt.figure(figsize=(15, 8))
gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1], wspace=0.28)
axF = fig.add_subplot(gs[0, 0])
axC = fig.add_subplot(gs[0, 1])

# Panel A — forest plot
y = np.arange(len(pts))
for i, (mdl, B, L, pct, kv) in enumerate(pts):
    axF.plot(pct, i, "o", color=MODEL_COLOR.get(mdl, "gray"), ms=7)
axF.axvline(0, color="k", lw=1)
axF.axvline(5, color="crimson", ls="--", lw=1.5, label="+5% break threshold")
axF.axvspan(-0.5, 5, color="seagreen", alpha=0.06)
axF.set_yticks(y)
axF.set_yticklabels([f"B{B}/L{L} ({kv//1000}k)" for (_, B, L, _, kv) in pts], fontsize=7)
axF.set_xlabel("ps1 decode TPOT vs the FASTEST page (%)  —  >0 would mean ps1 not fastest")
axF.set_xlim(-4, 8)
axF.set_title("FlashInfer + CUDA graph ON, TRUE batch: ps1 vs best page\n(every adversarial corner; right of red = a break)",
              fontweight="bold", fontsize=11)
handles = [plt.Line2D([], [], marker="o", ls="", color=MODEL_COLOR[m], label=MODEL_LABEL[m]) for m in MODEL_LABEL if any(p[0]==m for p in pts)]
handles.append(plt.Line2D([], [], color="crimson", ls="--", label="+5% break threshold"))
axF.legend(handles=handles, fontsize=8, loc="lower right")
axF.grid(axis="x", alpha=0.3)
mx = max((p[3] for p in pts), default=0)
axF.text(0.02, 0.98, f"max ps1 penalty across ALL {len(pts)} cells: +{mx:.1f}%  (no break)",
         transform=axF.transAxes, va="top", fontsize=9, fontweight="bold",
         bbox=dict(boxstyle="round", fc="lightyellow", ec="gray"))

# Panel B — page curves for the most extreme cells
def curve(mdl, B, L, label, color):
    d = rows.get((mdl, "flashinfer", B, L))
    if not d:
        return
    avail = {p: mn(d, p) for p in sorted(d) if mn(d, p) is not None}
    if 1 not in avail:
        return
    xs = list(avail.keys())
    ys = [(avail[p] / avail[1] - 1) * 100 for p in xs]
    axC.plot(range(len(xs)), ys, "o-", color=color, lw=2, ms=6, label=label)
    return xs

picks = [("q25_3b", 512, 64, "Qwen2.5-3B B512/L64 (32k)", "#d62728"),
         ("q3vl_2b", 256, 128, "Qwen3-VL-2B B256/L128 (32k)", "#1f77b4"),
         ("q3vl_2b", 1, 49152, "Qwen3-VL-2B B1/L49152 (49k)", "#9467bd"),
         ("q3vl_2b", 32, 256, "Qwen3-VL-2B B32/L256 (8k, L2-res)", "#ff7f0e")]
xref = None
for mdl, B, L, lab, col in picks:
    xs = curve(mdl, B, L, lab, col)
    if xs and (xref is None or len(xs) > len(xref)):
        xref = xs
axC.axhline(0, color="k", lw=1)
axC.axhline(5, color="crimson", ls="--", lw=1.2)
if xref:
    axC.set_xticks(range(len(xref)))
    axC.set_xticklabels([f"ps{p}" for p in xref])
axC.set_ylabel("decode TPOT vs ps1 (%)   —  >0 means slower than ps1")
axC.set_xlabel("page_size")
axC.set_title("Page curves at the extreme corners\n(ps1 is the minimum; larger pages flat/slightly slower)",
              fontweight="bold", fontsize=11)
axC.legend(fontsize=8)
axC.grid(alpha=0.3)

fig.suptitle("Adversarial search for a TRUE-batch FlashInfer ps1 penalty (CUDA graph ON) — NONE FOUND\n"
             "high batch (to 512) x short ctx, L2-resident small KV, long single-seq, GQA-2 model — RTX 5060 Ti",
             fontsize=13, fontweight="bold")
out = OUT / "fig_bob_adversarial.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out, f"({len(pts)} flashinfer cells, max ps1 penalty +{mx:.1f}%)")
