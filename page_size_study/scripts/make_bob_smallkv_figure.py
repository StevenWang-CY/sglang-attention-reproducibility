#!/usr/bin/env python3
"""Report 8 engine figure: small-KV TRUE-batch bench_one_batch (bobk_* tags).

Reads offline_batch_results/bench_one_batch_smallkv_5060ti/bobk_*.jsonl
  tag: bobk_r{round}_b{B}_l{L}_{backend}_ps{ps};  TPOT(ms) = median_decode_latency*1000.
Robust estimator: MIN across rounds (contention only inflates).
Writes ../report_8_smallkv_truebatch/fig_smallkv_engine.png:
  (A) Triton TPOT vs page_size (% vs ps128) across the footprint ladder.
  (B) FlashInfer TPOT vs page_size (% vs ps128).
  (C) per-decode-step DRAM-traffic roofline: model weights (4.10 GB) ALWAYS dominate the KV at
      small ctx -> engine is WEIGHT-bound -> page_size washes out (the flatness has nothing to do
      with cache-residency; that is the kernel story, fig_smallkv_kernel.png).
"""
import json, glob, re
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "offline_batch_results" / "bench_one_batch_smallkv_5060ti"
OUT = Path(__file__).resolve().parents[1] / "report_8_smallkv_truebatch"
OUT.mkdir(exist_ok=True)
TAG = re.compile(r"bobk_r(\d+)_b(\d+)_l(\d+)_(triton|flashinfer)_ps(\d+)")
WEIGHT_GB = 4.10
KV_B_PER_TOK = 114688

rows = defaultdict(lambda: defaultdict(list))
for f in sorted(glob.glob(str(DATA / "bobk_*.jsonl"))):
    m = TAG.search(Path(f).name)
    if not m: continue
    _, B, L, be, ps = int(m[1]), int(m[2]), int(m[3]), m[4], int(m[5])
    try:
        rec = json.loads(Path(f).read_text().strip().splitlines()[-1])
        rows[(be, B, L)][ps].append(rec["median_decode_latency"] * 1000.0)
    except Exception:
        pass
def tmin(be, B, L, ps):
    v = rows.get((be, B, L), {}).get(ps, [])
    return min(v) if v else None

# representative cells spanning the footprint ladder (small -> large KV)
CELLS = [(1, 256), (8, 256), (32, 256), (8, 512), (16, 512), (128, 128), (1, 4096)]
CELLS = [c for c in CELLS if (c[0], c[1]) in {(B, L) for (_, B, L) in rows}]
PAGES_FI = [1, 2, 4, 8, 16, 32, 128]
PAGES_TR = [1, 8, 32, 128]
def label(B, L): return f"B{B}/L{L} ({B*L//1000 if B*L>=1000 else B*L/1000:g}k KV)"

fig = plt.figure(figsize=(16, 5.2))
gs = fig.add_gridspec(1, 3, wspace=0.26, left=0.06, right=0.985, top=0.82, bottom=0.16)
axT, axF, axR = (fig.add_subplot(gs[0, i]) for i in range(3))
cmap = plt.cm.viridis(np.linspace(0.1, 0.9, max(1, len(CELLS))))

for ax, be, pages, title in [(axT, "triton", PAGES_TR, "Triton"),
                             (axF, "flashinfer", PAGES_FI, "FlashInfer")]:
    for ci, (B, L) in enumerate(CELLS):
        base = tmin(be, B, L, 128)
        if base is None: continue
        ys = [tmin(be, B, L, p) for p in pages]
        pct = [(y / base - 1) * 100 if y is not None else np.nan for y in ys]
        ax.plot(range(len(pages)), pct, "o-", color=cmap[ci], lw=2, ms=6, label=label(B, L))
    ax.axhline(0, color="k", lw=1); ax.axhline(5, color="crimson", ls="--", lw=1.2, label="+5% threshold")
    ax.set_xticks(range(len(pages))); ax.set_xticklabels([f"ps{p}" for p in pages], fontsize=8)
    ax.set_ylim(-6, 10); ax.set_xlabel("page_size"); ax.set_ylabel("decode TPOT vs ps128 (%)")
    ax.set_title(f"({'A' if be=='triton' else 'B'}) {title} — TRUE batch, small KV (graph ON)",
                 fontweight="bold", fontsize=10.5)
    ax.grid(alpha=0.3); ax.legend(fontsize=7.2, ncol=1, loc="upper left")

# Panel C: per-step DRAM-traffic roofline (weights vs KV)
rc = [(1, 256), (8, 256), (32, 256), (16, 512), (128, 128), (256, 128)]
rc = [c for c in rc if any((B, L) == c for (_, B, L) in rows)] or [(1,256),(8,256),(32,256),(16,512),(128,128)]
labels = [f"B{B}/L{L}" for (B, L) in rc]
kv = [B * L * KV_B_PER_TOK / 1e9 for (B, L) in rc]
x = np.arange(len(rc))
axR.bar(x, [WEIGHT_GB]*len(rc), 0.6, label="model weights (4.10 GB, fixed/step)", color="#8a8a8a")
axR.bar(x, kv, 0.6, bottom=[WEIGHT_GB]*len(rc), label="KV read (B·L·112 KB)", color="#c0392b")
for xi, (B, L) in enumerate(rc):
    frac = kv[xi] / (kv[xi] + WEIGHT_GB) * 100
    axR.annotate(f"KV {frac:.0f}%", (xi, WEIGHT_GB + kv[xi] + 0.1), ha="center", fontsize=8, fontweight="bold")
axR.set_xticks(x); axR.set_xticklabels(labels, fontsize=8, rotation=15)
axR.set_ylabel("DRAM bytes read per decode step (GB)")
axR.set_title("(C) WHY the engine is flat: weights dominate DRAM\ntraffic at small KV → weight-bound, not page-bound",
              fontweight="bold", fontsize=10.5)
axR.legend(fontsize=8.5, loc="upper left"); axR.grid(axis="y", alpha=0.3)

fig.suptitle("Report 8 — Engine bench_one_batch with small KV: page_size flat (weight-DRAM-bound), "
             "ps1 tied — Qwen3-VL-2B, RTX 5060 Ti, CUDA graph ON, independent seqs",
             fontsize=12.5, fontweight="bold", y=0.97)
out = OUT / "fig_smallkv_engine.png"
fig.savefig(out, dpi=130, bbox_inches="tight"); print("wrote", out)
