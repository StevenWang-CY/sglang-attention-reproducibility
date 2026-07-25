#!/usr/bin/env python3
"""Report 12 grid figure: (A) distinct L2-hit heatmap (B,L); (B) shared heatmap;
(C) hit/miss sector-count decomposition; (D) all reuse axes (shared-B, MLA heads, MHA) scatter.
Reads reuse_profile/*.csv (+ pagecost_profile/mla_*.csv for the MLA points)."""
import csv, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "studies" / "kv-cache-page-size" / "data" / "raw" / "reuse_profile"
MLA = ROOT / "studies" / "kv-cache-page-size" / "data" / "raw" / "pagecost_profile"
OUT = Path(__file__).resolve().parents[1] / "reports" / "12-reuse-not-residency"
OUT.mkdir(exist_ok=True)
TAG = re.compile(r"flashinfer_(distinct|shared)_(all|none)_ps(\d+)_bs(\d+)_kv(\d+)\.csv")


def load(p):
    lines = p.read_text(errors="replace").splitlines()
    try:
        hi = next(i for i, l in enumerate(lines) if l.startswith('"ID"'))
    except StopIteration:
        return None
    rows = list(csv.DictReader(lines[hi:]))
    def fn(s):
        try: return float(str(s).replace(",", ""))
        except: return None
    data = [r for r in rows if fn(r.get("gpu__time_duration.sum")) is not None]
    if not data: return None
    r = max(data, key=lambda r: fn(r["gpu__time_duration.sum"]))
    g = lambda k: fn(r.get(k))
    return dict(l2=g("lts__t_sector_hit_rate.pct"), dram=g("dram__throughput.avg.pct_of_peak_sustained_elapsed"),
                hit=g("lts__t_sectors_lookup_hit.sum"), miss=g("lts__t_sectors_lookup_miss.sum"),
                dramrd=g("dram__bytes_op_read.sum"))


D = {}
for p in DATA.glob("*.csv"):
    m = TAG.match(p.name)
    if not m: continue
    r = load(p)
    if r: D[(m.group(1), m.group(2), int(m.group(3)), int(m.group(4)), int(m.group(5)))] = r

Bs = [1, 4, 8, 16, 32]
Ls = [1024, 4096, 8192, 16384]

fig, ax = plt.subplots(2, 2, figsize=(14, 10))

# Panels A/B: heatmaps
for axi, mode in [(ax[0][0], "distinct"), (ax[0][1], "shared")]:
    M = np.full((len(Bs), len(Ls)), np.nan)
    for i, B in enumerate(Bs):
        for j, L in enumerate(Ls):
            r = D.get((mode, "all", 128, B, L))
            if r and r["l2"] is not None: M[i, j] = r["l2"]
    im = axi.imshow(M, vmin=0, vmax=100, cmap="viridis", aspect="auto", origin="lower")
    axi.set_xticks(range(len(Ls))); axi.set_xticklabels([f"{L//1024}k" for L in Ls])
    axi.set_yticks(range(len(Bs))); axi.set_yticklabels(Bs)
    axi.set_xlabel("context L"); axi.set_ylabel("batch B")
    for i in range(len(Bs)):
        for j in range(len(Ls)):
            if not np.isnan(M[i, j]):
                axi.text(j, i, f"{M[i,j]:.1f}", ha="center", va="center",
                         color="white" if M[i, j] < 55 else "black", fontsize=9)
    axi.set_title(f"({'A' if mode=='distinct' else 'B'}) {mode} L2-hit % (cold, ps128)\n"
                  + ("distinct ≈0% EVERYWHERE incl. sub-L2" if mode == "distinct" else "shared banded by B (=R), flat in L"))
    fig.colorbar(im, ax=axi, fraction=0.046, pad=0.04)

# Panel C: hit/miss sector counts (distinct vs shared-B8) at L1024, L8192
axc = ax[1][0]
cells = [("distinct", 8, 1024), ("shared", 8, 1024), ("distinct", 8, 8192), ("shared", 8, 8192)]
labels, hits, misses, dramsec = [], [], [], []
for mode, B, L in cells:
    r = D.get((mode, "all", 128, B, L))
    if not r: continue
    labels.append(f"{mode[:4]}\nB{B}L{L//1024}k")
    hits.append((r["hit"] or 0) / 1e6); misses.append((r["miss"] or 0) / 1e6)
    dramsec.append(((r["dramrd"] or 0) * 1e6 / 32) / 1e6 if r["dramrd"] else 0)  # dramrd is in MB
x = np.arange(len(labels)); w = 0.6
axc.bar(x, misses, w, label="L2 miss sectors (→DRAM)", color="#c0392b")
axc.bar(x, hits, w, bottom=misses, label="L2 hit sectors (reused)", color="#2e8b57")
axc.plot(x, dramsec, "ko", label="DRAM read bytes ÷ 32", zorder=4)
axc.set_xticks(x); axc.set_xticklabels(labels, fontsize=8)
axc.set_ylabel("sectors (millions)")
axc.set_title("(C) Count decomposition: distinct miss≈total≈DRAM, hit≈0;\nshared hit≈7/8 (rate is an honest ratio)")
axc.legend(fontsize=8); axc.grid(alpha=0.3, axis="y")

# Panel D: all reuse axes — L2-hit vs reuse factor
axd = ax[1][1]
# shared-B sweep (cross-sequence reuse)
shB = [(B, D[("shared", "all", 128, B, 1024)]["l2"]) for B in [1, 2, 4, 6, 8, 12, 16, 24, 32]
       if ("shared", "all", 128, B, 1024) in D]
if shB:
    axd.plot([b for b, _ in shB], [v for _, v in shB], "s-", color="#2e8b57", label="shared prefix (cross-seq, R=B)")
RR = np.array([1, 2, 4, 8, 16, 32])
axd.plot(RR, 100 * (RR - 1) / RR, "k--", lw=1, label="(R−1)/R model")
# MHA distinct (R=1) from reuse_profile
md = D.get(("distinct", "all", 128, 8, 1024))
if md: axd.scatter([1], [md["l2"]], c="#1f4e79", s=80, marker="o", zorder=5, label="MHA distinct (R=1)")
# MLA cross-head reuse from report 11 (pagecost_profile), distinct contig cold
def mla(H, B, L):
    p = MLA / f"mla_h{H}_contig_all_bs{B}_kv{L}.csv"
    return load(p)["l2"] if p.exists() and load(p) else None
for H, B, L, mk in [(16, 8, 131072, "D"), (128, 8, 131072, "P")]:
    v = mla(H, B, L)
    if v is not None:
        axd.scatter([H], [v], s=90, marker=mk, zorder=5,
                    label=f"MLA {H} heads (cross-head, distinct)")
axd.set_xscale("log", base=2); axd.set_xlabel("intra-kernel reuse factor (log)")
axd.set_ylabel("L2 hit %"); axd.set_ylim(-3, 100)
axd.set_title("(D) All reuse axes collapse onto 'L2-hit ↔ reuse'\n(MLA below model: head-reuse partly SMEM-absorbed)")
axd.legend(fontsize=7.5); axd.grid(alpha=0.3)

fig.suptitle("Report 12 — distinct true-batch decode L2-hit ≈0% because each KV byte is read once (R=1); reuse (shared prefix R=B, or MLA heads) is the only thing that produces L2 hits.", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = OUT / "fig_reuse_grid.png"
fig.savefig(out, dpi=130)
print("wrote", out)
