#!/usr/bin/env python3
"""Report 7 figure: shared-prefix versus distinct-KV memory regimes.

Sharing creates intra-kernel L2 reuse and moves decode off the DRAM wall. It does not by itself create a
page-size penalty; report 16 shows that page-dependent backend work is the additional requirement.

Reads:
  studies/kv-cache-page-size/data/raw/sharedprefix_profile/flashinfer_{shared,distinct}_ps{ps}_bs{B}_kv{L}.csv  (ncu counters)
  studies/kv-cache-page-size/data/raw/sharedprefix_profile/lat_{shared,distinct}.json                            (microbench latency)
Writes ../07-shared-prefix-mechanism/fig_sharedprefix.png:
  (A) the MECHANISM: DRAM% and L2-hit% per cell, shared vs distinct -> the DRAM<->cache flip.
  (B) the CONSEQUENCE: decode latency shared vs distinct (cache-resident reuse = 3x faster), page-flat.
"""
import csv, glob, json, re
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
D = ROOT / "studies" / "kv-cache-page-size" / "data" / "raw" / "sharedprefix_profile"
OUT = Path(__file__).resolve().parents[1] / "reports" / "07-shared-prefix-mechanism"
OUT.mkdir(exist_ok=True)

def load_ncu(p):
    lines = p.read_text(errors="replace").splitlines()
    hi = next(i for i, l in enumerate(lines) if l.startswith('"ID"'))
    rows = list(csv.DictReader(lines[hi:]))
    def fnum(s):
        try: return float(str(s).replace(",", ""))
        except: return None
    data = [r for r in rows if fnum(r.get("gpu__time_duration.sum")) is not None]
    if not data: return None
    r = max(data, key=lambda r: fnum(r["gpu__time_duration.sum"]))
    g = lambda k: fnum(r.get(k))
    return dict(dram=g("dram__throughput.avg.pct_of_peak_sustained_elapsed"),
                l2=g("lts__t_sector_hit_rate.pct"), sm=g("sm__throughput.avg.pct_of_peak_sustained_elapsed"))

TAG = re.compile(r"flashinfer_(shared|distinct)_ps(\d+)_bs(\d+)_kv(\d+)")
ncu = {}
for f in sorted(glob.glob(str(D / "flashinfer_*.csv"))):
    m = TAG.search(Path(f).name)
    if not m: continue
    r = load_ncu(Path(f))
    if r: ncu[(m[1], int(m[2]), int(m[3]), int(m[4]))] = r   # (mode,ps,B,L)

lat = {}
for mode in ("shared", "distinct"):
    j = json.loads((D / f"lat_{mode}.json").read_text())
    for k, v in j.items():
        if "latency_ms" in v:
            lat[(mode, v["page_size"], v["batch_size"], v["seq_len"])] = v["latency_ms"]

# cells to show (use ps128 for the counter panel; page barely matters within a mode)
CELLS = [(8, 6144), (8, 39168)]   # cells with BOTH ncu + latency (B16/L3072 ncu cited in text)
def lbl(B, L): return f"B{B}/L{L//1024 if L%1024==0 else L}\n({B*L//1000}k tok)"

fig = plt.figure(figsize=(14, 6.6))
gs = fig.add_gridspec(1, 2, wspace=0.25, top=0.74, bottom=0.10, left=0.07, right=0.98)
axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[0, 1])

# Panel A: DRAM% and L2hit% shared vs distinct (ps128 representative; ps1 ~identical)
labels = [lbl(B, L) for (B, L) in CELLS]
x = np.arange(len(CELLS)); w = 0.2
def vals(metric, mode):
    out = []
    for (B, L) in CELLS:
        r = ncu.get((mode, 128, B, L)) or ncu.get((mode, 1, B, L))
        out.append(r[metric] if r else np.nan)
    return out
axA.bar(x - 1.5*w, vals("dram", "distinct"), w, label="DRAM% — distinct (true batch)", color="#c0392b")
axA.bar(x - 0.5*w, vals("dram", "shared"),   w, label="DRAM% — shared prefix",        color="#e8a0a0")
axA.bar(x + 0.5*w, vals("l2", "distinct"),   w, label="L2-hit% — distinct",           color="#1f4e79")
axA.bar(x + 1.5*w, vals("l2", "shared"),     w, label="L2-hit% — shared prefix",      color="#7fb0d8")
axA.set_xticks(x); axA.set_xticklabels(labels, fontsize=8)
axA.set_ylabel("% (ncu hardware counters)"); axA.set_ylim(0, 105)
axA.axhline(90, color="gray", ls=":", lw=1)
axA.set_title("(A) MECHANISM: shared prefix flips decode\nDRAM-bound → L2-cache-bound",
              fontweight="bold", fontsize=11, pad=8)
axA.legend(fontsize=7.5, loc="center left")
for xi, (B, L) in enumerate(CELLS):
    dd = ncu.get(("distinct", 128, B, L)); ss = ncu.get(("shared", 128, B, L)) or ncu.get(("shared", 1, B, L))
    if dd: axA.annotate(f"{dd['dram']:.0f}", (xi-1.5*w, dd['dram']+1), ha="center", fontsize=7)
    if ss: axA.annotate(f"{ss['l2']:.0f}", (xi+1.5*w, ss['l2']+1), ha="center", fontsize=7)

# Panel B: latency shared vs distinct (ps1 & ps128 overlaid to show page-flatness)
def latv(mode, ps): return [lat.get((mode, ps, B, L), np.nan) for (B, L) in CELLS]
axB.bar(x - 1.5*w, latv("distinct", 1),   w, color="#c0392b", label="distinct ps1")
axB.bar(x - 0.5*w, latv("distinct", 128), w, color="#e8a0a0", label="distinct ps128")
axB.bar(x + 0.5*w, latv("shared", 1),     w, color="#1f4e79", label="shared ps1")
axB.bar(x + 1.5*w, latv("shared", 128),   w, color="#7fb0d8", label="shared ps128")
axB.set_xticks(x); axB.set_xticklabels(labels, fontsize=8)
axB.set_ylabel("decode kernel latency (ms)")
axB.set_title("(B) CONSEQUENCE: shared reuse is ~3× faster;\n"
              "FlashInfer remains page-flat in both regimes",
              fontweight="bold", fontsize=10, pad=8)
axB.legend(fontsize=8)
for xi, (B, L) in enumerate(CELLS):
    d = lat.get(("distinct", 128, B, L)); s = lat.get(("shared", 128, B, L))
    if d and s: axB.annotate(f"{d/s:.1f}×", (xi, max(d, s)*0.5), ha="center", fontsize=9, fontweight="bold", color="white")

fig.suptitle("Shared-prefix versus distinct-KV memory regimes — Qwen3-VL-2B, RTX 5060 Ti, ncu",
             fontsize=12.5, fontweight="bold", y=0.97)
fig.text(0.5, 0.90, "shared = 1 KV copy re-read by all B seqs (reuse → L2-cache-bound)   vs   "
         "distinct = B independent copies (no reuse → DRAM-bandwidth-bound)",
         ha="center", fontsize=9.5, style="italic")
out = OUT / "fig_sharedprefix.png"
fig.savefig(out, dpi=130)
print("wrote", out)
# print the table the report cites
print("\ncell           mode      DRAM%  L2hit%  SM%   lat(ps1)  lat(ps128)")
for (B, L) in CELLS:
    for mode in ("distinct", "shared"):
        r = ncu.get((mode, 128, B, L)) or ncu.get((mode, 1, B, L)) or {}
        print(f"B{B}/L{L:<7}{mode:<9}{r.get('dram',0):<7.1f}{r.get('l2',0):<8.1f}{r.get('sm',0):<6.1f}"
              f"{lat.get((mode,1,B,L),0):<10.3f}{lat.get((mode,128,B,L),0):<.3f}")
