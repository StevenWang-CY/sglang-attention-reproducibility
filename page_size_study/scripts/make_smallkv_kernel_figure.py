#!/usr/bin/env python3
"""Report 8 HEADLINE figure: isolated FlashInfer decode kernel, distinct (true-batch) KV,
swept across KV FOOTPRINT (L2-resident -> DRAM-bound). Does ps1 lose once off the ceiling? No.

Reads:
  offline_batch_results/smallkv_profile/lat_distinct_b{1,8,32}.json        (CUDA-event latency)
  offline_batch_results/smallkv_profile/flashinfer_distinct_{all,none}_ps{ps}_bs{B}_kv{L}.csv  (ncu)
Writes ../report_8_smallkv_truebatch/fig_smallkv_kernel.png:
  (A) ps1/ps128 latency ratio vs footprint (per batch) — flat <+2.5%, never >=5%, even L2-resident.
  (B) absolute kernel latency vs footprint (B8) — plateau (cache-resident) -> linear ramp (DRAM-bound).
  (C) ncu DRAM% / L2-hit% vs footprint (B8) — warm(none) vs cold(all): the regime, page-invariant.
"""
import csv, glob, json, re
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "offline_batch_results" / "smallkv_profile"
OUT = Path(__file__).resolve().parents[1] / "report_8_smallkv_truebatch"
OUT.mkdir(exist_ok=True)
L2_MB = 32.0
BYTES_PER_TOK_LAYER = 4096  # bf16 K+V, 8 KV heads * 128 dim * 2 bytes * 2(KV) ... = 4096 B/tok/layer

# ---- latency ----
lat = {}  # (B, L, ps) -> ms
for f in glob.glob(str(D / "lat_distinct_b*.json")):
    if "xqa" in f:
        continue
    for k, v in json.loads(Path(f).read_text()).items():
        if "latency_ms" in v:
            lat[(v["batch_size"], v["seq_len"], v["page_size"])] = v["latency_ms"]
Bs = sorted({k[0] for k in lat})
def fp_mb(B, L): return B * L * BYTES_PER_TOK_LAYER / 1e6

# ---- ncu ----
def load_ncu(p):
    lines = p.read_text(errors="replace").splitlines()
    try:
        hi = next(i for i, l in enumerate(lines) if l.startswith('"ID"'))
    except StopIteration:
        return None
    rows = list(csv.DictReader(lines[hi:]))
    def fnum(s):
        try: return float(str(s).replace(",", ""))
        except: return None
    data = [r for r in rows if fnum(r.get("gpu__time_duration.sum")) is not None]
    if not data: return None
    r = max(data, key=lambda r: fnum(r["gpu__time_duration.sum"]))
    g = lambda k: fnum(r.get(k))
    return dict(dram=g("dram__throughput.avg.pct_of_peak_sustained_elapsed"),
                l2=g("lts__t_sector_hit_rate.pct"),
                sm=g("sm__throughput.avg.pct_of_peak_sustained_elapsed"),
                dur=g("gpu__time_duration.sum"))
TAG = re.compile(r"flashinfer_distinct_(all|none)_ps(\d+)_bs(\d+)_kv(\d+)")
ncu = {}  # (cache, ps, B, L) -> metrics
for f in sorted(glob.glob(str(D / "flashinfer_distinct_*_ps*_bs*_kv*.csv"))):
    m = TAG.search(Path(f).name)
    if not m: continue
    r = load_ncu(Path(f))
    if r: ncu[(m[1], int(m[2]), int(m[3]), int(m[4]))] = r

fig = plt.figure(figsize=(16, 5.2))
gs = fig.add_gridspec(1, 3, wspace=0.27, left=0.06, right=0.985, top=0.84, bottom=0.16)
axA, axB, axC = (fig.add_subplot(gs[0, i]) for i in range(3))
cmap = {1: "#1f4e79", 8: "#c0392b", 32: "#2e8b57"}

# Panel A: ps1/ps128 ratio vs footprint, per batch
for B in Bs:
    pts = sorted({L for (b, L, p) in lat if b == B})
    xs, ys = [], []
    for L in pts:
        if (B, L, 1) in lat and (B, L, 128) in lat:
            xs.append(fp_mb(B, L)); ys.append((lat[(B, L, 1)] / lat[(B, L, 128)] - 1) * 100)
    axA.plot(xs, ys, "o-", color=cmap.get(B, "gray"), lw=2, ms=6, label=f"B={B}")
axA.axhline(0, color="k", lw=1); axA.axhline(5, color="crimson", ls="--", lw=1.3, label="+5% threshold")
axA.axvspan(0.5, L2_MB, color="#cfe8cf", alpha=0.5, label=f"L2-resident (<{L2_MB:.0f} MB)")
axA.set_xscale("log"); axA.set_xlabel("distinct KV footprint / layer (MB)")
axA.set_ylabel("ps1 latency vs ps128 (%)"); axA.set_ylim(-4, 8)
axA.set_title("(A) ps1 NEVER ≥5% slow vs ps128 —\neven when KV is L2-resident (off the DRAM ceiling)",
              fontweight="bold", fontsize=10.5)
axA.legend(fontsize=8.5, loc="upper right"); axA.grid(alpha=0.3, which="both")

# Panel B: absolute latency vs footprint (B8), ps1 & ps128
B = 8 if 8 in Bs else Bs[0]
pts = sorted({L for (b, L, p) in lat if b == B})
x = [fp_mb(B, L) for L in pts]
for ps, c, mk in [(1, "#1f4e79", "o"), (128, "#c0392b", "s")]:
    y = [lat.get((B, L, ps), np.nan) for L in pts]
    axB.plot(x, y, mk + "-", color=c, lw=2, ms=6, label=f"ps{ps}")
axB.axvline(L2_MB, color="gray", ls=":", lw=1.4); axB.text(L2_MB*1.05, axB.get_ylim()[1]*0.1, "L2=32MB", fontsize=8, color="gray")
axB.set_xscale("log"); axB.set_xlabel(f"distinct KV footprint / layer (MB), B={B}")
axB.set_ylabel("decode kernel latency (ms)")
axB.set_title("(B) Cache-resident plateau (~0.13 ms, flat) →\nDRAM-bound linear ramp; ps1≈ps128 throughout",
              fontweight="bold", fontsize=10.5)
axB.legend(fontsize=9); axB.grid(alpha=0.3, which="both")

# Panel C: ncu DRAM% / L2-hit% vs footprint (B8, ps128 representative), warm vs cold
B = 8
Ls = sorted({L for (cache, ps, b, L) in ncu if b == B})
def series(metric, cache, ps=128):
    xs, ys = [], []
    for L in Ls:
        r = ncu.get((cache, ps, B, L)) or ncu.get((cache, 1, B, L))
        if r and r.get(metric) is not None:
            xs.append(fp_mb(B, L)); ys.append(r[metric])
    return xs, ys
if ncu:
    for metric, c, lab in [("dram", "#c0392b", "DRAM%"), ("l2", "#1f4e79", "L2-hit%")]:
        xs, ys = series(metric, "none"); axC.plot(xs, ys, "o-", color=c, lw=2, ms=6, label=f"{lab} warm(resident)")
        xs, ys = series(metric, "all");  axC.plot(xs, ys, "s--", color=c, lw=1.6, ms=5, alpha=0.7, label=f"{lab} cold(stream)")
    axC.axvspan(0.5, L2_MB, color="#cfe8cf", alpha=0.5)
    axC.axhline(90, color="gray", ls=":", lw=1)
axC.set_xscale("log"); axC.set_xlabel(f"distinct KV footprint / layer (MB), B={B}")
axC.set_ylabel("% (ncu hardware counters)"); axC.set_ylim(0, 105)
axC.set_title("(C) ncu: warm small KV is OFF the DRAM ceiling\n(L2-resident); cold = real per-step streaming",
              fontweight="bold", fontsize=10.5)
axC.legend(fontsize=7.8, loc="center right"); axC.grid(alpha=0.3, which="both")

fig.suptitle("Report 8 — Shrinking the KV does NOT make page_size=1 lose in a true batch "
             "(isolated FlashInfer decode kernel, Qwen3-VL-2B GQA, RTX 5060 Ti, L2=32 MB)",
             fontsize=12.5, fontweight="bold", y=0.98)
out = OUT / "fig_smallkv_kernel.png"
fig.savefig(out, dpi=130); print("wrote", out)

# tables for the report
print("\n=== latency ps1/ps128 (%) vs footprint ===")
for B in Bs:
    for L in sorted({L for (b, L, p) in lat if b == B}):
        if (B, L, 1) in lat and (B, L, 128) in lat:
            print(f"  B{B}/L{L:<6} {fp_mb(B,L):>6.0f}MB  ps1={lat[(B,L,1)]:.4f} ps128={lat[(B,L,128)]:.4f}  "
                  f"{(lat[(B,L,1)]/lat[(B,L,128)]-1)*100:+.2f}%")
if ncu:
    print("\n=== ncu B8 (ps128): footprint  DRAM%(warm/cold)  L2hit%(warm/cold) ===")
    for L in Ls:
        w = ncu.get(("none", 128, 8, L), {}); c = ncu.get(("all", 128, 8, L), {})
        print(f"  L{L:<6} {fp_mb(8,L):>6.0f}MB  DRAM {w.get('dram',float('nan')):.0f}/{c.get('dram',float('nan')):.0f}"
              f"   L2 {w.get('l2',float('nan')):.0f}/{c.get('l2',float('nan')):.0f}")
