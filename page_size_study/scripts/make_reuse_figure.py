#!/usr/bin/env python3
"""Report 12 headline figure: L2-hit is governed by intra-kernel REUSE, not footprint.
Reads offline_batch_results/reuse_profile/*.csv (ncu raw CSV). Writes fig_reuse_law.png.
(A) reuse sweep at fixed 4MB footprint + (R-1)/R overlay; (B) fixed-R vs footprint (flat);
(C) cold-vs-warm distinct (residency artifact)."""
import csv, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "offline_batch_results" / "reuse_profile"
OUT = Path(__file__).resolve().parents[1] / "report_12_reuse_not_residency"
OUT.mkdir(exist_ok=True)
BYTES_PER_TOK = 4096  # K+V bf16, 8 kv heads*128*2B*2 (per layer)
L2_MB = 32
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
    return dict(l2=g("lts__t_sector_hit_rate.pct"),
                dram=g("dram__throughput.avg.pct_of_peak_sustained_elapsed"),
                hit=g("lts__t_sectors_lookup_hit.sum"), miss=g("lts__t_sectors_lookup_miss.sum"),
                dur=g("gpu__time_duration.sum"))


def collect():
    d = {}
    for p in DATA.glob("*.csv"):
        m = TAG.match(p.name)
        if not m: continue
        mode, cache, ps, B, L = m.group(1), m.group(2), int(m.group(3)), int(m.group(4)), int(m.group(5))
        r = load(p)
        if r: d[(mode, cache, ps, B, L)] = r
    return d


D = collect()
print(f"loaded {len(D)} reuse_profile cells")

fig, ax = plt.subplots(1, 3, figsize=(16, 4.7))
CR, BL, GR = "#c0392b", "#1f4e79", "#2e8b57"

# Panel A: reuse sweep (shared, cold, ps128, L=1024 -> 4MB fixed) vs B=R
Bs = [1, 2, 4, 6, 8, 12, 16, 24, 32]
xa, ya = [], []
for B in Bs:
    r = D.get(("shared", "all", 128, B, 1024))
    if r and r["l2"] is not None:
        xa.append(B); ya.append(r["l2"])
ax[0].plot(xa, ya, "o-", color=CR, label="measured (shared, 4 MB fixed)", zorder=3)
RR = np.array(sorted(set(xa)) or [1])
ax[0].plot(RR, 100 * (RR - 1) / RR, "k--", lw=1.2, label="(R−1)/R model")
ax[0].set_xscale("log", base=2)
ax[0].set_xticks(Bs); ax[0].set_xticklabels(Bs)
ax[0].set_xlabel("reuse factor R = batch B (shared prefix)")
ax[0].set_ylabel("L2 hit %"); ax[0].set_ylim(-3, 100)
ax[0].set_title("(A) L2-hit climbs ≈0→97% with reuse R (R=1 floors ~2.5%)\nfootprint PINNED at 4 MB (1/8 L2)")
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

# Panel B: fixed-R-vs-footprint (flat). shared B8 over L; distinct B8 over L.
def line(mode, B, Ls, fp_per_tok_factor):
    xs, ys = [], []
    for L in Ls:
        r = D.get((mode, "all", 128, B, L))
        if r and r["l2"] is not None:
            xs.append(fp_per_tok_factor * L * BYTES_PER_TOK / 1e6); ys.append(r["l2"])
    return xs, ys
xs, ys = line("shared", 8, [256, 1024, 4096, 8192, 16384, 39168], 1)   # shared footprint = L*4096
ax[1].plot(xs, ys, "s-", color=GR, label="shared, R=8 (footprint=1 copy)")
xd, yd = line("distinct", 8, [512, 1024, 2048, 4096, 8192, 16384], 8)  # distinct footprint = B*L*4096
ax[1].plot(xd, yd, "o-", color=CR, label="distinct, R=1 (footprint=B copies)")
ax[1].axvline(L2_MB, color="#888", ls=":", lw=1.2, label="L2 = 32 MB")
ax[1].set_xscale("log"); ax[1].set_xlabel("KV footprint / layer (MB, log)")
ax[1].set_ylabel("L2 hit %"); ax[1].set_ylim(-3, 100)
ax[1].set_title("(B) Fix R, sweep footprint 160× → FLAT\nlevel set by R, not by footprint")
ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)

# Panel C: cold vs warm distinct (residency artifact), B8 over footprint
def cwline(cache):
    xs, ys = [], []
    for L in [512, 1024, 2048, 4096, 8192, 16384]:
        r = D.get(("distinct", cache, 128, 8, L))
        if r and r["l2"] is not None:
            xs.append(8 * L * BYTES_PER_TOK / 1e6); ys.append(r["l2"])
    return xs, ys
xc, yc = cwline("all"); xw, yw = cwline("none")
ax[2].plot(xc, yc, "o-", color=BL, label="cold (--cache-control all) = faithful")
ax[2].plot(xw, yw, "^--", color=CR, label="warm (none) = cross-launch residency")
ax[2].axvline(L2_MB, color="#888", ls=":", lw=1.2, label="L2 = 32 MB")
ax[2].set_xscale("log"); ax[2].set_xlabel("distinct KV footprint / layer (MB, log)")
ax[2].set_ylabel("L2 hit %"); ax[2].set_ylim(-3, 100)
ax[2].set_title("(C) Distinct: cold ≈0% (no reuse);\nwarm 'hits' are a residency ARTIFACT")
ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)

fig.suptitle("Report 12 — L2 cache hit-rate is governed by intra-kernel REUSE (R), not footprint. Distinct true-batch decode = R=1 (each KV byte read once) → ≈0% even sub-L2.", fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = OUT / "fig_reuse_law.png"
fig.savefig(out, dpi=130)
print("wrote", out)

# print the reuse-sweep table
print("\n=== Exp B reuse sweep (shared, 4MB fixed) ===")
for B in Bs:
    r = D.get(("shared", "all", 128, B, 1024))
    if r: print(f"  R=B{B:>3}: L2-hit={r['l2']:.2f}%  (model (R-1)/R={100*(B-1)/B:.1f}%)  DRAM%={r['dram']:.1f}")
