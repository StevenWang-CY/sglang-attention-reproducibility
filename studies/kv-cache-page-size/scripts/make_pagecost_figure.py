#!/usr/bin/env python3
"""
Figure + summary for the decisive FlashInfer true-batch page-cost experiment (report 10).
Reads:
  studies/kv-cache-page-size/data/raw/pagecost_profile/pagecost_gather.json  (kernel: contig vs block128 vs scatter)
  studies/kv-cache-page-size/data/raw/pagecost_profile/pagecost_plan.json    (host plan cost vs index length)
  studies/kv-cache-page-size/data/raw/pagecost_profile/flashinfer_{contig,scatter}_all_bs{B}_kv{L}.csv  (ncu)
Writes report_10_.../fig_pagecost.png and prints the tables.
"""
import json, csv, re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "studies" / "kv-cache-page-size" / "data" / "raw" / "pagecost_profile"
OUT = Path(__file__).resolve().parents[1] / "reports" / "10-flashinfer-page-cost"
OUT.mkdir(exist_ok=True)

BYTES_PER_TOK = 4096  # K+V bf16, 8 kv heads*128*2B*2


def fnum(s):
    try:
        return float(str(s).replace(",", "").strip())
    except Exception:
        return None


def load_ncu(path):
    lines = path.read_text().splitlines()
    hdr = next(i for i, l in enumerate(lines) if l.startswith('"ID"'))
    rows = list(csv.DictReader(lines[hdr:]))
    # pick the row with max kernel duration (the real decode kernel)
    best, bestd = None, -1
    for r in rows:
        d = fnum(r.get("gpu__time_duration.sum"))
        if d is not None and d > bestd:
            best, bestd = r, d
    return best


def g(r, k):
    return fnum(r.get(k)) if r else None


# ---------- load ----------
gather = json.load(open(DATA / "pagecost_gather.json"))
plan = json.load(open(DATA / "pagecost_plan.json"))

# ---------- gather table ----------
print("\n=== KERNEL gather (FlashInfer decode, distinct true batch): contig vs block128 vs scatter ===")
print(f"{'B':>3} {'L':>8} {'tokens':>9} {'fp_GB':>6} {'contig_ms':>10} {'block128':>9} {'scatter':>9} {'scat-vs-contig':>14}")
cells = sorted({(v['batch_size'], v['seq_len']) for v in gather.values() if 'batch_size' in v},
               key=lambda c: (c[0], c[1]))
gather_rows = []
for (B, L) in cells:
    c = gather.get(f"gather_contig_bs{B}_kv{L}", {}).get("latency_ms")
    b = gather.get(f"gather_block128_bs{B}_kv{L}", {}).get("latency_ms")
    s = gather.get(f"gather_scatter_bs{B}_kv{L}", {}).get("latency_ms")
    N = B * L; fp = N * BYTES_PER_TOK / 1e9
    pen = (s / c - 1) * 100 if (c and s) else None
    gather_rows.append((B, L, N, fp, c, b, s, pen))
    print(f"{B:>3} {L:>8} {N:>9} {fp:>6.2f} {c:>10.4f} {b:>9.4f} {s:>9.4f} {pen:>+13.2f}%")

# ---------- plan table ----------
print("\n=== HOST plan() cost vs index length (per decode step) ===")
print(f"{'cell':>14} {'n_entries':>10} {'host_ms':>9} {'dev_ms':>8}")
plan_rows = []
for k, v in sorted(plan.items(), key=lambda kv: kv[1].get('n_entries', 0)):
    if 'error' in v:
        continue
    plan_rows.append((v['n_entries'], v['plan_host_ms'], v['plan_dev_ms'], v['page_label'], v['batch_size'], v['seq_len']))
    print(f"{k:>14} {v['n_entries']:>10} {v['plan_host_ms']:>9.4f} {v['plan_dev_ms']:>8.4f}")

# ---------- ncu table ----------
print("\n=== ncu (cold, --cache-control all): captured decode kernel, contig vs scatter ===")
print(f"{'cell':>16} {'pat':>8} {'dur_us':>9} {'DRAM%':>7} {'L2hit%':>7} {'sec/req':>8} {'SM%':>6}")
ncu_rows = []
for (B, L) in [(8, 32768), (8, 131072)]:
    for pat in ["contig", "scatter"]:
        p = DATA / f"flashinfer_{pat}_all_bs{B}_kv{L}.csv"
        if not p.exists():
            continue
        r = load_ncu(p)
        dur = g(r, "gpu__time_duration.sum")
        dram = g(r, "dram__throughput.avg.pct_of_peak_sustained_elapsed")
        l2 = g(r, "lts__t_sector_hit_rate.pct")
        secreq = g(r, "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio")
        sm = g(r, "sm__throughput.avg.pct_of_peak_sustained_elapsed")
        ncu_rows.append((B, L, pat, dur, dram, l2, secreq, sm))
        ds = f"{dur/1000:.1f}" if dur else "  -"
        print(f"{('B%d L%d'%(B,L)):>16} {pat:>8} {ds:>9} {('%.1f'%dram) if dram is not None else '-':>7} "
              f"{('%.1f'%l2) if l2 is not None else '-':>7} {('%.2f'%secreq) if secreq is not None else '-':>8} "
              f"{('%.1f'%sm) if sm is not None else '-':>6}")

# ---------- figure ----------
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
CR = "#c0392b"

# Panel A: scatter-vs-contig kernel penalty vs footprint
ax = axes[0]
xs = [r[3] for r in gather_rows if r[7] is not None]  # fp_GB
ys = [r[7] for r in gather_rows if r[7] is not None]   # scatter penalty
b128 = [(r[3], (r[5]/r[4]-1)*100) for r in gather_rows if r[4] and r[5]]
ax.plot(xs, ys, "o-", color=CR, label="scatter (worst-case ps1)")
ax.plot([x for x, _ in b128], [y for _, y in b128], "s--", color="#2e8b57", label="block128 (ps128-like)")
ax.axhline(5, color=CR, ls=":", lw=1.2, label="+5% threshold")
ax.axhline(0, color="#888", lw=0.8)
ax.axvspan(0, 0.032, color="#2e8b57", alpha=0.10)  # L2-resident (32 MB)
ax.set_xscale("log"); ax.set_xlabel("distinct KV footprint (GB, log)")
ax.set_ylabel("kernel latency vs contiguous (%)")
ax.set_ylim(-3, 7); ax.set_title("(A) FlashInfer decode kernel is gather-robust\nscattered kv_indices ≤ +1.3% everywhere")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Panel B: host plan cost vs index length
ax = axes[1]
ne = [r[0] for r in plan_rows]; hm = [r[1] for r in plan_rows]
ax.scatter(ne, hm, c="#1f4e79", zorder=3)
ax.set_xscale("log")
ax.set_xlabel("kv_indices length (entries, log)")
ax.set_ylabel("plan() host cost per step (ms)")
ax.set_ylim(0, max(hm) * 1.6)
ax.set_title("(B) Host plan() is length-insensitive\n~0.26 ms from 64 → 2.1M entries")
ax.grid(alpha=0.3)
ax.annotate("ps1 (1.05M) ≈ ps128 (8.2k)\nat B8×131072", xy=(0.5, 0.18), xycoords="axes fraction", fontsize=8, color="#1f4e79")

# Panel C: ncu DRAM% / L2-hit / sec-per-req contig vs scatter
ax = axes[2]
labels = [f"B{B}L{L}\n{pat}" for (B, L, pat, *_ ) in ncu_rows]
dram = [r[4] for r in ncu_rows]
l2 = [r[5] for r in ncu_rows]
import numpy as np
x = np.arange(len(ncu_rows)); w = 0.38
ax.bar(x - w/2, dram, w, label="DRAM % peak", color="#c0392b")
ax.bar(x + w/2, l2, w, label="L2 hit %", color="#1f4e79")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
ax.set_ylabel("%"); ax.set_ylim(0, 105)
ax.set_title("(C) ncu (cold): kernel stays DRAM-bound,\nscatter ≈ contig (token KV read = 4 KB burst)")
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

fig.suptitle("Report 10 — FlashInfer + CUDA graph + distinct KV: ps1 has no true-batch decode cost (host page-independent + kernel scatter-robust)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = OUT / "fig_pagecost.png"
fig.savefig(out, dpi=130)
print("\nwrote", out)
