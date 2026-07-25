#!/usr/bin/env python3
"""Figure + tables for the MLA caveat test (report 11). Reads pagecost_profile/mla_*.json
and the MLA ncu CSVs; writes report_11_.../fig_mla_pagecost.png."""
import json, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "studies" / "kv-cache-page-size" / "data" / "raw" / "pagecost_profile"
OUT = Path(__file__).resolve().parents[1] / "reports" / "11-mla-scatter-page-cost"
OUT.mkdir(exist_ok=True)
BYTES_PER_TOK = (512 + 64) * 2  # MLA latent bf16 = 1152 B/token


def fnum(s):
    try:
        return float(str(s).replace(",", "").strip())
    except Exception:
        return None


def load_ncu(path):
    lines = path.read_text().splitlines()
    hdr = next(i for i, l in enumerate(lines) if l.startswith('"ID"'))
    rows = list(csv.DictReader(lines[hdr:]))
    best, bestd = None, -1
    for r in rows:
        d = fnum(r.get("gpu__time_duration.sum"))
        if d is not None and d > bestd:
            best, bestd = r, d
    return best


def g(r, k):
    return fnum(r.get(k)) if r else None


gather = json.load(open(DATA / "mla_pagecost.json"))
worst = json.load(open(DATA / "mla_worstcase.json"))

print("\n=== MLA decode kernel: scatter vs contig (and block64) ===")
print(f"{'H':>4} {'B':>3} {'L':>8} {'tok':>9} {'fp_GB':>6} {'contig_ms':>10} {'blk64%':>8} {'scatter%':>9}")
rows = []
for k, v in gather.items():
    if "error" in v or v.get("pattern") != "contig":
        continue
    H, B, L = v["num_heads"], v["batch_size"], v["seq_len"]
    c = v["latency_ms"]
    b = gather.get(f"mla_h{H}_block64_bs{B}_kv{L}", {}).get("latency_ms")
    s = gather.get(f"mla_h{H}_scatter_bs{B}_kv{L}", {}).get("latency_ms")
    N = B * L
    bp = (b/c-1)*100 if (b and c) else None
    sp = (s/c-1)*100 if (s and c) else None
    rows.append((H, B, L, N, N*BYTES_PER_TOK/1e9, c, bp, sp))
for r in sorted(rows, key=lambda r: (r[0], r[1], r[2])):
    print(f"{r[0]:>4} {r[1]:>3} {r[2]:>8} {r[3]:>9} {r[4]:>6.2f} {r[5]:>10.4f} {r[6]:>+7.2f}% {r[7]:>+8.2f}%")

print("\n=== worst case: B1 h16 long single scattered sequence (does it cross 5%?) ===")
wrows = []
for k, v in worst.items():
    if "error" in v or v.get("pattern") != "contig":
        continue
    L = v["seq_len"]; c = v["latency_ms"]
    s = worst.get(f"mla_h16_scatter_bs1_kv{L}", {}).get("latency_ms")
    sp = (s/c-1)*100 if (s and c) else None
    wrows.append((L, v["n_tokens"]*BYTES_PER_TOK/1e9, c, sp))
for r in sorted(wrows):
    print(f"  L={r[0]:>8}  fp={r[1]:.2f}GB  contig={r[2]:.3f}ms  scatter {r[3]:+.2f}%")

print("\n=== ncu (cold): MLA kernel contig vs scatter ===")
print(f"{'cell':>20} {'pat':>8} {'DRAM%':>7} {'L2hit%':>7} {'sec/req':>8} {'SM%':>6} {'DRAMrd_GB':>10}")
ncu = {}
for (H, B, L) in [(16, 8, 131072), (128, 8, 131072), (16, 1, 524288)]:
    for pat in ["contig", "scatter"]:
        p = DATA / f"mla_h{H}_{pat}_all_bs{B}_kv{L}.csv"
        if not p.exists():
            continue
        r = load_ncu(p)
        dram = g(r, "dram__throughput.avg.pct_of_peak_sustained_elapsed")
        l2 = g(r, "lts__t_sector_hit_rate.pct")
        secreq = g(r, "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio")
        sm = g(r, "sm__throughput.avg.pct_of_peak_sustained_elapsed")
        rd = g(r, "dram__bytes_op_read.sum")
        ncu[(H, B, L, pat)] = (dram, l2, secreq, sm, rd)
        rdg = f"{rd/1e9:.3f}" if rd else "-"
        print(f"{('h%d B%d L%d'%(H,B,L)):>20} {pat:>8} "
              f"{('%.1f'%dram) if dram is not None else '-':>7} {('%.1f'%l2) if l2 is not None else '-':>7} "
              f"{('%.2f'%secreq) if secreq is not None else '-':>8} {('%.1f'%sm) if sm is not None else '-':>6} {rdg:>10}")

# ---------- figure ----------
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
CR = "#c0392b"

# Panel A: scatter penalty vs footprint, h16 vs h128 + worstcase
ax = axes[0]
for H, col, mk in [(16, "#c0392b", "o"), (128, "#1f4e79", "s")]:
    pts = sorted([(r[4], r[7]) for r in rows if r[0] == H and r[1] == 8], key=lambda t: t[0])
    ax.plot([x for x, _ in pts], [y for _, y in pts], mk + "-", color=col, label=f"scatter, {H} heads (B8)")
wpts = sorted([(r[1], r[3]) for r in wrows])
ax.plot([x for x, _ in wpts], [y for _, y in wpts], "^--", color="#8e44ad", label="scatter, 16 heads (B1 long, worst)")
ax.axhline(5, color=CR, ls=":", lw=1.2, label="+5% threshold")
ax.axhline(0, color="#888", lw=0.8)
ax.set_xscale("log"); ax.set_xlabel("MLA-KV footprint (GB, log)")
ax.set_ylabel("scatter vs contiguous (%)"); ax.set_ylim(-1, 7)
ax.set_title("(A) MLA scatter cost saturates ~+4%,\nnever reaches +5% (block64 ≈ contig, not shown)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Panel B: ncu sectors/request contig vs scatter (IDENTICAL → not a coalescing effect)
ax = axes[1]
cells = [(16, 8, 131072), (128, 8, 131072), (16, 1, 524288)]
labels = [f"h{H}\nB{B}L{L//1024}k" for (H, B, L) in cells]
x = np.arange(len(cells)); w = 0.38
sc_c = [ncu.get((H, B, L, "contig"), (None,)*5)[2] for (H, B, L) in cells]
sc_s = [ncu.get((H, B, L, "scatter"), (None,)*5)[2] for (H, B, L) in cells]
ax.bar(x - w/2, sc_c, w, label="sectors/req contig", color="#2e8b57")
ax.bar(x + w/2, sc_s, w, label="sectors/req scatter", color="#c0392b")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("global-load sectors / request"); ax.set_ylim(0, 18)
ax.set_title("(B) sectors/request UNCHANGED by scatter\n→ not coalescing; latent is an aligned burst")
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

# Panel C: regime DRAM% (low = not bandwidth-saturated → latency-bound, so scatter latency shows)
ax = axes[2]
dram_c = [ncu.get((H, B, L, "contig"), (None,)*5)[0] for (H, B, L) in cells]
sm_c = [ncu.get((H, B, L, "contig"), (None,)*5)[3] for (H, B, L) in cells]
ax.bar(x - w/2, dram_c, w, label="DRAM % peak", color="#c0392b")
ax.bar(x + w/2, sm_c, w, label="SM % peak", color="#1f4e79")
ax.axhline(96, color="#888", ls="--", lw=1, label="MHA DRAM% (saturated)")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("%"); ax.set_ylim(0, 105)
ax.set_title("(C) MLA decode is latency-bound (DRAM 15–38%,\nnot MHA's 96%) → scatter's DRAM latency isn't hidden")
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

fig.suptitle("Report 11 — MLA caveat: worst-case scatter costs ~3–4% (vs MHA ~1%), saturating below +5%. Mechanism: MLA decode is latency-bound (DRAM not saturated), so scattered-page DRAM-access latency shows — but sectors/request is unchanged (not coalescing). Realistic paging (block64) is free.", fontsize=9.2)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = OUT / "fig_mla_pagecost.png"
fig.savefig(out, dpi=130)
print("\nwrote", out)
