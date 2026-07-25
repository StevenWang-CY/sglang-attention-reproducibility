#!/usr/bin/env python3
"""Report 13 HEADLINE figure: constructing the off-the-DRAM-wall regime in a TRUE batch
(individual requests, distinct KV) and what page_size costs there.

Reads:
  studies/kv-cache-page-size/data/raw/offwall_profile/flashinfer_distinct_{all,none}_h{QH}k{KH}_ps{ps}_bs{B}_kv{L}.csv
  studies/kv-cache-page-size/data/raw/offwall_profile/pagecost_{pat}_all_h{QH}k{KH}_bs{B}_kv{L}.csv
  studies/kv-cache-page-size/data/raw/offwall_profile/bob_{model}_all_ps{ps}_bs{B}_kv{L}.csv     (engine ncu)
  studies/kv-cache-page-size/data/raw/offwall_profile/lat_offwall_h*_r*.json                     (CUDA-event ladder)
  studies/kv-cache-page-size/data/raw/offwall_profile/pagecost_offwall_h*_r*.json                (frag latency)
Writes ../13-true-batch-off-wall/fig_offwall.png:
  (A) DRAM% vs context per batch (GQA-8 solid / GQA-2 dashed) — the constructed region,
      with the engine-ncu (bench_one_batch) points overlaid as stars.
  (B) DRAM% vs total decode work B*L*kv_heads — both arms collapse onto one curve
      (work/parallelism-governed, not model-specific); L2-hit stays ~0 (cold) = not caching.
  (C) page_size cost off the wall: ncu kernel-duration ps1/ps128 ratio vs the cell's DRAM%
      (internal cause underdetermined) + wall-clock ladder ratio (launch-floor hidden).
  (D) worst-case ps1 fragmentation (scatter vs contig): ncu kernel duration + wall-clock,
      off-wall vs on-wall cells; the +5% bar.
"""
import csv, glob, json, re, statistics
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
D = ROOT / "studies" / "kv-cache-page-size" / "data" / "raw" / "offwall_profile"
OUT = Path(__file__).resolve().parents[1] / "reports" / "13-true-batch-off-wall"
OUT.mkdir(exist_ok=True)
HEAD_DIM = 128


def num(s):
    if s is None:
        return None
    s = str(s).strip().strip('"').replace(",", "")
    if s in ("", "N/A", "n/a", "<null>"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


UNIT_DUR = {"ns": 1e-3, "us": 1.0, "usecond": 1.0, "ms": 1e3, "s": 1e6}


def load_rows(path):
    lines = path.read_text(errors="replace").splitlines()
    hdr = next((i for i, l in enumerate(lines) if l.startswith('"ID"')), None)
    if hdr is None:
        return []
    rows = [dict(r) for r in csv.DictReader(lines[hdr:])]
    units = {}
    if rows and num(rows[0].get("gpu__time_duration.sum")) is None:
        units, rows = rows[0], rows[1:]
    out = []
    for r in rows:
        d = num(r.get("gpu__time_duration.sum"))
        if d is None:
            continue
        u = (units.get("gpu__time_duration.sum") or "").strip()
        out.append(dict(
            dur_us=d * UNIT_DUR.get(u, 1.0),
            dram=num(r.get("dram__throughput.avg.pct_of_peak_sustained_elapsed")),
            l2=num(r.get("lts__t_sector_hit_rate.pct")),
            sm=num(r.get("sm__throughput.avg.pct_of_peak_sustained_elapsed")),
            grid=num(r.get("launch__grid_size")),
            waves=num(r.get("launch__waves_per_multiprocessor")),
            kernel=r.get("Kernel Name", "?"),
        ))
    return out


def main_kernel(path):
    rows = load_rows(path)
    if not rows:
        return None
    attn = [r for r in rows if "BatchDecodeWithPagedKVCache" in r.get("kernel", "")
            or "BatchPrefillWithPagedKVCache" in r.get("kernel", "")]
    return max(attn or rows, key=lambda r: r["dur_us"])


def med_kernel(path):
    rows = load_rows(path)
    if not rows:
        return None
    return {k: statistics.median([r[k] for r in rows if r[k] is not None])
            for k in rows[0] if k != "kernel" and any(r[k] is not None for r in rows)}


# ---- load ncu grid ----
G = {}    # (kh, ps, B, L) -> metrics   (cold distinct microbench, CUDA-core decode kernel)
GTC = {}  # same, tensor-core wrapper arm (sglang's kernel for GQA group>=4)
for f in sorted(D.glob("flashinfer*_distinct_all_h16k*_ps*_bs*_kv*.csv")):
    m = re.search(r"flashinfer(tc)?_distinct_all_h16k(\d+)_ps(\d+)_bs(\d+)_kv(\d+)", f.name)
    r = main_kernel(f)
    if m and r:
        (GTC if m[1] else G)[(int(m[2]), int(m[3]), int(m[4]), int(m[5]))] = r

BOB = {}  # (model, ps, B, L) -> metrics  (engine ncu, per-launch median)
for f in sorted(D.glob("bob_*_all_ps*_bs*_kv*.csv")):
    m = re.search(r"bob_(q25_3b|q3vl_2b)_all_ps(\d+)_bs(\d+)_kv(\d+)", f.name)
    r = med_kernel(f)
    if m and r:
        BOB[(m[1], int(m[2]), int(m[3]), int(m[4]))] = r

PC = {}  # (kh_label, pat, B, L) -> metrics (pagecost ncu); kh_label "2tc" = tensor-core arm
# rep2/ (the clean verification pass) takes precedence: the first pagecosttc window
# overlapped concurrent engine runs and one cell was contention-dilated (2.1x phantom).
for base in (D, D / "rep2"):
    for f in sorted(base.glob("pagecost*_all_h16k*_bs*_kv*.csv")):
        m = re.search(r"pagecost(tc)?_(contig|block128|scatter)_all_h16k(\d+)_bs(\d+)_kv(\d+)", f.name)
        r = main_kernel(f)
        if m and r:
            PC[(m[3] + ("tc" if m[1] else ""), m[2], int(m[4]), int(m[5]))] = r

# ---- load wall-clock ladders ----
LAT = {}  # (kh, B, L, ps) -> [ms per round]   (CUDA-core arms; tc ladder analyzed in T2c)
for f in sorted(D.glob("lat_offwall_h16k*_r*.json")):
    m = re.search(r"h16k(\d+)(tc)?_r", f.name)
    if not m or m[2]:
        continue
    kh = int(m[1])
    for v in json.loads(f.read_text()).values():
        if "latency_ms" in v:
            LAT.setdefault((kh, v["batch_size"], v["seq_len"], v["page_size"]), []).append(v["latency_ms"])

PCL = {}  # (kh, B, L, pat) -> [ms per round]
for f in sorted(D.glob("pagecost_offwall_h16k*_r*.json")):
    m = re.search(r"h16k(\d+)(tc)?_r", f.name)
    if not m or m[2]:
        continue
    kh = int(m[1])
    for v in json.loads(f.read_text()).values():
        if "latency_ms" in v:
            PCL.setdefault((kh, v["batch_size"], v["seq_len"], v["pattern"]), []).append(v["latency_ms"])

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(16.5, 10.4))
gs = fig.add_gridspec(2, 2, wspace=0.22, hspace=0.46,
                      left=0.055, right=0.985, top=0.90, bottom=0.07)
axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
axC, axD = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
BLUE = {1: "#9ecae1", 2: "#4292c6", 4: "#2171b5", 8: "#08306b", 16: "#54278f", 32: "#3f007d"}
RED = {1: "#fcae91", 2: "#fb6a4a", 4: "#de2d26", 8: "#a50f15"}

# ---- A: DRAM% vs L, per B ----
for kh, cmap, ls, lab in [(8, BLUE, "-", "GQA-8 (Qwen3-VL-2B shape)"),
                          (2, RED, "--", "GQA-2 (Qwen2.5-3B shape)")]:
    Bs = sorted({B for (k, ps, B, L) in G if k == kh and ps == 128})
    for B in Bs:
        pts = sorted([(L, G[(kh, 128, B, L)]["dram"]) for (k, ps, b, L) in G
                      if k == kh and ps == 128 and b == B])
        if pts:
            axA.plot([p[0] for p in pts], [p[1] for p in pts], ls, marker="o", ms=4.5,
                     color=cmap.get(B, "gray"), lw=1.8,
                     label=f"B={B}" + (" GQA-2" if kh == 2 else ""))
# tensor-core arm (the kernel sglang actually dispatches for GQA group>=4) — B2 line
tc2 = sorted([(L, GTC[(2, 128, 2, L)]["dram"]) for (kh, ps, B, L) in GTC
              if kh == 2 and ps == 128 and B == 2])
if tc2:
    axA.plot([p[0] for p in tc2], [p[1] for p in tc2], ":", marker="D", ms=5,
             color="#7a0177", lw=2.2, label="B=2 GQA-2, tensor-core kernel (sglang's pick)")
# engine ncu stars (flagship cell only; the B8/L4096 engine window's per-step
# attribution is unresolved — see report §1 table — so it is not plotted)
for (mdl, ps, B, L), r in BOB.items():
    if ps != 128 or (B, L) != (2, 1024):
        continue
    c = "#d62728" if mdl == "q25_3b" else "#08306b"
    axA.scatter([L], [r["dram"]], marker="*", s=340, zorder=6, color=c, edgecolor="k", lw=0.8)
    axA.annotate(f"engine {('Qwen2.5-3B' if mdl == 'q25_3b' else 'Qwen3-VL-2B')}\nB{B}/L{L}: {r['dram']:.0f}%",
                 (L, r["dram"]), textcoords="offset points",
                 xytext=(10, -16) if mdl != "q25_3b" else (-64, 14), fontsize=8.5)
axA.axhline(50, color="crimson", ls=":", lw=2)
axA.text(140, 51.5, "DRAM 50% — the 'not-bandwidth-bound' bar", fontsize=9, color="crimson")
axA.axhspan(0, 50, color="#fde8e8", alpha=0.45, zorder=0)
axA.set_xscale("log", base=2)
axA.set_xlabel("context length L (tokens, log2)")
axA.set_ylabel("decode-attn kernel DRAM throughput (% of peak, ncu cold)")
axA.set_title("A — constructing the off-wall TRUE batch: small work leaves the DRAM wall\n"
              "(distinct KV, FlashInfer decode kernel, ps128; ★ = measured inside bench_one_batch)")
axA.set_ylim(0, 100)
h, l = axA.get_legend_handles_labels()
axA.legend(h, l, fontsize=7, ncol=7, loc="upper center",
           bbox_to_anchor=(0.5, -0.105), framealpha=0.95)

# ---- B: collapse on total work B*L*kh ----
for kh, cmap, mk in [(8, BLUE, "o"), (2, RED, "s")]:
    for (k, ps, B, L), r in G.items():
        if k != kh or ps != 128:
            continue
        axB.scatter(B * L * kh, r["dram"], marker=mk, s=42,
                    color=cmap.get(B, "gray"), zorder=4,
                    edgecolor="k", linewidth=0.3)
axB.axhline(50, color="crimson", ls=":", lw=2)
axB.set_xscale("log")
axB.set_xlabel("total decode work  B × L × KV-heads  (∝ KV bytes/layer = ×512 B)")
axB.set_ylabel("DRAM % of peak (ncu cold)")
axB.set_title("B — one curve: DRAM saturation is set by total work (bytes to stream ×\n"
              "available CTA parallelism), not by model or batch alone  [circles GQA-8, squares GQA-2]")
axB.set_ylim(0, 100)
# annotate L2-hit stays ~0
l2s = [r["l2"] for (k, ps, B, L), r in G.items() if ps == 128 and r["l2"] is not None]
if l2s:
    axB.text(0.03, 0.05, f"cold L2-hit in ALL cells: {min(l2s):.1f}–{max(l2s):.1f}%\n"
             "→ off-wall ≠ cache-resident; the missing DRAM% is idle\n"
             "(latency-bound: waves/SM < 1, occupancy ≈ 8–15%)",
             transform=axB.transAxes, fontsize=9, va="bottom",
             bbox=dict(boxstyle="round", fc="#fffde7", ec="#999"))

# ---- C: ps1 cost vs DRAM% ----
for kh, col, mk, lab in [(8, "#08306b", "o", "GQA-8"), (2, "#d62728", "s", "GQA-2")]:
    xs, ys = [], []
    for (k, ps, B, L), r in G.items():
        if k != kh or ps != 1:
            continue
        r128 = G.get((kh, 128, B, L))
        if not r128:
            continue
        xs.append(r128["dram"])
        ys.append((r["dur_us"] / r128["dur_us"] - 1) * 100)
    axC.scatter(xs, ys, marker=mk, s=48, color=col, label=f"ncu kernel duration ({lab})", zorder=4)
# wall-clock ladder ratio (median across rounds), same cells
for kh, col, mk in [(8, "#9ecae1", "o"), (2, "#fcae91", "s")]:
    xs, ys = [], []
    for (k, ps, B, L), r in G.items():
        if k != kh or ps != 128:
            continue
        l1 = LAT.get((kh, B, L, 1))
        l128 = LAT.get((kh, B, L, 128))
        if l1 and l128:
            xs.append(r["dram"])
            ys.append((statistics.median(l1) / statistics.median(l128) - 1) * 100)
    axC.scatter(xs, ys, marker=mk, s=26, color=col, alpha=0.75,
                label="wall-clock (CUDA-event, launch floor)" if kh == 8 else None, zorder=3)
axC.axhline(0, color="k", lw=1)
axC.axhline(5, color="crimson", ls="--", lw=1.4)
axC.text(2, 5.3, "+5% bar", color="crimson", fontsize=9)
axC.set_xlabel("cell's DRAM % of peak (ps128, ncu cold)  →  the wall is on the right")
axC.set_ylabel("ps1 vs ps128 (%)")
axC.set_title("C — page_size off the wall: native-plan microseconds differ by +1–5%\n"
              "as DRAM% falls (bytes identical; internal cause underdetermined);\n"
              "invisible at wall-clock (fixed ~0.13 ms launch floor dominates)")
axC.legend(fontsize=8.5, loc="upper right")
axC.set_ylim(-6, 8)

# ---- D: fragmentation ----
cells8 = [("8", 1, 512), ("8", 2, 1024), ("8", 4, 512), ("8", 8, 4096)]
cells2 = [("2", 2, 1024), ("2", 8, 1024), ("2", 8, 8192)]
cellstc = [("2tc", 2, 1024), ("2tc", 8, 8192)]
labels, ncu_v, lat_v = [], [], []
for cells in (cells8, cells2, cellstc):
    for (kh, B, L) in cells:
        c, s = PC.get((kh, "contig", B, L)), PC.get((kh, "scatter", B, L))
        nv = (s["dur_us"] / c["dur_us"] - 1) * 100 if c and s else np.nan
        khn = int(kh.replace("tc", ""))
        lc = PCL.get((khn, B, L, "contig")) if "tc" not in kh else None
        ls_ = PCL.get((khn, B, L, "scatter")) if "tc" not in kh else None
        lv = ((statistics.median(ls_) / statistics.median(lc) - 1) * 100) if lc and ls_ else np.nan
        dram = (c or {}).get("dram")
        labels.append(f"k{kh} B{B}\nL{L}" + (f"\n{dram:.0f}%" if dram else ""))
        ncu_v.append(nv)
        lat_v.append(lv)
x = np.arange(len(labels))
axD.bar(x - 0.19, ncu_v, 0.36, color="#2171b5", label="ncu kernel duration")
axD.bar(x + 0.19, lat_v, 0.36, color="#9ecae1", label="wall-clock (median of 3 rounds)")
axD.axhline(0, color="k", lw=1)
axD.axhline(5, color="crimson", ls="--", lw=1.4)
axD.set_xticks(x)
axD.set_xticklabels(labels, fontsize=8)
axD.set_ylabel("scatter vs contig (%)")
axD.set_title("D — worst-case ps1 fragmentation (random token order) stays <5% everywhere —\n"
              "off-wall and at the wall, CUDA-core and tensor-core kernels\n"
              "(cell's DRAM% under each label; tc cells verified in a clean repro pass)")
axD.legend(fontsize=8.5)
axD.set_ylim(-3.5, 11)

fig.suptitle("Report 13 — a TRUE batch (individual requests) leaves the DRAM wall at small work (B×L×heads);\n"
             "page_size then costs +1–5% kernel time but stays launch-floor-hidden — RTX 5060 Ti (sm120), FlashInfer 0.6.6, ncu cold",
             fontsize=11, y=0.985)
fig.savefig(OUT / "fig_offwall.png", dpi=140)
print("wrote", OUT / "fig_offwall.png")
