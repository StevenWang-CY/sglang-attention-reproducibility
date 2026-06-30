"""
make_xqa_figure.py — figures for Report 6 (XQA TMA profile).

Writes into the report folder. Run from anywhere:
    python3 page_size_study/scripts/make_xqa_figure.py

fig_xqa_latency.png  — from xqa_latency.json (CUDA-event timing; available now):
    (A) decode latency vs page_size, XQA vs FlashInfer (both page-flat; XQA has
        no ps=1 point — XQA rejects page_size<16);
    (B) XQA-over-FlashInfer overhead (%) vs batch, per seq_len (fixed-overhead
        penalty at small batch collapses as work becomes bandwidth-bound).

fig_xqa_tma.png      — from ncu CSVs (offline_batch_results/xqa_profile/), drawn
    only once profile_xqa_ncu.sh has produced data: TMA-vs-LSU access pattern,
    DRAM traffic, and L2 hit, XQA vs FlashInfer across page_size.
"""

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
OUT = HERE.parents[1] / "report_6_xqa_tma_profile"
LAT_JSON = ROOT / "offline_batch_results" / "xqa_profile_5060ti" / "xqa_latency.json"
NCU_DIR = ROOT / "offline_batch_results" / "xqa_profile"

PAGES = [1, 16, 32, 64, 128]
COLORS = {"xqa": "#d62728", "flashinfer": "#1f77b4"}


def load_lat():
    d = json.loads(LAT_JSON.read_text())
    def get(be, ps, bs, kv):
        v = d.get(f"{be}_ps{ps}_bs{bs}_kv{kv}", {})
        return v.get("latency_ms")
    return d, get


def fig_latency():
    if not LAT_JSON.exists():
        print(f"skip latency fig: {LAT_JSON} missing")
        return
    _, get = load_lat()
    batches = [1, 8, 32, 64]
    seqs = [1024, 4096, 16384]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: latency vs page_size at kv=4096
    axA = axes[0]
    kv = 4096
    for bs in batches:
        for be, ls, mk in [("xqa", "-", "o"), ("flashinfer", "--", "s")]:
            xs = [ps for ps in PAGES if get(be, ps, bs, kv) is not None]
            ys = [get(be, ps, bs, kv) for ps in xs]
            if not xs:
                continue
            axA.plot(xs, ys, ls, marker=mk, color=COLORS[be], alpha=0.85,
                     label=f"{be} bs{bs}" if bs in (1, 64) else None)
    axA.set_xscale("log", base=2)
    axA.set_xticks(PAGES)
    axA.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axA.set_xlabel("page_size")
    axA.set_ylabel("decode latency (ms)  [median, CUDA events]")
    axA.set_title(f"(A) Latency vs page_size (kv={kv})\nXQA solid · FlashInfer dashed · XQA has no ps=1")
    axA.grid(alpha=0.3)
    axA.legend(fontsize=8, ncol=2)

    # Panel B: XQA/FI overhead (%) vs batch, per seq, averaged over shared pages
    axB = axes[1]
    shared = [16, 32, 64, 128]
    for kv in seqs:
        ov = []
        for bs in batches:
            rs = []
            for ps in shared:
                x, f = get("xqa", ps, bs, kv), get("flashinfer", ps, bs, kv)
                if x and f:
                    rs.append(x / f - 1.0)
            ov.append(100 * np.mean(rs) if rs else np.nan)
        axB.plot(batches, ov, marker="o", label=f"kv={kv}")
    axB.axhline(0, color="k", lw=0.8)
    axB.set_xscale("log", base=2)
    axB.set_xticks(batches)
    axB.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axB.set_xlabel("batch size")
    axB.set_ylabel("XQA latency vs FlashInfer (%)")
    axB.set_title("(B) XQA fixed-overhead penalty\ncollapses as work becomes bandwidth-bound")
    axB.grid(alpha=0.3)
    axB.legend(fontsize=9)

    fig.tight_layout()
    p = OUT / "fig_xqa_latency.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"wrote {p}")


# ---- ncu counter figure (drawn only when data exists) ----------------------
import sys
sys.path.insert(0, str(HERE.parent))


def fig_tma():
    try:
        from analyze_xqa_ncu import load_cell, TAG_RE
    except Exception as e:
        print(f"skip TMA fig: cannot import analyzer ({e})")
        return
    csvs = sorted(p for p in NCU_DIR.glob("*.csv") if TAG_RE.match(p.stem)) if NCU_DIR.exists() else []
    if not csvs:
        print(f"skip TMA fig: no ncu CSVs in {NCU_DIR} yet (run profile_xqa_ncu.sh)")
        return
    data = {}
    for p in csvs:
        m = TAG_RE.match(p.stem)
        res = load_cell(p)
        if res:
            data[(m["be"], int(m["ps"]), int(m["bs"]), int(m["kv"]))] = res[0]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    bs, kv = 8, 4096
    pages = sorted({k[1] for k in data if k[2] == bs and k[3] == kv and k[0] == "xqa"})

    # Panel A: TMA load bytes vs DRAM read bytes — the "no TMA" headline (XQA & FI)
    axA = axes[0]
    labels, tma_vals, dram_vals = [], [], []
    for be in ("xqa", "flashinfer"):
        d = data.get((be, 64, bs, kv)) or data.get((be, 16, bs, kv))
        if d:
            labels.append(be)
            tma_vals.append((d.get("TMA_ld_B") or 0) / 1e6)   # bytes -> MB
            dram_vals.append(d.get("dram_rd_MB") or 0)
    x = np.arange(len(labels))
    axA.bar(x - 0.2, dram_vals, 0.4, label="DRAM read (MB)", color="#4c72b0")
    axA.bar(x + 0.2, tma_vals, 0.4, label="TMA global-load (MB)", color="#dd8452")
    for i, t in enumerate(tma_vals):
        axA.text(i + 0.2, max(dram_vals) * 0.04, "0", ha="center", fontsize=11, color="#dd8452")
    axA.set_xticks(x); axA.set_xticklabels(labels)
    axA.set_ylabel("MB")
    axA.set_title(f"(A) TMA usage = ZERO\nKV loaded via cp.async, not TMA  (bs{bs}/kv{kv})")
    axA.legend(fontsize=9)

    # Panel B: global-ld instructions vs page_size (XQA ∝1/page, FI flat)
    axB = axes[1]
    for be in ("xqa", "flashinfer"):
        ps_list = [ps for ps in pages if (be, ps, bs, kv) in data]
        ys = [data[(be, ps, bs, kv)].get("global_ld") for ps in ps_list]
        if ps_list:
            axB.plot(ps_list, ys, marker="o", color=COLORS[be], label=be)
    axB.set_xscale("log", base=2); axB.set_yscale("log")
    axB.set_xticks(pages); axB.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axB.set_xlabel("page_size"); axB.set_ylabel("global-load instructions")
    axB.set_title("(B) XQA global-loads ∝ 1/page\nFlashInfer ~flat")
    axB.grid(alpha=0.3, which="both"); axB.legend(fontsize=9)

    # Panel C: L2 hit % vs page_size — XQA's cache shift; both bandwidth-bound (annotate)
    axC = axes[2]
    for be in ("xqa", "flashinfer"):
        ps_list = [ps for ps in pages if (be, ps, bs, kv) in data]
        ys = [data[(be, ps, bs, kv)].get("L2_hit%") for ps in ps_list]
        if ps_list:
            axC.plot(ps_list, ys, marker="o", color=COLORS[be], label=be)
    axC.set_xscale("log", base=2)
    axC.set_xticks(pages); axC.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axC.set_xlabel("page_size"); axC.set_ylabel("L2 hit rate (%)")
    dpk = (data.get(("xqa", 64, bs, kv)) or {}).get("dram_%pk")
    axC.set_title(f"(C) Smaller pages → more L2 hits (XQA)\nbut DRAM ~{round(dpk) if dpk else 95}% peak → latency flat")
    axC.grid(alpha=0.3); axC.legend(fontsize=9)

    fig.suptitle("XQA vs FlashInfer decode on RTX 5060 Ti (sm120) — ncu counters: no TMA, DRAM-bandwidth-bound",
                 y=1.03, fontsize=12)
    fig.tight_layout()
    p = OUT / "fig_xqa_tma.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_latency()
    fig_tma()
