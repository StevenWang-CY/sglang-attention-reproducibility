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
TAG_RE = re.compile(r"(?P<be>xqa|flashinfer|triton)_ps(?P<ps>\d+)_bs(?P<bs>\d+)_kv(?P<kv>\d+)")


def fig_tma():
    import csv
    csvs = sorted(p for p in NCU_DIR.glob("*.csv") if TAG_RE.search(p.stem)) if NCU_DIR.exists() else []
    if not csvs:
        print(f"skip TMA fig: no ncu CSVs in {NCU_DIR} yet (run profile_xqa_ncu.sh)")
        return
    # minimal pivot reused from analyze_xqa_ncu
    def parse(path):
        rows = list(csv.DictReader(open(path, newline="")))
        from collections import defaultdict
        out = defaultdict(dict)
        for r in rows:
            kn = next((r[k] for k in r if k.strip().lower() == "kernel name"), "?")
            mn = next((r[k] for k in r if k.strip().lower() == "metric name"), None)
            mv = next((r[k] for k in r if k.strip().lower() == "metric value"), None)
            if mn is None:
                continue
            v = (mv or "").strip().strip('"').replace(",", "")
            try:
                out[kn][mn.strip()] = float(v)
            except ValueError:
                pass
        if not out:
            return {}
        return max(out.values(), key=lambda m: m.get("gpu__time_duration.sum", 0))
    data = {}
    for p in csvs:
        m = TAG_RE.search(p.stem)
        met = parse(p)
        if met:
            data[(m["be"], int(m["ps"]), int(m["bs"]), int(m["kv"]))] = met

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    bs, kv = 8, 4096
    pages = sorted({k[1] for k in data if k[2] == bs and k[3] == kv})
    panels = [
        ("sm__sass_inst_executed_op_global_ld.sum", "(A) scalar global-ld instructions"),
        ("dram__bytes_read.sum", "(B) DRAM bytes read"),
        ("lts__t_sector_hit_rate.pct", "(C) L2 hit rate (%)"),
    ]
    for ax, (metric, title) in zip(axes, panels):
        for be in ("xqa", "flashinfer"):
            xs = [ps for ps in pages if (be, ps, bs, kv) in data and data[(be, ps, bs, kv)].get(metric) is not None]
            ys = [data[(be, ps, bs, kv)][metric] for ps in xs]
            if xs:
                ax.plot(xs, ys, marker="o", color=COLORS[be], label=be)
        ax.set_xscale("log", base=2)
        ax.set_xticks(pages)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("page_size")
        ax.set_title(f"{title}\n(bs={bs}, kv={kv})")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("XQA (TMA bulk loads) vs FlashInfer (fused) — hardware counters", y=1.02)
    fig.tight_layout()
    p = OUT / "fig_xqa_tma.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_latency()
    fig_tma()
