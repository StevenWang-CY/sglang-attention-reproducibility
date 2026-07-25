#!/usr/bin/env python3
"""V2 report figures from the CLEAN datasets (GPU confirmed idle before+after each cell).
Each figure is guarded on data availability.

Data sources (all RTX 5060 Ti, Qwen3-VL-2B, ~10k ctx, 5060 Ti):
  FlashInfer graph-ON : studies/kv-cache-page-size/data/raw/ps_e2e_5060ti/results_ps*.json   (baseline; bs1/8/16, ps1/8/32/128)
  FlashInfer clean    : studies/kv-cache-page-size/data/raw/fi_clean_5060ti/fi_clean_{g,ng}_bs{1,8}_ps{1,128}.json
  Triton graph-ON     : studies/kv-cache-page-size/data/raw/triton_sweep_5060ti/tri_10k_g_bs{1,8,16}_ps*.json
  Triton no-graph clean: studies/kv-cache-page-size/data/raw/triton_clean_5060ti/tri_clean_ng_bs{8,16}_ps*.json
  FlashInfer no-graph CONTENDED (artifact): studies/kv-cache-page-size/data/raw/ps_nocudagraph_5060ti/ctx10k_ps*_nograph.json
"""
import json, glob, re
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]  # repo root
OUT = Path(__file__).resolve().parents[1] / "reports" / "02-low-batch-baseline"  # ../02-low-batch-baseline


def _med(path):
    j = json.load(open(path)); b = next(iter(j)); return j[b]["tpot_median_ms"]


def load_baseline():  # FI graph-ON: {bs: {ps: tpot}}
    out = {}
    for f in glob.glob(str(ROOT / "studies/kv-cache-page-size/data/raw/ps_e2e_5060ti/results_ps*.json")):
        ps = int(re.search(r"ps(\d+)", f).group(1)); j = json.load(open(f))
        for b in j:
            out.setdefault(int(b), {})[ps] = j[b]["tpot_median_ms"]
    return out


def load_named(dirname, pat):  # -> {(g/ng,bs,ps): tpot}
    out = {}
    for f in glob.glob(str(ROOT / dirname / "*.json")):
        m = re.search(pat, f.split("/")[-1])
        if m:
            out[(m.group(1), int(m.group(2)), int(m.group(3)))] = _med(f)
    return out


baseline = load_baseline()
fi = load_named("studies/kv-cache-page-size/data/raw/fi_clean_5060ti", r"fi_clean_(g|ng)_bs(\d+)_ps(\d+)")
tri_g = {}  # Triton graph-ON from triton_sweep
for f in glob.glob(str(ROOT / "studies/kv-cache-page-size/data/raw/triton_sweep_5060ti/tri_10k_g_*.json")):
    m = re.search(r"tri_10k_g_bs(\d+)_ps(\d+)", f.split("/")[-1])
    if m:
        tri_g[(int(m.group(1)), int(m.group(2)))] = _med(f)
tri_ng = load_named("studies/kv-cache-page-size/data/raw/triton_clean_5060ti", r"tri_clean_(ng)_bs(\d+)_ps(\d+)")


# ── FIG A: clean summary — normalized TPOT vs page_size, 4 configs ────────────
def fig_summary():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharey=True)
    PAGES = [1, 8, 32, 128]
    # (ax, title, getter(bs,ps)->tpot or None, batches)
    panels = [
        (axes[0, 0], "FlashInfer, CUDA graph ON", lambda bs, ps: baseline.get(bs, {}).get(ps), [1, 8, 16]),
        (axes[0, 1], "FlashInfer, no CUDA graph (clean)", lambda bs, ps: fi.get(("ng", bs, ps)), [1, 8]),
        (axes[1, 0], "Triton, CUDA graph ON", lambda bs, ps: tri_g.get((bs, ps)), [1, 8, 16]),
        (axes[1, 1], "Triton, no CUDA graph (clean)", lambda bs, ps: tri_ng.get(("ng", bs, ps)), [8, 16]),
    ]
    for ax, title, get, batches in panels:
        for bs in batches:
            xs, ys = [], []
            base = get(bs, 1)
            if base is None:
                continue
            for i, ps in enumerate(PAGES):
                v = get(bs, ps)
                if v is not None:
                    xs.append(i); ys.append(v / base * 100)
            if xs:
                ax.plot(xs, ys, marker="o", label=f"bs={bs}")
        ax.axhline(100, color="k", ls="--", lw=1, alpha=0.6)
        ax.set_xticks(range(len(PAGES))); ax.set_xticklabels(PAGES)
        ax.set_title(title, fontsize=11); ax.grid(alpha=0.25); ax.legend(fontsize=8)
        ax.set_xlabel("page_size")
    axes[0, 0].set_ylabel("TPOT rel. to page_size=1 (%)"); axes[1, 0].set_ylabel("TPOT rel. to page_size=1 (%)")
    fig.suptitle("Page size vs decode TPOT — clean (uncontended) measurement, all 4 SGLang configs\n"
                 "Every curve within ~3% of page_size=1: page size is near-neutral; no ≥5% scenario",
                 fontweight="bold")
    fig.text(0.5, 0.005, "FlashInfer favors page_size=1 (curves ≥100%); Triton favors larger pages (curves ≤100%) — "
             "opposite tiny signs, both <3%.", ha="center", fontsize=9, style="italic")
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    fig.savefig(OUT / "fig_v2_summary.png", dpi=150); plt.close(fig)
    print("wrote fig_v2_summary.png")


# ── FIG B: the contention-artifact lesson (FI no-graph bs1) ───────────────────
def fig_contention():
    # contended (Exp2) vs clean (fi_clean), FlashInfer no-graph bs1
    cont = {}
    for ps in (1, 128):
        p = ROOT / f"studies/kv-cache-page-size/data/raw/ps_nocudagraph_5060ti/ctx10k_ps{ps}_nograph.json"
        if p.exists():
            cont[ps] = _med(p)
    clean = {ps: fi.get(("ng", 1, ps)) for ps in (1, 128)}
    if not (cont.get(1) and clean.get(1)):
        print("skip fig_contention (missing data)"); return
    fig, ax = plt.subplots(figsize=(7.5, 5))
    x = np.arange(2); w = 0.35
    cbars = [cont[1], cont[128]]; kbars = [clean[1], clean[128]]
    ax.bar(x - w/2, cbars, w, color="#c44e52", label="measured during GPU contention")
    ax.bar(x + w/2, kbars, w, color="#55a868", label="clean (GPU idle confirmed)")
    for xi, v in zip(x - w/2, cbars):
        ax.text(xi, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    for xi, v in zip(x + w/2, kbars):
        ax.text(xi, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(["page_size=1", "page_size=128"])
    ax.set_ylabel("median TPOT (ms)")
    gap_c = (cont[128]/cont[1]-1)*100; gap_k = (clean[128]/clean[1]-1)*100
    ax.set_title(f"The 'big' page-size effects were GPU-contention artifacts\n"
                 f"FlashInfer no-graph bs1: contended gap +{gap_c:.0f}%  →  clean gap +{gap_k:.1f}%", fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "fig_v2_contention.png", dpi=150); plt.close(fig)
    print("wrote fig_v2_contention.png")


for fn in (fig_summary, fig_contention):
    try:
        fn()
    except Exception as e:
        print(f"{fn.__name__} error: {e}")
print("done")
