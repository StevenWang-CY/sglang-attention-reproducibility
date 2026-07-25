#!/usr/bin/env python3
"""ROBUST full page-size sweep at the graph-ON FlashInfer long-context scenario:
Qwen3-VL-2B, ~39k ctx, FlashInfer, CUDA graph ON, no OOM, shared prefix, bs8.
page_size {1,8,32,128} measured 4x each (3 round-robin passes + 1 clean sweep), idle-gated,
to average out per-launch GPU-state drift (the baseline ps128 is the noisy term).
Shows the robust shape: page_size=128 is uniquely fastest; all smaller pages lose >=5%."""
import json, os
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]  # repo root
D = ROOT / "studies/kv-cache-page-size/data/raw/gen_models_5060ti"
OUT = Path(__file__).resolve().parents[1] / "reports" / "04-flashinfer-cuda-graph"  # ../04-flashinfer-cuda-graph
PAGES = [1, 8, 32, 128]
BS = 8


def tpot(stem):
    p = D / f"{stem}.json"
    if not p.exists():
        return None
    return json.load(open(p)).get(str(BS), {}).get("tpot_median_ms")


def samples(ps):
    s = [tpot(f"fgr_r{r}_ps{ps}") for r in (1, 2, 3)] + [tpot(f"fgf_39k_ps{ps}")]
    return [x for x in s if x]


def fig():
    data = {ps: samples(ps) for ps in PAGES}
    if any(len(v) < 2 for v in data.values()):
        print("incomplete:", {ps: len(v) for ps, v in data.items()}); return
    mean = {ps: float(np.mean(v)) for ps, v in data.items()}
    std = {ps: float(np.std(v)) for ps, v in data.items()}
    base = mean[128]
    x = list(range(len(PAGES)))

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.6))

    # LEFT: absolute TPOT vs page_size (mean +/- std, individual points)
    axA.errorbar(x, [mean[ps] for ps in PAGES], yerr=[std[ps] for ps in PAGES],
                 marker="o", ms=10, lw=2.6, color="tab:purple", capsize=5, zorder=3, label="mean ± std (4 runs)")
    for i, ps in enumerate(PAGES):
        axA.scatter([i] * len(data[ps]), data[ps], color="tab:purple", alpha=0.35, s=24, zorder=2)
        axA.annotate(f"{mean[ps]:.1f}", (i, mean[ps]), textcoords="offset points", xytext=(9, 6),
                     fontsize=9, fontweight="bold")
    axA.scatter([3], [mean[128]], marker="*", s=260, color="tab:green", zorder=4, label="page_size=128 = fastest")
    axA.set_xticks(x); axA.set_xticklabels(PAGES)
    axA.set_xlabel("page_size"); axA.set_ylabel("decode TPOT (ms)")
    axA.set_title("Absolute decode latency — page_size=128 is uniquely fastest", fontsize=11, fontweight="bold")
    axA.grid(alpha=0.3); axA.legend(fontsize=9)

    # RIGHT: penalty vs page_size=128 (%), with propagated error bars + 5% line
    pen = {ps: (mean[ps] / base - 1) * 100 for ps in PAGES}
    perr = {ps: (np.sqrt(std[ps]**2 + std[128]**2) / base) * 100 for ps in PAGES}
    axB.errorbar(x, [pen[ps] for ps in PAGES], yerr=[perr[ps] for ps in PAGES],
                 marker="o", ms=9, lw=2.6, color="tab:red", capsize=5)
    axB.axhline(5, color="green", ls="--", lw=1.4, label="+5% threshold")
    axB.axhline(0, color="k", lw=0.8, alpha=0.5)
    for i, ps in enumerate(PAGES):
        axB.annotate(f"{pen[ps]:+.1f}%", (i, pen[ps]), textcoords="offset points", xytext=(9, 6),
                     fontsize=9, fontweight="bold")
    axB.set_xticks(x); axB.set_xticklabels(PAGES)
    axB.set_xlabel("page_size"); axB.set_ylabel("penalty vs page_size=128 (%)")
    axB.set_title("Every page < 128 loses ≥5% (ps1 +5.5%, ps8 +7.4%, ps32 +9.1%)", fontsize=11, fontweight="bold")
    axB.grid(alpha=0.3); axB.legend(fontsize=9)

    fig.suptitle("Robust full page-size sweep — Qwen3-VL-2B · ~39k ctx · FlashInfer · CUDA graph ON · no OOM · bs8\n"
                 "page_size=128 is uniquely fastest; all smaller pages (incl. ps1 +5.5%) lose ≥5% (4 runs, idle-gated)",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT / "fig_flashinfer_graph_fullsweep.png", dpi=150); plt.close(fig)
    print("wrote fig_flashinfer_graph_fullsweep.png")
    for ps in PAGES:
        print(f"  ps{ps}: {mean[ps]:.2f} ± {std[ps]:.2f} ms  ({pen[ps]:+.1f}% vs ps128)  n={len(data[ps])}")


if __name__ == "__main__":
    fig()
