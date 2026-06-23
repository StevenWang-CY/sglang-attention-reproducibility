#!/usr/bin/env python3
"""The graph-ON FlashInfer long-context finding (Qwen3-VL-2B, bs8):
with CUDA graph ON, FlashInfer, and NO OOM, page_size=1 is ~+5.5-8.8% slower than page_size=128
at ~39k context (reproduced 2x). Negligible at <=21k. A REAL but NARROW case (bs8 specifically).
Left: the confirmed scenario (ps1 vs ps128 absolute, both passes). Right: it is context-gated."""
import json, os
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]  # repo root
GEN = ROOT / "offline_batch_results/gen_models_5060ti"
DENSE = ROOT / "offline_batch_results/dense_5060ti"
OUT = Path(__file__).resolve().parents[1] / "report_4_flashinfer_graph_case"  # ../report_4_flashinfer_graph_case


def med(d, stem, bs, f="tpot_median_ms"):
    p = d / f"{stem}.json"
    if not p.exists():
        return None
    return json.load(open(p)).get(str(bs), {}).get(f)


def fig():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.5))

    # ---- LEFT: the confirmed scenario, 39k bs8, ps1 vs ps128, 2 passes (absolute TPOT) ----
    passes = [("pass 1", "fg_2b_39k"), ("pass 2", "fg2_2b_39k_p2")]
    labels, ps1v, ps1s, ps128v, ps128s, pens = [], [], [], [], [], []
    for name, tag in passes:
        a, sa = med(GEN, f"{tag}_ps1", 8), med(GEN, f"{tag}_ps1", 8, "tpot_std_ms")
        b, sb = med(GEN, f"{tag}_ps128", 8), med(GEN, f"{tag}_ps128", 8, "tpot_std_ms")
        if a and b:
            labels.append(name); ps1v.append(a); ps1s.append(sa); ps128v.append(b); ps128s.append(sb)
            pens.append((a / b - 1) * 100)
    x = np.arange(len(labels)); w = 0.36
    axL.bar(x - w/2, ps1v, w, yerr=ps1s, capsize=4, color="tab:red", label="page_size=1")
    axL.bar(x + w/2, ps128v, w, yerr=ps128s, capsize=4, color="tab:green", label="page_size=128")
    for i, p in enumerate(pens):
        axL.annotate(f"ps1 +{p:.1f}%", (i, max(ps1v[i], ps128v[i]) + 1), ha="center", fontsize=10, fontweight="bold")
    axL.set_xticks(x); axL.set_xticklabels(labels)
    axL.set_ylabel("decode TPOT (ms)")
    axL.set_title("Qwen3-VL-2B · 39k ctx · bs8 · FlashInfer · CUDA graph ON · no OOM\n"
                  "page_size=1 is ≥5% slower than page_size=128 (reproduced 2×)", fontsize=10, fontweight="bold")
    axL.legend(); axL.grid(axis="y", alpha=0.3); axL.set_ylim(0, max(ps1v + ps128v) * 1.18)

    # ---- RIGHT: context-gated — penalty at bs8 across natural long prompts (10k/21k/39k) ----
    ctx_pts = [(9.7, [("hb_fi_g", DENSE)]), (21.6, [("fg_2b_21k", GEN)]),
               (39.6, [("fg_2b_39k", GEN), ("fg2_2b_39k_p2", GEN)])]
    xs, ys, errs = [], [], []
    for ctx, srcs in ctx_pts:
        vals = []
        for tag, d in srcs:
            a, b = med(d, f"{tag}_ps1", 8), med(d, f"{tag}_ps128", 8)
            if a and b:
                vals.append((a / b - 1) * 100)
        if vals:
            xs.append(ctx); ys.append(float(np.mean(vals))); errs.append((max(vals)-min(vals))/2 if len(vals) > 1 else 0)
    axR.errorbar(xs, ys, yerr=errs, marker="o", ms=9, lw=2.5, color="tab:purple", capsize=4)
    axR.axhline(5, color="green", ls="--", lw=1.3, label="+5% threshold")
    axR.axhline(0, color="k", lw=0.8, alpha=0.5)
    for xx, yy in zip(xs, ys):
        axR.annotate(f"{yy:+.1f}%", (xx, yy), textcoords="offset points", xytext=(6, 6), fontsize=9, fontweight="bold")
    axR.set_xlabel("context length (k tokens)"); axR.set_ylabel("page_size=1 penalty vs page_size=128 (%)")
    axR.set_title("Context-gated: negligible ≤21k, crosses +5% by ~39k\n(FlashInfer, CUDA graph ON, bs8)",
                  fontsize=10, fontweight="bold")
    axR.legend(fontsize=9); axR.grid(alpha=0.3)

    fig.suptitle("FlashInfer is not immune: with CUDA graph ON + no OOM, page_size=1 loses at long context",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "fig_flashinfer_graph_longctx.png", dpi=150); plt.close(fig)
    print("wrote fig_flashinfer_graph_longctx.png; context points:", list(zip(xs, [round(y, 1) for y in ys])))


if __name__ == "__main__":
    fig()
