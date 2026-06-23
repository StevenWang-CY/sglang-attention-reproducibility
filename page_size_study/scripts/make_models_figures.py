#!/usr/bin/env python3
"""Cross-model generalization figures (CUDA graph ON, ~10k ctx, shared prefix).
Triton page_size=1 penalty vs ps128 across models + the layer-count driver. Plus vLLM block_size.
Data: offline_batch_results/gen_models_5060ti/B_* and dense_5060ti/hb_* (Qwen3-VL-2B reference)."""
import json, os, glob
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]  # repo root
GEN = ROOT / "offline_batch_results/gen_models_5060ti"
DENSE = ROOT / "offline_batch_results/dense_5060ti"
VLLM = ROOT / "offline_batch_results/vllm_blocksize_5060ti"
OUT = Path(__file__).resolve().parents[1] / "report_3_dense_sweep_MAIN"  # ../report_3_dense_sweep_MAIN

# model -> (dir, tri_ps1_stem, tri_ps128_stem, fi_ps1_stem, fi_ps128_stem, KV/token KB, layers, color)
MODELS = [
    ("Qwen2.5-3B (text)", GEN, "B_qwen25_3b_tri_ps1", "B_qwen25_3b_tri_ps128", "B_qwen25_3b_fi_ps1", "B_qwen25_3b_fi_ps128", 37, 36, "tab:green"),
    ("Qwen3-VL-2B", DENSE, "hb_tri_g_ps1", "hb_tri_g_ps128", "hb_fi_g_ps1", "hb_fi_g_ps128", 112, 28, "tab:blue"),
    ("Qwen3-VL-4B", GEN, "B_qwen4b_tri_ps1", "B_qwen4b_tri_ps128", "B_qwen4b_fi_ps1", "B_qwen4b_fi_ps128", 144, 36, "tab:red"),
    ("Qwen3-VL-8B-FP8", GEN, "B_qwen8bfp8_tri_ps1", "B_qwen8bfp8_tri_ps128", "B_qwen8bfp8_fi_ps1", "B_qwen8bfp8_fi_ps128", 144, 36, "tab:purple"),
]
BATCHES = [8, 16, 32]


def med(d, stem, bs):
    p = d / f"{stem}.json"
    if not p.exists():
        return None
    return json.load(open(p)).get(str(bs), {}).get("tpot_median_ms")


def pen(d, s1, s128, bs):
    a, b = med(d, s1, bs), med(d, s128, bs)
    return (a / b - 1) * 100 if (a and b) else None


def fig_generalization():
    fig, ax = plt.subplots(figsize=(9.5, 6))
    for name, d, t1, t128, f1, f128, kv, L, col in MODELS:
        tri = [pen(d, t1, t128, bs) for bs in BATCHES]
        fi = [pen(d, f1, f128, bs) for bs in BATCHES]
        if all(v is not None for v in tri):
            ax.plot(BATCHES, tri, "-o", color=col, lw=2, label=f"{name} (Triton, {L}L)")
        if all(v is not None for v in fi):
            ax.plot(BATCHES, fi, "--", color=col, lw=1, alpha=0.6)
    ax.axhline(5, color="green", ls=":", lw=1.2, label="+5% threshold")
    ax.axhline(0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel("batch size (shared prefix, ~10k ctx, CUDA graph ON)")
    ax.set_ylabel("page_size=1 penalty vs page_size=128 (%)")
    ax.set_title("The Triton page_size=1 penalty is UNIVERSAL across models (+15–40% at bs32)\n"
                 "solid = Triton (all climb); dashed = FlashInfer (all flat). Text & VL, GQA-2 & GQA-4, FP8.",
                 fontweight="bold")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(OUT / "fig_models_generalization.png", dpi=150); plt.close(fig)
    print("wrote fig_models_generalization.png")


def fig_driver():
    # bs32 penalty vs layers (the driver, r=0.97) and vs KV/token (no relationship)
    pts = []
    for name, d, t1, t128, f1, f128, kv, L, col in MODELS:
        p = pen(d, t1, t128, 32)
        if p is not None:
            pts.append((name, kv, L, p, col))
    if not pts:
        print("skip fig_driver"); return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    # left: vs layers
    ax = axes[0]
    for name, kv, L, p, col in pts:
        ax.scatter(L, p, s=120, color=col, zorder=3)
        ax.annotate(name, (L, p), textcoords="offset points", xytext=(6, 4), fontsize=8)
    xs = [L for _, _, L, _, _ in pts]; ys = [p for _, _, _, p, _ in pts]
    r = np.corrcoef(xs, ys)[0, 1]
    ax.set_xlabel("number of layers"); ax.set_ylabel("Triton ps1 penalty @ bs32 (%)")
    ax.set_title(f"Tracks LAYER COUNT  (Pearson r = {r:+.2f})", fontweight="bold")
    ax.grid(alpha=0.3)
    # right: vs KV/token
    ax = axes[1]
    for name, kv, L, p, col in pts:
        ax.scatter(kv, p, s=120, color=col, zorder=3)
        ax.annotate(name, (kv, p), textcoords="offset points", xytext=(6, 4), fontsize=8)
    xs = [kv for _, kv, _, _, _ in pts]
    r2 = np.corrcoef(xs, ys)[0, 1]
    ax.set_xlabel("KV-cache bytes per token (KB)"); ax.set_ylabel("Triton ps1 penalty @ bs32 (%)")
    ax.set_title(f"Does NOT track KV/token  (Pearson r = {r2:+.2f})", fontweight="bold")
    ax.grid(alpha=0.3)
    fig.suptitle("What drives the cross-model magnitude? — layers, not KV bytes "
                 "(smallest-KV model has the largest penalty)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "fig_models_driver.png", dpi=150); plt.close(fig)
    print(f"wrote fig_models_driver.png (r_layers={r:+.2f}, r_kv={r2:+.2f})")


def fig_vllm():
    files = glob.glob(str(VLLM / "*.json"))
    if not files:
        print("skip fig_vllm (no data yet)"); return
    # {bs: {blk: tpot}}
    data = {}
    for f in files:
        d = json.load(open(f))
        data.setdefault(d["batch_size"], {})[d["block_size"]] = d["tpot_median_ms"]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for bs in sorted(data):
        blks = sorted(data[bs]); ys = [data[bs][b] for b in blks]
        ax.plot(blks, ys, "-o", lw=2, label=f"bs={bs}")
    ax.set_xlabel("vLLM block_size"); ax.set_ylabel("decode TPOT (ms)")
    ax.set_title("vLLM block_size vs decode latency (sm120, FLASH_ATTN backend, eager)\n"
                 "block_size=1 is unsupported by vLLM (init fails) — it avoids the single-token-page pathology",
                 fontweight="bold")
    ax.set_xscale("log", base=2); ax.grid(alpha=0.3, which="both"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "fig_vllm_blocksize.png", dpi=150); plt.close(fig)
    print("wrote fig_vllm_blocksize.png")


if __name__ == "__main__":
    fig_generalization()
    fig_driver()
    fig_vllm()
