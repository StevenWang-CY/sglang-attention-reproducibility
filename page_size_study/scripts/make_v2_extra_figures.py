#!/usr/bin/env python3
"""
V2 figures for the 4-experiment page-size report. Each figure is GUARDED on data
availability (skips cleanly if its inputs aren't collected yet), so this can be
run repeatedly as Triton / vLLM data trickles in from the shared GPU.

Inputs:
  offline_batch_results/ps_e2e_5060ti/results_ps*.json        - baseline (FlashInfer, CUDA graph ON)
  offline_batch_results/ps_nocudagraph_5060ti/ctx*_ps*_*.json - Exp2 (FlashInfer, graph ON/OFF, long ctx, bs1)
  offline_batch_results/triton_sweep_5060ti/tri_*.json        - Exp3 (Triton)
  offline_batch_results/vllm_blocksize_5060ti/*.json          - Exp4 (vLLM)
Outputs: report_2_lowbatch_clean/fig6_from_zero.png, fig7_triton_vs_fi.png, fig8_graph_ctx.png,
         fig9_vllm.png, fig4b_synthesis_v2.png
"""
import json, glob, re
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]  # repo root
OUT = Path(__file__).resolve().parents[1] / "report_2_lowbatch_clean"  # ../report_2_lowbatch_clean
PAGES = [1, 2, 4, 8, 16, 32, 64, 128]
CTX_ORDER = {"ctx10k": 10, "ctx29k": 29, "ctx58k": 58}


def load_e2e(dirname):
    """baseline: results_ps{P}.json keyed by batch -> {ps: {bs: (tpot, std)}}"""
    out = {}
    for f in glob.glob(str(ROOT / dirname / "results_ps*.json")):
        ps = int(re.search(r"ps(\d+)", f.split("/")[-1]).group(1))
        j = json.load(open(f))
        out[ps] = {int(b): (j[b]["tpot_median_ms"], j[b]["tpot_std_ms"]) for b in j}
    return out


def load_nocg():
    """Exp2: ctx{N}_ps{P}_{mode}.json (bs1) -> {(ctxname, ps, mode): (tpot, std)}"""
    out = {}
    for f in glob.glob(str(ROOT / "offline_batch_results/ps_nocudagraph_5060ti/ctx*_ps*_*.json")):
        m = re.match(r"(ctx\w+?)_ps(\d+)_(\w+)\.json", f.split("/")[-1])
        if not m:
            continue
        j = json.load(open(f)); b = next(iter(j))
        out[(m.group(1), int(m.group(2)), m.group(3))] = (j[b]["tpot_median_ms"], j[b]["tpot_std_ms"])
    return out


def load_triton():
    """Exp3: tri_{ctx}_{g|ng}_bs{N}_ps{P}.json -> {(ctx, graph, bs, ps): (tpot, std)}"""
    out = {}
    for f in glob.glob(str(ROOT / "offline_batch_results/triton_sweep_5060ti/tri_*.json")):
        m = re.match(r"tri_(\w+?)_(g|ng)_bs(\d+)_ps(\d+)\.json", f.split("/")[-1])
        if not m:
            continue
        j = json.load(open(f)); b = next(iter(j))
        ok = j[b].get("concurrency_ok", True)
        ctx = "ctx" + m.group(1)  # file tag is "10k" -> normalize to "ctx10k"
        out[(ctx, m.group(2), int(m.group(3)), int(m.group(4)))] = (j[b]["tpot_median_ms"], j[b]["tpot_std_ms"], ok)
    return out


def load_vllm():
    """Exp4: bs{N}_bsz{B}.json (block_size B) -> {(bs, block): (tpot, std)} (schema flexible)"""
    out = {}
    for f in glob.glob(str(ROOT / "offline_batch_results/vllm_blocksize_5060ti/*.json")):
        try:
            j = json.load(open(f))
        except Exception:
            continue
        m = re.search(r"bs(\d+)_blk(\d+)", f.split("/")[-1])
        if m:
            bs, blk = int(m.group(1)), int(m.group(2))
            tpot = j.get("tpot_median_ms") or j.get(next(iter(j)), {}).get("tpot_median_ms")
            if tpot:
                out[(bs, blk)] = (tpot, j.get("tpot_std_ms", 0))
    return out


# ── FIG 6: absolute (from-0) axis — the PI's plotting point ───────────────────
def fig6():
    base = load_e2e("offline_batch_results/ps_e2e_5060ti")
    if not base:
        print("skip fig6 (no baseline)"); return
    ps_list = sorted(base); batches = sorted({b for ps in base for b in base[ps]})
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(ps_list)); w = 0.8 / max(1, len(batches))
    for i, bs in enumerate(batches):
        vals = [base[ps][bs][0] for ps in ps_list]
        ax.bar(x + (i - (len(batches)-1)/2) * w, vals, w, label=f"bs={bs}")
    ax.set_xticks(x); ax.set_xticklabels(ps_list); ax.set_ylim(0, None)
    ax.set_xlabel("page_size"); ax.set_ylabel("median TPOT (ms)")
    ax.set_title("Baseline engine TPOT on an absolute (from-0) axis\nmax spread +1.18% — visually flat (the PI's point)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "fig6_from_zero.png", dpi=150); plt.close(fig)
    print("wrote fig6_from_zero.png")


# ── FIG 8: page-size gap (ps128 vs ps1) vs context, CUDA graph ON vs OFF ───────
def fig8():
    nocg = load_nocg(); base = load_e2e("offline_batch_results/ps_e2e_5060ti")
    tri = load_triton()
    if not nocg:
        print("skip fig8 (no Exp2 data)"); return
    fig, ax = plt.subplots(figsize=(9, 5.5))
    # FlashInfer, graph OFF
    xs, ys = [], []
    for c in ["ctx10k", "ctx29k", "ctx58k"]:
        a = nocg.get((c, 1, "nograph")); b = nocg.get((c, 128, "nograph"))
        if a and b:
            xs.append(CTX_ORDER[c]); ys.append((b[0] / a[0] - 1) * 100)
    if xs:
        ax.plot(xs, ys, "o-", color="#d62728", lw=2.5, ms=7, label="FlashInfer, CUDA graph OFF")
    # FlashInfer, graph ON (from Exp2 controls if present, else baseline bs1 @10k)
    xs2, ys2 = [], []
    for c in ["ctx10k", "ctx29k", "ctx58k"]:
        a = nocg.get((c, 1, "graph")); b = nocg.get((c, 128, "graph"))
        if a and b:
            xs2.append(CTX_ORDER[c]); ys2.append((b[0] / a[0] - 1) * 100)
    if 1 in base and 128 in base and 1 in base[1]:  # baseline bs1 @ ~10k
        xs2 = [10] + xs2; ys2 = [(base[128][1][0] / base[1][1][0] - 1) * 100] + ys2
    if xs2:
        order = np.argsort(xs2)
        ax.plot(np.array(xs2)[order], np.array(ys2)[order], "s-", color="#2ca02c", lw=2.5, ms=7, label="FlashInfer, CUDA graph ON")
    # Triton if available
    for graph, style, col in [("ng", "^--", "#9467bd"), ("g", "v--", "#1f77b4")]:
        xs3, ys3 = [], []
        for c in ["ctx10k", "ctx29k", "ctx58k"]:
            a = tri.get((c, graph, 1, 1)); b = tri.get((c, graph, 1, 128))
            if a and b:
                xs3.append(CTX_ORDER[c]); ys3.append((b[0] / a[0] - 1) * 100)
        if xs3:
            lab = f"Triton, CUDA graph {'OFF' if graph=='ng' else 'ON'}"
            ax.plot(xs3, ys3, style, color=col, lw=2, ms=6, label=lab)
    ax.axhline(0, color="k", ls=":", lw=1)
    ax.axhline(5, color="grey", ls=":", lw=1); ax.text(58, 5.3, "+5% (PI threshold)", fontsize=8, color="grey", ha="right")
    ax.set_xlabel("context length (k tokens)"); ax.set_ylabel("ps128 TPOT vs ps1  (%)   [>0 = ps1 faster]")
    ax.set_title("Page-size gap vs context: CUDA graph collapses a large (~13%) large-page penalty\n(bs=1, Qwen3-VL-2B, RTX 5060 Ti)", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "fig8_graph_ctx.png", dpi=150); plt.close(fig)
    print("wrote fig8_graph_ctx.png")


# ── FIG 7: Triton vs FlashInfer engine TPOT (normalized to ps1), per batch ────
def fig7():
    tri = load_triton(); base = load_e2e("offline_batch_results/ps_e2e_5060ti")
    if not tri:
        print("skip fig7 (no Triton data yet)"); return
    batches = sorted({k[2] for k in tri if k[0] == "ctx10k" and k[1] == "g"})
    if not batches:
        print("skip fig7 (no Triton 10k graph-ON cells yet)"); return
    fig, axes = plt.subplots(1, len(batches), figsize=(5 * len(batches), 4.5), squeeze=False)
    for ax, bs in zip(axes[0], batches):
        # Triton
        ps_t = sorted({k[3] for k in tri if k[:3] == ("ctx10k", "g", bs)})
        if ps_t and (("ctx10k", "g", bs, 1) in tri):
            b1 = tri[("ctx10k", "g", bs, 1)][0]
            ax.plot(range(len(ps_t)), [tri[("ctx10k", "g", bs, p)][0]/b1*100 for p in ps_t], "o-", color="#1f77b4", label="Triton")
            ax.set_xticks(range(len(ps_t))); ax.set_xticklabels(ps_t)
        # FlashInfer baseline
        if base and 1 in base and bs in base.get(1, {}):
            ps_f = sorted(base); f1 = base[1][bs][0]
            ax.plot(range(len(ps_f)), [base[p][bs][0]/f1*100 for p in ps_f], "s--", color="#ff7f0e", label="FlashInfer")
        ax.axhline(100, color="k", ls=":", lw=1)
        ax.set_title(f"batch = {bs}"); ax.set_xlabel("page_size"); ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("TPOT rel. to page_size=1 (%)"); axes[0][0].legend()
    fig.suptitle("Triton vs FlashInfer engine TPOT (CUDA graph ON, ~10k ctx)", fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "fig7_triton_vs_fi.png", dpi=150); plt.close(fig)
    print("wrote fig7_triton_vs_fi.png")


# ── FIG 9: vLLM TPOT vs block_size ────────────────────────────────────────────
def fig9():
    v = load_vllm()
    if not v:
        print("skip fig9 (no vLLM data yet)"); return
    batches = sorted({k[0] for k in v}); blocks = sorted({k[1] for k in v})
    fig, ax = plt.subplots(figsize=(8, 5))
    for bs in batches:
        xs = [b for b in blocks if (bs, b) in v]
        ax.plot(range(len(xs)), [v[(bs, b)][0] for b in xs], "o-", label=f"bs={bs}")
        ax.set_xticks(range(len(xs))); ax.set_xticklabels(xs)
    ax.set_xlabel("block_size"); ax.set_ylabel("median TPOT (ms)")
    ax.set_title("vLLM: TPOT vs block_size (Qwen3-VL-2B, RTX 5060 Ti)", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "fig9_vllm.png", dpi=150); plt.close(fig)
    print("wrote fig9_vllm.png")


for fn in (fig6, fig8, fig7, fig9):
    try:
        fn()
    except Exception as e:
        print(f"{fn.__name__} error: {e}")
print("done")
