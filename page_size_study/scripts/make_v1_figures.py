#!/usr/bin/env python3
"""
Generate figures for the page-size latency report.

Inputs (relative to repo root):
  page_size_attn_bench.json          - isolated FlashInfer decode-attention kernel, datacenter GPU (repo)
  page_size_attn_bench_5060ti.json   - same microbenchmark re-run on RTX 5060 Ti (Blackwell sm_120)
  offline_batch_results/ps_e2e_5060ti/results_ps{1,8,32,128}.json - end-to-end SGLang engine TPOT, 5060 Ti

Outputs: report_1_kernel_microbench/fig*.png
"""
import json, glob, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]  # repo root
OUT = Path(__file__).resolve().parents[1] / "report_1_kernel_microbench"  # ../report_1_kernel_microbench
DATA = Path(__file__).resolve().parents[1] / "data"  # ../data (small local data)
OUT.mkdir(exist_ok=True)
PAGES = [1, 2, 4, 8, 16, 32, 64, 128]


def load_kernel(path):
    """Return {(bs,kv): {ps: latency_ms}}."""
    d = json.load(open(path))
    out = {}
    for v in d.values():
        if "latency_ms" not in v:
            continue
        out.setdefault((v["batch_size"], v["kv_len"]), {})[v["page_size"]] = v["latency_ms"]
    return out


def load_e2e():
    out = {}  # {ps: {bs: (tpot, std)}}
    for f in glob.glob(str(ROOT / "offline_batch_results/ps_e2e_5060ti/results_ps*.json")):
        ps = int(re.search(r"ps(\d+)", f).group(1))
        j = json.load(open(f))
        out[ps] = {int(bs): (j[bs]["tpot_median_ms"], j[bs]["tpot_std_ms"]) for bs in j}
    return out


dc = load_kernel(ROOT / "page_size_attn_bench.json")
ti = load_kernel(DATA / "page_size_attn_bench_5060ti.json")
e2e = load_e2e()

# The very first cell(s) of the full 5060 Ti microbench run (bs=1, kv=1024, page_size=1 & 2)
# were corrupted by a cold-start artifact (124 us / 70 us vs a true ~56 us). The regime was
# re-measured warm in a standalone confirmation run; substitute those verified clean values so
# no report figure is driven by a known-bad point. All other cells are warm/clean.
_clean = json.load(open(DATA / "confirm_A_5060ti.json"))
ti[(1, 1024)] = {v["page_size"]: v["latency_ms"]
                 for k, v in _clean.items() if v.get("batch_size") == 1 and v.get("kv_len") == 1024}

KV_COLORS = {1024: "#1f77b4", 4096: "#ff7f0e", 16384: "#2ca02c", 32768: "#d62728"}


# ─────────────────────────────────────────────────────────────────────────────
# FIG 1 & 2: kernel latency normalized to page_size=1, 2x2 grid by batch size
# ─────────────────────────────────────────────────────────────────────────────
def kernel_grid(data, title, fname, batches=(1, 8, 32, 64)):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for ax, bs in zip(axes.flat, batches):
        for kv in sorted({k[1] for k in data}):
            if (bs, kv) not in data:
                continue
            row = data[(bs, kv)]
            ps_have = [p for p in PAGES if p in row]
            base = row[1]
            y = [row[p] / base * 100 for p in ps_have]
            ax.plot(range(len(ps_have)), y, marker="o", ms=4,
                    color=KV_COLORS.get(kv), label=f"kv={kv}")
        ax.axhline(100, color="k", ls="--", lw=1, alpha=0.6)
        ax.set_title(f"batch size = {bs}", fontsize=11)
        ax.set_xticks(range(len(PAGES)))
        ax.set_xticklabels(PAGES)
        ax.grid(True, alpha=0.25)
        ax.set_ylabel("latency rel. to page_size=1  (%)")
    for ax in axes[1]:
        ax.set_xlabel("page_size")
    axes[0, 0].legend(title="KV length", fontsize=8)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.text(0.5, 0.005, "dashed line = page_size=1 baseline (100%).  Points BELOW the line "
             "= that page_size beats page_size=1.", ha="center", fontsize=9, style="italic")
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    fig.savefig(OUT / fname, dpi=150)
    plt.close(fig)
    print("wrote", fname)


kernel_grid(ti, "Isolated decode-attention kernel latency vs page_size — RTX 5060 Ti (Blackwell sm_120)",
            "fig1_kernel_5060ti.png")
kernel_grid(dc, "Isolated decode-attention kernel latency vs page_size — datacenter GPU (repo data)",
            "fig2_kernel_datacenter.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 3: end-to-end engine TPOT (absolute + normalized)
# ─────────────────────────────────────────────────────────────────────────────
def fig_e2e():
    batches = sorted({bs for ps in e2e for bs in e2e[ps]})
    ps_have = sorted(e2e)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))
    bs_colors = {1: "#1f77b4", 8: "#ff7f0e", 16: "#2ca02c"}
    # absolute
    for bs in batches:
        y = [e2e[ps][bs][0] for ps in ps_have]
        err = [e2e[ps][bs][1] for ps in ps_have]
        axA.errorbar(range(len(ps_have)), y, yerr=err, marker="o", capsize=3,
                     color=bs_colors.get(bs), label=f"batch={bs}")
    axA.set_yscale("log")
    axA.set_xticks(range(len(ps_have)))
    axA.set_xticklabels(ps_have)
    axA.set_xlabel("page_size")
    axA.set_ylabel("median TPOT (ms, log scale)")
    axA.set_title("End-to-end decode TPOT (absolute)")
    axA.grid(True, which="both", alpha=0.25)
    axA.legend()
    # normalized to ps1
    for bs in batches:
        base = e2e[1][bs][0]
        y = [e2e[ps][bs][0] / base * 100 for ps in ps_have]
        axB.plot(range(len(ps_have)), y, marker="o", color=bs_colors.get(bs), label=f"batch={bs}")
    axB.axhline(100, color="k", ls="--", lw=1, alpha=0.6)
    axB.set_xticks(range(len(ps_have)))
    axB.set_xticklabels(ps_have)
    axB.set_xlabel("page_size")
    axB.set_ylabel("TPOT rel. to page_size=1  (%)")
    axB.set_title("End-to-end decode TPOT (normalized to page_size=1)")
    axB.grid(True, alpha=0.25)
    axB.legend()
    fig.suptitle("SGLang engine end-to-end TPOT vs page_size — RTX 5060 Ti, Qwen3-VL-2B, ~9.7k-token prompt",
                 fontsize=12, fontweight="bold")
    fig.text(0.5, 0.005, "page_size=1 is lowest at every batch size; larger pages are progressively slower "
             "at batch>=8 (KV-page padding / tail over-read); batch 1 is flat (weight-bound).",
             ha="center", fontsize=9, style="italic")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(OUT / "fig3_e2e_5060ti.png", dpi=150)
    plt.close(fig)
    print("wrote fig3_e2e_5060ti.png")


fig_e2e()


# ─────────────────────────────────────────────────────────────────────────────
# FIG 4: synthesis — kernel vs engine, normalized to page_size=1 (bs=8)
# ─────────────────────────────────────────────────────────────────────────────
def fig_synthesis(bs=8, kv=4096):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    def norm_line(row, label, **kw):
        ps_have = [p for p in PAGES if p in row]
        base = row[1]
        ax.plot(range(len(ps_have)), [row[p] / base * 100 for p in ps_have],
                marker="o", label=label, **kw)

    if (bs, kv) in dc:
        norm_line(dc[(bs, kv)], f"datacenter kernel (bs={bs}, kv={kv})", color="#9467bd", lw=2)
    if (bs, kv) in ti:
        norm_line(ti[(bs, kv)], f"5060 Ti kernel (bs={bs}, kv={kv})", color="#1f77b4", lw=2)
    if all(bs in e2e[ps] for ps in e2e):
        ps_have = sorted(e2e)
        base = e2e[1][bs][0]
        ax.plot(range(len(ps_have)), [e2e[ps][bs][0] / base * 100 for ps in ps_have],
                marker="s", color="#d62728", lw=2.5, label=f"5060 Ti ENGINE end-to-end (bs={bs}, ctx~10k)")

    ax.axhline(100, color="k", ls="--", lw=1.2)
    ax.set_xticks(range(len(PAGES)))
    ax.set_xticklabels(PAGES)
    ax.set_xlabel("page_size")
    ax.set_ylabel("latency relative to page_size=1  (%)")
    ax.set_title(f"Synthesis: where page_size=1 wins vs loses (batch={bs})", fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    # annotate regions
    ymax = ax.get_ylim()[1]
    ax.text(0.15, 100 + (ymax - 100) * 0.55, "above 100% →\npage_size=1 is LOWEST (best)",
            fontsize=9, color="#d62728", va="center")
    ymin = ax.get_ylim()[0]
    ax.text(5.2, 100 - (100 - ymin) * 0.45, "below 100% →\npage_size=1 NOT lowest",
            fontsize=9, color="#9467bd", va="center")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_synthesis.png", dpi=150)
    plt.close(fig)
    print("wrote fig4_synthesis.png")


fig_synthesis()


# ─────────────────────────────────────────────────────────────────────────────
# FIG 5: robustness re-run — is the kernel "best page" real or noise?
# (needs ../data/noise_test/{fwd,rev}_*.json from the reversed-order control)
# ─────────────────────────────────────────────────────────────────────────────
def fig_robustness():
    import statistics
    from collections import Counter
    files = sorted(glob.glob(str(DATA / "noise_test" / "*.json")))
    if not files:
        print("skip fig5 (no noise_test/ data)")
        return
    runs = [load_kernel(f) for f in files]
    warm = [(8, 4096), (8, 16384), (32, 4096), (32, 16384)]  # exclude first-measured (cold-start) cells
    winners = Counter()
    norm = {p: [] for p in PAGES}   # latency as % of each cell's mean
    n = 0
    for r in runs:
        for c in warm:
            if c not in r:
                continue
            n += 1
            order = sorted((r[c][p], p) for p in PAGES if p in r[c])
            winners[order[0][1]] += 1
            cellmean = sum(r[c][p] for p in PAGES) / len(PAGES)
            for p in PAGES:
                norm[p].append(r[c][p] / cellmean * 100)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = ["#c44e52" if p <= 4 else "#4c72b0" for p in PAGES]
    ax1.bar([str(p) for p in PAGES], [winners.get(p, 0) for p in PAGES], color=colors)
    ax1.axhline(n / len(PAGES), color="k", ls="--", lw=1, label=f"uniform-noise expectation ({n/len(PAGES):.1f})")
    ax1.set_xlabel("page_size that was FASTEST")
    ax1.set_ylabel(f"# of (run, cell) it won  (of {n})")
    ax1.set_title("Small pages (<=4, red) win far less than chance\n=> a real, systematic penalty")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)
    means = [statistics.mean(norm[p]) for p in PAGES]
    errs = [statistics.pstdev(norm[p]) / len(norm[p]) ** 0.5 for p in PAGES]
    ax2.errorbar(range(len(PAGES)), means, yerr=errs, marker="o", capsize=3, color="#55a868")
    ax2.axhline(100, color="k", ls="--", lw=1)
    ax2.set_xticks(range(len(PAGES)))
    ax2.set_xticklabels(PAGES)
    ax2.set_xlabel("page_size")
    ax2.set_ylabel("mean latency (% of per-cell mean)")
    ax2.set_title("page_size <=4 sits ~0.15% above; >=8 flat\n(coalescing saturates at ~8)")
    ax2.grid(alpha=0.3)
    fig.suptitle("Robustness re-run: 5060 Ti kernel — real but tiny small-page penalty, noise among large pages",
                 fontweight="bold")
    fig.text(0.5, 0.005, "3 forward + 3 reversed-order runs, 4 warm cells. page_size<=4 is systematically ~0.15% slower "
             "(real); which of 8..128 wins is noise. Magnitude is still far below any 5-10% threshold.",
             ha="center", fontsize=8, style="italic")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(OUT / "fig5_robustness.png", dpi=150)
    plt.close(fig)
    print("wrote fig5_robustness.png")


fig_robustness()
print("\nAll figures written to", OUT)
