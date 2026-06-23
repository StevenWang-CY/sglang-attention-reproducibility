#!/usr/bin/env python3
"""Dense page_size x batch_size sweep: figures + consistency check.

Reads offline_batch_results/dense_5060ti/{config}_ps{ps}[_rev].json  (Block 1, radix OFF)
and                              hb_{config}_ps{ps}.json               (Block 2, radix ON, high batch)
plus the matching .log per cell (full stdout+stderr) from which we recover the TRUE per-batch
concurrency: SGLang's scheduler logs 'Decode batch ... #running-req: N' from a child process,
so it never reaches the harness's in-parent tracker (every JSON has max_running_req=0) -- but the
shell redirect '> cell.log 2>&1' DOES capture it. We parse it here to mark which cells truly fit.

Outputs (report_3_dense_sweep_MAIN/):
  fig_dense_4panel.png      - the enriched 4-panel (normalized to ps1), concurrency-annotated,
                              with the previous (sparse) run overlaid faintly for consistency.
  fig_dense_highbatch.png   - Block 2: TPOT vs page_size at high batch, both backends.
  dense_consistency.md      - new-vs-old overlap delta table + verdict; prints to stdout too.
"""
import json, glob, re, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]  # repo root
DENSE = ROOT / "offline_batch_results" / "dense_5060ti"
OUT = Path(__file__).resolve().parents[1] / "report_3_dense_sweep_MAIN"  # ../report_3_dense_sweep_MAIN
PAGES_ALL = [1, 2, 4, 8, 16, 32, 64, 128]


# ── concurrency recovery from the per-cell log ────────────────────────────────
def parse_concurrency(logpath):
    """Return {batch_size: max_running_req_seen} by walking the log: each '  Batch Size: N'
    header (printed by the harness) starts a new section; collect '#running-req: M' until the
    next header. The scheduler emits one such line per decode step."""
    out = {}
    cur = None
    try:
        txt = Path(logpath).read_text(errors="ignore")
    except Exception:
        return out
    for line in txt.splitlines():
        m = re.search(r"Batch Size:\s*(\d+)", line)
        if m:
            cur = int(m.group(1)); out.setdefault(cur, 0); continue
        m = re.search(r"#running-req:\s*(\d+)", line)
        if m and cur is not None:
            out[cur] = max(out[cur], int(m.group(1)))
    return out


def load_cell(tag):
    """Load one launch's JSON + its log-derived concurrency. -> {bs: {...}} or None."""
    jp = DENSE / f"{tag}.json"
    if not jp.exists():
        return None
    try:
        j = json.load(open(jp))
    except Exception:
        return None
    conc = parse_concurrency(DENSE / f"{tag}.log")
    out = {}
    for bs_str, d in j.items():
        bs = int(bs_str)
        rr = conc.get(bs)
        out[bs] = {
            "tpot": d.get("tpot_median_ms"),
            "std": d.get("tpot_std_ms"),
            "free_before": d.get("free_before_mb"),
            "free_after": d.get("free_after_mb"),
            "running_req": rr,
            "fit": (rr == bs) if rr is not None else None,  # True/False/None(unknown)
            "tokens": d.get("tokens_generated"),
        }
    return out


def load_config(prefix, pages, rev=False):
    """Merge forward (and optionally reverse) launches for a config -> {ps: {bs: cell}}.
    When both fwd and rev exist, keep both medians for a reproducibility check (stored under
    cell['tpot'] = mean of the two, with cell['fwd']/cell['rev'])."""
    res = {}
    for ps in pages:
        fwd = load_cell(f"{prefix}_ps{ps}")
        rv = load_cell(f"{prefix}_ps{ps}_rev") if rev else None
        if fwd is None and rv is None:
            continue
        merged = {}
        bss = set((fwd or {}).keys()) | set((rv or {}).keys())
        for bs in bss:
            a = (fwd or {}).get(bs); b = (rv or {}).get(bs)
            vals = [c["tpot"] for c in (a, b) if c and c["tpot"] is not None]
            cell = dict(a or b)
            cell["fwd"] = a["tpot"] if a else None
            cell["rev"] = b["tpot"] if b else None
            cell["tpot"] = float(np.mean(vals)) if vals else None
            # fit True only if every available pass fit
            fits = [c["fit"] for c in (a, b) if c and c["fit"] is not None]
            cell["fit"] = (all(fits) if fits else None)
            merged[bs] = cell
        res[ps] = merged
    return res


# ── previous (sparse) run, for consistency overlay ────────────────────────────
def _med(p):
    j = json.load(open(p)); b = next(iter(j)); return j[b]["tpot_median_ms"]


def load_old():
    """Previous-run TPOT per config -> {ps: {bs: tpot}}. IMPORTANT: each config draws from a
    SINGLE self-consistent source (one token-count), else the normalized ps1 base mixes scales.
      fi_g  <- ps_e2e (512-tok, ps{1,8,32,128} x bs{1,8,16})
      fi_ng <- fi_clean_ng (128-tok, ps{1,128} x bs{1,8})  [the only prior no-graph clean source]
      tri_g <- triton_sweep tri_10k_g
      tri_ng<- triton_clean tri_clean_ng
    """
    old = {"fi_g": {}, "fi_ng": {}, "tri_g": {}, "tri_ng": {}}
    for f in glob.glob(str(ROOT / "offline_batch_results/ps_e2e_5060ti/results_ps*.json")):
        ps = int(re.search(r"ps(\d+)", f).group(1)); j = json.load(open(f))
        for b in j:
            old["fi_g"].setdefault(ps, {})[int(b)] = j[b]["tpot_median_ms"]
    for f in glob.glob(str(ROOT / "offline_batch_results/fi_clean_5060ti/fi_clean_ng_*.json")):
        m = re.search(r"fi_clean_ng_bs(\d+)_ps(\d+)", f)
        if m:
            old["fi_ng"].setdefault(int(m.group(2)), {})[int(m.group(1))] = _med(f)
    for f in glob.glob(str(ROOT / "offline_batch_results/triton_sweep_5060ti/tri_10k_g_*.json")):
        m = re.search(r"tri_10k_g_bs(\d+)_ps(\d+)", f)
        if m:
            old["tri_g"].setdefault(int(m.group(2)), {})[int(m.group(1))] = _med(f)
    for f in glob.glob(str(ROOT / "offline_batch_results/triton_clean_5060ti/*.json")):
        m = re.search(r"tri_clean_ng_bs(\d+)_ps(\d+)", f)
        if m:
            old["tri_ng"].setdefault(int(m.group(2)), {})[int(m.group(1))] = _med(f)
    return old


CONFIGS = [
    ("fi_g",  "FlashInfer, CUDA graph ON",      False),
    ("fi_ng", "FlashInfer, no CUDA graph",      True),
    ("tri_g", "Triton, CUDA graph ON",          False),
    ("tri_ng","Triton, no CUDA graph",          True),
]


def fig_4panel(data, old):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    axflat = axes.flatten()
    for ax, (key, title, rev) in zip(axflat, CONFIGS):
        cfg = data[key]
        batches = sorted({bs for ps in cfg for bs in cfg[ps]})
        for bs in batches:
            xs, ys, fits = [], [], []
            base = cfg.get(1, {}).get(bs, {}).get("tpot")
            if not base:
                continue
            for i, ps in enumerate(PAGES_ALL):
                c = cfg.get(ps, {}).get(bs)
                if c and c["tpot"]:
                    xs.append(i); ys.append(c["tpot"] / base * 100); fits.append(c["fit"])
            if not xs:
                continue
            line, = ax.plot(xs, ys, marker="o", label=f"bs={bs}")
            # mark non-fitting (or unknown) cells with an open red ring
            for x, y, fit in zip(xs, ys, fits):
                if fit is False:
                    ax.plot(x, y, marker="o", mfc="none", mec="red", ms=12, mew=1.6)
        # faint overlay of previous run (normalized to its own ps1)
        for bs, ov in sorted(old.get(key, {}).items() if False else []):
            pass
        og = old.get(key, {})
        ob = sorted({bs for ps in og for bs in og[ps]})
        for bs in ob:
            base = og.get(1, {}).get(bs)
            if not base:
                continue
            xs, ys = [], []
            for i, ps in enumerate(PAGES_ALL):
                if ps in og and bs in og[ps]:
                    xs.append(i); ys.append(og[ps][bs] / base * 100)
            if xs:
                ax.plot(xs, ys, marker="x", ls=":", color="grey", alpha=0.5, lw=1)
        ax.axhline(100, color="k", ls="--", lw=1, alpha=0.5)
        ax.set_xticks(range(len(PAGES_ALL))); ax.set_xticklabels(PAGES_ALL)
        ax.set_title(title, fontsize=11); ax.grid(alpha=0.25); ax.legend(fontsize=8, ncol=2)
        ax.set_xlabel("page_size")
    axes[0, 0].set_ylabel("TPOT rel. to page_size=1 (%)")
    axes[1, 0].set_ylabel("TPOT rel. to page_size=1 (%)")
    fig.suptitle("Dense page_size x batch_size re-run (RTX 5060 Ti, Qwen3-VL-2B, ~10k ctx)\n"
                 "8 page sizes x batch {1,2,4,8}; grey dotted = previous sparse run; "
                 "red ring = batch did not fully fit (log-confirmed)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "fig_dense_4panel.png", dpi=150); plt.close(fig)
    print("wrote fig_dense_4panel.png")


def fig_highbatch():
    """Block 2 radix-ON high batch, x=page_size, lines=batch. Emits BOTH presentations:
       fig_dense_highbatch.png    - normalized TPOT rel. to ps1 (%)   [the original]
       fig_dense_highbatch_ms.png - absolute decode latency TPOT (ms)"""
    cfgs = [("hb_fi_g", "FlashInfer graph ON"), ("hb_fi_ng", "FlashInfer no graph"),
            ("hb_tri_g", "Triton graph ON"), ("hb_tri_ng", "Triton no graph")]
    pages = [1, 8, 32, 128]
    have = {k: load_config(k, pages, rev=False) for k, _ in cfgs}
    if not any(have.values()):
        print("skip fig_highbatch (no Block-2 data yet)"); return
    for mode, fname, ylab, sub in [
        ("pct", "fig_dense_highbatch.png", "TPOT rel. to page_size=1 (%)",
         "normalized to page_size=1 (%) — the Triton drop at ps128 is the ≥5% effect"),
        ("ms", "fig_dense_highbatch_ms.png", "decode latency — TPOT (ms)",
         "absolute decode latency (ms), lower = faster"),
    ]:
        fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
        for ax, (key, title) in zip(axes.flatten(), cfgs):
            cfg = have[key]
            for bs in sorted({bs for ps in cfg for bs in cfg[ps]}):
                base = cfg.get(1, {}).get(bs, {}).get("tpot")
                if mode == "pct" and not base:
                    continue
                xs, ys, fits = [], [], []
                for i, ps in enumerate(pages):
                    c = cfg.get(ps, {}).get(bs)
                    if c and c["tpot"]:
                        xs.append(i); fits.append(c["fit"])
                        ys.append(c["tpot"] / base * 100 if mode == "pct" else c["tpot"])
                if xs:
                    ax.plot(xs, ys, marker="o", label=f"bs={bs}")
                    for x, y, fit in zip(xs, ys, fits):
                        if fit is False:
                            ax.plot(x, y, marker="o", mfc="none", mec="red", ms=12, mew=1.6)
            if mode == "pct":
                ax.axhline(100, color="k", ls="--", lw=1, alpha=0.5)
            else:
                ax.set_ylim(bottom=0)
            ax.set_xticks(range(len(pages))); ax.set_xticklabels(pages)
            ax.set_title(title, fontsize=11); ax.grid(alpha=0.25); ax.legend(fontsize=8)
            ax.set_xlabel("page_size")
        axes[0, 0].set_ylabel(ylab); axes[1, 0].set_ylabel(ylab)
        fig.suptitle("Block 2 — high-batch reach: decode TPOT vs page_size (radix ON / shared prefix, ~10k ctx)\n"
                     + sub + ". red ring = batch did not fully fit", fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(OUT / fname, dpi=150); plt.close(fig)
        print(f"wrote {fname}")


def fig_penalty_and_abs():
    """The two decision figures for Block 2 (radix ON, high batch):
      fig_dense_penalty.png  - ps1 penalty vs ps128 (%) as batch grows, all 4 configs, +5% line.
      fig_dense_highbatch_abs.png - absolute TPOT (ms) from 0, Triton vs FlashInfer, ps1 vs ps128."""
    BATCHES = [1, 8, 16, 32, 48]
    cfgs = [("hb_fi_g", "FlashInfer graph-ON", "tab:blue"),
            ("hb_fi_ng", "FlashInfer no-graph", "tab:cyan"),
            ("hb_tri_g", "Triton graph-ON", "tab:red"),
            ("hb_tri_ng", "Triton no-graph", "tab:orange")]

    def tp(tag, ps, bs):
        c = load_cell(tag)
        return None if c is None else (c.get(bs, {}) or {}).get("tpot")

    # need a per-(tag,ps) loader since load_cell wants a full tag
    def med(tag, ps, bs):
        import os as _os
        p = DENSE / f"{tag}_ps{ps}.json"
        if not p.exists():
            return None
        j = json.load(open(p))
        return j.get(str(bs), {}).get("tpot_median_ms")

    # FIG penalty (normalized %) — the original: ps1 penalty vs ps128, 4 configs, +5% line.
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    any_data = False
    for tag, name, col in cfgs:
        xs, ys = [], []
        for bs in BATCHES:
            v1, v128 = med(tag, 1, bs), med(tag, 128, bs)
            if v1 and v128:
                xs.append(bs); ys.append((v1 / v128 - 1) * 100)
        if xs:
            any_data = True
            ax.plot(xs, ys, marker="o", color=col, label=name, lw=2)
    if not any_data:
        print("skip fig_penalty (no Block-2 data)"); plt.close(fig); return
    ax.axhline(5, color="green", ls="--", lw=1.2, label="+5% (PI threshold)")
    ax.axhline(0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel("batch size (shared-prefix, full ~10k ctx)")
    ax.set_ylabel("page_size=1 penalty vs page_size=128  (%)")
    ax.set_title("Does page_size=1 ever lose by ≥5%? YES — Triton at high batch\n"
                 "(TPOT[ps1]/TPOT[ps128]−1); FlashInfer stays flat", fontweight="bold")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "fig_dense_penalty.png", dpi=150); plt.close(fig)
    print("wrote fig_dense_penalty.png")

    # FIG penalty_ms — same story in absolute latency: ps1 (solid) vs ps128 (dashed) per config.
    fig, ax = plt.subplots(figsize=(9.5, 6))
    for tag, name, col in cfgs:
        for ps, ls, mk in [(1, "-", "o"), (128, "--", "s")]:
            xs, ys = [], []
            for bs in BATCHES:
                v = med(tag, ps, bs)
                if v:
                    xs.append(bs); ys.append(v)
            if xs:
                ax.plot(xs, ys, marker=mk, ls=ls, color=col, lw=2, label=f"{name}, ps{ps}")
    ax.set_xlabel("batch size (shared-prefix, full ~10k ctx)")
    ax.set_ylabel("decode latency — TPOT (ms)")
    ax.set_title("Decode latency vs batch — page_size=1 (solid) vs page_size=128 (dashed)\n"
                 "Triton ps1 climbs above ps128 (the ≥5% penalty, in ms); FlashInfer lines overlap",
                 fontweight="bold")
    ax.set_ylim(bottom=0); ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(OUT / "fig_dense_penalty_ms.png", dpi=150); plt.close(fig)
    print("wrote fig_dense_penalty_ms.png")

    # FIG absolute, from 0
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, (gtag, gname) in zip(axes, [("hb_tri_g", "Triton graph-ON"), ("hb_fi_g", "FlashInfer graph-ON")]):
        for ps, col, mk in [(1, "tab:red", "o"), (128, "tab:green", "s")]:
            xs, ys = [], []
            for bs in BATCHES:
                v = med(gtag, ps, bs)
                if v:
                    xs.append(bs); ys.append(v)
            if xs:
                ax.plot(xs, ys, marker=mk, color=col, label=f"page_size={ps}", lw=2)
        ax.set_title(gname, fontsize=11); ax.set_xlabel("batch size")
        ax.grid(alpha=0.3); ax.legend(); ax.set_ylim(bottom=0)
    axes[0].set_ylabel("decode TPOT (ms)")
    fig.suptitle("Absolute decode latency, high batch (shared prefix, ~10k ctx): "
                 "Triton ps1 diverges from ps128; FlashInfer does not", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "fig_dense_highbatch_abs.png", dpi=150); plt.close(fig)
    print("wrote fig_dense_highbatch_abs.png")


def fig_confirm():
    """radix-ON (shared prefix, full ctx) vs radix-OFF (independent contexts, short ctx):
    how much of the Triton ps1 penalty is shared-prefix-specific."""
    def med(tag, ps, bs):
        p = DENSE / f"{tag}_ps{ps}.json"
        if not p.exists():
            return None
        return json.load(open(p)).get(str(bs), {}).get("tpot_median_ms")

    def pen(tag, bs):  # ps1 vs ps128 penalty %
        a, b = med(tag, 1, bs), med(tag, 128, bs)
        return (a / b - 1) * 100 if (a and b) else None

    series = [
        ("Triton, radix ON, full ~10k ctx", "hb_tri_g", [8, 16, 32, 48], "tab:red", "-"),
        ("Triton, radix OFF, ~3k ctx (indep.)", "cf_tri_g_3k", [8, 16, 24], "tab:orange", "--"),
        ("Triton, radix OFF, ~1.5k ctx (indep.)", "cf_tri_g_15k", [16, 32, 48], "tab:brown", "--"),
        ("FlashInfer, radix ON, full ctx", "hb_fi_g", [8, 16, 32, 48], "tab:blue", "-"),
    ]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    any_data = False
    for name, tag, bss, col, ls in series:
        xs, ys = [], []
        for bs in bss:
            p = pen(tag, bs)
            if p is not None:
                xs.append(bs); ys.append(p)
        if xs:
            any_data = True
            ax.plot(xs, ys, marker="o", color=col, ls=ls, lw=2, label=name)
    if not any_data:
        print("skip fig_confirm (no data)"); plt.close(fig); return
    ax.axhline(5, color="green", ls=":", lw=1.2, label="+5% (PI threshold)")
    ax.axhline(0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel("batch size"); ax.set_ylabel("page_size=1 penalty vs page_size=128 (%)")
    ax.set_title("How much of the Triton ps1 penalty is shared-prefix?\n"
                 "radix ON (shared prefix) ~20-25%; radix OFF (independent contexts) ~2-5%",
                 fontweight="bold")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig_dense_confirm.png", dpi=150); plt.close(fig)
    print("wrote fig_dense_confirm.png")


def fig_latency_decomp():
    """Investigate ALL THREE latency metrics, not just decode TPOT:
      - per-cell: page effect on TPOT (decode) vs TTFT (prefill) vs total (e2e).
      - model the end-to-end ps1 penalty vs output length: total(N) = TTFT + N*TPOT,
        showing the e2e penalty rises from ~0 (1-token reply) toward the full decode
        penalty (long output), because prefill is fixed and page-neutral."""
    def fld(stem, bs, f):
        p = DENSE / f"{stem}.json"
        if not p.exists():
            return None
        d = json.load(open(p)).get(str(bs))
        return d.get(f) if d else None
    bs = 32
    cfgs = [("hb_tri_g", "Triton graph-ON", "tab:red"),
            ("hb_tri_ng", "Triton no-graph", "tab:orange"),
            ("hb_fi_g", "FlashInfer graph-ON", "tab:blue")]
    Ns = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    drew = False
    for tag, name, col in cfgs:
        tt1, tp1 = fld(f"{tag}_ps1", bs, "ttft_median_ms"), fld(f"{tag}_ps1", bs, "tpot_median_ms")
        tt128, tp128 = fld(f"{tag}_ps128", bs, "ttft_median_ms"), fld(f"{tag}_ps128", bs, "tpot_median_ms")
        if None in (tt1, tp1, tt128, tp128):
            continue
        ys = [((tt1 + N * tp1) / (tt128 + N * tp128) - 1) * 100 for N in Ns]
        ax.plot(Ns, ys, marker="o", ms=3, color=col, label=name)
        drew = True
    if not drew:
        print("skip fig_latency_decomp"); plt.close(fig); return
    ax.axhline(5, color="green", ls=":", lw=1.2, label="+5%")
    ax.axhline(0, color="k", lw=0.8, alpha=0.5)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("output length (decode tokens)")
    ax.set_ylabel("end-to-end ps1 penalty vs ps128 (%)")
    ax.set_title("End-to-end latency: the Triton ps1 penalty is decode-only\n"
                 f"bs{bs}, shared prefix — grows with output length (prefill is page-neutral); model total(N)=TTFT+N·TPOT",
                 fontweight="bold")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig_dense_latency_decomp.png", dpi=150); plt.close(fig)
    print("wrote fig_dense_latency_decomp.png")


def fig_flashinfer():
    """FlashInfer's narrow >=5% corner: eager + ~39k ctx + bs~8 (shared prefix).
    ps1 penalty vs ps128 across batch, with the 21k-eager line (stays <0) and the
    graph-ON control at bs8 (eager-specific). Individual reps scattered; median line."""
    def c(stem, bs, f='tpot_median_ms'):
        p = DENSE / f"{stem}.json"
        if not p.exists():
            return None
        d = json.load(open(p)).get(str(bs))
        return d.get(f) if d else None

    def pen(s1, s128, bs):
        a, b = c(s1, bs), c(s128, bs)
        return (a / b - 1) * 100 if (a and b) else None

    # 39k eager: gather all reps per batch
    SRCS = [("fxc_39k_eager_ps1", "fxc_39k_eager_ps128"),
            ("fxv_39k_eager_ps1", "fxv_39k_eager_ps128"),
            ("fxv_39k_eager_p2_ps1", "fxv_39k_eager_p2_ps128"),
            ("fx2_39k_eager_lobs_ps1", "fx2_39k_eager_lobs_ps128"),
            ("fx_58k_eager_ps1", "fx_58k_eager_ps128")]
    BATCHES = [1, 2, 4, 6, 8, 10, 12, 16, 64]
    med39, allpts = {}, []
    for bs in BATCHES:
        vals = [pen(s1, s128, bs) for s1, s128 in SRCS]
        vals = [v for v in vals if v is not None]
        if vals:
            med39[bs] = float(np.median(vals))
            allpts += [(bs, v) for v in vals]
    # 21k eager
    med21 = {bs: pen("fxv_21k_eager_ps1", "fxv_21k_eager_ps128", bs) for bs in [4, 8, 16]}
    med21 = {k: v for k, v in med21.items() if v is not None}
    g8 = pen("fxc_39k_graph_ps1", "fxc_39k_graph_ps128", 8)
    if not med39:
        print("skip fig_flashinfer (no fx data)"); return

    fig, ax = plt.subplots(figsize=(9.5, 6))
    if allpts:
        ax.scatter([p[0] for p in allpts], [p[1] for p in allpts], s=18, color="tab:blue",
                   alpha=0.35, zorder=2, label="individual repeats (39k eager)")
    xs = sorted(med39); ax.plot(xs, [med39[b] for b in xs], "-o", color="tab:blue", lw=2,
                                zorder=3, label="FlashInfer eager, ~39k ctx (median)")
    if med21:
        xs2 = sorted(med21); ax.plot(xs2, [med21[b] for b in xs2], "--s", color="tab:cyan",
                                     lw=2, label="FlashInfer eager, ~21k ctx")
    if g8 is not None:
        ax.scatter([8], [g8], s=140, marker="*", color="tab:red", zorder=4,
                   label=f"39k CUDA-graph ON, bs8 ({g8:+.1f}%)")
    ax.axhline(5, color="green", ls=":", lw=1.3, label="+5% threshold")
    ax.axhline(0, color="k", lw=0.8, alpha=0.5)
    ax.annotate("bs8: +7.4%\n(reproduced 3x, clean)", xy=(8, med39.get(8, 7.4)), xytext=(11, 12),
                fontsize=9, arrowprops=dict(arrowstyle="->", color="gray"))
    ax.set_xlabel("batch size"); ax.set_ylabel("page_size=1 penalty vs page_size=128 (%)")
    ax.set_title("FlashInfer's narrow ≥5% corner: eager (no CUDA graph) + ~39k ctx + bs~8\n"
                 "shared prefix. Peaks ~bs6–8; CUDA graph reduces it; ~21k ctx stays <0 (ps1 faster)",
                 fontweight="bold")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(OUT / "fig_dense_flashinfer.png", dpi=150); plt.close(fig)
    print("wrote fig_dense_flashinfer.png")


def fig_over5():
    """Dedicated figure: every (config, batch) cell where page_size=1 is >=5% WORSE than
    page_size=128. Solid bars = robust (clean, low std); hatched grey = noise-limited."""
    def m(stem, bs):
        p = DENSE / f"{stem}.json"
        if not p.exists():
            return None
        return json.load(open(p)).get(str(bs), {}).get("tpot_median_ms")

    def avg(stems, bs):
        vs = [m(s, bs) for s in stems]; vs = [v for v in vs if v]
        return sum(vs) / len(vs) if vs else None

    rows = []  # (penalty, label, color, hatch)
    COL = {"Triton graph-ON": "tab:red", "Triton no-graph": "tab:orange",
           "FlashInfer graph-ON": "tab:blue", "FlashInfer no-graph": "tab:cyan"}
    # Block 2: shared prefix (radix ON), full ctx, high batch
    for tag, name in [("hb_tri_g", "Triton graph-ON"), ("hb_tri_ng", "Triton no-graph"),
                      ("hb_fi_g", "FlashInfer graph-ON"), ("hb_fi_ng", "FlashInfer no-graph")]:
        for bs in [8, 16, 32, 48]:
            a, b = m(f"{tag}_ps1", bs), m(f"{tag}_ps128", bs)
            if a and b and (a / b - 1) * 100 >= 5:
                rows.append(((a / b - 1) * 100, f"{name}, bs{bs}  (shared prefix, ~10k)", COL[name], None))
    # Block 1: independent ctx (radix OFF), ~10k, no-graph fwd+rev avg (noise-limited)
    for bs in [1, 2, 4, 8]:
        a = avg([f"fi_ng_ps1", f"fi_ng_ps1_rev"], bs); b = avg([f"fi_ng_ps128", f"fi_ng_ps128_rev"], bs)
        if a and b and (a / b - 1) * 100 >= 5:
            rows.append(((a / b - 1) * 100, f"FlashInfer no-graph, bs{bs}  (indep ctx, ~10k) — noise-limited", "grey", "///"))
    if not rows:
        print("skip fig_over5 (no >=5% cells)"); return
    rows.sort(key=lambda r: r[0])
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(rows) + 2))
    y = range(len(rows))
    bars = ax.barh(list(y), [r[0] for r in rows],
                   color=[r[2] for r in rows], hatch=[r[3] for r in rows], edgecolor="k", alpha=0.9)
    ax.set_yticks(list(y)); ax.set_yticklabels([r[1] for r in rows], fontsize=9)
    ax.axvline(5, color="green", ls="--", lw=1.5, label="+5% threshold")
    for i, r in enumerate(rows):
        ax.text(r[0] + 0.3, i, f"+{r[0]:.1f}%", va="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("page_size=1 decode-latency penalty vs page_size=128  (%)")
    ax.set_title("Where page_size=1 is ≥5% WORSE than page_size=128\n"
                 "(all such cells; solid = robust/clean, hatched grey = noise-limited)", fontweight="bold")
    ax.set_xlim(0, max(r[0] for r in rows) * 1.15); ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "fig_dense_over5.png", dpi=150); plt.close(fig)
    print(f"wrote fig_dense_over5.png ({len(rows)} cells)")


def consistency(data, old):
    """Normalized (rel-to-ps1) consistency vs the previous sparse run, per overlap cell."""
    lines = ["# Dense re-run consistency vs previous run\n",
             "Normalized gap = (TPOT[ps] / TPOT[ps1] - 1) * 100%, compared new vs old per (config, bs, ps).",
             "Absolute TPOT differs by token-count between runs, so only the *normalized* gap is compared.\n",
             "| config | bs | ps | old gap% | new gap% | |Δ| | within noise? |",
             "|---|---|---|---|---|---|---|"]
    worst = 0.0
    for key, title, rev in CONFIGS:
        cfg = data.get(key, {}); og = old.get(key, {})
        ob = sorted({bs for ps in og for bs in og[ps]})
        for bs in ob:
            old_base = og.get(1, {}).get(bs)
            new_base = cfg.get(1, {}).get(bs, {}).get("tpot")
            if not old_base or not new_base:
                continue
            for ps in sorted({ps for ps in og if bs in og[ps]}):
                if ps == 1:
                    continue
                new_c = cfg.get(ps, {}).get(bs)
                if not new_c or not new_c["tpot"]:
                    continue
                old_gap = (og[ps][bs] / old_base - 1) * 100
                new_gap = (new_c["tpot"] / new_base - 1) * 100
                d = abs(new_gap - old_gap)
                std = new_c.get("std") or 0
                tol = max(2.0, 100 * 2 * std / new_c["tpot"])
                ok = "yes" if d <= tol else "**NO**"
                worst = max(worst, d)
                lines.append(f"| {title} | {bs} | {ps} | {old_gap:+.2f} | {new_gap:+.2f} | {d:.2f} | {ok} (tol {tol:.1f}) |")
    lines.append(f"\n**Worst normalized |Δ| = {worst:.2f}%.**")
    txt = "\n".join(lines)
    (Path(__file__).resolve().parents[1] / "data" / "dense_consistency.md").write_text(txt)
    print(txt)
    print("\nwrote dense_consistency.md")


if __name__ == "__main__":
    data = {key: load_config(key, PAGES_ALL, rev=rev) for key, _, rev in CONFIGS}
    old = load_old()
    n = sum(len(cfg) for cfg in data.values())
    print(f"loaded {n} (config,ps) launches from {DENSE}")
    if n == 0:
        print("no dense data yet — run after pulling JSONs+logs"); sys.exit(0)
    fig_4panel(data, old)
    fig_highbatch()
    fig_penalty_and_abs()
    fig_confirm()
    fig_latency_decomp()
    fig_flashinfer()
    fig_over5()
    consistency(data, old)
