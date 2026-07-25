#!/usr/bin/env python3
"""Generate report 16's page-size-by-memory-regime mechanism figure."""

import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STUDY_ROOT = Path(__file__).resolve().parents[1]
MECHANISM_DATA = STUDY_ROOT / "data" / "raw" / "report-16-mechanism-sweep"
OUTPUT_FILE = (
    STUDY_ROOT / "reports" / "16-trtllm-mha-xqa" / "fig_page_regime_mechanism.png"
)


def metric_value(path: Path, name_fragment: str):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    column = next((key for key in rows[0] if name_fragment in key), None)
    if not column:
        return None
    parsed = []
    for row in rows:
        try:
            parsed.append(float(row[column].replace(",", "")))
        except (AttributeError, TypeError, ValueError):
            continue
    if not parsed:
        return None
    return sum(parsed) if ".sum" in name_fragment else sum(parsed) / len(parsed)

# ---- XQA sweep ----
xqa = {}
for path in MECHANISM_DATA.glob("xqa_*.csv"):
    match = re.match(
        r"xqa_(shared|distinct)_ps(\d+)_bs(\d+)_kv(\d+)", path.name
    )
    if not match:
        continue
    regime, page, batch, kv_length = (
        match.group(1),
        int(match.group(2)),
        int(match.group(3)),
        int(match.group(4)),
    )
    xqa[(regime, batch, kv_length, page)] = {
        "dur": metric_value(path, "gpu__time_duration.sum"),
        "dram": metric_value(path, "dram__throughput.avg.pct_of_peak"),
        "gld": metric_value(path, "sass_inst_executed_op_global_ld.sum"),
    }

fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.6))

# Panel A: page-sensitivity vs DRAM%
for km, col, mk in [("shared", "#d62728", "o"), ("distinct", "#1f77b4", "s")]:
    xs, ys, labs = [], [], []
    for (k_km, bs, kv) in sorted({(k[0], k[1], k[2]) for k in xqa if k[0] == km and k[1] == 32}):
        d16, d64 = xqa.get((km, bs, kv, 16)), xqa.get((km, bs, kv, 64))
        if not (d16 and d64 and d16["dur"] and d64["dur"] and d64["dram"]): continue
        xs.append(d64["dram"]); ys.append((d16["dur"] / d64["dur"] - 1) * 100); labs.append(kv)
    axA.scatter(xs, ys, c=col, marker=mk, s=70, label=f"{km} (bs32)", zorder=3, edgecolor="k", linewidth=0.4)
    for x, y, l in zip(xs, ys, labs):
        if km == "distinct":
            label_y = {
                2048: 0.35,
                4096: -0.15,
                8192: -0.65,
                16384: -1.15,
            }[l]
            axA.annotate(
                f"L{l // 1024}k",
                (x, y),
                xytext=(88.5, label_y),
                textcoords="data",
                fontsize=7,
                arrowprops={"arrowstyle": "-", "color": col, "lw": 0.5},
            )
        else:
            offsets = {
                2048: (3, 5),
                4096: (3, -10),
                8192: (3, 7),
                16384: (3, 3),
            }
            axA.annotate(
                f"L{l // 1024}k",
                (x, y),
                fontsize=7,
                xytext=offsets[l],
                textcoords="offset points",
            )
axA.axhline(0, color="gray", lw=0.8, ls=":")
axA.set_ylim(-1.6, 11.0)
axA.set_xlabel("DRAM throughput (% of peak)  — 'the wall' = 100%")
axA.set_ylabel("XQA page-sensitivity: decode-time ps16 vs ps64  (%)")
axA.set_title("A. Page effect vanishes as decode hits the DRAM wall")
axA.legend(fontsize=9); axA.grid(alpha=0.25)

# Panel B: global loads vs page_size (XQA shared) + Triton/FI flat reference
pages = [16, 32, 64, 128]
gld_xqa = [xqa.get(("shared", 32, 8192, p), {}).get("gld") for p in pages]
base = gld_xqa[-1] if gld_xqa[-1] else 1
axB.plot(pages, [g / base if g else None for g in gld_xqa], "o-", c="#d62728",
         label="XQA: measured page-dependent loads", lw=2)
axB.plot(pages, [1, 1, 1, 1], "s--", c="#2ca02c", label="Triton / FlashInfer: page-invariant (1.00×)", lw=1.8)
axB.plot(pages, [128 / p for p in pages], ":", c="gray", label="1/page reference (not measured law)", lw=1)
axB.set_xscale("log", base=2); axB.set_xticks(pages); axB.set_xticklabels(pages)
axB.set_xlabel("page_size"); axB.set_ylabel("global-load instructions (normalized to ps128)")
axB.set_title("B. XQA load work changes with page size;\nTriton/FI work is page-invariant")
axB.legend(fontsize=9); axB.grid(alpha=0.25)

fig.suptitle("Report 16 — page_size × regime mechanism (RTX 5060 Ti sm120, ncu): "
             "page matters off the wall (shared), washes out on it (true batch)", fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.96])
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT_FILE, dpi=130)
print(f"saved {OUTPUT_FILE}")
