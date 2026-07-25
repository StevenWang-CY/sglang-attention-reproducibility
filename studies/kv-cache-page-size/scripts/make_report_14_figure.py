#!/usr/bin/env python3
"""Generate report 14's whole-call-versus-steady-decode reconciliation figure."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STUDY_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = STUDY_ROOT / "data" / "curated" / "report-14-figure-values.csv"
OUTPUT_FILE = (
    STUDY_ROOT
    / "reports"
    / "14-shared-prefix-benchmark-artifact"
    / "fig_reconcile.png"
)


def load_rows() -> list[dict[str, str]]:
    with DATA_FILE.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def values(
    rows: list[dict[str, str]],
    *,
    panel: str,
    metric: str,
    page_size: str,
    order_field: str,
) -> list[float]:
    selected = [
        row
        for row in rows
        if row["panel"] == panel
        and row["metric"] == metric
        and row["page_size"] == page_size
    ]
    selected.sort(key=lambda row: int(row[order_field]))
    return [float(row["value"]) for row in selected]


def main() -> None:
    rows = load_rows()

    batch_sizes = [8, 16, 32, 48]
    reported_ps1 = values(
        rows, panel="A", metric="reported_tpot", page_size="1", order_field="batch_size"
    )
    reported_ps128 = values(
        rows, panel="A", metric="reported_tpot", page_size="128", order_field="batch_size"
    )
    steady_ps1 = values(
        rows, panel="A", metric="steady_decode", page_size="1", order_field="batch_size"
    )
    steady_ps128 = values(
        rows, panel="A", metric="steady_decode", page_size="128", order_field="batch_size"
    )

    max_tokens = [96, 512]
    reported_delta = values(
        rows,
        panel="B",
        metric="reported_delta",
        page_size="1-vs-128",
        order_field="max_tokens",
    )
    steady_delta = values(
        rows,
        panel="B",
        metric="steady_delta",
        page_size="1-vs-128",
        order_field="max_tokens",
    )

    panel_c = [row for row in rows if row["panel"] == "C"]
    configurations = ["matched-prompt9k-ctx12288", "longer-prompt16k-ctx20480"]

    def config_values(metric: str, page_size: str) -> list[float]:
        lookup = {
            (row["configuration"], row["metric"], row["page_size"]): float(row["value"])
            for row in panel_c
        }
        return [lookup[(config, metric, page_size)] for config in configurations]

    config_reported_ps1 = config_values("reported_tpot", "1")
    config_reported_ps128 = config_values("reported_tpot", "128")
    config_steady_ps1 = config_values("steady_decode", "1")
    config_steady_ps128 = config_values("steady_decode", "128")

    fig, (axis_a, axis_b, axis_c) = plt.subplots(1, 3, figsize=(15, 4.6))

    x = np.arange(len(batch_sizes))
    width = 0.2
    axis_a.bar(x - 1.5 * width, reported_ps1, width, label="whole-call ps1", color="#d62728")
    axis_a.bar(
        x - 0.5 * width, reported_ps128, width, label="whole-call ps128", color="#ff9896"
    )
    axis_a.bar(x + 0.5 * width, steady_ps1, width, label="steady ps1", color="#1f77b4")
    axis_a.bar(
        x + 1.5 * width, steady_ps128, width, label="steady ps128", color="#aec7e8"
    )
    for index in range(len(batch_sizes)):
        reported_pct = (
            (reported_ps1[index] - reported_ps128[index]) / reported_ps128[index] * 100
        )
        steady_pct = (
            (steady_ps1[index] - steady_ps128[index]) / steady_ps128[index] * 100
        )
        axis_a.text(
            x[index] - width,
            reported_ps1[index] + 1.5,
            f"{reported_pct:+.0f}%",
            ha="center",
            fontsize=8,
            color="#d62728",
            fontweight="bold",
        )
        axis_a.text(
            x[index] + width,
            max(steady_ps1[index], steady_ps128[index]) + 1.5,
            f"{steady_pct:+.1f}%",
            ha="center",
            fontsize=8,
            color="#1f77b4",
            fontweight="bold",
        )
    axis_a.set_xticks(x)
    axis_a.set_xticklabels([f"bs{batch}" for batch in batch_sizes])
    axis_a.set_ylabel("Latency (ms/token or ms/step)")
    axis_a.set_title(
        "A. Whole-call metric shows +14–32%;\nsteady decode is page-flat", fontsize=10
    )
    axis_a.legend(fontsize=7, ncol=2, loc="upper left")
    axis_a.grid(axis="y", alpha=0.3)

    x_b = np.arange(len(max_tokens))
    axis_b.bar(
        x_b - 0.2,
        reported_delta,
        0.4,
        label="whole-call ps1−ps128",
        color="#d62728",
    )
    axis_b.bar(
        x_b + 0.2,
        steady_delta,
        0.4,
        label="steady ps1−ps128",
        color="#1f77b4",
    )
    for index, delta in enumerate(reported_delta):
        axis_b.text(
            x_b[index] - 0.2,
            delta + 0.7,
            f"{delta:+.1f}%",
            ha="center",
            fontsize=9,
            fontweight="bold",
            color="#d62728",
        )
    axis_b.set_xticks(x_b)
    axis_b.set_xticklabels([f"max-tokens {tokens}" for tokens in max_tokens])
    axis_b.set_ylabel("ps1 − ps128 (%)")
    axis_b.set_title(
        "B. Amortization: whole-call difference\ncollapses as decode length grows", fontsize=10
    )
    axis_b.axhline(0, color="black", linewidth=0.6)
    axis_b.legend(fontsize=8)
    axis_b.grid(axis="y", alpha=0.3)

    x_c = np.arange(len(configurations))
    axis_c.bar(
        x_c - 1.5 * width,
        config_reported_ps1,
        width,
        label="whole-call ps1",
        color="#d62728",
    )
    axis_c.bar(
        x_c - 0.5 * width,
        config_reported_ps128,
        width,
        label="whole-call ps128",
        color="#ff9896",
    )
    axis_c.bar(
        x_c + 0.5 * width,
        config_steady_ps1,
        width,
        label="steady ps1",
        color="#1f77b4",
    )
    axis_c.bar(
        x_c + 1.5 * width,
        config_steady_ps128,
        width,
        label="steady ps128",
        color="#aec7e8",
    )
    axis_c.set_xticks(x_c)
    axis_c.set_xticklabels(["matched prompt9k\nctx12288", "longer prompt16k\nctx20480"], fontsize=8)
    axis_c.set_ylabel("Latency (ms/token or ms/step)")
    axis_c.set_title(
        "C. Whole-call artifact is configuration-dependent;\nsteady decode stays page-flat",
        fontsize=10,
    )
    axis_c.legend(fontsize=7, ncol=2, loc="upper left")
    axis_c.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Report 14 — whole-call shared-prefix metric versus steady decode",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FILE, dpi=130, bbox_inches="tight")
    print(f"wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
