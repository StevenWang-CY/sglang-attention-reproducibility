#!/usr/bin/env python3
"""Print the report-14 Triton stage-1 metrics from Nsight Compute CSV exports."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


STUDY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = (
    STUDY_ROOT / "data" / "raw" / "report-14-shared-prefix-page-kernel-rtx5060ti"
)


def metric(row: dict[str, str], fragment: str, default: str = "?") -> str:
    for key, value in row.items():
        if fragment in key and value:
            return value
    return default


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "patterns", nargs="*", default=["pc_triton_shared_*_kv6144.csv"]
    )
    args = parser.parse_args()

    files: list[Path] = []
    for pattern in args.patterns:
        files.extend(sorted(args.data.glob(pattern)))

    print(
        "cell".ljust(44),
        "dur_us".rjust(8),
        "DRAM".rjust(6),
        "L2hit".rjust(7),
        "L2thr".rjust(7),
        "SM".rjust(5),
        "sec_req".rjust(8),
    )
    for path in files:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = [
                row
                for row in csv.DictReader(handle)
                if row.get("ID", "").isdigit() and "stage1" in row.get("Kernel Name", "")
            ]
        if not rows:
            continue
        row = rows[0]
        tag = (
            path.stem.replace("pc_triton_shared_", "").replace("_h16k8", "")
        )
        try:
            print(
                tag.ljust(44),
                metric(row, "gpu__time_duration.avg").rjust(8),
                f"{float(metric(row, 'dram__throughput.avg')):.1f}".rjust(6),
                f"{float(metric(row, 'lts__t_sector_hit_rate')):.1f}".rjust(7),
                f"{float(metric(row, 'lts__throughput.avg')):.1f}".rjust(7),
                f"{float(metric(row, 'sm__throughput.avg')):.1f}".rjust(5),
                metric(
                    row,
                    "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio",
                ).rjust(8),
            )
        except (TypeError, ValueError) as error:
            print(tag, "PARSE-ERR", error)


if __name__ == "__main__":
    main()
