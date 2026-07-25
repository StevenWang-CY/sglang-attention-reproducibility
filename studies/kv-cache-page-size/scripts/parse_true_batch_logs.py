"""Parse report-16 true-batch ``bench_one_batch`` JSONL results.

TPOT is ``median_decode_latency * 1000`` in ms/token. The input filename convention is
``bt_<model>_r<round>_b<batch>_l<length>_<backend>_ps<page>.jsonl``.
"""

import argparse
import json
import re
import statistics
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
STUDY_ROOT = SCRIPT_DIR.parent
DEFAULT_DATA = STUDY_ROOT / "data" / "raw" / "report-16-true-batch-engine"
TAG_RE = re.compile(
    r"bt_(.+?)_r(\d+)_b(\d+)_l(\d+)_(.+?)_ps(\d+)\.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_DATA,
        help=f"Directory containing bt_*.jsonl files (default: {DEFAULT_DATA})",
    )
    return parser.parse_args()


args = parse_args()
rows = {}  # (model, batch, length, backend, page) -> [TPOT per round]
for path in sorted(args.data_dir.glob("bt_*.jsonl")):
    match = TAG_RE.fullmatch(path.name)
    if not match:
        continue
    model, _round, batch, length, backend, page = match.groups()
    try:
        last_line = path.read_text(encoding="utf-8").splitlines()[-1]
        record = json.loads(last_line)
        tpot = float(record["median_decode_latency"]) * 1000.0
    except (IndexError, OSError, ValueError, KeyError, json.JSONDecodeError):
        continue
    key = (model, int(batch), int(length), backend, int(page))
    rows.setdefault(key, []).append(tpot)

# median across rounds
cell = {key: statistics.median(values) for key, values in rows.items()}
groups = {}
for (model, batch, length, backend, page), milliseconds in cell.items():
    groups.setdefault((model, batch, length), {}).setdefault(backend, {})[
        page
    ] = milliseconds

for model, batch, length in sorted(groups):
    print(f"\n=== {model}  B{batch}  L{length} ===")
    for backend in sorted(groups[(model, batch, length)]):
        pages = groups[(model, batch, length)][backend]
        base = pages.get(128) or pages.get(64) or pages[min(pages)]
        cells = " ".join(
            f"ps{page}={milliseconds:.3f}({(milliseconds / base - 1) * 100:+.1f}%)"
            for page, milliseconds in sorted(pages.items())
        )
        spread = (max(pages.values()) / min(pages.values()) - 1) * 100
        round_count = max(
            len(values)
            for key, values in rows.items()
            if key[:4] == (model, batch, length, backend)
        )
        print(
            f"  {backend:11s}: {cells}   "
            f"[spread {spread:.1f}%, {round_count}r]"
        )
    # cross-backend best page
    parts = []
    for backend, pages in sorted(groups[(model, batch, length)].items()):
        lowest = min(pages.values())
        best_page = min(pages, key=pages.get)
        parts.append(f"{backend} best ps{best_page}={lowest:.3f}")
    print("  -> " + " | ".join(parts))
