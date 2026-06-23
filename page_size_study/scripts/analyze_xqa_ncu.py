"""
analyze_xqa_ncu.py — parse ncu --csv output from profile_xqa_ncu.sh into
per-cell hardware-counter tables for Report 6.

Each <tag>.csv is ncu's long CSV (one row per kernel x metric). Tags encode
the cell: {backend}_ps{page}_bs{batch}_kv{seqlen}.csv. We pivot metrics to
columns and emit:
  - the TMA-vs-LSU contrast  (global-ld instructions, TMA/async counts if present)
  - memory traffic (DRAM read/write bytes, DRAM %peak, L2 / L1 hit rate)
  - timing + occupancy
and the key questions: does page_size move XQA's counters? XQA vs FlashInfer?

Paths self-computed: run from anywhere.
    python3 page_size_study/scripts/analyze_xqa_ncu.py [--data <dir>]
"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "offline_batch_results" / "xqa_profile"

TAG_RE = re.compile(r"(?P<be>xqa|flashinfer|triton)_ps(?P<ps>\d+)_bs(?P<bs>\d+)_kv(?P<kv>\d+)")

# metric -> short label (only those we report; others ignored)
LABELS = {
    "gpu__time_duration.sum": "dur_us",
    "dram__bytes_read.sum": "dram_rd_B",
    "dram__bytes_write.sum": "dram_wr_B",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "dram_%pk",
    "lts__t_sector_hit_rate.pct": "L2_hit%",
    "l1tex__t_sector_hit_rate.pct": "L1_hit%",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "sm_%pk",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "occ%",
    "sm__sass_inst_executed_op_global_ld.sum": "global_ld",
}


def _to_num(s):
    if s is None:
        return None
    s = s.strip().strip('"').replace(",", "")
    if s in ("", "N/A", "n/a"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_csv(path):
    """Return {kernel_name: {metric_name: value}} for one ncu csv."""
    rows = list(csv.DictReader(open(path, newline="")))
    if not rows:
        return {}
    # tolerate column-name variants
    def col(d, *names):
        for n in names:
            for k in d:
                if k.strip().lower() == n:
                    return d[k]
        return None
    out = defaultdict(dict)
    for r in rows:
        kn = col(r, "kernel name") or "?"
        mn = col(r, "metric name")
        mv = col(r, "metric value")
        if mn is None:
            continue
        out[kn][mn.strip()] = _to_num(mv)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    args = ap.parse_args()
    data = Path(args.data)
    csvs = sorted(p for p in data.glob("*.csv") if TAG_RE.search(p.stem))
    if not csvs:
        print(f"no ncu csv cells found in {data} (run profile_xqa_ncu.sh first)")
        return

    cells = {}
    tma_metrics = set()
    for p in csvs:
        m = TAG_RE.search(p.stem)
        kerns = parse_csv(p)
        if not kerns:
            continue
        # pick the kernel with the largest duration (the attention kernel)
        def dur(k):
            return kerns[k].get("gpu__time_duration.sum") or 0
        kn = max(kerns, key=dur)
        met = kerns[kn]
        for mm in met:
            if re.search(r"tma|async|bulk|tensor_memory", mm, re.I):
                tma_metrics.add(mm)
        cells[(m["be"], int(m["ps"]), int(m["bs"]), int(m["kv"]))] = (kn, met)

    cols = list(LABELS) + sorted(tma_metrics)
    hdr = ["backend", "ps", "bs", "kv"] + [LABELS.get(c, c.split(".")[0][:14]) for c in cols]
    print("  ".join(f"{h:>12}" for h in hdr))
    print("-" * (14 * len(hdr)))
    for key in sorted(cells):
        be, ps, bs, kv = key
        _, met = cells[key]
        row = [be, ps, bs, kv]
        for c in cols:
            v = met.get(c)
            row.append("-" if v is None else (f"{v:.3g}" if abs(v) < 1e4 else f"{v:.3e}"))
        print("  ".join(f"{str(x):>12}" for x in row))

    # ---- key question 1: page_size effect on XQA counters (per bs,kv) -------
    print("\n=== XQA: counter spread across page_size (max-min)/min, per (bs,kv) ===")
    by_cfg = defaultdict(dict)
    for (be, ps, bs, kv), (_, met) in cells.items():
        if be == "xqa":
            by_cfg[(bs, kv)][ps] = met
    for cfg in sorted(by_cfg):
        d = by_cfg[cfg]
        durs = {ps: d[ps].get("gpu__time_duration.sum") for ps in d}
        vals = [v for v in durs.values() if v]
        if len(vals) >= 2:
            spread = (max(vals) - min(vals)) / min(vals) * 100
            print(f"  bs={cfg[0]:>3} kv={cfg[1]:>6}  dur spread {spread:5.1f}%  "
                  f"{ {ps: round(durs[ps],1) for ps in sorted(durs) if durs[ps]} }")

    # ---- key question 2: XQA vs FlashInfer at matched (ps,bs,kv) -----------
    print("\n=== XQA vs FlashInfer (matched cells): global_ld and dram_rd ratio ===")
    for (be, ps, bs, kv) in sorted(cells):
        if be != "xqa":
            continue
        fi = cells.get(("flashinfer", ps, bs, kv))
        if not fi:
            continue
        xm = cells[(be, ps, bs, kv)][1]
        fm = fi[1]
        def ratio(metric):
            a, b = xm.get(metric), fm.get(metric)
            return f"{a/b:.2f}x" if a and b else "-"
        print(f"  ps={ps:>3} bs={bs:>3} kv={kv:>6}  "
              f"global_ld XQA/FI={ratio('sm__sass_inst_executed_op_global_ld.sum')}  "
              f"dram_rd XQA/FI={ratio('dram__bytes_read.sum')}  "
              f"dur XQA/FI={ratio('gpu__time_duration.sum')}")

    if tma_metrics:
        print(f"\nTMA/async metrics captured: {sorted(tma_metrics)}")
    else:
        print("\n(no TMA/async-named metrics in CSVs; see tma_metric_names.txt + full-set .ncu-rep)")


if __name__ == "__main__":
    main()
