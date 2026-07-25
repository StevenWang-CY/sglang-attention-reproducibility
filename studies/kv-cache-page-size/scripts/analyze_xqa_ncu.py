"""
analyze_xqa_ncu.py — parse ncu CSVs (exported via `ncu --import --page raw --csv`)
from the XQA TMA profile into per-cell hardware-counter tables for Report 6.

CSV format: WIDE — ncu banner (`==PROF==` lines) then a header row whose columns
are metric names (`gpu__time_duration.sum`, `...op_tma_ld.sum`, ...), one data row
per kernel invocation. Files: {backend}_ps{page}_bs{batch}_kv{seqlen}.csv
(+ full_* full-set reports, ignored here).

Answers: does XQA use TMA? does page_size move its counters? XQA vs FlashInfer.

Run from anywhere:  python3 studies/kv-cache-page-size/scripts/analyze_xqa_ncu.py [--data <dir>]
"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA = ROOT / "studies" / "kv-cache-page-size" / "data" / "raw" / "xqa_profile"
TAG_RE = re.compile(r"^(?P<be>xqa|flashinfer|triton)_ps(?P<ps>\d+)_bs(?P<bs>\d+)_kv(?P<kv>\d+)$")

# metric column -> short label
COLS = {
    "gpu__time_duration.sum": "dur_us",            # ns -> us below
    "dram__bytes_op_read.sum": "dram_rd_MB",       # bytes -> MB below
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "dram_%pk",
    "lts__t_sector_hit_rate.pct": "L2_hit%",
    "l1tex__t_sector_hit_rate.pct": "L1_hit%",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "sm_%pk",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "occ%",
    "sm__sass_inst_executed_op_global_ld.sum": "global_ld",
    "l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum": "TMA_ld_B",
    "l1tex__m_l1tex2xbar_req_cycles_active_op_tma.sum": "TMA_cyc",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ldgsts.sum": "ldgsts",
}


def num(s):
    if s is None:
        return None
    s = str(s).strip().strip('"').replace(",", "")
    if s in ("", "N/A", "n/a", "<null>"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


UNIT_DUR = {"ns": 1e-3, "us": 1.0, "ms": 1e3, "s": 1e6}                      # -> us
UNIT_BYTE = {"byte": 1e-6, "Kbyte": 1e-3, "Mbyte": 1.0, "Gbyte": 1e3, "Tbyte": 1e6}  # -> MB


def load_cell(path):
    """Return (metrics, kernel_name) for the max-duration kernel.
    ncu --import --page raw --csv emits a UNITS row (us/Mbyte/byte/inst) before
    data rows and auto-scales per cell; we read it and normalize dur->us, dram->MB."""
    lines = path.read_text(errors="replace").splitlines()
    hdr_i = next((i for i, l in enumerate(lines) if l.startswith('"ID"') or l.startswith("ID,")), None)
    if hdr_i is None:
        return None
    rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(lines[hdr_i:])]
    if not rows:
        return None
    units = {}
    if num(rows[0].get("gpu__time_duration.sum")) is None:   # first row is the units row
        units, rows = rows[0], rows[1:]
    rows = [r for r in rows if num(r.get("gpu__time_duration.sum")) is not None]
    if not rows:
        return None
    best = max(rows, key=lambda r: num(r.get("gpu__time_duration.sum")))
    out = {}
    for c, lab in COLS.items():
        v = num(best.get(c))
        u = (units.get(c) or "").strip()
        if v is not None:
            if lab == "dur_us":
                v *= UNIT_DUR.get(u, 1.0)
            elif lab == "dram_rd_MB":
                v *= UNIT_BYTE.get(u, 1.0)
        out[lab] = v
    return out, best.get("Kernel Name", "?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    args = ap.parse_args()
    data = Path(args.data)

    cells = {}
    kerns = {}
    for p in sorted(data.glob("*.csv")):
        m = TAG_RE.match(p.stem)
        if not m:
            continue
        res = load_cell(p)
        if not res:
            print(f"  (unparsed: {p.name})")
            continue
        met, kn = res   # already normalized (dur->us, dram->MB) in load_cell
        key = (m["be"], int(m["ps"]), int(m["bs"]), int(m["kv"]))
        cells[key] = met
        kerns[key] = kn

    if not cells:
        print(f"no parseable cells in {data}")
        return

    order = ["dur_us", "dram_rd_MB", "dram_%pk", "L2_hit%", "L1_hit%", "sm_%pk",
             "occ%", "global_ld", "ldgsts", "TMA_ld_B", "TMA_cyc"]
    hdr = ["backend", "ps", "bs", "kv"] + order
    print("  ".join(f"{h:>10}" for h in hdr))
    print("-" * (12 * len(hdr)))
    def fmt(v):
        if v is None:
            return "-"
        if abs(v) >= 1e4:
            return f"{v:.3e}"
        return f"{v:.3g}"
    for key in sorted(cells):
        be, ps, bs, kv = key
        m = cells[key]
        print("  ".join(f"{str(x):>10}" for x in [be, ps, bs, kv] + [fmt(m.get(c)) for c in order]))

    # ---- Q1: does XQA ever use TMA? -----------------------------------------
    print("\n=== Q1: XQA TMA usage across ALL cells ===")
    xqa_tma = [(k, cells[k].get("TMA_ld_B"), cells[k].get("TMA_cyc")) for k in cells if k[0] == "xqa"]
    nonzero = [k for k, b, c in xqa_tma if (b or 0) > 0 or (c or 0) > 0]
    print(f"  XQA cells: {len(xqa_tma)}; with ANY TMA activity (load-bytes>0 or tma-cycles>0): {len(nonzero)}")
    if nonzero:
        for k in nonzero:
            print(f"    TMA>0 at {k}: load_B={cells[k]['TMA_ld_B']} cyc={cells[k]['TMA_cyc']}")
    else:
        print("  -> XQA uses NO TMA in any cell (kernel_mha loads KV via cp.async/LDGSTS).")
    fi_tma_nonzero = [k for k in cells if k[0] == "flashinfer" and ((cells[k].get("TMA_ld_B") or 0) > 0 or (cells[k].get("TMA_cyc") or 0) > 0)]
    print(f"  FlashInfer cells with TMA activity: {len(fi_tma_nonzero)}")

    # ---- Q2: XQA counter spread across page_size (per bs,kv) ----------------
    print("\n=== Q2: XQA dur/dram spread across page_size {16,32,64,128} per (bs,kv) ===")
    bycfg = defaultdict(dict)
    for (be, ps, bs, kv), m in cells.items():
        if be == "xqa":
            bycfg[(bs, kv)][ps] = m
    for cfg in sorted(bycfg):
        d = bycfg[cfg]
        durs = {ps: d[ps].get("dur_us") for ps in d if d[ps].get("dur_us")}
        if len(durs) >= 2:
            sp = (max(durs.values()) - min(durs.values())) / min(durs.values()) * 100
            dram = {ps: round(d[ps].get("dram_%pk") or 0) for ps in sorted(d)}
            print(f"  bs={cfg[0]:>3} kv={cfg[1]:>6}: dur spread {sp:5.1f}%  dram%pk={dram}")

    # ---- Q3: XQA vs FlashInfer matched cells -------------------------------
    print("\n=== Q3: XQA vs FlashInfer (matched ps,bs,kv) ===")
    print(f"  {'cell':>22}  {'dur XQA/FI':>11}  {'dram%pk X|F':>12}  {'global_ld X|F':>16}")
    for (be, ps, bs, kv) in sorted(cells):
        if be != "xqa":
            continue
        fi = cells.get(("flashinfer", ps, bs, kv))
        if not fi:
            continue
        x = cells[(be, ps, bs, kv)]
        dr = (x.get("dur_us") and fi.get("dur_us")) and x["dur_us"] / fi["dur_us"]
        print(f"  ps{ps:>3} bs{bs:>3} kv{kv:>6}  {('%.3f' % dr) if dr else '-':>11}  "
              f"{round(x.get('dram_%pk') or 0)}|{round(fi.get('dram_%pk') or 0):<6}  "
              f"{int(x.get('global_ld') or 0)}|{int(fi.get('global_ld') or 0)}")


if __name__ == "__main__":
    main()
