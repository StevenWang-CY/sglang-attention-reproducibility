#!/usr/bin/env python3
"""
analyze_offwall.py — tables for Report 13 ("true batch off the DRAM wall").

Reads offline_batch_results/offwall_profile/ (ncu CSVs + latency/pagecost JSONs)
and offline_batch_results/bench_one_batch_offwall_5060ti/ (engine JSONLs), prints:

  T1  Phase-1 ncu grid (distinct, cold): DRAM% / L2-hit / SM% / occupancy / grid /
      waves / effective GB/s per (heads, B, L) — with the <50% DRAM cells marked.
  T2  page ladder (native page_size 1..128, FlashInfer): per-cell latency by page,
      ps1-vs-ps128 and max spread, median & min across rounds.
  T3  worst-case ps1 fragmentation (contig / block128 / scatter), per cell x rounds.
  T4  ncu attribution for T3 patterns + ps1-vs-ps128 counters (sectors/request etc).
  T5  XQA ladder (page>=16).
  T6  engine TPOT by page (bench_one_batch, graph ON), paired per round.
  T7  engine ncu (decode-attn kernel inside bench_one_batch, per-launch median).

Run from anywhere:  python3 page_size_study/scripts/analyze_offwall.py
"""
import csv, glob, json, re, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "offline_batch_results" / "offwall_profile"
DBOB = ROOT / "offline_batch_results" / "bench_one_batch_offwall_5060ti"
HEAD_DIM = 128
SMS = 36            # RTX 5060 Ti (GB206) multiprocessors
DRAM_THRESH = 50.0  # the user's "not DRAM-bound" bar

UNIT_DUR = {"ns": 1e-3, "us": 1.0, "usecond": 1.0, "ms": 1e3, "s": 1e6}
UNIT_BYTE = {"byte": 1e-6, "Kbyte": 1e-3, "Mbyte": 1.0, "Gbyte": 1e3, "Tbyte": 1e6}

COLS = {
    "gpu__time_duration.sum": "dur_us",
    "dram__bytes_op_read.sum": "dram_rd_MB",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "dram_pct",
    "lts__t_sector_hit_rate.pct": "l2_hit",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed": "l2_bw",
    "l1tex__t_sector_hit_rate.pct": "l1_hit",
    "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio": "sect_per_req",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "sm_pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "occ_pct",
    "launch__grid_size": "grid",
    "launch__block_size": "block",
    "launch__waves_per_multiprocessor": "waves",
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


def load_rows(path):
    """All numeric data rows of a raw-page ncu CSV, normalized (dur->us, bytes->MB)."""
    lines = path.read_text(errors="replace").splitlines()
    hdr = next((i for i, l in enumerate(lines) if l.startswith('"ID"') or l.startswith("ID,")), None)
    if hdr is None:
        return []
    rows = [dict(r) for r in csv.DictReader(lines[hdr:])]
    if not rows:
        return []
    units = {}
    if num(rows[0].get("gpu__time_duration.sum")) is None:
        units, rows = rows[0], rows[1:]
    out = []
    for r in rows:
        if num(r.get("gpu__time_duration.sum")) is None:
            continue
        m = {}
        for c, lab in COLS.items():
            v = num(r.get(c))
            u = (units.get(c) or "").strip()
            if v is not None:
                if lab == "dur_us":
                    v *= UNIT_DUR.get(u, 1.0)
                elif lab == "dram_rd_MB":
                    v *= UNIT_BYTE.get(u, 1.0)
            m[lab] = v
        m["kernel"] = r.get("Kernel Name", "?")
        out.append(m)
    return out


def main_kernel(path):
    """The attention kernel's row (microbench --single). Prefer rows whose kernel name
    is the decode/prefill attention kernel (at tiny tensor-core cells the split-merge
    kernel can out-last the main kernel, breaking the bare max-duration convention);
    fall back to max duration."""
    rows = load_rows(path)
    if not rows:
        return None
    attn = [r for r in rows if "BatchDecodeWithPagedKVCache" in r.get("kernel", "")
            or "BatchPrefillWithPagedKVCache" in r.get("kernel", "")]
    return max(attn or rows, key=lambda r: r["dur_us"])


def med_kernel(path):
    """Per-launch median over all rows (engine ncu: -k filtered, N launches).
    Also returns the launch-grid distribution — the engine may split one logical
    decode into several fixed-size launches (chunked split-KV), so per-launch
    medians describe a launch, not necessarily the whole (B,L) cell."""
    rows = load_rows(path)
    if not rows:
        return None, 0
    m = {}
    for lab in list(COLS.values()):
        vs = [r[lab] for r in rows if r.get(lab) is not None]
        m[lab] = statistics.median(vs) if vs else None
    from collections import Counter
    m["grid_dist"] = dict(Counter(int(r["grid"]) for r in rows if r.get("grid") is not None))
    m["sum_rd_MB"] = sum(r["dram_rd_MB"] or 0 for r in rows)
    return m, len(rows)


def gbps(m):
    if m and m.get("dram_rd_MB") and m.get("dur_us"):
        return m["dram_rd_MB"] * 1e3 / m["dur_us"]
    return None


def fmt(v, nd=1):
    return "-" if v is None else (f"{v:,.{nd}f}" if isinstance(v, float) else str(v))


# ---------------- T1: phase-1 ncu grid ----------------
def t1():
    pat = re.compile(r"flashinfer(tc)?_(distinct|shared)_(all|none)_h(\d+)k(\d+)_ps(\d+)_bs(\d+)_kv(\d+)\.csv$")
    cells = {}
    for f in sorted(D.glob("flashinfer*_h*_ps*_bs*_kv*.csv")):
        m = pat.search(f.name)
        if not m:
            continue
        k = (m[1] or "", m[2], m[3], int(m[4]), int(m[5]), int(m[6]), int(m[7]), int(m[8]))
        r = main_kernel(f)
        if r:
            cells[k] = r
    if not cells:
        print("T1: no grid csvs yet")
        return
    print("\n=== T1. ncu grid — distinct decode kernel (max-duration row; tc = tensor-core wrapper) ===")
    print(f"{'heads':>9} {'cache':>5} {'ps':>4} {'B':>3} {'L':>6} {'fp_MB':>7} | "
          f"{'dur_us':>8} {'DRAM%':>6} {'GB/s':>6} {'L2hit%':>6} {'SM%':>5} {'occ%':>5} "
          f"{'grid':>6} {'waves':>6} | off-wall?")
    for k in sorted(cells, key=lambda k: (k[3], k[4], k[0], k[1], k[2], k[5], k[6], k[7])):
        tc, mode, cache, qh, kh, ps, B, L = k
        m = cells[k]
        fp = B * L * 2 * kh * HEAD_DIM * 2 / 1e6
        off = "  <-- OFF-WALL" if (m["dram_pct"] or 100) < DRAM_THRESH else ""
        print(f" h{qh}k{kh}{tc:<2} {cache:>5} {ps:>4} {B:>3} {L:>6} {fp:>7.1f} | "
              f"{fmt(m['dur_us']):>8} {fmt(m['dram_pct']):>6} {fmt(gbps(m),0):>6} "
              f"{fmt(m['l2_hit']):>6} {fmt(m['sm_pct']):>5} {fmt(m['occ_pct']):>5} "
              f"{fmt(m['grid'],0):>6} {fmt(m['waves'],2):>6} |{off}")
    qual = [(k, cells[k]["dram_pct"]) for k in cells
            if k[2] == "all" and (cells[k]["dram_pct"] or 100) < DRAM_THRESH]
    print(f"\n  cold cells with DRAM% < {DRAM_THRESH:.0f}: {len(qual)}")
    for k, d in sorted(qual, key=lambda x: x[1]):
        print(f"    h{k[3]}k{k[4]}{k[0]} ps{k[5]} B{k[6]} L{k[7]}  ->  {d:.1f}%")


# ---------------- T2: native page ladder ----------------
def lat_files(prefix):
    out = {}
    for f in sorted(D.glob(f"{prefix}_r*.json")):
        r = int(re.search(r"_r(\d+)\.json$", f.name)[1])
        out[r] = json.loads(f.read_text())
    return out


def t2(prefix, title):
    rounds = lat_files(prefix)
    if not rounds:
        print(f"{title}: no data yet")
        return
    cells = {}
    for r, data in rounds.items():
        for v in data.values():
            if "latency_ms" not in v:
                continue
            cells.setdefault((v["batch_size"], v["seq_len"]), {}).setdefault(
                v["page_size"], {})[r] = v["latency_ms"]
    pss = sorted({ps for c in cells.values() for ps in c})
    print(f"\n=== {title} (median across {len(rounds)} rounds; per-cell best page ==*) ===")
    print(f"{'B':>3} {'L':>6} | " + " ".join(f"ps{p:>4}" for p in pss) +
          " | ps1/ps128-1  maxspread  [min-across-rounds ps1/ps128-1]")
    for (B, L) in sorted(cells):
        med = {ps: statistics.median(rs.values()) for ps, rs in cells[(B, L)].items()}
        mn = {ps: min(rs.values()) for ps, rs in cells[(B, L)].items()}
        best = min(med.values())
        row = " ".join((f"{med[p]:.4f}" + ("*" if med[p] == best else " ")) if p in med else "   -   "
                       for p in pss)
        r1 = (med[1] / med[128] - 1) * 100 if 1 in med and 128 in med else None
        r1m = (mn[1] / mn[128] - 1) * 100 if 1 in mn and 128 in mn else None
        sp = (max(med.values()) / min(med.values()) - 1) * 100
        print(f"{B:>3} {L:>6} | {row} | {fmt(r1,2):>10}%  {sp:>8.2f}%  [{fmt(r1m,2):>7}%]")


# ---------------- T3: fragmentation patterns ----------------
def t3():
    print("\n=== T3. worst-case ps1 fragmentation (kernel latency vs contig; per round) ===")
    for heads in ("h16k8", "h16k2"):
        rounds = lat_files(f"pagecost_offwall_{heads}")
        if not rounds:
            continue
        cells = {}
        for r, data in rounds.items():
            for v in data.values():
                if "latency_ms" not in v:
                    continue
                cells.setdefault((v["batch_size"], v["seq_len"]), {}).setdefault(
                    v["pattern"], {})[r] = v["latency_ms"]
        print(f"  -- {heads} --")
        for (B, L) in sorted(cells):
            pats = cells[(B, L)]
            if "contig" not in pats:
                continue
            line = f"   B{B:>3} L{L:>6}  contig={statistics.median(pats['contig'].values()):.4f}ms"
            for p in ("block128", "scatter"):
                if p in pats:
                    per = [(pats[p][r] / pats["contig"][r] - 1) * 100
                           for r in pats[p] if r in pats["contig"]]
                    line += f"  {p}: {statistics.median(per):+.2f}% (rounds: " + \
                            ",".join(f"{x:+.1f}" for x in sorted(per)) + ")"
            print(line)


# ---------------- T4: ncu attribution (patterns + ps1 vs ps128) ----------------
def t4():
    pat = re.compile(r"pagecost(tc)?_(contig|block128|scatter)_(all|none)_h(\d+)k(\d+)_bs(\d+)_kv(\d+)\.csv$")
    cells = {}
    # rep2/ overlays the first pass: one pagecosttc window overlapped concurrent engine
    # runs (contention-dilated 2.1x phantom on the scatter cell — refuted by the clean pass)
    for base in (D, D / "rep2"):
        for f in sorted(base.glob("pagecost*_h*_bs*_kv*.csv")):
            m = pat.search(f.name)
            if not m:
                continue
            r = main_kernel(f)
            if r:
                cells[(f"h{m[4]}k{m[5]}{m[1] or ''}", m[2], int(m[6]), int(m[7]))] = r
    if cells:
        print("\n=== T4a. ncu — contig vs scatter (cold), token-index ps1 pool ===")
        print(f"{'heads':>7} {'B':>3} {'L':>6} {'pattern':>9} | {'dur_us':>8} {'DRAM%':>6} "
              f"{'L2hit%':>6} {'sect/req':>8} {'SM%':>5} {'grid':>6}")
        for k in sorted(cells):
            h, p, B, L = k
            m = cells[k]
            print(f"{h:>7} {B:>3} {L:>6} {p:>9} | {fmt(m['dur_us']):>8} {fmt(m['dram_pct']):>6} "
                  f"{fmt(m['l2_hit']):>6} {fmt(m['sect_per_req'],2):>8} {fmt(m['sm_pct']):>5} "
                  f"{fmt(m['grid'],0):>6}")
    # ps1 vs ps128 counter deltas from the grid files (incl. tensor-core arm)
    pat2 = re.compile(r"flashinfer(tc)?_distinct_all_h(\d+)k(\d+)_ps(\d+)_bs(\d+)_kv(\d+)\.csv$")
    g = {}
    for f in sorted(D.glob("flashinfer*_distinct_all_h*_ps*_bs*_kv*.csv")):
        m = pat2.search(f.name)
        if m:
            r = main_kernel(f)
            if r:
                g[(f"h{m[2]}k{m[3]}{m[1] or ''}", int(m[4]), int(m[5]), int(m[6]))] = r
    pairs = sorted({(h, B, L) for (h, ps, B, L) in g if ps == 1 and (h, 128, B, L) in g})
    if pairs:
        print("\n=== T4b. ncu — ps1 vs ps128 (cold): duration & DRAM-read ratios ===")
        print(f"{'heads':>7} {'B':>3} {'L':>6} | {'dur ps1/128':>11} {'rdMB ps1/128':>12} "
              f"{'DRAM% 1|128':>12} {'grid 1|128':>12}")
        for (h, B, L) in pairs:
            a, b = g[(h, 1, B, L)], g[(h, 128, B, L)]
            print(f"{h:>7} {B:>3} {L:>6} | {a['dur_us']/b['dur_us']:>11.3f} "
                  f"{(a['dram_rd_MB']/b['dram_rd_MB']) if b['dram_rd_MB'] else 0:>12.3f} "
                  f"{fmt(a['dram_pct']):>5}|{fmt(b['dram_pct']):<6} "
                  f"{fmt(a['grid'],0):>5}|{fmt(b['grid'],0):<6}")


# ---------------- T6: engine TPOT ----------------
def t6():
    files = sorted(DBOB.glob("bobo_*.jsonl")) if DBOB.exists() else []
    if not files:
        print("T6: no engine jsonl yet")
        return
    pat = re.compile(r"bobo_(q25_3b|q3vl_2b)_r(\d+)_b(\d+)_l(\d+)_(flashinfer|triton)_ps(\d+)")
    cells = {}
    for f in files:
        m = pat.search(f.name)
        if not m:
            continue
        try:
            recs = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        except json.JSONDecodeError:
            continue
        lat = next((r["median_decode_latency"] for r in recs if "median_decode_latency" in r), None)
        if lat is None:
            continue
        mdl, r, B, L, be, ps = m[1], int(m[2]), int(m[3]), int(m[4]), m[5], int(m[6])
        cells.setdefault((mdl, B, L, be), {}).setdefault(ps, {})[r] = lat * 1000
    print("\n=== T6. engine bench_one_batch TPOT (ms/token; median across rounds, * = best) ===")
    for k in sorted(cells):
        mdl, B, L, be = k
        med = {ps: statistics.median(rs.values()) for ps, rs in cells[k].items()}
        mn = {ps: min(rs.values()) for ps, rs in cells[k].items()}
        best = min(med.values())
        pss = sorted(med)
        row = "  ".join(f"ps{p}={med[p]:.3f}{'*' if med[p]==best else ''}" for p in pss)
        extra = ""
        if 1 in med and 128 in med:
            extra = (f"  | ps1 vs ps128: med {(med[1]/med[128]-1)*100:+.2f}% "
                     f"min {(mn[1]/mn[128]-1)*100:+.2f}%")
        print(f"  {mdl:8s} B{B:<3} L{L:<5} {be:10s} {row}{extra}")


# ---------------- T7: engine ncu ----------------
def t7():
    pat = re.compile(r"bob_(q25_3b|q3vl_2b)_all_ps(\d+)_bs(\d+)_kv(\d+)\.csv$")
    found = False
    for f in sorted(D.glob("bob_*_all_ps*_bs*_kv*.csv")):
        m = pat.search(f.name)
        if not m:
            continue
        med, n = med_kernel(f)
        if not med:
            continue
        if not found:
            print("\n=== T7. ncu INSIDE bench_one_batch (decode-attn kernel, per-launch median) ===")
            print(f"{'model':>8} {'ps':>4} {'B':>3} {'L':>6} {'#launch':>7} | {'dur_us':>8} "
                  f"{'DRAM%':>6} {'GB/s':>6} {'L2hit%':>6} {'SM%':>5} {'occ%':>5} {'grid':>6} {'waves':>6}")
            found = True
        off = "  <-- OFF-WALL" if (med["dram_pct"] or 100) < DRAM_THRESH else ""
        print(f"{m[1]:>8} {m[2]:>4} {m[3]:>3} {m[4]:>6} {n:>7} | {fmt(med['dur_us']):>8} "
              f"{fmt(med['dram_pct']):>6} {fmt(gbps(med),0):>6} {fmt(med['l2_hit']):>6} "
              f"{fmt(med['sm_pct']):>5} {fmt(med['occ_pct']):>5} {fmt(med['grid'],0):>6} "
              f"{fmt(med['waves'],2):>6} |{off}  grids={med['grid_dist']} sum_rd={med['sum_rd_MB']:.1f}MB")
    if not found:
        print("T7: no engine ncu csvs yet")


if __name__ == "__main__":
    t1()
    t2("lat_offwall_h16k8", "T2a. FlashInfer page ladder — GQA-8 (16q/8kv), distinct")
    t2("lat_offwall_h16k2", "T2b. FlashInfer page ladder — GQA-2 (16q/2kv = Qwen2.5-3B), distinct")
    t2("lat_offwall_h16k2tc", "T2c. FlashInfer page ladder — GQA-2 TENSOR-CORE kernel (sglang's pick)")
    t3()
    t4()
    t2("lat_offwall_xqa", "T5. XQA page ladder (page>=16), GQA-8 distinct")
    t6()
    t7()
