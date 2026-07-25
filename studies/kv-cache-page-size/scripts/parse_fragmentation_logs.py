"""Steady-state per-step decode parser for the shared-prefix arms (reports 15 & 16).

Reads fr_<tag>.log engine debug logs and computes the correct steady-state decode latency:
  ms/step = B / (gen throughput tok/s) * 1000, median over full-concurrency (#running-req == B) steps.
NOT the harness whole-batch TPOT (which folds the cold-prefill admission ramp; see reports 14/15).

Tag convention (report 16): <backend>_ps<page>_bs<bs>  e.g. trtllm_ps16_bs32, FI_ps1_bs8, T_ps128_bs16.
Batch size B is parsed from the tag ('bs<N>'), defaulting to 32 for report-15 legacy tags
(T_ps1_frag, FI_ps128_frag, ...). The '#running-req: <B>' literal is the full-concurrency filter;
the throughput floor (4*B tok/s) drops cold/JIT-compile spike steps that survive at full concurrency.
"""
import argparse
import json
import re
import statistics
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = STUDY_ROOT / "data" / "raw" / "report-16-shared-prefix-engine" / "a1logs"

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
args = parser.parse_args()

def bs_of(tag):
    m = re.search(r"bs(\d+)", tag)
    return int(m.group(1)) if m else 32

def parse_arm(path, B):
    lat = []; walls = []; runreq = set()
    pat = re.compile(r"Decode batch, #running-req: (\d+),.*gen throughput \(token/s\): ([0-9.]+)")
    for line in open(path, errors="ignore"):
        m = pat.search(line)
        if m:
            rr = int(m.group(1)); thr = float(m.group(2))
            if rr == B and thr > 4 * B:      # full concurrency + drop ramp/JIT spikes
                lat.append(B / thr * 1000.0)
                runreq.add(rr)
        w = re.search(r"\[PHASE_B\] round \d+/\d+ wall=([0-9.]+)s", line)
        if w: walls.append(float(w.group(1)))
    return lat, walls, runreq

res = {}   # tag -> (median_ms, n, B)
print("arm".ljust(22), "steady ms/step (n)".rjust(22), "B".rjust(4), "walls s".rjust(22))
for path in sorted(args.data.glob("fr_*.log")):
    tag = path.stem[3:]
    B = bs_of(tag)
    lat, walls, runreq = parse_arm(path, B)
    st = statistics.median(lat) if lat else None
    res[tag] = (st, len(lat), B)
    flag = "" if lat else "  <-- NO DATA (check #running-req/backend)"
    print(tag.ljust(22),
          (f"{st:.2f} ({len(lat)})" if st else "NO DATA").rjust(22),
          str(B).rjust(4), str([round(w, 1) for w in walls]).rjust(22), flag)

# ---- Grouped page-flatness + ratios, per (backend, bs). Baseline = ps128 if present else min page. ----
def parse_tag(tag):
    m = re.match(r"([A-Za-z]+)_ps(\d+)_bs(\d+)", tag)
    if not m: return None
    return m.group(1), int(m.group(2)), int(m.group(3))

groups = {}   # (backend, bs) -> {page: ms}
for tag, (st, n, B) in res.items():
    p = parse_tag(tag)
    if p and st:
        be, page, bs = p
        groups.setdefault((be, bs), {})[page] = st

if groups:
    print("\n== page-flatness per (backend, bs): baseline=ps128 ==")
    for (be, bs) in sorted(groups):
        pages = groups[(be, bs)]
        base = pages.get(128) or pages[min(pages)]
        cells = " ".join(f"ps{pg}={ms:.2f}({(ms/base-1)*100:+.1f}%)" for pg, ms in sorted(pages.items()))
        spread = (max(pages.values()) / min(pages.values()) - 1) * 100
        print(f"  {be} bs{bs}: {cells}   [max spread {spread:.1f}%]")

# ---- Cross-backend contrast at matched bs: is ps1 (FI/Triton) faster than trtllm's best page? ----
by_bs = {}
for (be, bs), pages in groups.items():
    by_bs.setdefault(bs, {})[be] = pages
if by_bs:
    print("\n== cross-backend at matched bs (ms/step) ==")
    for bs in sorted(by_bs):
        parts = []
        for be, pages in sorted(by_bs[bs].items()):
            lo = min(pages.values()); lop = min(pages, key=pages.get)
            parts.append(f"{be}: best ps{lop}={lo:.2f}")
        print(f"  bs{bs}: " + " | ".join(parts))

# ---- report-15 legacy explicit pairs (back-compat) ----
legacy = [("T_ps1_frag","T_ps128_frag","Triton frag ps1-vs-ps128"),
          ("T_ps1_fresh","T_ps128_fresh","Triton fresh ps1-vs-ps128"),
          ("FI_ps1_frag","FI_ps128_frag","FI frag ps1-vs-ps128")]
printed = False
for a, b, lbl in legacy:
    if res.get(a, (None,))[0] and res.get(b, (None,))[0]:
        if not printed: print("\n== legacy report-15 pairs =="); printed = True
        va, vb = res[a][0], res[b][0]
        print(f"  {lbl}: {va:.2f} vs {vb:.2f} = {(va-vb)/vb*100:+.2f}%")

# ---- kvidx contiguity dumps (report 15 verification arms), if present ----
for f in (args.data / "fr_kvidx_frag.json", args.data / "fr_kvidx_fresh.json"):
    try:
        d = json.load(f.open())
        print(f.name, json.dumps(d)[:400])
    except Exception:
        pass
