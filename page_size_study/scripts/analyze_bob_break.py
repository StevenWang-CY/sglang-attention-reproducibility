#!/usr/bin/env python3
"""Analyze the adversarial break-search data (bobx_* tags): fine page grid, high batch / short ctx etc.

Tag: bobx_r{round}_b{B}_l{L}_{backend}_ps{ps}.  TPOT(ms) = median_decode_latency*1000.
Robust estimator: MIN across rounds (contention only inflates). For each (backend,B,L):
  - TPOT per page (min across rounds) + per-round spread for ps1 vs the winning page
  - the winning (fastest) page, ps1-vs-winner %, ps1-vs-ps128 %
  - FLAG cells where ps1 is NOT the fastest by >=2% (candidate) / >=5% (BREAK).
Run from anywhere.
"""
import json, glob, re
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "offline_batch_results" / "bench_one_batch_5060ti"
# bobx_ (Qwen3-VL-2B break ladder) and bobm_<model>_ (cross-model)
TAGX = re.compile(r"bobx_r(\d+)_b(\d+)_l(\d+)_(triton|flashinfer)_ps(\d+)")
TAGM = re.compile(r"bobm_([a-z0-9_]+?)_r(\d+)_b(\d+)_l(\d+)_(triton|flashinfer)_ps(\d+)")

rows = defaultdict(lambda: defaultdict(list))  # (model,be,B,L) -> ps -> [tpot per round]
for f in sorted(glob.glob(str(DATA / "bobx_*.jsonl")) + glob.glob(str(DATA / "bobm_*.jsonl"))):
    name = Path(f).name
    mm = TAGM.search(name)
    if mm:
        mdl, rnd, B, L, be, ps = mm[1], int(mm[2]), int(mm[3]), int(mm[4]), mm[5], int(mm[6])
    else:
        m = TAGX.search(name)
        if not m:
            continue
        mdl, rnd, B, L, be, ps = "q3vl_2b", int(m[1]), int(m[2]), int(m[3]), m[4], int(m[5])
    try:
        rec = json.loads(Path(f).read_text().strip().splitlines()[-1])
    except Exception:
        continue
    rows[(mdl, be, B, L)][ps].append(rec["median_decode_latency"] * 1000.0)

def mn(d, p):
    v = d.get(p, [])
    return min(v) if v else None

candidates = []
groups = sorted({(k[0], k[1]) for k in rows})
for (mdl, be) in groups:
    keys = sorted([k for k in rows if k[0] == mdl and k[1] == be], key=lambda k: (k[3], k[2]))
    if not keys:
        continue
    print(f"\n========== {mdl} / {be.upper()} ==========")
    for (m2, b2, B, L) in keys:
        d = rows[(mdl, be, B, L)]
        pages = sorted(d.keys())
        mins = {p: mn(d, p) for p in pages}
        avail = {p: v for p, v in mins.items() if v is not None}
        if not avail:
            continue
        win = min(avail, key=lambda p: avail[p])
        ps1 = avail.get(1)
        line = " ".join(f"ps{p}={avail[p]:.2f}" for p in pages if p in avail)
        tag = ""
        if ps1 is not None and win != 1:
            gap = (ps1 / avail[win] - 1) * 100
            tag = f"  <<< ps1 +{gap:.1f}% vs ps{win}"
            if gap >= 5:
                tag += "  *** BREAK ***"
                candidates.append((mdl, be, B, L, win, gap, "BREAK"))
            elif gap >= 2:
                tag += "  (candidate)"
                candidates.append((mdl, be, B, L, win, gap, "cand"))
        p128 = avail.get(128)
        rel128 = f" | ps1/ps128={ (ps1/p128-1)*100:+.1f}%" if (ps1 and p128) else ""
        print(f"B{B:<4}/L{L:<6} KV={B*L//1000}k  best=ps{win}  {line}{rel128}{tag}")

print("\n=== SUMMARY ===")
if candidates:
    for mdl, be, B, L, win, gap, kind in candidates:
        print(f"  [{kind}] {mdl} {be} B{B}/L{L}: ps1 +{gap:.1f}% slower than ps{win}")
else:
    print("  No cell where ps1 is >=2% slower than the best page. ps1 is fastest/tied everywhere.")
