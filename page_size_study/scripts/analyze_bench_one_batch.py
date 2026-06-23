#!/usr/bin/env python3
"""Analyze bench_one_batch (TRUE-batch) page_size sweep.

Reads offline_batch_results/bench_one_batch_5060ti/bob_*.jsonl
Tag format: bob_r{round}_b{B}_l{L}_{backend}_ps{ps}
Decode TPOT (ms/tok) = median_decode_latency * 1000.

Reports, per (backend, B, L):
  - TPOT per page_size, per round
  - ps{1,8,32}-vs-ps128 PAIRED within each round, then mean across rounds

Run from anywhere: paths computed from __file__ (parents[2] = repo root).
"""
import json, glob, re
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]  # repo root (sglang_log/)
DATA = ROOT / "offline_batch_results" / "bench_one_batch_5060ti"

TAG = re.compile(r"bob_r(\d+)_b(\d+)_l(\d+)_(triton|flashinfer)_ps(\d+)")

# rows[(backend,B,L)][round][ps] = tpot_ms
rows = defaultdict(lambda: defaultdict(dict))
for f in sorted(glob.glob(str(DATA / "bob_*.jsonl"))):
    m = TAG.search(Path(f).name)
    if not m:
        continue
    rnd, B, L, be, ps = int(m[1]), int(m[2]), int(m[3]), m[4], int(m[5])
    try:
        rec = json.loads(Path(f).read_text().strip().splitlines()[-1])
    except Exception:
        continue
    tpot = rec["median_decode_latency"] * 1000.0
    rows[(be, B, L)][rnd][ps] = tpot

PAGES = [1, 8, 32, 128]
print(f"{'config':<28}{'rnd':<4}" + "".join(f"ps{p:<8}" for p in PAGES)
      + "  ps1/128  ps8/128  ps32/128")
print("-" * 100)

# aggregate paired ratios across rounds
agg = defaultdict(lambda: defaultdict(list))  # (be,B,L) -> ps -> [ratio% vs 128]
for key in sorted(rows.keys(), key=lambda k: (k[0], k[1] * k[2])):
    be, B, L = key
    for rnd in sorted(rows[key]):
        d = rows[key][rnd]
        cells = "".join(f"{d.get(p, float('nan')):<10.3f}" if p in d else f"{'--':<10}" for p in PAGES)
        ratios = {}
        base = d.get(128)
        for p in (1, 8, 32):
            if base and p in d:
                r = (d[p] / base - 1) * 100
                ratios[p] = r
                agg[key][p].append(r)
        rstr = "".join(f"{ratios[p]:+6.1f}%  " if p in ratios else f"{'--':<8}" for p in (1, 8, 32))
        print(f"{be+' B'+str(B)+' L'+str(L):<28}{rnd:<4}{cells}  {rstr}")

print("\n=== PAIRED mean across rounds (ps vs ps128, %) ===")
print(f"{'config':<30}{'ps1':<12}{'ps8':<12}{'ps32':<12}{'KV(B*L)':<10}")
print("-" * 80)
for key in sorted(agg.keys(), key=lambda k: (k[0], k[1] * k[2])):
    be, B, L = key
    def fmt(p):
        v = agg[key].get(p, [])
        if not v:
            return "--"
        mean = sum(v) / len(v)
        if len(v) > 1:
            sd = (sum((x - mean) ** 2 for x in v) / (len(v) - 1)) ** 0.5
            return f"{mean:+.1f}±{sd:.1f}"
        return f"{mean:+.1f}"
    print(f"{be+' B'+str(B)+' L'+str(L):<30}{fmt(1):<12}{fmt(8):<12}{fmt(32):<12}{B*L:<10}")
