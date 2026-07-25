#!/usr/bin/env python3
"""Analyze the Report-8 small-KV TRUE-batch bench_one_batch sweep (bobk_* tags).

Reads studies/kv-cache-page-size/data/raw/bench_one_batch_smallkv_5060ti/bobk_*.jsonl
Tag: bobk_r{round}_b{B}_l{L}_{backend}_ps{ps};  decode TPOT(ms) = median_decode_latency*1000.

For each (backend,B,L):
  - TPOT per page_size (min across rounds, robust; contention only inflates)
  - ps{1,8,32}-vs-ps128 PAIRED within each round, then mean±std across rounds
  - KV footprint/layer (B*L*4096 B; L2=32MB <=> B*L~8192) and the per-step DRAM-traffic
    ROOFLINE split (weights 4.10 GB vs KV B*L*114688 B) that explains the flatness.

Constants from a gray engine log (Qwen3-VL-2B, RTX 5060 Ti):
  weight mem usage = 4.10 GB ;  KV = 114688 B/token (28 layers * 4096 B/tok/layer).
Run from anywhere.
"""
import json, glob, re
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "studies" / "kv-cache-page-size" / "data" / "raw" / "bench_one_batch_smallkv_5060ti"
TAG = re.compile(r"bobk_r(\d+)_b(\d+)_l(\d+)_(triton|flashinfer)_ps(\d+)")

WEIGHT_GB = 4.10            # Qwen3-VL-2B decoder weights re-read per decode step (gray log)
KV_B_PER_TOK = 114688       # bytes/token across all 28 layers
L2_MB = 32

# rows[(be,B,L)][ps] = [tpot per round] ;  per-round paired ratios
rows = defaultdict(lambda: defaultdict(list))
perround = defaultdict(lambda: defaultdict(dict))   # (be,B,L)->rnd->{ps:tpot}
for f in sorted(glob.glob(str(DATA / "bobk_*.jsonl"))):
    m = TAG.search(Path(f).name)
    if not m:
        continue
    rnd, B, L, be, ps = int(m[1]), int(m[2]), int(m[3]), m[4], int(m[5])
    try:
        rec = json.loads(Path(f).read_text().strip().splitlines()[-1])
        tpot = rec["median_decode_latency"] * 1000.0
    except Exception:
        continue
    rows[(be, B, L)][ps].append(tpot)
    perround[(be, B, L)][rnd][ps] = tpot

def mn(d, p):
    v = d.get(p, [])
    return min(v) if v else None

def paired(key, p, base=128):
    """mean,std of (ps_p/ps_base - 1)*100 paired within each round."""
    rs = []
    for rnd, d in perround[key].items():
        if p in d and base in d and d[base]:
            rs.append((d[p] / d[base] - 1) * 100)
    if not rs:
        return None
    mean = sum(rs) / len(rs)
    sd = (sum((x - mean) ** 2 for x in rs) / (len(rs) - 1)) ** 0.5 if len(rs) > 1 else 0.0
    return mean, sd, len(rs)

PAGES = [1, 2, 4, 8, 16, 32, 128]
print(f"weights/step={WEIGHT_GB:.2f} GB   KV={KV_B_PER_TOK} B/tok   L2={L2_MB} MB "
      f"(per-layer L2-resident <=> B*L<=8192)\n")
for be in ("flashinfer", "triton"):
    keys = sorted([k for k in rows if k[0] == be], key=lambda k: (k[1] * k[2], k[1]))
    if not keys:
        continue
    print(f"================= {be.upper()} — TRUE batch, small KV =================")
    print(f"{'B/L':<12}{'KVlay':>7}{'KV/(KV+W)':>10}  "
          + "".join(f"ps{p:<7}" for p in PAGES) + "  ps1/128(paired)")
    for (b2, B, L) in keys:
        d = rows[(be, B, L)]
        kv_lay_mb = B * L * 4096 / 1e6
        kv_step_gb = B * L * KV_B_PER_TOK / 1e9
        frac = kv_step_gb / (kv_step_gb + WEIGHT_GB) * 100
        cells = "".join(f"{mn(d,p):<9.3f}" if mn(d, p) is not None else f"{'--':<9}" for p in PAGES)
        pr = paired((be, B, L), 1)
        prs = f"{pr[0]:+.2f}±{pr[1]:.2f}% (n{pr[2]})" if pr else "--"
        res = "  [L2-resident]" if B * L <= 8192 else ""
        print(f"B{B}/L{L:<8}{kv_lay_mb:>6.0f}M{frac:>9.0f}%  {cells}  {prs}{res}")
    print()

# headline summary: worst ps1-vs-best across all small-KV cells, per backend
print("=== HEADLINE: max ps1 penalty vs the fastest page (min-across-rounds) ===")
for be in ("flashinfer", "triton"):
    worst = None
    for (b2, B, L) in [k for k in rows if k[0] == be]:
        d = rows[(be, B, L)]
        avail = {p: mn(d, p) for p in d if mn(d, p) is not None}
        if 1 not in avail or len(avail) < 2:
            continue
        win = min(avail, key=lambda p: avail[p])
        gap = (avail[1] / avail[win] - 1) * 100
        if worst is None or gap > worst[0]:
            worst = (gap, B, L, win)
    if worst:
        g, B, L, win = worst
        verdict = "NO >=5% ps1 penalty" if g < 5 else "*** >=5% BREAK ***"
        print(f"  {be:<11}: worst ps1 = +{g:.2f}% vs ps{win} at B{B}/L{L}  -> {verdict}")
