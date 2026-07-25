#!/usr/bin/env python3
"""Analyze Report-9 vLLM TRUE-batch block_size sweep (FlashInfer backend, CUDA graph ON, no prefix sharing).

Reads studies/kv-cache-page-size/data/raw/vllm_smallkv_5060ti/vllm_b{B}_l{L}_blk{blk}.json (+ probe_blk*.json).
Each json: {block_size, batch_size, input_len, backend, enforce_eager, tpot_median_ms, tpot_std_ms, tpots,...}
vLLM-FlashInfer supports block_size in {16,32,64,128} (1 and 8 rejected: 'block_size not supported').

Per (B,L): TPOT per block_size, and smallest-supported-block (16) vs the FASTEST block (%) — the
analog of the 'ps1 disadvantage' the PI asked for (ps1 itself is inexpressible in vLLM-FlashInfer).
Flags any cell where block16 is >=5% slower than the best block. Run from anywhere.
"""
import json, glob, re
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[3]
D = ROOT / "studies" / "kv-cache-page-size" / "data" / "raw" / "vllm_smallkv_5060ti"
TAG = re.compile(r"vllm_b(\d+)_l(\d+)_blk(\d+)")
BLK = [16, 32, 64, 128]

cells = defaultdict(dict)   # (B,L) -> {blk: (tpot, std)}
for f in sorted(glob.glob(str(D / "vllm_b*_l*_blk*.json"))):
    m = TAG.search(Path(f).name)
    if not m:
        continue
    B, L, blk = int(m[1]), int(m[2]), int(m[3])
    try:
        d = json.loads(Path(f).read_text())
    except Exception:
        continue
    if d.get("tpot_median_ms", 0) > 0:
        cells[(B, L)][blk] = (d["tpot_median_ms"], d.get("tpot_std_ms", 0.0))

# probe (block_size support)
print("=== block_size support (vLLM + FlashInfer, sm120) ===")
for b in [1, 8, 16, 32, 64, 128]:
    f = D / f"probe_blk{b}.json"
    if f.exists():
        d = json.loads(f.read_text())
        ok = d.get("tpot_median_ms", 0) > 0
        print(f"  block_size={b:<4} {'OK' if ok else 'REJECTED (' + str(d.get('error','?')) + ')'}")

print("\n=== TRUE-batch TPOT (ms) vs block_size; CUDA graph ON; FlashInfer; no prefix sharing ===")
print(f"{'B / L':<14}{'KV tok':>8}  " + "".join(f"blk{b:<8}" for b in BLK) + "  blk16-vs-best  best")
print("-" * 86)
worst = None
for (B, L) in sorted(cells, key=lambda k: (k[0], k[1])):
    d = cells[(B, L)]
    row = "".join(f"{d[b][0]:<10.2f}" if b in d else f"{'--':<10}" for b in BLK)
    avail = {b: v[0] for b, v in d.items()}
    tag = ""
    if avail:
        best = min(avail, key=lambda b: avail[b])
        if 16 in avail:
            gap = (avail[16] / avail[best] - 1) * 100
            tag = f"{gap:+.1f}% vs blk{best}"
            if worst is None or gap > worst[0]:
                worst = (gap, B, L, best)
            if gap >= 5:
                tag += "  *** >=5% ***"
    print(f"B{B}/L{L:<8}{B*L:>8}  {row}  {tag}")

print("\n=== VERDICT ===")
if worst:
    g, B, L, best = worst
    v = "NO >=5% small-block penalty" if g < 5 else "*** >=5% BREAK ***"
    print(f"  worst block16 = {g:+.1f}% vs blk{best} (at B{B}/L{L})  -> {v}")
print("  NOTE: block_size=1 and 8 are REJECTED by vLLM-FlashInfer on sm120 ('block_size not supported'),")
print("  so the literal ps1 pathology is structurally inexpressible; smallest expressible page is 16.")
