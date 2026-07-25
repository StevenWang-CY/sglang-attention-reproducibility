#!/usr/bin/env python3
"""
Report-14 analysis tables for shared-prefix placement, synthetic layout controls, and the historical
whole-call-versus-steady-decode discrepancy.

Reads studies/kv-cache-page-size/data/raw/report-14-shared-prefix-page-kernel-rtx5060ti/:
  * pc_<be>_<reuse>_ps<ps>_<pattern>_h<QH>k<KH>_bs<B>_kv<L>.csv  -- ncu single-kernel
  * lat_tri_<reuse>.json / lat_*.json                            -- CUDA-event latency sweeps
and (optional) /tmp-style report-3 reproduction JSONs + Step-0 kvidx JSONs if passed.

Prints:
  K1  synthetic scatter-vs-block128 layout contrast per backend x reuse x cell. These layouts bound
      sensitivity; they are not assumed to equal the real engine's ps1/ps128 placement.
  K2  engine shared-prefix kv_indices contiguity at ps1 vs ps128.
  K3  report-3 whole-call TPOT reproduction, interpreted alongside per-step steady decode.

Reuses analyze_offwall.py's ncu CSV loader (COLS/num/load_rows/main_kernel).
Run:  python3 studies/kv-cache-page-size/scripts/analyze_report_14_shared_prefix.py
"""
import csv, glob, json, re, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
D = ROOT / "studies" / "kv-cache-page-size" / "data" / "raw" / "report-14-shared-prefix-page-kernel-rtx5060ti"

# -- reuse the ncu CSV parsing from analyze_offwall.py -----------------------------
import importlib.util
_ao = importlib.util.spec_from_file_location("_ao", Path(__file__).with_name("analyze_offwall.py"))
_aom = importlib.util.module_from_spec(_ao); _ao.loader.exec_module(_aom)
load_rows, num = _aom.load_rows, _aom.num


def kern_row(path):
    """The attention kernel row: prefer the Triton stage1/stage2 or FlashInfer decode/prefill
    kernel; fall back to the max-duration row. For Triton, sum stage1+stage2 durations."""
    rows = load_rows(path)
    if not rows:
        return None
    tri = [r for r in rows if "_fwd_" in r.get("kernel", "")]
    fi = [r for r in rows if "BatchDecodeWithPagedKVCache" in r.get("kernel", "")
          or "BatchPrefillWithPagedKVCache" in r.get("kernel", "")]
    if tri:
        # combine stage1 (the gather) + stage2 (reduce); report stage1 counters (the gather)
        s1 = max((r for r in tri if "stage1" in r["kernel"]), key=lambda r: r["dur_us"], default=None)
        dur = sum(r["dur_us"] for r in tri if r.get("dur_us"))
        base = dict(s1 or tri[0]); base["dur_us"] = dur
        base["kernel"] = "triton_stage1+2"
        return base
    return max(fi or rows, key=lambda r: r["dur_us"])


PAT = re.compile(r"pc_(?P<be>\w+?)_(?P<reuse>shared|distinct)_ps(?P<ps>\d+)_(?P<pat>contig|block128|scatter)_h(?P<qh>\d+)k(?P<kh>\d+)_bs(?P<b>\d+)_kv(?P<l>\d+)\.csv")


def k1_microbench_pk():
    print("\n=== K1  synthetic kernel layout contrast (scatter vs block128) ===")
    print("   backend  reuse    B     L | contig  block128 scatter |  p_k%  | sect/req c->s | DRAM% | L2hit%")
    cells = {}
    for f in sorted(D.glob("pc_*.csv")):
        m = PAT.match(f.name)
        if not m:
            continue
        r = kern_row(f)
        if not r:
            continue
        key = (m["be"], m["reuse"], int(m["b"]), int(m["l"]))
        cells.setdefault(key, {})[m["pat"]] = r
    for (be, reuse, B, L), pats in sorted(cells.items()):
        c, bk, sc = pats.get("contig"), pats.get("block128"), pats.get("scatter")
        def dur(x): return x["dur_us"] if x else None
        pk = (dur(sc) / dur(bk) - 1) * 100 if sc and bk else None
        spr_c = c["sect_per_req"] if c and c.get("sect_per_req") is not None else None
        spr_s = sc["sect_per_req"] if sc and sc.get("sect_per_req") is not None else None
        dram = sc["dram_pct"] if sc else None
        l2 = sc["l2_hit"] if sc else None
        print(f"   {be:8} {reuse:8} {B:3} {L:5} | "
              f"{dur(c) or 0:6.1f} {dur(bk) or 0:8.1f} {dur(sc) or 0:7.1f} | "
              f"{pk if pk is not None else 0:+5.1f} | "
              f"{spr_c or 0:5.2f}->{spr_s or 0:5.2f} | {dram or 0:5.1f} | {l2 or 0:5.1f}")
    print("   interpretation: sectors/request distinguishes transaction-shape changes from"
          " latency effects. Do not map synthetic scatter to engine page size without K2's"
          " measured index-placement evidence.")


def k2_step0(ps1_json=None, ps128_json=None):
    p1 = Path(ps1_json) if ps1_json else (D / "kvidx_ps1.json")
    p128 = Path(ps128_json) if ps128_json else (D / "kvidx_ps128.json")
    if not (p1.exists() and p128.exists()):
        return
    print("\n=== K2  Step-0: measured engine shared-prefix kv_indices placement ===")
    print("   ps   bs    L | frac_contig  n_runs  max_run  pages128/exp -> verdict")
    for ps, p in [(1, p1), (128, p128)]:
        d = json.load(open(p))
        for b, s in sorted(d.get("per_batch", {}).items(), key=lambda x: int(x[0])):
            fc = s["frac_contig"]
            v = "SCATTERED" if fc < 0.5 else ("MIXED" if fc < 0.95 else "CONTIGUOUS")
            print(f"   {ps:3} {int(b):3} {s['seq_len']:5} | {fc:10.4f}  {s['n_runs']:6}  "
                  f"{s['max_run']:7}  {s['pages128']:4}/{s['expected_pages128']:<4} -> {v}")


def k3_report3(ps1_json=None, ps128_json=None):
    p1 = Path(ps1_json) if ps1_json else (D / "r3_ps1.json")
    p128 = Path(ps128_json) if ps128_json else (D / "r3_ps128.json")
    if not (p1.exists() and p128.exists()):
        return
    print("\n=== K3  report-3 reproduction: end-to-end TPOT ps1 vs ps128 (radix ON, Triton) ===")

    def tpot_by_bs(path):
        d = json.load(open(path))
        out = {}
        # harness JSON keyed by batch size; TPOT median in ms
        for k, v in (d.items() if isinstance(d, dict) else []):
            if isinstance(v, dict):
                b = v.get("batch_size") or (int(k) if str(k).isdigit() else None)
                t = v.get("tpot_median_ms") or v.get("tpot_ms") or v.get("median_tpot_ms")
                if b and t:
                    out[int(b)] = t
        return out
    t1, t128 = tpot_by_bs(p1), tpot_by_bs(p128)
    print("    B | TPOT ps1 (ms)  TPOT ps128 (ms) |  ps1-vs-ps128 %")
    for b in sorted(set(t1) & set(t128)):
        pen = (t1[b] / t128[b] - 1) * 100
        print(f"   {b:2} | {t1[b]:12.4f}  {t128[b]:14.4f} | {pen:+7.1f}%")
    print("   (Step 0 shows kv_indices CONTIGUOUS at both ps => any penalty here is NOT the"
          " attention-kernel gather; see nsys per-kernel decomposition.)")


if __name__ == "__main__":
    import sys
    a = sys.argv[1:]
    k1_microbench_pk()
    k2_step0(*(a[0:2] if len(a) >= 2 else []))
    k3_report3(*(a[2:4] if len(a) >= 4 else []))
