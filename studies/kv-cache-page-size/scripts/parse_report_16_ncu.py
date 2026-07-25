"""Report 16 Arm C ncu parser — reads report-16-ncu-primary/*.csv (ncu --page raw --csv) and prints, per cell,
the metrics that explain the XQA page-sensitivity: kernel duration, DRAM %-peak, L2-hit%, sectors/request,
global-load instruction count (∝ 1/page if the kernel reads page-granular), and TMA-load bytes (=0 → no TMA).
Usage: python parse_report_16_ncu.py <dir>"""
import sys, glob, os, csv
d = sys.argv[1] if len(sys.argv) > 1 else "."
WANT = {
    "dur": "gpu__time_duration",
    "dram%": "dram__throughput.avg.pct_of_peak",
    "L2hit%": "lts__t_sector_hit_rate",
    "sec/req": "average_t_sectors_per_request",
    "gld": "sass_inst_executed_op_global_ld",
    "tma_ld": "op_tma_ld",
}
def kname(row): return row.get("Kernel Name", "")[:22]
print(f"{'cell':42s} {'kernel':22s} {'dur':>9} {'dram%':>7} {'L2hit%':>7} {'sec/req':>8} {'gld':>12} {'tma_ld':>8}")
for f in sorted(glob.glob(os.path.join(d, "*.csv"))):
    tag = os.path.basename(f)[:-4]
    try:
        rows = list(csv.DictReader(open(f)))
    except Exception:
        continue
    if not rows:
        print(f"{tag:42s} (empty)"); continue
    # collapse: take the (single) profiled kernel row; map metric name -> value (last numeric)
    cols = rows[0].keys()
    def col_for(sub):
        for c in cols:
            if sub in c:
                return c
        return None
    r = rows[-1]
    vals = {}
    for k, sub in WANT.items():
        c = col_for(sub)
        v = r.get(c, "") if c else ""
        vals[k] = v.replace(",", "") if isinstance(v, str) else v
    print(f"{tag:42s} {kname(r):22s} {vals['dur']:>9} {vals['dram%']:>7} {vals['L2hit%']:>7} {vals['sec/req']:>8} {vals['gld']:>12} {vals['tma_ld']:>8}")
