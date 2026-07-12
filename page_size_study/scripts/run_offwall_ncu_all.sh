#!/usr/bin/env bash
# Report 13 driver: all ncu sweeps for the "true batch off the DRAM wall" study, via
# profile_offwall_ncu.sh (idle-gated, skip-guarded => resumable). Run as:
#   echo PW | sudo -S -E bash ~/sglang_log/run_offwall_ncu_all.sh
set -u
S=/home/wangcy07/sglang_log/profile_offwall_ncu.sh

# ---- Phase 1a — GQA-8 (Qwen3-VL-2B shape, 16q/8kv) (B,L) grid, distinct cold, ps{1,128} --
# maps DRAM% from the wall (~95) down to the off-wall corner; anchor B8/6144 = report-7 cell
CACHE=all TOOL=xqa QH=16 KH=8 MODES=distinct PAGES="1 128" \
  CELLS="1 128|1 256|1 512|1 1024|1 2048|1 4096|2 128|2 256|2 512|2 1024|2 2048|2 4096|4 128|4 256|4 512|4 1024|4 2048|4 4096|8 128|8 256|8 512|8 1024|8 2048|8 4096|16 128|16 256|16 512|32 128|32 256|8 6144" \
  bash "$S"

# ---- Phase 1b — GQA-2 arm (Qwen2.5-3B shape, 16q/2kv): the user's literal bs2/L1k case ----
CACHE=all TOOL=xqa QH=16 KH=2 MODES=distinct PAGES="1 128" \
  CELLS="1 512|1 1024|1 2048|1 4096|2 512|2 1024|2 2048|2 4096|2 8192|4 512|4 1024|4 2048|4 4096|8 512|8 1024|8 2048|8 4096|8 8192" \
  bash "$S"

# ---- Phase 1c — warm contrast at flagship cells (the report-8/12 residency-artifact echo) --
CACHE=none TOOL=xqa QH=16 KH=8 MODES=distinct PAGES="128" CELLS="2 1024|1 512" bash "$S"
CACHE=none TOOL=xqa QH=16 KH=2 MODES=distinct PAGES="128" CELLS="2 1024" bash "$S"

# ---- Phase 2 — worst-case ps1 fragmentation (sglang-faithful token index), off-wall vs wall
CACHE=all TOOL=pagecost QH=16 KH=8 PATTERNS="contig block128 scatter" \
  CELLS="1 512|2 1024|4 512|8 4096" bash "$S"
CACHE=all TOOL=pagecost QH=16 KH=2 PATTERNS="contig block128 scatter" \
  CELLS="2 1024|2 4096|8 1024|8 8192" bash "$S"

# ---- Phase 3 — tensor-core arm (the wrapper sglang dispatches for GQA group>=4) ------------
# NB: run with the GPU otherwise idle — do NOT overlap the engine TPOT tier (report 13 §2b
# documents a 2.1x contention phantom from exactly that overlap).
CACHE=all TOOL=xqa QH=16 KH=2 TC=1 MODES=distinct PAGES="1 128" \
  CELLS="1 512|1 1024|1 2048|1 4096|2 512|2 1024|2 2048|2 4096|2 8192|4 512|4 1024|4 2048|4 4096|8 512|8 1024|8 2048|8 4096|8 8192" \
  bash "$S"
CACHE=all TOOL=pagecost QH=16 KH=2 TC=1 PATTERNS="contig block128 scatter" \
  CELLS="2 1024|8 8192" bash "$S"

echo "ALL OFFWALL NCU SWEEPS DONE"
