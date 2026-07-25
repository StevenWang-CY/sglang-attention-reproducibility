#!/usr/bin/env bash
# Report 12 driver: runs all reuse-profile ncu sweeps via profile_reuse_ncu.sh (idle-gated,
# skip-guarded so it's resumable). Run as: echo PW | sudo -S -E bash run_reuse_all.sh
set -u
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
S="${PROFILE_SCRIPT:-$SCRIPT_DIR/profile_reuse_ncu.sh}"

# Exp A — reproducibility grid: B{1,4,8,16,32} x L{1024,4096,8192,16384}, distinct+shared, cold ps128
CACHE=all MODES="distinct shared" PAGES=128 \
  CELLS="1 1024|1 4096|1 8192|1 16384|4 1024|4 4096|4 8192|4 16384|8 1024|8 4096|8 8192|8 16384|16 1024|16 4096|16 8192|16 16384|32 1024|32 4096|32 8192|32 16384" \
  bash "$S"

# Exp B — reuse sweep (shared, L=1024 fixed=4MB) + dual fixed-R footprint sweep (B8 shared)
CACHE=all MODES="shared" PAGES=128 \
  CELLS="2 1024|6 1024|12 1024|24 1024|8 256|8 39168" \
  bash "$S"

# Exp A confound C4 — ps1 diagonal, both modes
CACHE=all MODES="distinct shared" PAGES=1 CELLS="8 1024|8 8192" bash "$S"

# Exp C cold extras (distinct B8 L512,L2048; others already in grid)
CACHE=all MODES="distinct" PAGES=128 CELLS="8 512|8 2048" bash "$S"

# Exp C warm (distinct B8, CACHE=none) — the residency artifact
CACHE=none MODES="distinct" PAGES=128 CELLS="8 512|8 1024|8 2048|8 4096|8 8192|8 16384" bash "$S"

echo "ALL REUSE SWEEPS DONE"
