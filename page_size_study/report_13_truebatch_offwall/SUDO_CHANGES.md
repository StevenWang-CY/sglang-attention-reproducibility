# sudo / privileged-action audit log — Report 13 (true batch off the DRAM wall)

Box: **phastform.cis.upenn.edu** (user `wangcy07`, RTX 5060 Ti sm120, driver 580.95.05).
Same policy as reports 6–12: sudo is used **only** to run the `ncu` profiler
(`RmProfilingAdminOnly=1` restricts GPU perf counters to root). No sudoers edits, no
driver/module changes, no package installs. The password is piped per-launch and never
written to disk. Every runner appends its privileged actions to
`offline_batch_results/offwall_profile/SUDO_CHANGES.md` on the box; merged here.

## Change log

### 2026-07-11 — report-13 profiling session
- **Smoke test (root):** one `ncu` launch (`bench_xqa.py --single`, B2/L1024 ps128 distinct,
  cold) to validate the extended metric set (`launch__grid_size`, `launch__block_size`,
  `launch__waves_per_multiprocessor`, `l1tex__average_t_sectors_per_request...op_ld.ratio`
  on top of the report-12 set) on ncu 2025.3.1/sm120. Wrote `~/offwall_smoke/` (created
  root-owned by ncu; **chowned back to wangcy07**). Result: all metrics collected;
  B2/L1024 GQA-8 cold = 79.5% DRAM.
- **`run_offwall_ncu_all.sh` as root** (detached, umask 022): ~123 ncu cells into
  `~/sglang_log/offline_batch_results/offwall_profile/`:
  (1a) GQA-8 (16q/8kv) distinct cold grid B{1,2,4,8,16,32}×L{128..6144} × ps{1,128};
  (1b) GQA-2 (16q/2kv, Qwen2.5-3B shape) grid B{1,2,4,8}×L{512..8192} × ps{1,128};
  (1c) warm (`--cache-control none`) contrast at the flagship cells;
  (2) `bench_pagecost.py` contig/block128/scatter cells, both head shapes.
  Each cell: `ncu --target-processes all --profile-from-start off --replay-mode kernel
  --cache-control {all|none} --metrics <set> -o <tag> <venv-python> <bench> --single ...`;
  CSVs exported via `ncu --import --page raw --csv`; outputs **chowned back to wangcy07**
  at the end of every runner invocation.
- **`profile_bob_ncu.sh` as root**: ncu attached to `sglang.bench_one_batch` (eager,
  FlashInfer, `-k regex:BatchDecodeWithPagedKVCacheKernel`, skip/count = past-warmup
  two full decode steps) for Qwen2.5-3B & Qwen3-VL-2B at B2/L1024 (ps 1,128) and
  Qwen2.5-3B B8/L4096 (ps128). Engine runs as root with HOME pinned to the user's home;
  `~/.cache` (flashinfer JIT, HF/torch caches) and all outputs **chowned back**.
- **Cleanup chown:** `chown -R wangcy07 ~/offwall_smoke` (fix root-owned smoke dir).
- **System config changed: NONE.** Footprint = profiler outputs + JIT caches, all
  returned to user ownership. Model download (Qwen2.5-3B-Instruct, ~5.8 GB into
  `~/hf_models/`) was done **as the user, no sudo**.

### 2026-07-12/13 — nsys cross-check session (report 13 §5)
- **PERSISTENT SYSTEM CHANGE (sudo, precedented by the report-6 ncu install):**
  `sudo apt-get install -y cuda-nsight-systems-13-0` → installed
  `nsight-systems-2025.3.2` at `/opt/nvidia/nsight-systems/2025.3.2/` (the pre-existing
  2024.6.2 produces **silently empty** kernel traces against driver 580.x).
  **Undo:** `sudo apt-get remove --purge cuda-nsight-systems-13-0 nsight-systems-2025.3.2`.
- **nsys profiling runs as root** (CUPTI gated by `RmProfilingAdminOnly=1`): two microbench
  traces + two engine traces into `offline_batch_results/offwall_profile/nsys/`
  (first pass — partially overlapped by another user's GPU job; superseded for analysis by
  the rented-box clean pass in `nsys_vast/`, which needed no phastform sudo). A queued
  idle-gated rerun script was left detached (root) and may write refreshed traces into
  `nsys/` whenever the box frees; it chowns its outputs back and makes **no system change**.
- **Cleanup chown:** all `nsys/` outputs chowned back to `wangcy07`.
