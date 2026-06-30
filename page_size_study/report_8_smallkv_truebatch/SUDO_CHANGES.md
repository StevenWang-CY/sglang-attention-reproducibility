[2026-06-25 05:11:45] ================ profile_smallkv_ncu.sh START (report 8, CACHE=all) ================
[2026-06-25 05:11:45] host=phastform uid=0 (root via sudo for ncu perf counters)
[2026-06-25 05:11:45] PY=/home/wangcy07/venvs/bench_sglang/bin/python NCU=/opt/nvidia/nsight-compute/2025.3.1/ncu BENCH=/home/wangcy07/sglang_log/bench_xqa.py OUTDIR=/home/wangcy07/sglang_log/offline_batch_results/smallkv_profile BACKEND=flashinfer CACHE=all
[2026-06-25 05:11:45] ncu: NVIDIA (R) Nsight Compute Command Line Profiler
[2026-06-25 05:11:45] ncu flashinfer_distinct_all_ps1_bs8_kv256
[2026-06-25 05:11:54] ncu flashinfer_distinct_all_ps128_bs8_kv256
[2026-06-25 05:12:02] ncu flashinfer_distinct_all_ps1_bs8_kv512
[2026-06-25 05:12:11] ncu flashinfer_distinct_all_ps128_bs8_kv512
[2026-06-25 05:12:19] ncu flashinfer_distinct_all_ps1_bs8_kv1024
[2026-06-25 05:12:27] ncu flashinfer_distinct_all_ps128_bs8_kv1024
[2026-06-25 05:12:36] ncu flashinfer_distinct_all_ps1_bs8_kv2048
[2026-06-25 05:12:44] ncu flashinfer_distinct_all_ps128_bs8_kv2048
[2026-06-25 05:12:52] ncu flashinfer_distinct_all_ps1_bs8_kv6144
[2026-06-25 05:13:01] ncu flashinfer_distinct_all_ps128_bs8_kv6144
[2026-06-25 05:13:09] ncu flashinfer_distinct_all_ps1_bs8_kv16384
[2026-06-25 05:13:18] ncu flashinfer_distinct_all_ps128_bs8_kv16384
[2026-06-25 05:13:26] ================ profile_smallkv_ncu.sh DONE (CACHE=all); chowned to wangcy07 ================
[2026-06-25 05:13:26] ================ profile_smallkv_ncu.sh START (report 8, CACHE=none) ================
[2026-06-25 05:13:26] host=phastform uid=0 (root via sudo for ncu perf counters)
[2026-06-25 05:13:26] PY=/home/wangcy07/venvs/bench_sglang/bin/python NCU=/opt/nvidia/nsight-compute/2025.3.1/ncu BENCH=/home/wangcy07/sglang_log/bench_xqa.py OUTDIR=/home/wangcy07/sglang_log/offline_batch_results/smallkv_profile BACKEND=flashinfer CACHE=none
[2026-06-25 05:13:26] ncu: NVIDIA (R) Nsight Compute Command Line Profiler
[2026-06-25 05:13:26] ncu flashinfer_distinct_none_ps1_bs8_kv256
[2026-06-25 05:13:34] ncu flashinfer_distinct_none_ps128_bs8_kv256
[2026-06-25 05:13:43] ncu flashinfer_distinct_none_ps1_bs8_kv512
[2026-06-25 05:13:51] ncu flashinfer_distinct_none_ps128_bs8_kv512
[2026-06-25 05:13:59] ncu flashinfer_distinct_none_ps1_bs8_kv1024
[2026-06-25 05:14:08] ncu flashinfer_distinct_none_ps128_bs8_kv1024
[2026-06-25 05:14:16] ncu flashinfer_distinct_none_ps1_bs8_kv2048
[2026-06-25 05:14:25] ncu flashinfer_distinct_none_ps128_bs8_kv2048
[2026-06-25 05:14:33] ncu flashinfer_distinct_none_ps1_bs8_kv6144
[2026-06-25 05:14:41] ncu flashinfer_distinct_none_ps128_bs8_kv6144
[2026-06-25 05:14:49] ncu flashinfer_distinct_none_ps1_bs8_kv16384
[2026-06-25 05:14:58] ncu flashinfer_distinct_none_ps128_bs8_kv16384
[2026-06-25 05:15:06] ================ profile_smallkv_ncu.sh DONE (CACHE=none); chowned to wangcy07 ================
