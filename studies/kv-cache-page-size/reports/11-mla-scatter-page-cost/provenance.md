# Report 11 privileged profiling log

Report 10's MHA profiling entries are retained in
[`../10-flashinfer-page-cost/provenance.md`](../10-flashinfer-page-cost/provenance.md). The entries below
cover the MLA-specific continuation. Absolute paths record the original environment.

[2026-06-26 10:25:44] === profile_mla_pagecost_ncu.sh start (CACHE=all, ncu=/usr/local/cuda-12.8/bin/ncu, user=wangcy07) ===
[2026-06-26 10:25:45] ncu mla_h16_contig_all_bs8_kv131072 : --cache-control all on /home/wangcy07/sglang_log/bench_mla_pagecost.py --single h=16 contig B8 L131072
[2026-06-26 10:25:54] ncu mla_h16_scatter_all_bs8_kv131072 : --cache-control all on /home/wangcy07/sglang_log/bench_mla_pagecost.py --single h=16 scatter B8 L131072
[2026-06-26 10:26:03] ncu mla_h128_contig_all_bs8_kv131072 : --cache-control all on /home/wangcy07/sglang_log/bench_mla_pagecost.py --single h=128 contig B8 L131072
[2026-06-26 10:26:12] ncu mla_h128_scatter_all_bs8_kv131072 : --cache-control all on /home/wangcy07/sglang_log/bench_mla_pagecost.py --single h=128 scatter B8 L131072
[2026-06-26 10:26:22] ncu mla_h16_contig_all_bs1_kv524288 : --cache-control all on /home/wangcy07/sglang_log/bench_mla_pagecost.py --single h=16 contig B1 L524288
[2026-06-26 10:26:31] ncu mla_h16_scatter_all_bs1_kv524288 : --cache-control all on /home/wangcy07/sglang_log/bench_mla_pagecost.py --single h=16 scatter B1 L524288
[2026-06-26 10:26:39] === profile_mla_pagecost_ncu.sh done; chown -R wangcy07 /home/wangcy07/sglang_log/studies/kv-cache-page-size/data/raw/pagecost_profile ===
