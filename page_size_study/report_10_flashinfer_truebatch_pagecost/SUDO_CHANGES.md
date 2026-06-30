[2026-06-26 09:54:56] === profile_pagecost_ncu.sh start (CACHE=all, ncu=/usr/local/cuda-12.8/bin/ncu, user=wangcy07) ===
[2026-06-26 09:54:56] ncu flashinfer_contig_all_bs8_kv32768 : /usr/local/cuda-12.8/bin/ncu --cache-control all on /home/wangcy07/sglang_log/bench_pagecost.py --single --pattern contig -b 8 -l 32768
[2026-06-26 09:55:05] ncu flashinfer_scatter_all_bs8_kv32768 : /usr/local/cuda-12.8/bin/ncu --cache-control all on /home/wangcy07/sglang_log/bench_pagecost.py --single --pattern scatter -b 8 -l 32768
[2026-06-26 09:55:14] ncu flashinfer_contig_all_bs8_kv131072 : /usr/local/cuda-12.8/bin/ncu --cache-control all on /home/wangcy07/sglang_log/bench_pagecost.py --single --pattern contig -b 8 -l 131072
[2026-06-26 09:55:22] ncu flashinfer_scatter_all_bs8_kv131072 : /usr/local/cuda-12.8/bin/ncu --cache-control all on /home/wangcy07/sglang_log/bench_pagecost.py --single --pattern scatter -b 8 -l 131072
[2026-06-26 09:55:32] === profile_pagecost_ncu.sh done; chown -R wangcy07 /home/wangcy07/sglang_log/offline_batch_results/pagecost_profile ===
