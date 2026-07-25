#!/usr/bin/env python3
"""Generate a continuous co-tenant GPU compute load for measurement-hygiene controls.

This utility models compute contention with a small memory footprint. It does not establish
that contention caused any particular historical page-size result.
"""
import torch, time
dev = "cuda"
a = torch.randn(4096, 4096, dtype=torch.float16, device=dev)
b = torch.randn(4096, 4096, dtype=torch.float16, device=dev)
print("gpu_load running (compute contention, ~1.5GB)", flush=True)
while True:
    for _ in range(200):
        a = (a @ b) * 0.0001 + 0.1
    torch.cuda.synchronize()
