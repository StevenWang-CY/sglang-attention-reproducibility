#!/usr/bin/env python3
"""
Simple CUDA kernel test to verify nsys can capture kernel names.
"""

import torch

print("Creating tensors...")
a = torch.randn(1000, 1000, device='cuda')
b = torch.randn(1000, 1000, device='cuda')

print("Running matmul (should launch CUDA kernels)...")
for i in range(5):
    c = torch.matmul(a, b)
    torch.cuda.synchronize()

print("Done!")
