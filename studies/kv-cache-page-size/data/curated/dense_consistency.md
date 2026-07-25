# Dense re-run consistency vs previous run

Normalized gap = (TPOT[ps] / TPOT[ps1] - 1) * 100%, compared new vs old per (config, bs, ps).
Absolute TPOT differs by token-count between runs, so only the *normalized* gap is compared.

| config | bs | ps | old gap% | new gap% | |Δ| | within noise? |
|---|---|---|---|---|---|---|
| FlashInfer, CUDA graph ON | 1 | 8 | +0.05 | +0.03 | 0.03 | yes (tol 2.0) |
| FlashInfer, CUDA graph ON | 1 | 32 | +0.07 | +0.01 | 0.06 | yes (tol 2.0) |
| FlashInfer, CUDA graph ON | 1 | 128 | +0.05 | -0.01 | 0.05 | yes (tol 2.0) |
| FlashInfer, CUDA graph ON | 8 | 8 | +0.08 | +0.12 | 0.05 | yes (tol 2.0) |
| FlashInfer, CUDA graph ON | 8 | 32 | +0.64 | +0.10 | 0.54 | yes (tol 2.0) |
| FlashInfer, CUDA graph ON | 8 | 128 | +0.77 | -0.32 | 1.09 | yes (tol 2.0) |
| FlashInfer, no CUDA graph | 1 | 128 | +2.87 | -5.02 | 7.89 | **NO** (tol 2.0) |
| FlashInfer, no CUDA graph | 8 | 128 | -0.14 | -1.82 | 1.68 | yes (tol 2.0) |
| Triton, CUDA graph ON | 1 | 8 | -0.13 | +0.25 | 0.37 | yes (tol 2.0) |
| Triton, CUDA graph ON | 1 | 32 | -0.10 | +0.23 | 0.33 | yes (tol 2.0) |
| Triton, CUDA graph ON | 1 | 128 | -0.39 | +0.22 | 0.61 | yes (tol 2.0) |
| Triton, CUDA graph ON | 8 | 8 | -0.22 | -0.46 | 0.24 | yes (tol 2.0) |
| Triton, CUDA graph ON | 8 | 32 | -0.26 | -0.48 | 0.21 | yes (tol 2.0) |
| Triton, CUDA graph ON | 8 | 128 | -1.27 | -2.50 | 1.22 | yes (tol 2.0) |
| Triton, no CUDA graph | 8 | 8 | +1.40 | -0.52 | 1.92 | yes (tol 2.0) |
| Triton, no CUDA graph | 8 | 32 | -0.59 | -1.24 | 0.64 | yes (tol 2.0) |
| Triton, no CUDA graph | 8 | 128 | -0.03 | -3.86 | 3.83 | **NO** (tol 2.0) |

**Worst normalized |Δ| = 7.89%.**