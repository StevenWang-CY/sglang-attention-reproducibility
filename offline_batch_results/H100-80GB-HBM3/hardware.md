# Hardware — NVIDIA H100 80GB HBM3

Captured: 2026-04-23

## GPU

| Field | Value |
|---|---|
| Model | NVIDIA H100 80GB HBM3 |
| Compute capability | 9.0 (SM90, Hopper) |
| SM count | 132 |
| Total memory | 81,559 MiB (~79.18 GiB usable from PyTorch) |
| Max SM clock | 1980 MHz |
| Max memory clock | 2619 MHz |
| Max power limit | 700 W |
| PCI bus ID | 00000000:08:00.0 |
| Count | 1 |
| Driver version | 580.126.20 |

## CPU

| Field | Value |
|---|---|
| Model | Intel Xeon Platinum 8480+ |
| Architecture | x86_64 |
| Sockets × cores × threads | 1 × 52 × 2 (104 logical CPUs) |
| NUMA nodes | 1 |

## Memory

| Field | Value |
|---|---|
| System RAM | 885 GiB total (816 GiB available at capture) |

## Software stack

| Field | Value |
|---|---|
| OS | Ubuntu 24.04.4 LTS |
| Kernel | Linux 6.8.0-1046-nvidia x86_64 |
| CUDA toolkit (nvcc) | 13.0.88 |
| CUDA runtime (torch.version.cuda) | 12.8 |
| cuDNN | 91002 |
| PyTorch | 2.9.1+cu128 |
| Python | 3.12 (conda env `sglang` at /venv/sglang) |
| Hostname | a2145cc11a5b |

## Notes

- The 32B bf16 Qwen3-VL weights occupy ~62.3 GiB, leaving ~16 GiB for KV + activations on a single H100.
- Sweeps on this hardware use `--kv-cache-dtype fp8_e5m2`, `--context-length 102400`, `--mem-fraction-static 0.95` to fit the KV pool. Effective `max_total_num_tokens` ≈ 100,468.
- Input lengths above ~100k are infeasible for Qwen3-VL-32B at batch 1 on one 80 GB H100; would need TP≥2 or a smaller model.
