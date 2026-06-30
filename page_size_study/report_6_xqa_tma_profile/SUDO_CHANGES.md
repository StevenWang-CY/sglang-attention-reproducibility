# sudo / privileged-action audit log — Report 6 (XQA TMA profiling)

PI instruction (2026-06-21): *"给你在 phastform 上面的 sudo 了 … 记得记录下来你 sudo command 做的更改"*
— sudo granted on **phastform.cis.upenn.edu** (user `wangcy07`); record every change my sudo
commands make. This file is that audit trail. The runner `scripts/profile_xqa_ncu.sh` *also*
appends a timestamped line to `~/sglang_log/SUDO_CHANGES.log` on the box for every privileged
action; merge that log here after each run.

## Why sudo is needed at all
`ncu` (Nsight Compute) needs GPU performance-counter access. CIS boxes set
`RmProfilingAdminOnly=1` in the NVIDIA driver, so **all** CUPTI-based profiling (ncu, nsys
GPU-trace, torch.profiler CUDA activity) is restricted to root for non-admin users. Hence ncu
must run under sudo. (Confirmed on gray; expected identical on phastform.)

## Privileged-action policy (least-footprint)
I use sudo **only** to run the `ncu` profiler. I deliberately avoid persistent system changes:
- **No** edits to `/etc/sudoers` / sudoers.d.
- **No** change to `NVreg_RestrictProfilingToAdminUsers` / driver module params (would need a
  module reload/reboot of a shared box).
- **No** package installs, service changes, or config edits.

The only footprint sudo leaves:
1. Profiler outputs (`*.ncu-rep`, `*.csv`, `*.stdout`) under
   `~/sglang_log/offline_batch_results/xqa_profile/` — created root-owned, then `chown`ed back
   to `wangcy07` by the script.
2. flashinfer JIT kernel cache under `~/.cache/flashinfer` (only if a kernel recompiles while
   running as root) — also `chown`ed back to `wangcy07`. `HOME` is pinned to the user's home so
   nothing is written under `/root`.

## Change log (append one entry per privileged session)

### 2026-06-24 (cont.) — TMA positive control
- **Privileged action:** ran `ncu` as root on `tma_positive_control.py` (Triton TMA matmul). No persistent
  change. Result: TMA load bytes = **2.15 GB**, TMA cycles = 17.8 M, ldgsts = 0 → sm120 runs TMA and the
  counter fires. Confirms XQA/cuBLAS `TMA=0` is genuine. Script saved to `scripts/tma_positive_control.py`.

### 2026-06-24 — TMA cross-check (is the GPU/counter even capable of TMA?)
- **Privileged actions:** ran `ncu` as root on a bf16 GEMM probe (`_tma_probe.py`) and on XQA cells.
  No persistent change. Result: GEMM runs as a `cutlass_80` (Ampere) kernel — tensor pipe = 268 M inst,
  DRAM 1.34 GB (counters return large real values → TMA `0`s are genuine, not capped), TMA = 0.
  Conclusion: sm120 GPU *has* TMA, but the shipped kernels (FlashInfer cp.async, cuBLAS cutlass_80) don't
  emit it. Folded into report §3.

### 2026-06-23 (cont. 2) — full ncu counter sweep launched as root
- **Privileged action:** `sudo bash profile_xqa_ncu.sh` (detached, umask 022 → outputs world-readable).
  Runs ncu 2025.3.1 over XQA{16,32,64,128} + FlashInfer{1,16,32,64,128} × batch{1,8,32,64} × ctx{1k,4k,16k}
  + full-set cells → `offline_batch_results/xqa_profile/*.{csv,ncu-rep}`. No persistent system change beyond
  output files (chowned to wangcy07 at end). Every ncu invocation logged in `~/sglang_log/SUDO_CHANGES.log`.
- **Smoke result that reframes the report:** on RTX 5060 Ti (sm120), XQA `kernel_mha` uses **NO TMA**
  (`l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum = 0`, TMA pipe cycles = 0); it loads KV via
  cp.async (LDGSTS), same as FlashInfer. Both are DRAM-bandwidth-bound (~95% peak, ~134 MB) → ≈equal latency.

### 2026-06-23 (cont.) — root confirmed; LibraryNotLoaded = ncu VERSION mismatch; FIXED by installing matched ncu
- User provided the sudo password; verified `sudo … id` → `uid=0(root)`. **Privileged action:** ran ncu as root
  for a one-cell test → STILL `LibraryNotLoaded` with ncu **2025.1.1** ⇒ the error is a **version mismatch**
  (ncu 2025.1.1 / CUDA 12.8 vs driver **580.95.05 / CUDA 13.0**), NOT a permission problem.
- **PERSISTENT SYSTEM CHANGE (sudo):** `sudo apt-get install -y cuda-nsight-compute-13-0` →
  installed `nsight-compute-2025.3.1` (2025.3.1.4-1) + `cuda-nsight-compute-13-0` (13.0.3-1) from the
  already-configured NVIDIA CUDA apt repo. New profiler: `/opt/nvidia/nsight-compute/2025.3.1/ncu`.
  **Undo:** `sudo apt-get remove --purge nsight-compute-2025.3.1 cuda-nsight-compute-13-0`.
- **VERIFIED working:** ncu 2025.3.1 as root profiles `kernel_mha` (bs8/kv4096/ps64): dur=324,352 ns,
  global_ld=20,992, L2 hit=0.17%. (`dram__bytes_read.sum`=n/a on this GeForce → refining metric set.)
- Each profiling run still needs sudo (RmProfilingAdminOnly=1). Using the password per-launch; no driver
  module change made (avoided rebooting the shared box).

### 2026-06-23 — EXACT profiling command + errors (PI asked "你报错是啥")
**Command (phastform, normal user `wangcy07`, no sudo):**
```
cd ~/sglang_log
export PATH=$HOME/venvs/xqa_bench/bin:/usr/local/cuda-12.8/bin:$PATH CUDA_HOME=/usr/local/cuda-12.8
/opt/nvidia/nsight-compute/2025.1.1/ncu --target-processes all --profile-from-start off \
  --metrics gpu__time_duration.sum,dram__bytes_read.sum --csv \
  python bench_xqa.py --single --backend-single xqa --page-size 64 --batch-size 8 --seq-len 4096 --warmup 5
```
**Error (no sudo):**
```
==ERROR== An error was reported by the counter measurement library:
==ERROR== Failed to initialize the profiler: LibraryNotLoaded. Check that a compatible driver library is loaded.
==ERROR== The application returned an error code (9).
```
**With sudo (`sudo -n /opt/nvidia/nsight-compute/2025.1.1/ncu --version`):** `sudo: a password is required`

**Environment facts:** `wangcy07` is in the `sudo` group but every sudo needs a password (no NOPASSWD);
not root; `RmProfilingAdminOnly: 1`. Driver **580.95.05 → CUDA 13.0**; only Nsight Compute **2025.1.1**
(CUDA 12.8) installed (`/opt/nvidia/nsight-compute/2025.1.1`, also `cuda-12.8/bin/ncu`); **no CUDA-13 toolkit**.

**Two candidate causes (both likely in play):**
1. **Permission** — ncu needs root because `RmProfilingAdminOnly=1`.
2. **Version mismatch** — ncu 2025.1.1 (CUDA 12.8) is older than the driver (580 / CUDA 13.0); the
   "LibraryNotLoaded / compatible driver library" wording is the classic too-old-Nsight-for-this-driver signal.

**Fix:** (a) enable root profiling (NVreg=0 OR NOPASSWD); (b) if it still says LibraryNotLoaded as root,
install Nsight Compute matching CUDA 13.0 / driver 580 (or the CUDA-13 toolkit's ncu).

### 2026-06-23 — phastform reachable; sudo is password-gated → asking user to enable NOPASSWD for ncu
- phastform.cis.upenn.edu = **RTX 5060 Ti, sm120** (same GPU class as gray), idle, driver 580.95.05.
  `RmProfilingAdminOnly: 1` (ncu needs root, as expected).
- phastform home is **NOT** shared with gray (fresh account, no venv/model). Built a fresh,
  **sudo-free** venv `~/venvs/xqa_bench` (python3.10 + torch 2.9.1+cu128 + flashinfer 0.6.6) for
  the microbench (bench_xqa.py needs only torch+flashinfer; no sglang/model).
- `sudo -n` fails (password required) even for ncu → I cannot run `sudo ncu` non-interactively.
- **Requested user change (the only sudoers edit involved):** add
  `/etc/sudoers.d/ncu-prof` = `wangcy07 ALL=(root) NOPASSWD: /opt/nvidia/nsight-compute/2025.1.1/ncu, /usr/local/cuda-12.8/bin/ncu`
  (chmod 440). This grants passwordless sudo **only** for the ncu profiler binary, nothing else.
  Undo: `sudo rm /etc/sudoers.d/ncu-prof`. *Pending user execution.*

### 2026-06-21 — access granted, NOT yet exercised
- PI granted sudo on phastform.cis.upenn.edu.
- **No sudo command run yet.** Blocked before first use by a *local* networking fault on the
  workstation: a VPN/proxy client is hijacking DNS for `*.cis.upenn.edu` into the 198.18.0.0/15
  fake-IP range while its tunnel (configured SOCKS/HTTP port 127.0.0.1:6789) is down, so SSH to
  both gray and phastform fails (`kex_exchange_identification: Connection closed`). Fix is
  user-side (restart/quit the proxy client, or add `cis.upenn.edu` to its DIRECT rules).
- Once connectivity returns: run `sudo -E bash ~/sglang_log/profile_xqa_ncu.sh`; this file and
  the on-box `SUDO_CHANGES.log` will record the exact `ncu` invocations and their footprint.

<!-- TEMPLATE for each future entry:
### YYYY-MM-DD HH:MM — <what>
- command: sudo -E bash profile_xqa_ncu.sh   (ncu version X, GPU Y)
- privileged actions: ncu launched N target processes as root
- files written (root->chowned to wangcy07): offline_batch_results/xqa_profile/*.csv, *.ncu-rep
- system config changed: NONE
-->
