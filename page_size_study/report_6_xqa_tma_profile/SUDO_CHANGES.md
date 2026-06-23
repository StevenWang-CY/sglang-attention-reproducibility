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
