#!/usr/bin/env python3
"""
Patch SGLang to add logging around QKV projection.
This will show us EXACTLY what's happening without needing profilers.
"""

import fileinput
import sys
from pathlib import Path

def patch_qwen3_attention():
    """Add logging to qwen3.py attention forward."""

    qwen3_file = Path("/vast/projects/liuv/pennnetworks/jiaheng/sglang_my/python/sglang/srt/models/qwen3.py")

    # Backup original
    backup = qwen3_file.with_suffix(".py.backup")
    if not backup.exists():
        import shutil
        shutil.copy(qwen3_file, backup)
        print(f"✓ Created backup: {backup}")

    # Read file
    with open(qwen3_file, 'r') as f:
        lines = f.readlines()

    # Find the line with timer.start
    modified_lines = []
    for i, line in enumerate(lines):
        modified_lines.append(line)

        # After timer start, add logging
        if 'if t: t.start(f"L{lid}:qkv_proj")' in line:
            indent = "        "  # 8 spaces to match indentation
            modified_lines.append(f'{indent}# DEBUG: Log QKV projection details\n')
            modified_lines.append(f'{indent}import time, os\n')
            modified_lines.append(f'{indent}_qkv_start = time.perf_counter()\n')
            modified_lines.append(f'{indent}_batch_size = hidden_states.shape[0]\n')
            modified_lines.append(f'{indent}_log_file = os.environ.get("QKV_DEBUG_LOG", "/tmp/qkv_debug.log")\n')
            modified_lines.append(f'{indent}_msg = f"[QKV_DEBUG] Layer {{lid}}, Batch {{_batch_size}}, Shape {{hidden_states.shape}}"\n')
            modified_lines.append(f'{indent}print(_msg)\n')
            modified_lines.append(f'{indent}with open(_log_file, "a") as _f: _f.write(_msg + "\\n")\n')

        # Before timer stop, add logging
        if 'if t: t.stop(f"L{lid}:qkv_proj")' in line:
            indent = "        "  # 8 spaces
            # Insert BEFORE the stop line
            modified_lines.pop()  # Remove the stop line we just added
            modified_lines.append(f'{indent}_qkv_end = time.perf_counter()\n')
            modified_lines.append(f'{indent}_qkv_elapsed_ms = (_qkv_end - _qkv_start) * 1000\n')
            modified_lines.append(f'{indent}_msg2 = f"[QKV_DEBUG] Layer {{lid}}, Direct timing: {{_qkv_elapsed_ms:.3f}} ms"\n')
            modified_lines.append(f'{indent}print(_msg2)\n')
            modified_lines.append(f'{indent}with open(_log_file, "a") as _f: _f.write(_msg2 + "\\n")\n')
            modified_lines.append(line)  # Now add the stop line back

    # Write modified file
    with open(qwen3_file, 'w') as f:
        f.writelines(modified_lines)

    print(f"✓ Patched {qwen3_file}")
    print(f"  Added logging around QKV projection")


def patch_decode_timer():
    """Add verbose logging to decode_timer.py."""

    timer_file = Path("/vast/projects/liuv/pennnetworks/jiaheng/sglang_my/python/sglang/srt/layers/attention/tree_sparse/decode_timer.py")

    # Backup
    backup = timer_file.with_suffix(".py.backup")
    if not backup.exists():
        import shutil
        shutil.copy(timer_file, backup)
        print(f"✓ Created backup: {backup}")

    # Read file
    with open(timer_file, 'r') as f:
        lines = f.readlines()

    # Find the finish_step method and add logging
    modified_lines = []
    for i, line in enumerate(lines):
        modified_lines.append(line)

        # After elapsed_time calculation, add logging
        if 'elapsed = start_ev.elapsed_time(end_ev)  # milliseconds' in line:
            indent = " " * 12  # Match indentation
            modified_lines.append(f'{indent}if "qkv_proj" in name:\n')
            modified_lines.append(f'{indent}    import os\n')
            modified_lines.append(f'{indent}    _log_file = os.environ.get("QKV_DEBUG_LOG", "/tmp/qkv_debug.log")\n')
            modified_lines.append(f'{indent}    _msg = f"[TIMER_DEBUG] {{name}}: {{elapsed:.3f}} ms"\n')
            modified_lines.append(f'{indent}    print(_msg)\n')
            modified_lines.append(f'{indent}    with open(_log_file, "a") as _f: _f.write(_msg + "\\n")\n')

    # Write
    with open(timer_file, 'w') as f:
        f.writelines(modified_lines)

    print(f"✓ Patched {timer_file}")
    print(f"  Added verbose logging for qkv_proj timing")


def restore_backups():
    """Restore original files from backups."""

    files = [
        "/vast/projects/liuv/pennnetworks/jiaheng/sglang_my/python/sglang/srt/models/qwen3.py",
        "/vast/projects/liuv/pennnetworks/jiaheng/sglang_my/python/sglang/srt/layers/attention/tree_sparse/decode_timer.py",
    ]

    for file_path in files:
        file_path = Path(file_path)
        backup = file_path.with_suffix(".py.backup")

        if backup.exists():
            import shutil
            shutil.copy(backup, file_path)
            backup.unlink()
            print(f"✓ Restored {file_path}")
        else:
            print(f"⚠ No backup found for {file_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--restore', action='store_true', help='Restore original files from backups')
    args = parser.parse_args()

    if args.restore:
        print("Restoring original files...")
        restore_backups()
    else:
        print("=" * 80)
        print("Patching SGLang with QKV logging")
        print("=" * 80)
        print()

        try:
            patch_qwen3_attention()
            patch_decode_timer()

            print()
            print("=" * 80)
            print("SUCCESS! Files patched with logging")
            print("=" * 80)
            print()
            print("Logs will be written to: /tmp/qkv_debug.log (or set QKV_DEBUG_LOG env var)")
            print()
            print("Now run your benchmark:")
            print("  export QKV_DEBUG_LOG=/vast/projects/liuv/pennnetworks/jiaheng/sglang_log/qkv_debug.log")
            print("  python measure_batch_latency_offline.py ... --batch-sizes 16 32")
            print()
            print("View the log:")
            print("  cat /vast/projects/liuv/pennnetworks/jiaheng/sglang_log/qkv_debug.log")
            print()
            print("To restore original files:")
            print("  python patch_sglang_logging.py --restore")
            print("=" * 80)

        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            print("\nRestoring backups...")
            restore_backups()
            sys.exit(1)
