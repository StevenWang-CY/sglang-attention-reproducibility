#!/usr/bin/env python3
"""
Analyze QKV debug log to compare batch 16 vs 32 timing.
"""

import re
from collections import defaultdict

def analyze_log(log_file):
    """Parse log file and extract timing info."""

    batch_16_direct = defaultdict(list)
    batch_32_direct = defaultdict(list)
    batch_16_timer = defaultdict(list)
    batch_32_timer = defaultdict(list)

    with open(log_file, 'r') as f:
        lines = f.readlines()

    current_batch = None
    current_layer = None

    for line in lines:
        # Parse batch size and layer
        match = re.search(r'Layer (\d+), Batch (\d+)', line)
        if match:
            current_layer = int(match.group(1))
            current_batch = int(match.group(2))

        # Parse direct timing
        match = re.search(r'Layer (\d+), Direct timing: ([\d.]+) ms', line)
        if match:
            layer = int(match.group(1))
            time_ms = float(match.group(2))

            # Figure out batch from context (last seen batch)
            if current_batch == 16:
                batch_16_direct[layer].append(time_ms)
            elif current_batch == 32:
                batch_32_direct[layer].append(time_ms)

        # Parse timer debug
        match = re.search(r'\[TIMER_DEBUG\] L(\d+):qkv_proj: ([\d.]+) ms', line)
        if match:
            layer = int(match.group(1))
            time_ms = float(match.group(2))

            # Use current_batch
            if current_batch == 16:
                batch_16_timer[layer].append(time_ms)
            elif current_batch == 32:
                batch_32_timer[layer].append(time_ms)

    return batch_16_direct, batch_32_direct, batch_16_timer, batch_32_timer


def print_analysis(b16_direct, b32_direct, b16_timer, b32_timer):
    """Print analysis comparing batch sizes."""

    print("=" * 80)
    print("QKV Projection Timing Analysis: Batch 16 vs 32")
    print("=" * 80)
    print()

    # Layer 0 detailed analysis
    print("LAYER 0 (First Layer) - Detailed Analysis:")
    print("-" * 80)

    if 0 in b16_direct and b16_direct[0]:
        avg_16_direct = sum(b16_direct[0]) / len(b16_direct[0])
        print(f"Batch 16 - Direct timing:")
        print(f"  Samples: {len(b16_direct[0])}")
        print(f"  Average: {avg_16_direct:.3f} ms")
        print(f"  Min: {min(b16_direct[0]):.3f} ms")
        print(f"  Max: {max(b16_direct[0]):.3f} ms")

    if 0 in b16_timer and b16_timer[0]:
        avg_16_timer = sum(b16_timer[0]) / len(b16_timer[0])
        print(f"Batch 16 - Timer recorded:")
        print(f"  Samples: {len(b16_timer[0])}")
        print(f"  Average: {avg_16_timer:.3f} ms")

    print()

    if 0 in b32_direct and b32_direct[0]:
        avg_32_direct = sum(b32_direct[0]) / len(b32_direct[0])
        print(f"Batch 32 - Direct timing:")
        print(f"  Samples: {len(b32_direct[0])}")
        print(f"  Average: {avg_32_direct:.3f} ms")
        print(f"  Min: {min(b32_direct[0]):.3f} ms")
        print(f"  Max: {max(b32_direct[0]):.3f} ms")

    if 0 in b32_timer and b32_timer[0]:
        avg_32_timer = sum(b32_timer[0]) / len(b32_timer[0])
        print(f"Batch 32 - Timer recorded:")
        print(f"  Samples: {len(b32_timer[0])}")
        print(f"  Average: {avg_32_timer:.3f} ms")

    print()
    print("-" * 80)

    # Compare direct timing
    if 0 in b16_direct and 0 in b32_direct and b16_direct[0] and b32_direct[0]:
        avg_16 = sum(b16_direct[0]) / len(b16_direct[0])
        avg_32 = sum(b32_direct[0]) / len(b32_direct[0])
        ratio = avg_32 / avg_16 if avg_16 > 0 else 0

        print(f"COMPARISON (Direct Timing - Ground Truth):")
        print(f"  Batch 16: {avg_16:.3f} ms")
        print(f"  Batch 32: {avg_32:.3f} ms")
        print(f"  Ratio: {ratio:.2f}x (expected: ~2.0x for doubling batch)")

        if ratio < 1.5:
            print(f"  ⚠️  ANOMALY: Batch 32 is only {ratio:.2f}x, not 2.0x!")
        elif ratio > 2.5:
            print(f"  ⚠️  ANOMALY: Batch 32 is {ratio:.2f}x, more than expected!")

    print()

    # Compare timer vs direct
    if (0 in b16_direct and 0 in b16_timer and
        b16_direct[0] and b16_timer[0]):
        avg_direct = sum(b16_direct[0]) / len(b16_direct[0])
        avg_timer = sum(b16_timer[0]) / len(b16_timer[0])
        diff = avg_direct - avg_timer
        pct_diff = 100 * abs(diff) / avg_direct if avg_direct > 0 else 0

        print(f"BATCH 16 - Timer Accuracy:")
        print(f"  Direct timing: {avg_direct:.3f} ms")
        print(f"  Timer recorded: {avg_timer:.3f} ms")
        print(f"  Difference: {diff:.3f} ms ({pct_diff:.1f}%)")
        if abs(diff) < 0.1:
            print(f"  ✅ Timer is accurate!")

    print()

    if (0 in b32_direct and 0 in b32_timer and
        b32_direct[0] and b32_timer[0]):
        avg_direct = sum(b32_direct[0]) / len(b32_direct[0])
        avg_timer = sum(b32_timer[0]) / len(b32_timer[0])
        diff = avg_direct - avg_timer
        pct_diff = 100 * abs(diff) / avg_direct if avg_direct > 0 else 0

        print(f"BATCH 32 - Timer Accuracy:")
        print(f"  Direct timing: {avg_direct:.3f} ms")
        print(f"  Timer recorded: {avg_timer:.3f} ms")
        print(f"  Difference: {diff:.3f} ms ({pct_diff:.1f}%)")
        if abs(diff) > 0.5:
            print(f"  ❌ TIMER INACCURACY DETECTED!")
            print(f"     Timer missed {diff:.3f} ms of work!")

    print()
    print("=" * 80)

    # All layers summary
    print("\nALL LAYERS SUMMARY:")
    print("-" * 80)

    all_layers = sorted(set(list(b16_direct.keys()) + list(b32_direct.keys())))

    total_16_direct = 0
    total_32_direct = 0
    total_16_timer = 0
    total_32_timer = 0

    for layer in all_layers:
        if layer in b16_direct and b16_direct[layer]:
            total_16_direct += sum(b16_direct[layer]) / len(b16_direct[layer])
        if layer in b32_direct and b32_direct[layer]:
            total_32_direct += sum(b32_direct[layer]) / len(b32_direct[layer])
        if layer in b16_timer and b16_timer[layer]:
            total_16_timer += sum(b16_timer[layer]) / len(b16_timer[layer])
        if layer in b32_timer and b32_timer[layer]:
            total_32_timer += sum(b32_timer[layer]) / len(b32_timer[layer])

    print(f"Total QKV time across all layers:")
    print(f"  Batch 16 (direct): {total_16_direct:.3f} ms")
    print(f"  Batch 16 (timer):  {total_16_timer:.3f} ms")
    print(f"  Batch 32 (direct): {total_32_direct:.3f} ms")
    print(f"  Batch 32 (timer):  {total_32_timer:.3f} ms")

    print()
    print(f"Missing time in timer:")
    print(f"  Batch 16: {total_16_direct - total_16_timer:.3f} ms")
    print(f"  Batch 32: {total_32_direct - total_32_timer:.3f} ms")

    print()
    print("=" * 80)


if __name__ == '__main__':
    log_file = '/vast/projects/liuv/pennnetworks/jiaheng/sglang_log/qkv_debug.log'

    print("Analyzing QKV debug log...")
    print(f"Log file: {log_file}")
    print()

    b16_direct, b32_direct, b16_timer, b32_timer = analyze_log(log_file)

    print_analysis(b16_direct, b32_direct, b16_timer, b32_timer)
