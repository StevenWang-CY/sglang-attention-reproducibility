#!/usr/bin/env python3
"""
Instrument PyTorch to capture GEMM kernel selection and execution details.

This monkey-patches PyTorch's Linear layer to log:
1. When QKV projections happen
2. What batch sizes trigger different behavior
3. Actual execution time
4. Stack traces to see calling code
5. Any kernel selection logic
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import wraps
import traceback
import os


# Global log file
LOG_FILE = os.environ.get('KERNEL_LOG', 'kernel_selection.log')


def log_message(msg):
    """Thread-safe logging."""
    with open(LOG_FILE, 'a') as f:
        f.write(msg + '\n')
    print(msg)  # Also print to console


# Instrument torch.nn.functional.linear
_original_f_linear = F.linear

def instrumented_linear(input, weight, bias=None):
    """Instrumented version of F.linear that logs execution details."""

    batch_size = input.shape[0] if input.dim() > 1 else 1
    in_features = input.shape[-1]
    out_features = weight.shape[0]

    # Check if this looks like a QKV projection (output = 3x input)
    is_qkv = (out_features == 3 * in_features)

    # Only log for batch sizes we care about and potential QKV ops
    if batch_size in [16, 32] and is_qkv:
        log_message(f"\n{'='*80}")
        log_message(f"POTENTIAL QKV PROJECTION DETECTED")
        log_message(f"{'='*80}")
        log_message(f"Batch size: {batch_size}")
        log_message(f"Input shape: {input.shape}")
        log_message(f"Weight shape: {weight.shape}")
        log_message(f"Output features: {out_features} (3x input: {in_features * 3})")
        log_message(f"Is QKV: {is_qkv}")

        # Log calling context (last 5 stack frames)
        log_message("\nCalling stack:")
        stack = traceback.extract_stack()[:-1]  # Exclude current frame
        for frame in stack[-5:]:
            log_message(f"  {frame.filename}:{frame.lineno} in {frame.name}")

        # Time the execution
        if input.is_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            result = _original_f_linear(input, weight, bias)
            end.record()

            torch.cuda.synchronize()
            elapsed_ms = start.elapsed_time(end)

            log_message(f"\n✓ Execution time: {elapsed_ms:.6f} ms")
            log_message(f"{'='*80}\n")
        else:
            result = _original_f_linear(input, weight, bias)
    else:
        # Not a QKV projection, execute normally
        result = _original_f_linear(input, weight, bias)

    return result


# Instrument torch.matmul to catch direct GEMM calls
_original_matmul = torch.matmul

def instrumented_matmul(input, other, *, out=None):
    """Instrumented matmul to catch direct matrix multiplications."""

    # Only log for matrices (not vectors) with our target batch sizes
    if input.dim() >= 2 and other.dim() >= 2:
        batch_size = input.shape[0] if input.dim() > 2 else 1

        if batch_size in [16, 32]:
            log_message(f"\nDirect matmul: batch={batch_size}, "
                       f"input={input.shape}, other={other.shape}")

            if input.is_cuda:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                start.record()
                result = _original_matmul(input, other, out=out)
                end.record()

                torch.cuda.synchronize()
                elapsed_ms = start.elapsed_time(end)

                log_message(f"  Time: {elapsed_ms:.6f} ms")
            else:
                result = _original_matmul(input, other, out=out)
        else:
            result = _original_matmul(input, other, out=out)
    else:
        result = _original_matmul(input, other, out=out)

    return result


# Instrument nn.Linear.forward
_original_linear_forward = nn.Linear.forward

def instrumented_linear_forward(self, input):
    """Instrumented Linear layer forward pass."""

    batch_size = input.shape[0] if input.dim() > 1 else 1

    # Check if this is a QKV projection layer
    is_qkv = (self.out_features == 3 * self.in_features)

    if batch_size in [16, 32] and is_qkv:
        log_message(f"\n{'='*80}")
        log_message(f"nn.Linear (QKV) - batch={batch_size}")
        log_message(f"  Layer: in={self.in_features}, out={self.out_features}")
        log_message(f"  Input shape: {input.shape}")
        log_message(f"  Weight shape: {self.weight.shape}")
        if self.bias is not None:
            log_message(f"  Bias shape: {self.bias.shape}")

        if input.is_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            result = _original_linear_forward(self, input)
            end.record()

            torch.cuda.synchronize()
            elapsed_ms = start.elapsed_time(end)

            log_message(f"  ✓ Time: {elapsed_ms:.6f} ms")
            log_message(f"{'='*80}\n")
        else:
            result = _original_linear_forward(self, input)
    else:
        result = _original_linear_forward(self, input)

    return result


def install_instrumentation():
    """Install all instrumentation patches."""

    # Clear log file
    with open(LOG_FILE, 'w') as f:
        f.write("="*80 + "\n")
        f.write("KERNEL SELECTION INSTRUMENTATION LOG\n")
        f.write("="*80 + "\n\n")

    # Apply patches
    torch.nn.functional.linear = instrumented_linear
    torch.matmul = instrumented_matmul
    torch.nn.Linear.forward = instrumented_linear_forward

    log_message("✓ Instrumentation installed")
    log_message(f"✓ Logging to: {LOG_FILE}\n")


def uninstall_instrumentation():
    """Remove all instrumentation patches."""
    torch.nn.functional.linear = _original_f_linear
    torch.matmul = _original_matmul
    torch.nn.Linear.forward = _original_linear_forward

    log_message("\n✓ Instrumentation removed")


if __name__ == '__main__':
    print("This module provides instrumentation for PyTorch GEMM operations.")
    print("\nUsage in your script:")
    print("  from instrument_kernel_selection import install_instrumentation")
    print("  install_instrumentation()")
    print("  # ... run your model ...")
    print("\nOr set as environment variable:")
    print("  export KERNEL_LOG=/path/to/log.txt")
