#!/bin/bash

# Quick script to analyze existing qwen3vl logs

echo "Analyzing existing logs from qwen3vl-log directory..."
echo "=" | head -c 70 | tr '\n' '='
echo

# Check if the log directory exists
if [ ! -d "qwen3vl-log" ]; then
    echo "Error: qwen3vl-log directory not found"
    echo "This script should be run from the sglang_log directory"
    exit 1
fi

# Check if there are log files
LOG_COUNT=$(find qwen3vl-log -name "*.log" | wc -l)
echo "Found $LOG_COUNT log files in qwen3vl-log/"
echo

if [ $LOG_COUNT -eq 0 ]; then
    echo "No log files found! Please run experiments first."
    exit 1
fi

# Check if logs contain timing data
echo "Checking for timing data in logs..."
TIMING_BLOCKS=$(grep -c "DECODE STEP TIMING" qwen3vl-log/*.log 2>/dev/null | awk -F: '{sum += $2} END {print sum}')
echo "Found $TIMING_BLOCKS decode timing blocks"
echo

if [ $TIMING_BLOCKS -eq 0 ]; then
    echo "Warning: No decode timing blocks found in logs"
    echo "The logs might not contain the detailed timing information needed"
    echo
fi

# Parse the logs
echo "Step 1: Parsing logs..."
echo "-" | head -c 70 | tr '\n' '-'
echo
python parse_timing_logs.py qwen3vl-log/

# Check if parsing was successful
if [ ! -f "qwen3vl-log/timing_data.json" ]; then
    echo
    echo "Error: Failed to generate timing_data.json"
    echo "This might be because the log filenames don't contain batch size information"
    echo
    echo "Expected filename format: batch_<N>_<timestamp>.log"
    echo "Example: batch_10_20260329_145205.log"
    echo
    echo "Current log files:"
    ls -1 qwen3vl-log/*.log 2>/dev/null | head -10
    exit 1
fi

echo
echo "Step 2: Generating plots..."
echo "-" | head -c 70 | tr '\n' '-'
echo
python plot_latency_breakdown.py qwen3vl-log/

echo
echo "=" | head -c 70 | tr '\n' '='
echo
echo "Analysis complete!"
echo
echo "Results:"
echo "  - Timing data: qwen3vl-log/timing_data.json"
echo "  - Simple plot: qwen3vl-log/latency_breakdown_simple.png"
echo "  - Detailed plot: qwen3vl-log/latency_breakdown_detailed.png"
echo
echo "To view the plots:"
echo "  display qwen3vl-log/latency_breakdown_simple.png"
echo "  display qwen3vl-log/latency_breakdown_detailed.png"
echo
