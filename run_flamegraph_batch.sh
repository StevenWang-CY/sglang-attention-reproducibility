#!/bin/bash

# Batch generate flamegraph visualizations for a range of layers, heads, and focus tokens
# Usage: ./run_flamegraph_batch.sh [options]
#
# Options:
#   --layer-start N       Start layer (default: 0)
#   --layer-end N         End layer inclusive (default: 0)
#   --head-start N        Start head (default: 0)
#   --head-end N          End head inclusive (default: 0)
#   --token-start N       Start focus token (default: -1, last token)
#   --token-end N         End focus token inclusive (default: -1)
#   --token-step N        Step size for focus tokens (default: 1)
#   --base-dir DIR        Base directory for attention weights (default: attention_weights/archive_20251215_112126)
#   --output-dir DIR      Output directory for generated files (default: current directory)
#   --format FORMAT       Output format: html, svg, or both (default: both)
#   --parallel N          Number of parallel jobs (default: 1)

# Default values
LAYER_START=1
LAYER_END=1
HEAD_START=1
HEAD_END=1
TOKEN_START=8426
TOKEN_END=8650
TOKEN_STEP=2
BASE_DIR="attention_weights/archive_20251215_112126"
OUTPUT_DIR="."
FORMAT="html"
PARALLEL=1

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --layer-start)
            LAYER_START="$2"
            shift 2
            ;;
        --layer-end)
            LAYER_END="$2"
            shift 2
            ;;
        --head-start)
            HEAD_START="$2"
            shift 2
            ;;
        --head-end)
            HEAD_END="$2"
            shift 2
            ;;
        --token-start)
            TOKEN_START="$2"
            shift 2
            ;;
        --token-end)
            TOKEN_END="$2"
            shift 2
            ;;
        --token-step)
            TOKEN_STEP="$2"
            shift 2
            ;;
        --base-dir)
            BASE_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --format)
            FORMAT="$2"
            shift 2
            ;;
        --parallel)
            PARALLEL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run with --help for usage information"
            exit 1
            ;;
    esac
done

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Count total jobs
total_jobs=0
for layer in $(seq $LAYER_START $LAYER_END); do
    for head in $(seq $HEAD_START $HEAD_END); do
        # Count tokens
        if [ $TOKEN_START -ge 0 ] && [ $TOKEN_END -ge 0 ]; then
            num_tokens=$(( (TOKEN_END - TOKEN_START) / TOKEN_STEP + 1 ))
        else
            num_tokens=1
        fi
        total_jobs=$((total_jobs + num_tokens))
    done
done

echo "=========================================="
echo "Batch Flamegraph Generation"
echo "=========================================="
echo "Layers: $LAYER_START to $LAYER_END"
echo "Heads: $HEAD_START to $HEAD_END"
echo "Focus tokens: $TOKEN_START to $TOKEN_END (step: $TOKEN_STEP)"
echo "Base directory: $BASE_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Format: $FORMAT"
echo "Parallel jobs: $PARALLEL"
echo "Total jobs: $total_jobs"
echo "=========================================="
echo ""

# Function to run a single flamegraph generation
run_flamegraph() {
    local layer=$1
    local head=$2
    local token=$3
    local base_dir=$4
    local output_dir=$5
    local format=$6

    # Format layer and head with zero padding
    layer_padded=$(printf "%02d" $layer)
    head_padded=$(printf "%02d" $head)

    # Build file paths
    csv_file="${base_dir}/attn_weights_request_000001_layer_${layer_padded}_head_${head_padded}.csv"
    tokens_file="${base_dir}/attn_weights_request_000001_layer_${layer_padded}_head_${head_padded}_tokens.csv"
    output_file="${output_dir}/attention_flamegraph_L${layer_padded}_H${head_padded}"

    # Check if files exist
    if [ ! -f "$csv_file" ]; then
        echo "⚠ Skipping layer $layer, head $head: CSV file not found: $csv_file"
        return 1
    fi

    if [ ! -f "$tokens_file" ]; then
        echo "⚠ Skipping layer $layer, head $head: Tokens file not found: $tokens_file"
        return 1
    fi

    echo "▶ Generating: Layer $layer, Head $head, Token $token"

    # Run the flamegraph generation
    python3 visualize_attention_flamegraph.py \
        --csv "$csv_file" \
        --tokens "$tokens_file" \
        --layer $layer \
        --head $head \
        --focus-token $token \
        --output "$output_file" \
        --format "$format" 2>&1 | grep -v "Storage Quotas\|parcc_quota\|parcc_du" || true

    if [ $? -eq 0 ]; then
        echo "✓ Completed: Layer $layer, Head $head, Token $token"
        return 0
    else
        echo "✗ Failed: Layer $layer, Head $head, Token $token"
        return 1
    fi
}

export -f run_flamegraph

# Build job list
job_count=0
jobs_file=$(mktemp)

for layer in $(seq $LAYER_START $LAYER_END); do
    for head in $(seq $HEAD_START $HEAD_END); do
        if [ $TOKEN_START -ge 0 ] && [ $TOKEN_END -ge 0 ]; then
            for token in $(seq $TOKEN_START $TOKEN_STEP $TOKEN_END); do
                echo "$layer $head $token $BASE_DIR $OUTPUT_DIR $FORMAT" >> "$jobs_file"
                job_count=$((job_count + 1))
            done
        else
            # Use default token (-1 = last token)
            echo "$layer $head $TOKEN_START $BASE_DIR $OUTPUT_DIR $FORMAT" >> "$jobs_file"
            job_count=$((job_count + 1))
        fi
    done
done

echo "Starting $job_count jobs with $PARALLEL parallel workers..."
echo ""

# Run jobs in parallel
if command -v parallel &> /dev/null && [ $PARALLEL -gt 1 ]; then
    # Use GNU parallel if available
    cat "$jobs_file" | parallel --jobs $PARALLEL --colsep ' ' run_flamegraph {1} {2} {3} {4} {5} {6}
else
    # Fall back to sequential processing
    if [ $PARALLEL -gt 1 ]; then
        echo "⚠ GNU parallel not found, falling back to sequential processing"
    fi

    while read -r layer head token base_dir output_dir format; do
        run_flamegraph "$layer" "$head" "$token" "$base_dir" "$output_dir" "$format"
    done < "$jobs_file"
fi

# Clean up
rm -f "$jobs_file"

echo ""
echo "=========================================="
echo "Batch generation complete!"
echo "Output files are in: $OUTPUT_DIR"
echo "=========================================="
