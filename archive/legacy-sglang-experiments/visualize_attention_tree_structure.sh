#!/bin/bash

# Script to run visualize_attention_tree_structure.py
# This script visualizes attention weights as a tree structure based on ChatML/HTML

# Directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/visualize_attention_tree_structure.py"

# Default parameters - MODIFY THESE FOR YOUR DATA
CSV_FILE="attention_weights/archive_20251215_112126/attn_weights_request_000001_layer_01_head_01.csv"
TOKEN_FILE="attention_weights/archive_20251215_112126/attn_weights_request_000001_layer_01_head_01_tokens.csv"
LAYER=1
HEAD=1
FOCUS_TOKEN=-1  # -1 means last token
OUTPUT_FILE=""  # Leave empty for auto-generation

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --csv)
            CSV_FILE="$2"
            shift 2
            ;;
        --tokens)
            TOKEN_FILE="$2"
            shift 2
            ;;
        --layer)
            LAYER="$2"
            shift 2
            ;;
        --head)
            HEAD="$2"
            shift 2
            ;;
        --focus-token)
            FOCUS_TOKEN="$2"
            shift 2
            ;;
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --csv FILE          Path to CSV file with attention weights (required)"
            echo "  --tokens FILE       Path to token mapping CSV file (required)"
            echo "  --layer NUM         Layer number for display (default: 0)"
            echo "  --head NUM          Head number for display (default: 0)"
            echo "  --focus-token NUM   Token index to focus on, -1 for last (default: -1)"
            echo "  --output FILE       Output HTML file (default: auto-generate)"
            echo "  -h, --help          Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --csv attn_weights_request_000001_layer_00_head_00.csv \\"
            echo "     --tokens attn_weights_request_000001_layer_00_head_00_tokens.csv \\"
            echo "     --layer 0 --head 0"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found at $PYTHON_SCRIPT"
    exit 1
fi

# Build the Python command
PYTHON_CMD="python3 \"$PYTHON_SCRIPT\" --csv \"$CSV_FILE\" --tokens \"$TOKEN_FILE\" --layer $LAYER --head $HEAD --focus-token $FOCUS_TOKEN"

if [ -n "$OUTPUT_FILE" ]; then
    PYTHON_CMD="$PYTHON_CMD --output \"$OUTPUT_FILE\""
fi

# Display the command being run
echo "Running visualization script..."
echo "CSV file: $CSV_FILE"
echo "Token file: $TOKEN_FILE"
echo "Layer: $LAYER, Head: $HEAD"
echo "Focus token: $FOCUS_TOKEN"
echo ""

# Run the Python script
eval $PYTHON_CMD

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Visualization completed successfully!"
else
    echo ""
    echo "✗ Error occurred during visualization"
    exit 1
fi
