#!/bin/bash
# Batch generate attention weight visualizations for all CSV files

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCHIVE_DIR="${1:-attention_weights/archive_20251207_230914}"

# Change to the archive directory
cd "$SCRIPT_DIR/$ARCHIVE_DIR" || exit 1

echo "Generating visualizations for all attention weight files in: $ARCHIVE_DIR"
echo "Note: Token mapping CSV files (*_tokens.csv) must exist for each attention CSV"
echo ""

# Find all attention weight CSV files (not max_norm, not token mapping)
csv_files=$(find . -name "attn_weights_request_*.csv" ! -name "*max_norm*" ! -name "*_tokens.csv" | sort)

count=0
total=$(echo "$csv_files" | wc -l)

for csv_file in $csv_files; do
    count=$((count + 1))

    # Extract layer and head from filename
    # Format: attn_weights_request_000001_layer_NN_head_NN.csv
    if [[ $csv_file =~ layer_([0-9]+)_head_([0-9]+) ]]; then
        layer="${BASH_REMATCH[1]}"
        head="${BASH_REMATCH[2]}"

        # Remove leading zeros
        layer=$((10#$layer))
        head=$((10#$head))

        # Generate token mapping filename from CSV filename
        base_name=$(basename "$csv_file" .csv)
        token_file="${base_name}_tokens.csv"

        echo "[$count/$total] Processing Layer $layer, Head $head..."

        # Check if token mapping file exists
        if [ ! -f "$token_file" ]; then
            echo "  ⚠ Warning: Token mapping file not found: $token_file"
            echo "  Run: python ../../translate_tokens.py $csv_file"
            echo "  Skipping..."
            continue
        fi

        # Run visualization (output will be auto-generated)
        python "$SCRIPT_DIR/visualize_attention_hierarchical.py" \
            --csv "$csv_file" \
            --tokens "$token_file" \
            --layer "$layer" \
            --head "$head" \
            --min-weight 0.0001

        if [ $? -eq 0 ]; then
            # Output filename is auto-generated as: {csv_stem}_focus{N}_viz.html
            # Note: focus token is -1 (last token) by default
            echo "  ✓ Generated visualization (check for *_focus*_viz.html)"
        else
            echo "  ✗ Failed to generate visualization"
        fi
        echo ""
    else
        echo "Skipping $csv_file (couldn't parse layer/head)"
    fi
done

echo "Done! Generated $count visualizations."
echo ""
echo "To view, open any HTML file in your browser:"
echo "  firefox $ARCHIVE_DIR/attention_layer0_head0_hierarchical.html"
echo ""
echo "Or transfer to your local machine:"
echo "  scp -r \"$(whoami)@$(hostname):$(pwd)/*.html\" ."