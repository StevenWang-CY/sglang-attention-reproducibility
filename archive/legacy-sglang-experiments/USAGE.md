# Attention Visualization Usage Guide

## Quick Start

### Step 1: Generate Token Mapping CSV

First, translate your attention weight CSV to include readable token text:

```bash
cd /vast/projects/liuv/pennnetworks/jiaheng/sglang_log

# This will create *_tokens.csv file automatically
python translate_tokens.py attention_weights/archive_20251207_230914/attn_weights_request_000001_layer_00_head_00.csv
```

**Output:**
- `attn_weights_request_000001_layer_00_head_00_tokens.csv` - Token mapping
- `attn_weights_request_000001_layer_00_head_00_context.txt` - Full context (optional)

### Step 2: Generate Visualization

```bash
python visualize_attention_hierarchical.py \
    --csv attention_weights/archive_20251207_230914/attn_weights_request_000001_layer_00_head_00.csv \
    --tokens attention_weights/archive_20251207_230914/attn_weights_request_000001_layer_00_head_00_tokens.csv \
    --output attention_layer0_head0.html \
    --layer 0 \
    --head 0
```

```bash
python visualize_attention_hierarchical.py \
    --csv attention_weights/archive_20251207_230914/attn_weights_request_000001_layer_00_head_00.csv \
    --tokens attention_weights/archive_20251207_230914/attn_weights_request_000001_layer_00_head_00_tokens.csv \
    --show-all \
    --output attention_layer0_head0_all_tokens.html \
    --layer 0 \
    --head 0
```

### Step 3: View in Browser

```bash
# Option 1: If you have a browser on the server
firefox attention_layer0_head0.html

# Option 2: Transfer to your local machine
scp user@server:/vast/projects/liuv/pennnetworks/jiaheng/sglang_log/attention_layer0_head0.html .
```

---

## Batch Processing

### Generate Token Mappings for All Files

```bash
cd /vast/projects/liuv/pennnetworks/jiaheng/sglang_log/attention_weights/archive_20251207_230914

# Process all attention weight CSV files
for csv in attn_weights_request_*_layer_*_head_*.csv; do
    if [[ ! "$csv" =~ "max_norm" && ! "$csv" =~ "_tokens" ]]; then
        echo "Processing $csv..."
        python ../../translate_tokens.py "$csv"
    fi
done
```

### Generate All Visualizations

```bash
cd /vast/projects/liuv/pennnetworks/jiaheng/sglang_log

# Run batch visualization (will process all layers/heads)
./batch_visualize.sh attention_weights/archive_20251207_230914
```

---

## File Structure

```
/vast/projects/liuv/pennnetworks/jiaheng/sglang_log/
├── translate_tokens.py              # Token ID → text translation
├── visualize_attention_hierarchical.py  # Main visualization script
├── visualize_attention_flamegraph.py    # Alternative flame graph style
├── visualize_attention_heatmap.py       # Heatmap visualization
├── batch_visualize.sh                   # Batch process all files
└── attention_weights/
    └── archive_20251207_230914/
        ├── attn_weights_request_000001_layer_00_head_00.csv        # Raw attention
        ├── attn_weights_request_000001_layer_00_head_00_tokens.csv # Token mapping
        ├── attn_weights_request_000001_layer_00_head_00_context.txt # Full context
        └── attention_layer0_head0.html                             # Visualization
```

---

## Command Reference

### translate_tokens.py

```bash
python translate_tokens.py INPUT.csv [OPTIONS]

Options:
  --model MODEL           Tokenizer model (default: Qwen/Qwen3-VL-30B-A3B-Instruct)
  --text-output, -t FILE  Output text file (default: auto-generate)
  --token-map, -m FILE    Output token CSV (default: auto-generate)
  --no-tokenizer          Skip loading tokenizer
```

### visualize_attention_hierarchical.py

```bash
python visualize_attention_hierarchical.py [OPTIONS]

Required:
  --csv FILE              Attention weights CSV file
  --tokens FILE           Token mapping CSV file

Optional:
  --output FILE           Output HTML file (default: attention_tree.html)
  --focus-token N         Token index to analyze (default: -1 = last token)
  --min-weight FLOAT      Minimum attention weight (default: 0.0001)
  --layer N               Layer number for display (default: 0)
  --head N                Head number for display (default: 0)
```

---

## Examples

### Example 1: Single Visualization

```bash
# Generate token mapping
python translate_tokens.py \\
    attention_weights/archive_20251207_230914/attn_weights_request_000001_layer_16_head_01.csv

# Create visualization
python visualize_attention_hierarchical.py \\
    --csv attention_weights/archive_20251207_230914/attn_weights_request_000001_layer_16_head_01.csv \\
    --tokens attention_weights/archive_20251207_230914/attn_weights_request_000001_layer_16_head_01_tokens.csv \\
    --output layer16_head1.html \\
    --layer 16 \\
    --head 1
```

### Example 2: Focus on Specific Token

```bash
# Analyze attention FROM token at position 1000
python visualize_attention_hierarchical.py \\
    --csv attention_weights/archive_20251207_230914/attn_weights_request_000001_layer_00_head_00.csv \\
    --tokens attention_weights/archive_20251207_230914/attn_weights_request_000001_layer_00_head_00_tokens.csv \\
    --focus-token 1000 \\
    --output token_1000_attention.html
```

### Example 3: High Attention Weights Only

```bash
# Only show attention weights >= 0.01
python visualize_attention_hierarchical.py \\
    --csv attention_weights/archive_20251207_230914/attn_weights_request_000001_layer_23_head_00.csv \\
    --tokens attention_weights/archive_20251207_230914/attn_weights_request_000001_layer_23_head_00_tokens.csv \\
    --min-weight 0.01 \\
    --output layer23_head0_high_weights.html
```

---

## Troubleshooting

### "Token mapping file not found"

You forgot to generate the token CSV. Run:
```bash
python translate_tokens.py YOUR_ATTENTION_FILE.csv
```

### "Newlines appear as empty in CSV"

This is fixed in the latest version. The newlines are escaped as `\\n` in the CSV.
Make sure you're using the updated `translate_tokens.py`.

### "Visualization is slow/large"

Try filtering with higher `--min-weight`:
```bash
--min-weight 0.001  # or higher
```

### "Tokens show as [ID:1234]"

The tokenizer couldn't decode that token ID. This is normal for some special tokens.
You can specify a different tokenizer with `--model`.

---

## Tips

1. **Always generate token mappings first** before visualization
2. **Use batch scripts** for processing multiple files
3. **Increase `--min-weight`** to reduce file size and improve performance
4. **Compare layers** by opening multiple HTML files in different tabs
5. **Share visualizations** - HTML files are self-contained and portable
