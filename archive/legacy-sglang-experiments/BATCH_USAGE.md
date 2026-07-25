# Batch Flamegraph Generation Usage Guide

The `run_flamegraph_batch.sh` script allows you to generate flamegraphs for multiple layers, heads, and focus tokens in a single run.

## Quick Examples

### Example 1: Generate for a single layer/head, multiple focus tokens
```bash
./run_flamegraph_batch.sh \
  --layer-start 1 --layer-end 1 \
  --head-start 1 --head-end 1 \
  --token-start 8400 --token-end 8450 --token-step 10
```
This generates flamegraphs for layer 1, head 1, and tokens 8400, 8410, 8420, 8430, 8440, 8450.

### Example 2: Generate for multiple layers, single head
```bash
./run_flamegraph_batch.sh \
  --layer-start 0 --layer-end 5 \
  --head-start 0 --head-end 0 \
  --token-start -1 --token-end -1
```
This generates flamegraphs for layers 0-5, head 0, using the last token (-1).

### Example 3: Generate for all combinations of layers and heads
```bash
./run_flamegraph_batch.sh \
  --layer-start 0 --layer-end 2 \
  --head-start 0 --head-end 3 \
  --token-start -1 --token-end -1 \
  --output-dir flamegraphs_output
```
This generates flamegraphs for:
- Layers 0, 1, 2
- Heads 0, 1, 2, 3
- Last token only
- Total: 3 × 4 = 12 files

### Example 4: Generate with parallel processing
```bash
./run_flamegraph_batch.sh \
  --layer-start 0 --layer-end 3 \
  --head-start 0 --head-end 7 \
  --token-start 8000 --token-end 8500 --token-step 100 \
  --parallel 4 \
  --format html
```
This runs 4 jobs in parallel (requires GNU `parallel` installed).

### Example 5: Different data directory
```bash
./run_flamegraph_batch.sh \
  --layer-start 0 --layer-end 1 \
  --head-start 0 --head-end 1 \
  --base-dir attention_weights/archive_20251220_120000 \
  --output-dir results/flamegraphs \
  --format svg
```

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--layer-start N` | Starting layer number | 0 |
| `--layer-end N` | Ending layer number (inclusive) | 0 |
| `--head-start N` | Starting head number | 0 |
| `--head-end N` | Ending head number (inclusive) | 0 |
| `--token-start N` | Starting focus token | -1 (last token) |
| `--token-end N` | Ending focus token (inclusive) | -1 |
| `--token-step N` | Step size between tokens | 1 |
| `--base-dir DIR` | Base directory for CSV files | `attention_weights/archive_20251215_112126` |
| `--output-dir DIR` | Output directory | `.` (current dir) |
| `--format FORMAT` | Output format: `html`, `svg`, or `both` | `both` |
| `--parallel N` | Number of parallel jobs | 1 |

## Output Files

Files are named: `attention_flamegraph_L{layer}_H{head}_token{token}.{html|svg}`

Example output for layer 1, head 2, token 8425:
- `attention_flamegraph_L01_H02_token8425.html`
- `attention_flamegraph_L01_H02_token8425.svg`

## Tips

1. **Start small**: Test with a small range first to make sure everything works
2. **Use parallel processing**: If you have GNU `parallel` installed, use `--parallel` to speed up generation
3. **Output directory**: Use `--output-dir` to organize your results
4. **Format selection**: Use `--format html` or `--format svg` if you only need one format

## File Naming Convention

The script expects CSV files in this format:
```
{base-dir}/attn_weights_request_000001_layer_{LL}_head_{HH}.csv
{base-dir}/attn_weights_request_000001_layer_{LL}_head_{HH}_tokens.csv
```

Where `{LL}` is the zero-padded layer number and `{HH}` is the zero-padded head number.
