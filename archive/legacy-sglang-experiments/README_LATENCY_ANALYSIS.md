# SGLang Latency Breakdown Analysis

This directory contains scripts to run batch size experiments, extract timing data, and visualize decode latency breakdowns.

## Files Overview

1. **run_batch_experiments.sh** - Runs experiments with different batch sizes
2. **parse_timing_logs.py** - Extracts timing data from log files
3. **plot_latency_breakdown.py** - Creates visualization plots

## Quick Start

### Step 1: Run Experiments

```bash
./run_batch_experiments.sh
```

This will:
- Test batch sizes: 1, 2, 4, 8, 10, 16, 32, 64
- Run 5 repetitions for each batch size
- Save logs to `batch_experiment_logs/`

**Configuration (edit in script if needed):**
- Request file: `html_request/request_005_20260316_221014/request.json`
- Model: `Qwen3-VL-8B-Instruct`
- Max tokens: 256
- Repetitions: 5

### Step 2: Parse Logs

```bash
python parse_timing_logs.py batch_experiment_logs/
```

This will:
- Parse all `.log` files in the directory
- Extract decode timing breakdowns
- Generate `batch_experiment_logs/timing_data.json`
- Print a summary table

**Output example:**
```
Batch Size   Latency (ms)    Attention    MLP          Others       Samples
--------------------------------------------------------------------------------
1            85.34           52.10        30.20        3.04         50
2            88.67           53.45        31.10        4.12         50
4            95.23           58.20        32.80        4.23         50
...
```

### Step 3: Generate Plots

```bash
python plot_latency_breakdown.py batch_experiment_logs/timing_data.json
```

Or simply:
```bash
python plot_latency_breakdown.py batch_experiment_logs/
```

This generates two plots:

1. **latency_breakdown_simple.png**
   - High-level view with 3 components:
     - Attention block (green)
     - MLP block (light blue)
     - Others (dark blue-gray)

2. **latency_breakdown_detailed.png**
   - Detailed view with 5 components:
     - QKV projection (red)
     - Attention (green)
     - Output projection (tan)
     - MLP (light blue)
     - Others (dark blue-gray)

## One-Liner Workflow

```bash
# Run everything in sequence
./run_batch_experiments.sh && \
python parse_timing_logs.py batch_experiment_logs/ && \
python plot_latency_breakdown.py batch_experiment_logs/
```

## Analyzing Existing qwen3vl Logs

If you already have SGLang server logs in `qwen3vl-log/`:

```bash
# Parse qwen3vl logs (automatically detects batch sizes from log content)
python parse_qwen3vl_logs.py qwen3vl-log/

# Plot the results
python plot_latency_breakdown.py qwen3vl-log/timing_data.json
```

**Note:** The `parse_qwen3vl_logs.py` script automatically:
- Extracts batch size from `Decode batch, #running-req: N` lines
- Groups timing data by batch size
- Handles multiple batch sizes in a single log file
- Requires logs with `DECODE STEP TIMING` blocks

## Customizing Batch Sizes

Edit `run_batch_experiments.sh` and modify the `BATCH_SIZES` array:

```bash
# Default
BATCH_SIZES=(1 2 4 8 10 16 32 64)

# Custom example
BATCH_SIZES=(1 4 16 64 128)
```

## Output Files

```
batch_experiment_logs/
├── batch_1_20260329_145205.log       # Raw logs for each batch size
├── batch_2_20260329_145210.log
├── ...
├── timing_data.json                   # Parsed timing data
├── latency_breakdown_simple.png       # Simple visualization
└── latency_breakdown_detailed.png     # Detailed visualization
```

## Troubleshooting

### No timing data found

Ensure your logs contain decode timing blocks:
```bash
grep "DECODE STEP TIMING" batch_experiment_logs/*.log
```

### Cannot extract batch size

Log files must follow naming convention: `batch_<N>_<timestamp>.log`

Example: `batch_10_20260329_145205.log`

### Missing dependencies

Install required Python packages:
```bash
pip install numpy matplotlib
```

## Data Format

The `timing_data.json` structure:
```json
{
  "10": {
    "batch_size": 10,
    "decode_latency_ms": 85.34,
    "breakdown_simple": {
      "attention_block": 52.10,
      "mlp_block": 30.20,
      "others": 3.04
    },
    "breakdown_detailed": {
      "qkv_proj": 15.20,
      "attention": 25.30,
      "o_proj": 11.60,
      "mlp": 30.20
    },
    "num_samples": 50
  }
}
```

## Tips

1. **Run multiple times**: The script averages across repetitions for more stable results
2. **Increase repetitions**: Edit `REPEAT=5` to higher value for better accuracy
3. **Batch size selection**: Use powers of 2 (1, 2, 4, 8, 16, 32, 64) for typical GPU optimization
4. **Monitor resources**: Large batch sizes may require more GPU memory

## Example Workflow

```bash
# 1. Run experiments (takes ~10-30 minutes depending on batch sizes)
./run_batch_experiments.sh

# 2. Check results while running
tail -f batch_experiment_logs/batch_*.log

# 3. After completion, parse and plot
python parse_timing_logs.py batch_experiment_logs/
python plot_latency_breakdown.py batch_experiment_logs/

# 4. View the plots
open batch_experiment_logs/latency_breakdown_simple.png
open batch_experiment_logs/latency_breakdown_detailed.png
```
