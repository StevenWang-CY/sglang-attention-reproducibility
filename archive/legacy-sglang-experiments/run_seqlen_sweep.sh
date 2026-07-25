#!/bin/bash
set -euo pipefail

# Sweep input sequence length to see how TPOT scales with context length.
#
# Creates synthetic prompts of varying lengths by repeating simple text.
# Use this to measure sglang's dense attention TPOT at different context lengths.
#
# Usage:
#   bash run_seqlen_sweep.sh                    # default: BS=1
#   BATCH_SIZE=8 bash run_seqlen_sweep.sh       # BS=8
#   BATCH_SIZE=128 bash run_seqlen_sweep.sh     # BS=128

cd /vast/projects/liuv/pennnetworks/jiaheng/sglang_log

MODEL_PATH="/vast/projects/liuv/pennnetworks/hf_models/Qwen/Qwen3-VL-8B-Instruct"
NUM_DECODE_TOKENS=256
REPEAT=2
BATCH_SIZE="${BATCH_SIZE:-1}"
IGNORE_EOS="--ignore-eos"

# Sequence lengths to test (in tokens, approximate)
# The script will pad/repeat the base prompt to reach these lengths
SEQ_LENGTHS=(2048 4096 8192 16384 32768)

ts=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="batch_results/seqlen_sweep/${ts}"
mkdir -p "$OUTPUT_DIR"

echo "========================================================================"
echo "Sequence Length Sweep at BS=$BATCH_SIZE"
echo "========================================================================"
echo "Date: $(date)"
echo "Base request: $REQUEST_FILE"
echo "Sequence lengths: ${SEQ_LENGTHS[*]}"
echo "Batch size: $BATCH_SIZE"
echo "Top-k: $TOP_K"
echo "Decode tokens: $NUM_DECODE_TOKENS"
echo "Repetitions: $REPEAT"
echo "Output directory: $OUTPUT_DIR"
echo "========================================================================"
echo ""

# First, generate synthetic request files of varying lengths
echo "Generating synthetic prompts..."
python -u -c "
import json, os

# Simple repeating sentence (~10 tokens each)
filler = 'The quick brown fox jumps over the lazy dog. '

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('$MODEL_PATH', trust_remote_code=True)
filler_tokens = len(tok.encode(filler))
print(f'Filler: {filler_tokens} tokens per sentence')

seq_lengths = [${SEQ_LENGTHS[*]// /, }]
output_dir = '$OUTPUT_DIR'

for target_len in seq_lengths:
    repeats = target_len // filler_tokens + 1
    text = filler * repeats
    actual = len(tok.encode(text))
    print(f'  seq_len={target_len}: generated {actual} tokens')

    out_file = os.path.join(output_dir, f'request_seqlen_{target_len}.json')
    with open(out_file, 'w') as f:
        json.dump({'text': text}, f)

print('Done.')
" 2>&1

echo ""

START_TIME=$(date +%s)

for SEQ_LEN in "${SEQ_LENGTHS[@]}"; do
    REQ_FILE="${OUTPUT_DIR}/request_seqlen_${SEQ_LEN}.json"

    if [ ! -f "$REQ_FILE" ]; then
        echo "WARNING: $REQ_FILE not found, skipping"
        continue
    fi

    echo ""
    echo "======================================================================"
    echo "SEQ_LEN=$SEQ_LEN  BS=$BATCH_SIZE"
    echo "======================================================================"

    RESULT_FILE="${OUTPUT_DIR}/results_seqlen_${SEQ_LEN}.json"
    LOG_FILE="${OUTPUT_DIR}/runtime_seqlen_${SEQ_LEN}.log"

    python -u measure_batch_latency_offline.py "$REQ_FILE" \
        --model-path "$MODEL_PATH" \
        --batch-sizes $BATCH_SIZE \
        --max-tokens $NUM_DECODE_TOKENS \
        --repeat $REPEAT \
        --output "$RESULT_FILE" \
        $IGNORE_EOS \
        2>&1 | tee "$LOG_FILE"

    # Check for errors
    ERROR_COUNT=$(grep -c "Error\|CUDA error\|OutOfMemoryError" "$LOG_FILE" 2>/dev/null || echo 0)
    if [ "$ERROR_COUNT" -gt 0 ]; then
        echo "!! WARNING: Errors detected for seq_len=$SEQ_LEN !!"
        grep "Error\|CUDA error\|OutOfMemoryError" "$LOG_FILE" | head -3
    fi

    sleep 2
done

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "========================================================================"
echo "SEQUENCE LENGTH SWEEP COMPLETE (${DURATION}s)"
echo "========================================================================"
echo "Results in: $OUTPUT_DIR"
echo ""

# ─── Print summary table ──────────────────────────────────────────────────────
echo "─────────────────────────────────────────────────────────────────────────────"
printf "%-10s %-10s %-12s %-12s\n" "seq_len" "BS" "TPOT(ms)" "tok/s"
echo "─────────────────────────────────────────────────────────────────────────────"

for SEQ_LEN in "${SEQ_LENGTHS[@]}"; do
    RESULT_FILE="${OUTPUT_DIR}/results_seqlen_${SEQ_LEN}.json"
    [ ! -f "$RESULT_FILE" ] && continue

    TPOT=$(python3 -c "
import json
d = json.load(open('$RESULT_FILE'))
v = list(d.values())
print(f\"{v[0]['tpot_mean_ms']:.2f}\" if v and 'tpot_mean_ms' in v[0] else 'N/A')
" 2>/dev/null)

    THROUGHPUT=$(python3 -c "
import json
d = json.load(open('$RESULT_FILE'))
v = list(d.values())
print(f\"{v[0]['throughput_tokens_per_sec']:.1f}\" if v and 'throughput_tokens_per_sec' in v[0] else 'N/A')
" 2>/dev/null)

    printf "%-10s %-10s %-12s %-12s\n" "$SEQ_LEN" "$BATCH_SIZE" "${TPOT:-N/A}" "${THROUGHPUT:-N/A}"
done

echo "─────────────────────────────────────────────────────────────────────────────"
