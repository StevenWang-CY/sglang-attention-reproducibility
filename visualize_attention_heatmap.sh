# python visualize_attention_heatmap.py attention_weights/attn_weights_request_000007_layer_00_head_00.csv --no-tokenizer
# python visualize_attention_heatmap.py attention_weights/archive_20251204_173002/attn_weights_request_000001_layer_00_head_00.csv --format png --start-row 0 --start-col 0
# python visualize_attention_heatmap.py attention_weights/archive_20251204_173002/attn_weights_request_000001_layer_00_head_00_normalized.csv --format png --start-row 0 --start-col 0

# python visualize_attention_heatmap.py attention_weights/archive_20251204_173002/attn_weights_request_000001_layer_00_head_00_max_norm.csv --format png --start-row 0 --start-col 0

# python visualize_attention_heatmap.py attention_weights/archive_20251204_173002/attn_weights_request_000001_layer_00_head_00_minmax_norm.csv --format png --start-row 0 --start-col 0

# python visualize_attention_heatmap.py attention_weights/archive_20251204_214346/attn_weights_request_000007_layer_00_head_00.csv --no-tokenizer --start-row 0 --end-row 100 --start-col 0 --end-col 100 --compress-level 0
# python visualize_attention_heatmap.py attention_weights/archive_20251204_214346/attn_weights_request_000007_layer_00_head_00.csv --no-tokenizer --start-row 0 --start-col 0 --compress-level 0


# python visualize_attention_heatmap.py attention_weights/archive_20251204_214346/attn_weights_request_000007_layer_00_head_00_max_norm.csv --format png --start-row 0 --start-col 0 --no-tokenizer


# python visualize_attention_heatmap.py attention_weights/archive_20251207_230914/attn_weights_request_000001_layer_00_head_00.csv --format png --start-row 0 --start-col 0 --no-tokenizer
# python visualize_attention_heatmap.py attention_weights/archive_20251207_230914/attn_weights_request_000001_layer_00_head_00_max_norm.csv --format png --start-row 0 --start-col 0 --no-tokenizer

# Loop over specific layers and heads
# Layers: 00, 01, 09, 16, 23
# Heads: 00, 01
for layer in 00 01 09 16 23; do
  for head in 00 01; do
    echo "Processing layer $layer, head $head..."
    python visualize_attention_heatmap.py \
      attention_weights/archive_20251215_112126/attn_weights_request_000001_layer_${layer}_head_${head}_max_norm.csv \
      --format png \
      --start-row 0 \
      --start-col 0 \
      --no-tokenizer
  done
done


# for layer in 00 01 09 16 23; do
#   for head in 00 01; do
#     echo "Processing layer $layer, head $head..."
#     python visualize_attention_heatmap.py \
#       attention_weights/archive_20251215_112126/attn_weights_request_000001_layer_${layer}_head_${head}.csv \
#       --format png \
#       --start-row 0 \
#       --start-col 0 \
#       --no-tokenizer
#   done
# done
