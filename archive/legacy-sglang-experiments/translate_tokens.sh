# Basic usage - auto-generates output filenames
# python translate_tokens.py attention_weights/archive_20251207_230914/attn_weights_request_000001_layer_01_head_01.csv

# python translate_tokens.py attention_weights/attn_weights_request_000000_layer_00_head_00.csv

python translate_tokens.py attention_weights/archive_20251215_112126/attn_weights_request_000001_layer_01_head_01.csv


# # Specify custom output paths
# python translate_tokens.py attention_weights/attn_weights_request_000002_layer_01_head_01.csv \
#     --output translated.csv \
#     --text-output context.txt

# # Use different model tokenizer
# python translate_tokens.py attention_weights/attn_weights_request_000002_layer_01_head_01.csv \
#     --model Qwen/Qwen3-VL-30B-A3B-Instruct

# # Skip tokenizer loading (just shows token IDs)
# python translate_tokens.py attention_weights/attn_weights_request_000002_layer_01_head_01.csv \
#     --no-tokenizer