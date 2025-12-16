#!/bin/bash

# Generate flamegraph visualization 
python3 visualize_attention_flamegraph.py \
  --csv attention_weights/archive_20251215_112126/attn_weights_request_000001_layer_01_head_01.csv \
  --tokens attention_weights/archive_20251215_112126/attn_weights_request_000001_layer_01_head_01_tokens.csv \
  --layer 1 \
  --head 1 \
  --focus-token 7346 \
  --output attention_flamegraph.html

echo "Done! Open attention_flamegraph.html in your browser"
