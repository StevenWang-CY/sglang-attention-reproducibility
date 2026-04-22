#!/usr/bin/env python3
"""Generate dummy request JSON files with specific input token lengths.

Uses the model's tokenizer to ensure accurate token counts.

Usage:
    python generate_dummy_requests.py --lengths 1024 2048 4096 8192 16384 32768
    python generate_dummy_requests.py --lengths 1024 4096 16384 --model-path /path/to/model
"""

import argparse
import json
import os
from pathlib import Path


def generate_dummy_text(target_tokens: int, tokenizer) -> str:
    """Generate dummy text that tokenizes to approximately target_tokens tokens.

    Fast approach: tokenize a small chunk once, repeat the token IDs to the
    target length, then decode. Avoids tokenizing huge strings.
    """
    chunk = (
        "The quick brown fox jumps over the lazy dog. "
        "A large language model processes input tokens sequentially. "
        "Attention mechanisms compute pairwise interactions between all positions. "
    )

    # Tokenize the small chunk once
    chunk_tokens = tokenizer.encode(chunk, add_special_tokens=False)

    # Repeat token IDs to reach target length
    repeats = (target_tokens // len(chunk_tokens)) + 1
    tokens = (chunk_tokens * repeats)[:target_tokens]

    # Decode back to text
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    actual = len(tokenizer.encode(text))
    return text, actual


def main():
    parser = argparse.ArgumentParser(description="Generate dummy requests with specific token lengths")
    parser.add_argument('--lengths', type=int, nargs='+', required=True,
                        help='Target input token lengths')
    parser.add_argument('--model-path', type=str,
                        default='/vast/projects/liuv/pennnetworks/hf_models/Qwen/Qwen3-VL-8B-Instruct',
                        help='Model path for tokenizer')
    parser.add_argument('--output-dir', type=str, default='dummy_requests',
                        help='Output directory for request JSON files')
    args = parser.parse_args()

    from transformers import AutoTokenizer
    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating dummy requests:")
    for target in args.lengths:
        text, actual = generate_dummy_text(target, tokenizer)

        request = {
            "messages": [
                {"role": "user", "content": text}
            ],
            "_meta": {
                "target_tokens": target,
                "actual_tokens": actual,
                "description": f"Dummy request with ~{target} input tokens"
            }
        }

        filename = output_dir / f"dummy_{target}tok.json"
        with open(filename, 'w') as f:
            json.dump(request, f, indent=2)

        print(f"  {target:>6} tokens -> {filename} (actual: {actual} tokens, {len(text)} chars)")

    print(f"\nAll requests saved to: {output_dir}/")


if __name__ == '__main__':
    main()
