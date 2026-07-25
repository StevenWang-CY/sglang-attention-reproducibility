#!/usr/bin/env python3
"""Step 0 driver: boot sgl.Engine (radix ON => shared prefix, Triton, eager) and run a
batch sweep of identical long prompts in ONE process (so the pool fragments across batches,
replicating report 3). Capture is done by the env-gated sitecustomize hook (subprocess-safe).
Set SGLANG_DUMP_KVIDX=<path> before running."""
import argparse, sglang as sgl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--page-size", type=int, required=True)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8, 16, 32])
    ap.add_argument("--prompt-tokens", type=int, default=4096)
    ap.add_argument("--gen", type=int, default=6)
    ap.add_argument("--mem-fraction-static", type=float, default=0.85)
    ap.add_argument("--context-length", type=int, default=12288)
    args = ap.parse_args()

    engine = sgl.Engine(
        model_path=args.model_path, attention_backend="triton", trust_remote_code=True,
        disable_radix_cache=False, disable_cuda_graph=True,
        page_size=args.page_size, mem_fraction_static=args.mem_fraction_static,
        context_length=args.context_length, log_level="warning",
    )
    word = "the quick brown fox jumps over the lazy dog and then keeps running onward "
    reps = max(1, args.prompt_tokens // 12)
    prompt = ("Summarize the following passage in detail. " + word * reps)[: args.prompt_tokens * 6]
    sp = {"temperature": 0.0, "max_new_tokens": args.gen}
    for B in args.batches:
        try:
            engine.generate([prompt] * B, sampling_params=sp)
            print(f"[ps{args.page_size}] generated bs{B}", flush=True)
        except Exception as ex:
            print(f"[ps{args.page_size}] bs{B} ERROR {ex}", flush=True)
    engine.shutdown()


if __name__ == "__main__":
    main()
