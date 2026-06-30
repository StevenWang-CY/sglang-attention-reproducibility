#!/usr/bin/env python3
"""Report 9 — vLLM decode TPOT vs block_size in a TRUE batch (NO prefix sharing),
FlashInfer backend, CUDA graph ON. Hunt for a >=5% page_size=1 (smallest block) latency
disadvantage on sm120.

Controls that make it a true batch with no shared prefix:
  - B DISTINCT prompts built from RANDOM token ids (every sequence a different KV) — never
    identical prompts (which prefix-caching would dedupe).
  - LLM(enable_prefix_caching=False) — no radix/prefix reuse.
  - VLLM_ATTENTION_BACKEND=FLASHINFER (set by the runner before this imports vllm).
  - enforce_eager=False  -> CUDA graph ON (the whole point; report 9).

Version-robust TPOT: two timed generates (max_tokens=1 ~ prefill+1; max_tokens=N);
TPOT=(tN-t1)/(N-1) cancels the identical prefill. One JSON per (block_size, batch_size).
"""
import argparse, json, time, statistics, os, random, traceback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--block-size", type=int, required=True)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--input-len", type=int, default=2048)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--attention-backend", default="FLASHINFER",
                    help="vLLM 0.23 EngineArgs.attention_backend (replaces the removed "
                         "VLLM_ATTENTION_BACKEND env var). FLASHINFER|FLASH_ATTN|TRITON_ATTN")
    ap.add_argument("--enforce-eager", action="store_true", default=False,
                    help="default OFF => CUDA graph ON")
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    from transformers import AutoConfig
    from vllm import LLM, SamplingParams
    try:
        from vllm import TokensPrompt
    except Exception:
        TokensPrompt = None

    cfg = AutoConfig.from_pretrained(a.model_path, trust_remote_code=True)
    vocab = int(getattr(cfg, "vocab_size", None) or getattr(getattr(cfg, "text_config", cfg), "vocab_size", 32000))
    rng = random.Random(1234)
    # B DISTINCT random-token prompts of exactly input-len tokens -> distinct KV, no sharing.
    def make_prompt(seed_off):
        r = random.Random(1234 + seed_off)
        ids = [r.randrange(0, max(1000, vocab - 100)) for _ in range(a.input_len)]
        if TokensPrompt is not None:
            return TokensPrompt(prompt_token_ids=ids)
        return {"prompt_token_ids": ids}
    prompts = [make_prompt(i) for i in range(a.batch_size)]

    info = {"block_size": a.block_size, "batch_size": a.batch_size, "input_len": a.input_len,
            "backend": a.attention_backend,
            "enforce_eager": a.enforce_eager, "vocab": vocab}
    try:
        llm = LLM(model=a.model_path, block_size=a.block_size,
                  gpu_memory_utilization=a.gpu_mem_util, max_model_len=a.max_model_len,
                  trust_remote_code=True, disable_log_stats=True,
                  enable_prefix_caching=False,          # NO prefix sharing
                  attention_backend=a.attention_backend,  # vLLM 0.23 EngineArgs (force FlashInfer)
                  enforce_eager=a.enforce_eager)        # graph ON by default
    except Exception:
        print("VLLM_LLM_INIT_FAILED — full traceback:")
        traceback.print_exc()
        info["error"] = "init_failed"
        json.dump(info, open(a.output, "w"), indent=2)
        raise

    def gen(n):
        sp = SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True)
        t = time.perf_counter()
        o = llm.generate(prompts, sp, use_tqdm=False)
        return time.perf_counter() - t, o

    gen(8)  # warmup
    tpots = []
    for _ in range(a.repeat):
        t1, _ = gen(1)
        tN, o = gen(a.max_tokens)
        ntok = min(len(x.outputs[0].token_ids) for x in o)
        if ntok > 1:
            tpots.append((tN - t1) / (ntok - 1) * 1000)
    info.update({
        "tpot_median_ms": statistics.median(tpots) if tpots else 0.0,
        "tpot_mean_ms": statistics.mean(tpots) if tpots else 0.0,
        "tpot_std_ms": statistics.pstdev(tpots) if len(tpots) > 1 else 0.0,
        "repeat": a.repeat, "tpots": [round(x, 3) for x in tpots],
    })
    json.dump(info, open(a.output, "w"), indent=2)
    print(f"backend={info['backend']} graph={'OFF' if a.enforce_eager else 'ON'} "
          f"block_size={a.block_size} bs={a.batch_size} L={a.input_len} "
          f"TPOT={info['tpot_median_ms']:.2f}ms tpots={info['tpots']}")


if __name__ == "__main__":
    main()
