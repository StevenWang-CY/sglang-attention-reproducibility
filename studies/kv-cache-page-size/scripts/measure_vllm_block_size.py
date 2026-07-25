#!/usr/bin/env python3
"""Measure vLLM decode TPOT vs block_size (Qwen3-VL-2B). Version-robust TPOT via
two timed generates (max_tokens=1 ~ prefill+1; max_tokens=N): TPOT=(tN-t1)/(N-1),
so the (identical) prefill cost cancels. One JSON per (block_size, batch_size)."""
import argparse, json, time, statistics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("request_file")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--block-size", type=int, required=True)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--gpu-mem-util", type=float, default=0.6)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--enforce-eager", action="store_true", default=False,
                    help="skip CUDA graph capture (avoids engine-init capture crashes)")
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    import traceback
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(a.model_path, trust_remote_code=True)
    req = json.load(open(a.request_file))
    prompt = tok.apply_chat_template(req["messages"], tokenize=False, add_generation_prompt=True)
    prompts = [prompt] * a.batch_size

    try:
        llm = LLM(model=a.model_path, block_size=a.block_size,
                  gpu_memory_utilization=a.gpu_mem_util, max_model_len=a.max_model_len,
                  trust_remote_code=True, disable_log_stats=True,
                  enforce_eager=a.enforce_eager)
    except Exception:
        print("VLLM_LLM_INIT_FAILED — full traceback:")
        traceback.print_exc()
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
        ntok = len(o[0].outputs[0].token_ids)
        if ntok > 1:
            tpots.append((tN - t1) / (ntok - 1) * 1000)

    res = {
        "block_size": a.block_size, "batch_size": a.batch_size,
        "tpot_median_ms": statistics.median(tpots) if tpots else 0.0,
        "tpot_mean_ms": statistics.mean(tpots) if tpots else 0.0,
        "tpot_std_ms": statistics.pstdev(tpots) if len(tpots) > 1 else 0.0,
        "repeat": a.repeat, "tpots": tpots,
    }
    json.dump(res, open(a.output, "w"), indent=2)
    print(f"block_size={a.block_size} bs={a.batch_size} TPOT median={res['tpot_median_ms']:.2f}ms "
          f"tpots={[round(x, 2) for x in tpots]}")


if __name__ == "__main__":
    main()
