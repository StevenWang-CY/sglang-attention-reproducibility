#!/usr/bin/env python3
"""Report 15 — constructed max-amplification experiment.

Goal: the *best case* for a real end-to-end page_size=1 penalty, with every known amplifier
stacked: Triton backend x shared prefix (L2-reuse regime, off the DRAM wall, f_attn large)
x high batch x a FRAGMENTED ps1 pool (the one lever that changes the kernel's work).

Fragmentation construction (realistic, mirrors long-running serving):
  Phase A (churn): waves of `--churn-bs` CONCURRENT decodes. Concurrent requests allocate
  1 token/step each, round-robin -> their KV interleaves token-by-token in the pool. Waves
  exceed the pool -> LRU evicts earlier leaves -> the free list becomes token-strided at ps1.
  At ps128 the paged allocator keeps within-page 128-contiguity (pattern ~ block128 ~ contig).
  Phase B: admit `--batch` identical shared-prefix requests (prefix allocates from the
  fragmented free list) and decode `--gen` tokens x `--rounds`. Steady-state per-step decode
  is read from the engine's `Decode batch ... gen throughput` log lines (decode_log_interval=1),
  NOT from any whole-batch formula.

Verification arm: run with --eager and SGLANG_DUMP_KVIDX=<path> (r14hook .pth installed) to
dump the phase-B decode kv_indices contiguity (frac_contig, mean run length) - proves the
fragmentation actually happened before any conclusion is drawn from the timing.
"""
import argparse, random, time
import sglang as sgl


def words_prompt(n_words, seed):
    rng = random.Random(seed)
    ws = ["alpha","bravo","charlie","delta","echo","foxtrot","golf","hotel","india","juliet",
          "kilo","lima","mike","november","oscar","papa","quebec","romeo","sierra","tango",
          "uniform","victor","whiskey","xray","yankee","zulu"]
    return " ".join(rng.choice(ws) for _ in range(n_words))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--backend", choices=["triton", "flashinfer", "trtllm_mha"], default="triton")
    ap.add_argument("--page-size", type=int, required=True)
    ap.add_argument("--frag", action="store_true", help="run the churn phase first")
    ap.add_argument("--churn-waves", type=int, default=3)
    ap.add_argument("--churn-bs", type=int, default=16)
    ap.add_argument("--churn-gen", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--prefix-words", type=int, default=3630)  # ~6.1k tokens
    ap.add_argument("--gen", type=int, default=256)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--mem-fraction-static", type=float, default=0.85)
    ap.add_argument("--context-length", type=int, default=8192)
    ap.add_argument("--eager", action="store_true")
    args = ap.parse_args()

    # report 16: trtllm_mha on sm120 = XQA decode kernel; engine allows page in {16,32,64}
    # (128 is silently coerced to 64), and its PREFILL is SM100-only -> use split backends:
    # decode=trtllm_mha, prefill=flashinfer. It also can't do ps1 (radix off / XQA page floor).
    kw = dict(
        model_path=args.model_path, trust_remote_code=True,
        disable_radix_cache=False, page_size=args.page_size,
        mem_fraction_static=args.mem_fraction_static, context_length=args.context_length,
        log_level="debug", decode_log_interval=1,  # per-step decode lines
    )
    if args.backend == "trtllm_mha":
        if args.page_size not in (16, 32, 64):
            raise SystemExit(f"trtllm_mha (engine): page_size must be in {{16,32,64}} "
                             f"(128 is coerced to 64; ps1 disables radix). Got {args.page_size}.")
        if args.frag:
            raise SystemExit("trtllm_mha: --frag is a ps1-only lever (report 15); N/A to a page>=16 backend.")
        kw["decode_attention_backend"] = "trtllm_mha"
        kw["prefill_attention_backend"] = "flashinfer"   # trtllm_mha prefill is SM100-only
    else:
        kw["attention_backend"] = args.backend
    if args.eager:
        kw["disable_cuda_graph"] = True
    else:
        kw["disable_cuda_graph"] = False
        try:
            engine = sgl.Engine(**kw, cuda_graph_max_bs_decode=args.batch)
        except TypeError:
            engine = sgl.Engine(**kw, cuda_graph_max_bs=args.batch)
        kw = None
    if kw is not None:
        engine = sgl.Engine(**kw)

    # ---- Phase A: churn -> token-interleaved pool + LRU-evicted strided free list ----
    if args.frag:
        for w in range(args.churn_waves):
            prompts = [words_prompt(48, 1000 * w + i) for i in range(args.churn_bs)]
            sp = {"temperature": 0.0, "max_new_tokens": args.churn_gen, "ignore_eos": True}
            t0 = time.time()
            engine.generate(prompts, sampling_params=sp)
            print(f"[CHURN] wave {w+1}/{args.churn_waves} done in {time.time()-t0:.1f}s", flush=True)

    # ---- Phase B: shared-prefix batch on the (possibly fragmented) pool ----
    prefix = "Summarize the following list of callsigns in detail. " + words_prompt(args.prefix_words, 7)
    tails = [words_prompt(12, 90000 + i) for i in range(args.batch)]
    prompts = [prefix + " " + t for t in tails]
    sp = {"temperature": 0.0, "max_new_tokens": args.gen, "ignore_eos": True}
    for r in range(args.rounds):
        t0 = time.time()
        engine.generate(prompts, sampling_params=sp)
        print(f"[PHASE_B] round {r+1}/{args.rounds} wall={time.time()-t0:.2f}s "
              f"bs={args.batch} gen={args.gen}", flush=True)
    engine.shutdown()
    print("[DONE]", flush=True)


if __name__ == "__main__":
    main()
