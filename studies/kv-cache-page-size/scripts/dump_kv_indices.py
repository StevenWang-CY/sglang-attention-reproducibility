#!/usr/bin/env python3
"""
Capture the real engine's shared-prefix decode ``kv_indices`` placement for report 14.

The later report-15 closure shows that report 3's +11–25% whole-call result was an admission-ramp metric
artifact, not a steady Triton decode penalty. This tool remains useful for the independent placement
question: whether a fresh or sequentially reused pool gives contiguous, blockwise, or scattered physical
slots at each engine page size. A fresh shared prefix was observed contiguous at both page sizes; heavy
pool-overcommit can construct fragmentation, but that must be measured rather than inferred from page size.

Method: monkeypatch TritonAttnBackend._fill_kv_indptr_and_indices (which fills the per-token
kv_indices in place via create_flashinfer_kv_indices_triton) to capture, per decode batch,
the physical-slot contiguity of one sequence's KV.  Radix ON + B identical long prompts =>
shared prefix.  Run once with --page-size 1 and once with 128; compare.

Contiguity stats (per sequence, decode step):
  frac_contig  = mean(idx[i+1]==idx[i]+1)          1.0 = perfectly contiguous, ~0 = scattered
  n_runs       = # of maximal consecutive runs      1 = contiguous, ~L = scattered
  max_run      = longest consecutive run            L = contiguous, 1 = scattered
  mean_run     = L / n_runs
  pages128     = #distinct (slot//128)              ceil(L/128) = block-packed, ~L = scattered

Usage (phastform or gray; no root needed):
  python dump_kv_indices.py --model-path ~/hf_models/Qwen3-VL-2B-Instruct \
     --page-size 1   --batches 1 8 16 32 48 --prompt-tokens 4096 --gen 6 \
     --output kvidx_ps1.json
  python dump_kv_indices.py ... --page-size 128 ... --output kvidx_ps128.json
"""
import argparse, json, os, sys


def contiguity_stats(idx):
    """idx: 1-D python list / tensor of physical slot ids for ONE sequence."""
    import torch
    if not torch.is_tensor(idx):
        idx = torch.as_tensor(idx)
    idx = idx.to(torch.int64).flatten()
    L = int(idx.numel())
    if L < 2:
        return dict(L=L, frac_contig=1.0, n_runs=1, max_run=L, mean_run=float(L), pages128=1)
    d = (idx[1:] - idx[:-1])
    contig = (d == 1)
    frac_contig = float(contig.float().mean().item())
    breaks = int((~contig).sum().item())
    n_runs = breaks + 1
    # longest consecutive run (vectorised): max #True-steps in a run, +1 for the span in elements
    c = contig.to(torch.int8)
    z = torch.cat([torch.zeros(1, dtype=c.dtype), c, torch.zeros(1, dtype=c.dtype)])
    dz = z[1:] - z[:-1]
    starts = (dz == 1).nonzero().flatten()
    ends = (dz == -1).nonzero().flatten()
    max_true_run = int((ends - starts).max().item()) if len(starts) else 0
    max_run = max_true_run + 1
    pages128 = int(torch.unique(idx // 128).numel())
    return dict(L=L, frac_contig=round(frac_contig, 6), n_runs=n_runs,
                max_run=max_run, mean_run=round(L / n_runs, 3), pages128=pages128,
                expected_pages128=(L + 127) // 128,
                slot_min=int(idx.min().item()), slot_max=int(idx.max().item()))


def install_hook(state, min_seq_len):
    """Patch TritonAttnBackend.forward_decode to capture the per-decode kv_indices contiguity,
       keyed by batch size. Works on the experiment build (~/latency_bench/sglang_CW), where
       forward_decode reads kv_indices from self.forward_metadata.kv_indices."""
    import torch
    from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
    orig = TritonAttnBackend.forward_decode

    def patched(self, q, k, v, layer, forward_batch, *a, **kw):
        try:
            state["_calls"] = state.get("_calls", 0) + 1
            fm = getattr(self, "forward_metadata", None)
            kv_indptr = getattr(fm, "kv_indptr", None)
            kv_indices = getattr(fm, "kv_indices", None)
            bs = int(getattr(forward_batch, "batch_size", 0) or 0)
            if kv_indices is not None and kv_indptr is not None and bs >= 1:
                seg0_len = int(kv_indptr[1].item() - kv_indptr[0].item())
                state["_max_seg"] = max(state.get("_max_seg", 0), seg0_len)
                prev = state.get(bs)
                # keep, per batch size, the decode capture with the LONGEST context (post-prefill)
                if seg0_len >= 8 and (prev is None or seg0_len >= prev.get("seq_len", 0)):
                    seg0 = kv_indices[:seg0_len].detach().to("cpu")
                    st = contiguity_stats(seg0)
                    all_same = True
                    if bs > 1:
                        a1, a2 = int(kv_indptr[1].item()), int(kv_indptr[2].item())
                        seg1 = kv_indices[a1:a2].detach().to("cpu")
                        m = min(len(seg0), len(seg1))
                        all_same = bool((seg0[:m] == seg1[:m]).all().item())
                    st["all_B_identical_slots"] = all_same
                    st["bs"] = bs; st["seq_len"] = seg0_len
                    state[bs] = st
        except Exception as ex:  # never break the engine
            state.setdefault("_errors", []).append(repr(ex))
        return orig(self, q, k, v, layer, forward_batch, *a, **kw)

    TritonAttnBackend.forward_decode = patched
    return orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--page-size", type=int, required=True)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8, 16, 32, 48])
    ap.add_argument("--prompt-tokens", type=int, default=4096,
                    help="shared-prefix length (approx tokens of the identical prompt)")
    ap.add_argument("--gen", type=int, default=6, help="decode steps to trigger (max_new_tokens)")
    ap.add_argument("--mem-fraction-static", type=float, default=0.85)
    ap.add_argument("--context-length", type=int, default=12288)
    ap.add_argument("--output", default="kvidx.json")
    args = ap.parse_args()

    import torch
    import sglang as sgl

    state = {}
    install_hook(state, min_seq_len=max(64, args.prompt_tokens // 2))

    engine = sgl.Engine(
        model_path=args.model_path,
        attention_backend="triton",
        trust_remote_code=True,
        disable_radix_cache=False,          # radix ON => shared prefix
        disable_cuda_graph=True,            # eager decode => per-step forward_decode w/ real metadata
        page_size=args.page_size,
        mem_fraction_static=args.mem_fraction_static,
        context_length=args.context_length,
        log_level="warning",
    )

    # A single long shared prompt (identical across the batch => radix shares the prefix).
    # Build ~prompt_tokens of text; exact token count is not critical for the contiguity test.
    word = "the quick brown fox jumps over the lazy dog and then keeps running onward "
    reps = max(1, args.prompt_tokens // 12)
    shared_prompt = ("Summarize the following passage in detail. " + word * reps)[: args.prompt_tokens * 6]
    sp = {"temperature": 0.0, "max_new_tokens": args.gen}

    # Replay the historical batch order in one engine process to observe any placement evolution.
    for B in args.batches:
        prompts = [shared_prompt] * B
        try:
            engine.generate(prompts, sampling_params=sp)
            snap = state.get(B, {})
            print(f"[ps{args.page_size}] bs{B}: {json.dumps(snap)}", flush=True)
        except Exception as ex:
            print(f"[ps{args.page_size}] bs{B}: GENERATE ERROR {ex}", flush=True)

    engine.shutdown()

    out = dict(page_size=args.page_size, prompt_tokens=args.prompt_tokens,
               batches=args.batches,
               debug=dict(forward_decode_calls=state.get("_calls", 0),
                          max_seg_len=state.get("_max_seg", 0)),
               per_batch={str(k): v for k, v in state.items()
                          if isinstance(k, int)})
    if "_errors" in state:
        out["hook_errors"] = state["_errors"][:5]
    print("DEBUG forward_decode_calls=", state.get("_calls", 0),
          "max_seg_len=", state.get("_max_seg", 0), flush=True)
    json.dump(out, open(args.output, "w"), indent=1)
    print("wrote", args.output, flush=True)
    # one-line verdict
    for B in args.batches:
        s = state.get(B)
        if s:
            verdict = "SCATTERED" if s["frac_contig"] < 0.5 else \
                      ("MIXED" if s["frac_contig"] < 0.95 else "CONTIGUOUS")
            print(f"  ps{args.page_size} bs{B}: frac_contig={s['frac_contig']:.3f} "
                  f"n_runs={s['n_runs']} max_run={s['max_run']} pages128={s['pages128']}"
                  f"/{s['expected_pages128']} -> {verdict}", flush=True)


if __name__ == "__main__":
    main()
