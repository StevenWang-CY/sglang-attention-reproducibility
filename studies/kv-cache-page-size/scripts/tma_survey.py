"""Survey: which sm120 kernels actually use TMA. Each op = warmup + one profiled
launch (cudaProfilerStart/Stop) so ncu --profile-from-start off captures it.
Robust per-op try/except; prints [ok]/[skip]."""
import torch, torch.cuda.profiler as prof
import flashinfer

dev = "cuda"; bf = torch.bfloat16
def region(name, fn):
    try:
        for _ in range(3): fn()
        torch.cuda.synchronize()
        prof.start(); fn(); torch.cuda.synchronize(); prof.stop()
        print(f"[ok]   {name}", flush=True)
    except Exception as e:
        print(f"[skip] {name}: {type(e).__name__}: {str(e).splitlines()[0][:110]}", flush=True)

M = N = K = 4096
a = torch.randn(M, K, dtype=bf, device=dev); b = torch.randn(K, N, dtype=bf, device=dev)
ah = a.half(); bh = b.half()

# ---- GEMM family ----
region("torch.matmul_bf16(cuBLAS)", lambda: a @ b)
region("torch.matmul_fp16(cuBLAS)", lambda: ah @ bh)
try:
    a8 = a.to(torch.float8_e4m3fn)
    b8 = torch.randn(N, K, dtype=bf, device=dev).to(torch.float8_e4m3fn).t()  # [K,N] col-major
    sc = torch.tensor(1.0, device=dev)
    region("torch._scaled_mm_fp8", lambda: torch._scaled_mm(a8, b8, scale_a=sc, scale_b=sc, out_dtype=bf))
except Exception as e:
    print("[skip] fp8 setup:", e)
for be in ["cutlass", "cudnn", "tgv", "auto"]:
    region(f"fi.gemm.mm_bf16[{be}]", lambda be=be: flashinfer.gemm.mm_bf16(a, b, backend=be))
region("fi.gemm.bmm_bf16[cutlass]",
       lambda: flashinfer.gemm.bmm_bf16(a.unsqueeze(0), b.unsqueeze(0), backend="cutlass"))

# ---- Triton TMA control ----
try:
    import tma_positive_control as tpc
    A = torch.randn(4096, 4096, dtype=bf, device=dev); B = torch.randn(4096, 4096, dtype=bf, device=dev)
    C = torch.empty(4096, 4096, dtype=bf, device=dev)
    g = (4096 // 128, 4096 // 128)
    region("triton_TMA_matmul(control)", lambda: tpc.matmul_tma[g](A, B, C, 4096, 4096, 4096, 128, 128, 64))
except Exception as e:
    print("[skip] triton control:", e)

# ---- Attention family ----
qs = torch.randn(1, 16, 2048, 128, dtype=bf, device=dev); ks = qs.clone(); vs = qs.clone()
region("torch.SDPA_bf16", lambda: torch.nn.functional.scaled_dot_product_attention(qs, ks, vs, is_causal=True))
qp = torch.randn(2048, 16, 128, dtype=bf, device=dev)
kp = torch.randn(2048, 16, 128, dtype=bf, device=dev); vp = kp.clone()
region("fi.single_prefill", lambda: flashinfer.single_prefill_with_kv_cache(qp, kp, vp, causal=True))

# ---- negative control: norm ----
x = torch.randn(8192, 4096, dtype=bf, device=dev); w = torch.randn(4096, dtype=bf, device=dev)
region("fi.rmsnorm(neg-control)", lambda: flashinfer.norm.rmsnorm(x, w))
print("SURVEY DONE", flush=True)
