"""fp8 GEMM TMA profile (kernels already cached by survey2)."""
import torch, torch.cuda.profiler as prof, flashinfer
dev="cuda"; bf=torch.bfloat16; f8=torch.float8_e4m3fn
def region(name, fn):
    try:
        for _ in range(3): fn()
        torch.cuda.synchronize(); prof.start(); fn(); torch.cuda.synchronize(); prof.stop()
        print(f"[ok] {name}", flush=True)
    except Exception as e:
        print(f"[skip] {name}: {str(e).splitlines()[0][:100]}", flush=True)
A=torch.randn(1,2048,2048,dtype=bf,device=dev).to(f8); B=torch.randn(1,2048,2048,dtype=bf,device=dev).to(f8)
As=torch.ones(1,device=dev); Bs=torch.ones(1,device=dev)
for be in ["cutlass","cudnn","cublas"]:
    region(f"fi.bmm_fp8[{be}]", lambda be=be: flashinfer.gemm.bmm_fp8(A,B,As,Bs,bf,backend=be))
a=torch.randn(16,7168,dtype=bf,device=dev).to(f8); b=torch.randn(256,7168,dtype=bf,device=dev).to(f8)
region("fi.mm_fp8[trtllm_ll]", lambda: flashinfer.gemm.mm_fp8(a,b))
print("DONE", flush=True)
