"""Isolated profile of the trtllm-gen low-latency fp8 GEMM (mm_fp8). Cached."""
import torch, torch.cuda.profiler as prof, flashinfer
dev="cuda"; bf=torch.bfloat16; f8=torch.float8_e4m3fn
a=torch.randn(16,7168,dtype=bf,device=dev).to(f8)
b=torch.randn(256,7168,dtype=bf,device=dev).to(f8)
def run(): return flashinfer.gemm.mm_fp8(a,b)
for _ in range(5): run()
torch.cuda.synchronize()
prof.start(); run(); torch.cuda.synchronize(); prof.stop()
print("done", flush=True)
