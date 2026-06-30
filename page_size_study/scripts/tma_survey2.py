"""Survey round 2: FlashInfer fp8 GEMM + MoE (sm120 cutlass) TMA candidates."""
import sys; sys.path.insert(0, "/home/wangcy07/sglang_log")
import torch, torch.cuda.profiler as prof, flashinfer
dev="cuda"; bf=torch.bfloat16; f8=torch.float8_e4m3fn
def region(name, fn):
    try:
        for _ in range(3): fn()
        torch.cuda.synchronize(); prof.start(); fn(); torch.cuda.synchronize(); prof.stop()
        print(f"[ok]   {name}", flush=True)
    except Exception as e:
        print(f"[skip] {name}: {type(e).__name__}: {str(e).splitlines()[0][:120]}", flush=True)

# --- fp8 GEMM via FlashInfer (bmm_fp8: cublas/cutlass/cudnn) ---
try:
    A=torch.randn(1,2048,2048,dtype=bf,device=dev).to(f8); B=torch.randn(1,2048,2048,dtype=bf,device=dev).to(f8)
    As=torch.ones(1,device=dev); Bs=torch.ones(1,device=dev)
    for be in ["cutlass","cudnn","cublas"]:
        region(f"fi.bmm_fp8[{be}]", lambda be=be: flashinfer.gemm.bmm_fp8(A,B,As,Bs,bf,backend=be))
except Exception as e: print("[skip] bmm_fp8 setup:",str(e)[:100])

# --- mm_fp8 trtllm_low_latency (the mm_M1_16 special path) ---
try:
    a=torch.randn(16,7168,dtype=bf,device=dev).to(f8); b=torch.randn(256,7168,dtype=bf,device=dev).to(f8)
    region("fi.mm_fp8[trtllm_ll]", lambda: flashinfer.gemm.mm_fp8(a,b))
except Exception as e: print("[skip] mm_fp8 setup:",str(e)[:100])

# --- cutlass fused MoE bf16 (sm120 cutlass module) ---
try:
    nt,hid,ne,tk,inter=256,2048,8,2,1024
    inp=torch.randn(nt,hid,dtype=bf,device=dev)
    sel=torch.randint(0,ne,(nt,tk),device=dev,dtype=torch.int)
    sc=torch.rand(nt,tk,device=dev,dtype=torch.float32)
    fc1=torch.randn(ne,2*inter,hid,dtype=bf,device=dev)
    fc2=torch.randn(ne,hid,inter,dtype=bf,device=dev)
    region("fi.cutlass_fused_moe_bf16",
           lambda: flashinfer.fused_moe.cutlass_fused_moe(inp,sel,sc,fc1,fc2,bf,[]))
except Exception as e: print("[skip] cutlass_moe setup:",str(e)[:120])

# --- trtllm_bf16_moe ---
try:
    nt,hid,ne,tk,inter=256,2048,8,2,1024
    rl=torch.randn(nt,ne,dtype=bf,device=dev)
    hs=torch.randn(nt,hid,dtype=bf,device=dev)
    g1=torch.randn(ne,2*inter,hid,dtype=bf,device=dev); g2=torch.randn(ne,hid,inter,dtype=bf,device=dev)
    region("fi.trtllm_bf16_moe",
           lambda: flashinfer.fused_moe.trtllm_bf16_moe(rl,None,hs,g1,g2,ne,tk,None,None,inter,0,ne))
except Exception as e: print("[skip] trtllm_bf16_moe setup:",str(e)[:120])
print("SURVEY2 DONE", flush=True)
