"""
tma_positive_control.py — Report 6 positive control: a Triton matmul that
*explicitly* uses TMA (`tl.make_tensor_descriptor` → cp.async.bulk.tensor),
to prove (a) sm120 can run TMA and (b) the ncu TMA counters fire (so the TMA=0
seen for XQA/cuBLAS is genuine, not a GeForce measurement cap).

Profiled on RTX 5060 Ti (sm120) with ncu 2025.3.1:
  l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum = 2.15 GB  (NONZERO)
  l1tex__m_l1tex2xbar_req_cycles_active_op_tma.sum        = 17.8 M
  l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ldgsts.sum = 0       (no cp.async)
vs XQA kernel_mha: TMA=0, ldgsts~20k;  cuBLAS bf16 GEMM (cutlass_80): TMA=0.

Run:  python tma_positive_control.py        (correctness, sudo-free)
Profile (root):  ncu --profile-from-start off --metrics \
  l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum,... python tma_positive_control.py
"""
import torch, triton, triton.language as tl
import torch.cuda.profiler as prof


def alloc_fn(size, alignment, stream):       # device-side TMA descriptors need scratch
    return torch.empty(size, dtype=torch.int8, device="cuda")
triton.set_allocator(alloc_fn)


@triton.jit
def matmul_tma(a_ptr, b_ptr, c_ptr, M, N, K,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    a_desc = tl.make_tensor_descriptor(a_ptr, [M, K], [K, 1], [BM, BK])
    b_desc = tl.make_tensor_descriptor(b_ptr, [K, N], [N, 1], [BK, BN])
    c_desc = tl.make_tensor_descriptor(c_ptr, [M, N], [N, 1], [BM, BN])
    om = pid_m * BM; on_ = pid_n * BN
    acc = tl.zeros([BM, BN], tl.float32)
    for k in range(0, K, BK):
        a = a_desc.load([om, k])     # TMA load
        b = b_desc.load([k, on_])    # TMA load
        acc = tl.dot(a, b, acc)
    c_desc.store([om, on_], acc.to(tl.bfloat16))


def main():
    M = N = K = 4096
    BM = BN = 128; BK = 64
    a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
    c = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    grid = (M // BM, N // BN)
    for _ in range(5):
        matmul_tma[grid](a, b, c, M, N, K, BM, BN, BK)
    torch.cuda.synchronize()
    ref = a.float() @ b.float()
    rel = (c.float() - ref).abs().max().item() / ref.abs().max().item()
    print("max rel err vs torch:", rel)
    prof.start()
    matmul_tma[grid](a, b, c, M, N, K, BM, BN, BK)
    torch.cuda.synchronize()
    prof.stop()
    print("done")


if __name__ == "__main__":
    main()
