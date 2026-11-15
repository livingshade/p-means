"""Microbenchmark of the coarse step: o-means' kernel against a sort-based variant."""
import sys
import os
import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                "contrib", "p_means", "gpu"))
from faiss.contrib.o_means import triton_two_level as tl2


@triton.jit
def _coarse_noatomic(x_ptr, idx_ptr, g_ptr, gn_ptr, pc_ptr, R, D, sx, sg,
                     HAS_IDX: tl.constexpr, P: tl.constexpr, GP: tl.constexpr,
                     BLOCK_B: tl.constexpr, BLOCK_D: tl.constexpr):
    pid = tl.program_id(0)
    r = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    rmask = r < R
    if HAS_IDX:
        rows = tl.load(idx_ptr + r, mask=rmask, other=0).to(tl.int64)
    else:
        rows = r.to(tl.int64)
    xbase = x_ptr + rows[:, None] * sx
    gid = tl.arange(0, GP)
    gbase = g_ptr + gid[:, None].to(tl.int64) * sg
    acc = tl.zeros((BLOCK_B, GP), tl.float32)
    for d0 in range(0, D, BLOCK_D):
        col = d0 + tl.arange(0, BLOCK_D)
        xt = tl.load(xbase + col[None, :], mask=rmask[:, None], other=0.0)
        gt = tl.load(gbase + col[None, :])
        acc = tl.dot(xt, tl.trans(gt), acc=acc, out_dtype=tl.float32)
    score = tl.load(gn_ptr + gid)[None, :] - 2.0 * acc
    key = tl2._pack_keys(score, gid, GP)
    for s in tl.static_range(P):
        kmin = tl.min(key, axis=1)
        bj = kmin & (GP - 1)
        key = tl.where(key == kmin[:, None], 2147483647, key)
        tl.store(pc_ptr + r * P + s, bj, mask=rmask)


def bench(f, n=20):
    f(); torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n):
        f()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / n * 1000


R, dp = 524288, 128
for G in (512, 256):
    for P in (1, 4, 8):
        GP = G
        x = torch.randn(4 * R, dp, device="cuda").half()
        idx = torch.randperm(4 * R, device="cuda")[:R]
        g = torch.randn(GP, dp, device="cuda").half()
        gn = g.float().square().sum(1)
        cnt = torch.zeros(GP, dtype=torch.int32, device="cuda")
        bucket = torch.empty((G, R), dtype=torch.int32, device="cuda")
        pc = torch.empty(R * P, dtype=torch.int32, device="cuda")
        out = {}
        for rows, warps in ((32, 8), (64, 8)):
            def orig():
                cnt.zero_()
                tl2._coarse_kernel[(triton.cdiv(R, rows),)](
                    x, idx, g, gn, cnt, bucket, R, dp, x.stride(0), g.stride(0), R,
                    HAS_IDX=True, P=P, GP=GP, BLOCK_B=rows, BLOCK_D=32, num_warps=warps)
            def noat():
                _coarse_noatomic[(triton.cdiv(R, rows),)](
                    x, idx, g, gn, pc, R, dp, x.stride(0), g.stride(0),
                    HAS_IDX=True, P=P, GP=GP, BLOCK_B=rows, BLOCK_D=32, num_warps=warps)
            def noat_sort():
                noat()
                o = torch.argsort(pc)
                c = torch.bincount(pc, minlength=G)
                return o, c
            out["orig%d" % rows] = bench(orig)
            out["noatomic%d" % rows] = bench(noat)
            out["noatomic+sort%d" % rows] = bench(noat_sort)
        print(G, P, {k: "%.0fus" % v for k, v in out.items()}, flush=True)
