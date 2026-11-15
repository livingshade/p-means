"""accumulate_sorted must give the same int64 sums and counts as o-means' accumulate."""
import os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                "contrib", "p_means", "gpu"))
from p_means_gpu import accumulate_sorted
from faiss.contrib.o_means.triton_kernels import accumulate
for n, d, dp, k, R in [(5000, 96, 96, 300, 4000), (20000, 1000, 1024, 16384, 65536),
                       (300000, 100, 128, 65536, 262144), (100, 7, 32, 3, 1)]:
    x = torch.randn(n, dp, device="cuda").half()
    idx = torch.randint(0, n, (R,), device="cuda")
    lab = torch.randint(0, k, (R,), device="cuda", dtype=torch.int32)
    for use_idx in (True, False):
        xi = x if use_idx else x[:R] if R <= n else None
        if xi is None:
            continue
        a_s = torch.zeros(k, dp, dtype=torch.int64, device="cuda"); a_c = torch.zeros(k, dtype=torch.int32, device="cuda")
        b_s = torch.zeros_like(a_s); b_c = torch.zeros_like(a_c)
        accumulate(xi, lab, a_s, a_c, d, idx=idx if use_idx else None)
        accumulate_sorted(xi, lab, b_s, b_c, d, idx=idx if use_idx else None)
        assert torch.equal(a_s, b_s) and torch.equal(a_c, b_c), (n, d, k, R, use_idx)
        f = lambda fn: fn(xi, lab, a_s, a_c, d, idx=idx if use_idx else None)
        for fn in (accumulate, accumulate_sorted):
            f(fn); torch.cuda.synchronize()
            s, e = torch.cuda.Event(True), torch.cuda.Event(True); s.record()
            for _ in range(20): f(fn)
            e.record(); torch.cuda.synchronize()
            print(n, d, k, R, use_idx, fn.__name__, "%.0fus" % (s.elapsed_time(e) / 20 * 1000))
print("accumulate_sorted matches")
