# p-means

[`gpu/p_means_gpu.py`](gpu/p_means_gpu.py) is p-means for one CUDA GPU. It sets each centroid to
the ratio of an EMA of the vector sums assigned to it to an EMA of their counts. This is
the o-means recency-weighted closing mean, applied at every budget and over a window of half a
pass (`close_len=0.5`). The training loop, two-level assignment, EMA steps and split revival come
from [o-means](../o_means/README.md) and [e-means](../e_means/README.md). The changes are a larger
batch, fewer probes, a cell rebuild that does not synchronize with the host, coarse-kernel tiles
retuned for the H100, and, for d ≥ 512, wider dimension slices and label-sorted accumulation.

It needs CUDA-enabled PyTorch with Triton, an NVIDIA GPU with compute capability 8.0 or newer, and
the `faiss.contrib.o_means` package (copy `contrib/e_means` and `contrib/o_means` into the
installed `faiss/contrib`).

```python
import sys
import torch
sys.path.insert(0, "contrib/p_means/gpu")
from p_means_gpu import Pmeans

x = torch.randn(1_000_000, 96, device="cuda")
init = x[torch.randperm(len(x), device="cuda")[:16384]]
pm = Pmeans(96, 16384, nepochs=2.4, seed=10, nprobe=4, batch_size=65536, close_len=0.5)
centroids = pm.train(x, init)
```

Every setting is a keyword: `nprobe`, `nlist`, `batch_size`, `alpha0`, `alpha_scale`,
`close_len`, `check_every`, `rebuilds_per_pass`, `unused_threshold`, `final_nprobe` /
`final_frac`, and the kernel launch shapes in `launch`. The per-dataset values and the H100
results, measured with the e-means paper's protocol against its stored Flash-KMeans and e-means
numbers, are in [`benchs/p_means`](../../benchs/p_means/README.md).
