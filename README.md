# p-means: o-means with a recency-weighted mean at every budget

p-means is a k-means trainer built on [o-means](contrib/o_means/README.md), which is built on
[e-means](contrib/e_means/README.md). It keeps o-means' two-level assignment, EMA update, batch
and step laws and split revival, and changes three things: the centroids are always the ratio of
an EMA of vector sums to an EMA of counts, the batches are larger and the probes fewer, and every
setting is exposed and set by one rule for all datasets. On one H100 it reached the held-out
error of Flash-KMeans@25 faster than e-means on all eight datasets, with a **2.76× geometric
mean speedup**. On DINOv3 10M the speedup over Flash-KMeans@25 was **218.5×**.

[Interactive demo and results](https://livingshade.github.io/p-means/)

![Time to the target error on DINOv3 10M, followed by the H100 results for all eight datasets.](benchs/p_means/assets/h100-race.gif)

## When to use it

p-means is for reproducing the table under Results. It was evaluated on the eight datasets of
the e-means paper's Table 3, with k of 16,384 and 65,536, d from 96 to 1,024, n of 1 and 10
million, and seeds 10 to 19 (10 to 12 at 10 million rows). Its rule was selected on seed 10.

Use e-means when you need its paper's guarantees, the CPU, several GPUs, or no hyperparameters.
Use o-means when you need defaults that were not selected on the evaluation data.

## Quick start

The GPU version needs CUDA-enabled PyTorch, Triton and the o-means package. It was developed and
measured on an NVIDIA H100 80GB HBM3.

```python
import sys
sys.path.insert(0, "contrib/p_means/gpu")
from p_means_gpu import Pmeans

pm = Pmeans(1024, 65536, nepochs=1.4, close_len=0.5, nprobe=4, batch_size=262144)
centroids = pm.train(xg, init_centroids=start)    # (65536, 1024) float32
print(pm.recipe)
```

These are the rule's settings at 10 million rows, with DINOv3 10M's budget. At 1 million rows
the rule uses `batch_size=65536`. It is `RULE` in
[`benchs/p_means/crossing.py`](benchs/p_means/crossing.py).

## How it works

The training loop and the two-level assignment are o-means'. The o-means README describes them.

**Closing mean.** p-means sets each centroid to an EMA of the vector sums assigned to it divided
by an EMA of their counts. This is o-means' recency-weighted closing mean with a window
of half a pass (`close_len=0.5`), applied at every budget. o-means applies it below 2 epochs only.

**Rule.** All eight datasets use `nprobe=4` and `close_len=0.5`, with `batch_size=65536` at
1 million rows and `262144` at 10 million. The rule was found by sweeping `nprobe` and
`batch_size` on seed 10 of each dataset against that seed's Flash-KMeans@25 error and keeping the
setting that crossed it in the least time on most datasets. Seed 10 is one of the evaluated
seeds.

**Kernels.** Four changes to the o-means kernels, each found by profiling on the H100:

1. The cell rebuild counts with `index_add_` instead of `torch.bincount`, which read a value back
   to the host at every rebuild.
2. The coarse kernel processes 64 rows per program instead of 32 when there are more than 128
   cells. At 512 cells and 8 probes this is 1.6 times faster.
3. For d ≥ 512 both scans step through the dimension in slices of 128 instead of 32.
4. For d ≥ 512 the batch is sorted by label and runs of equal labels are summed with a segmented
   scan before the int64 atomics. The sums are bit-identical to o-means'. At d = 1,024 this step
   is 1.9 times faster.

These changes also apply to o-means. The o-means numbers below were measured without them.

## Parameters

| Setting | Default | What it does |
|---|---|---|
| `d`, `k` | required | Dimension and number of clusters |
| `nepochs` | `1.0` | Passes over the data |
| `seed` | `1234` | As in o-means |
| `batch_size` | o-means' batch law | Rows per step |
| `nprobe` | o-means' rule | Cells each row probes |
| `nlist` | o-means' rule | Number of coarse cells |
| `close_len` | `1.0` | Length of the closing-mean window, in passes |
| `alpha0`, `alpha_scale` | o-means' step law | EMA step size |
| `rebuilds_per_pass`, `first_pass_rebuild_every` | `2`, every revival check | Cell rebuild cadence |
| `unused_threshold`, `check_every` | as in o-means | Split revival |
| `launch` | H100 tiles | Triton block sizes |
| `final_nprobe`, `final_frac` | off | More probes for the last part of the run |

All of them were available for tuning. The ones used are listed under Results.

## Results

One NVIDIA H100 80GB HBM3, with the e-means paper's data, held-out split,
starting rows, warm-up, timer and exact scorer
([`benchs/e_means/paper_xb.py`](benchs/e_means/paper_xb.py)). Each method is run at budgets of
1.0, 1.2, 1.4, ... epochs, and for each seed the first budget whose test error is at or below that
seed's Flash-KMeans@25 error counts, as in the paper's Table 3. Lloyd and e-means were not
re-run: Flash-KMeans' per-seed errors and times are the paper's
([`paper_targets.json`](benchs/e_means/paper_targets.json)), timed on the same GPU model, and
e-means' speedups are its Table 3.

| Dataset | e-means (paper): epochs, speedup | o-means: epochs, speedup | p-means: epochs, speedup | p-means / e-means |
| --- | ---: | ---: | ---: | ---: |
| DEEP* 1M | 2.0, 12.8× | 1.8, 20.1× | 2.4, 29.2× | 2.28× |
| DINOv3/4096 1M | 2.0, 12.7× | 1.8, 20.1× | 2.8, 30.5× | 2.40× |
| SPACEV 1M | 2.4, 9.5× | 2.2, 13.2× | 3.8, 15.1× | 1.59× |
| SimSearchNet++ 1M | 1.4, 12.9× | 1.2, 21.1× | 1.6, 30.8× | 2.39× |
| DEEP* 10M | 2.0, 27.8× | 1.6, 48.5× | 3.0, 131.3× | 4.72× |
| DINOv3/4096 10M | 1.4, 23.3× | 1.0, 66.2× | 1.4, 218.5× | 9.38× |
| SPACEV 10M | 2.0, 19.2× | 2.6, 23.7× | 10.2, 23.2× | 1.21× |
| SimSearchNet++ 10M | 1.6, 23.8× | 2.4, 31.9× | 4.0, 72.5× | 3.05× |

Geometric means over the eight datasets: p-means 2.76× e-means, o-means 1.62× e-means, p-means
46.0× Flash-KMeans@25.

p-means ran the rule above on every dataset. o-means ran its defaults. The two SPACEV rows were
measured in a second session on the same machine, where a reference run was about 8% slower than in
the first. A SPACEV-specific setting (`nprobe=8`) measured 16.2× at 1M rows and 42.5× at 10M; it
is not used, because it is not the rule.

The [benchmark README](benchs/p_means/README.md) has the commands, the raw results and the
profiles.

## Limitations

- p-means needs more epochs than e-means on seven of the eight datasets, up to 5.1 times as
  many (10.2 against 2.0 on SPACEV 10M). Its speedup comes from a lower cost per epoch.
- On SPACEV 10M the rule is slower than o-means' defaults (23.2× against 23.7×).
- The rule was selected on seed 10, which is also evaluated.
- At 10M rows, 3 seeds were evaluated. Table 3 uses 10.
- Our H100 was about 9% faster than the paper's: e-means took 0.518 s on DEEP* 10M here, against
  0.569 s implied by Table 3. The speedups over Flash-KMeans and e-means include this.
- Test error at a fixed budget varied by up to about 0.1% between identical runs, which is larger
  than several of the margins that decided a crossing. At 1M rows the run time depends on the
  CPU that launches the kernels. One DEEP* 1M run measured 35.3×; the table
  reports the final run, 29.2×.
- o-means reported 4.49× over e-means on an RTX 5090, against e-means' Triton kernels. Against
  the paper's CuTe e-means on an H100 it measured 1.62×.

## Faiss

This repository is a fork of [Faiss](https://github.com/facebookresearch/faiss), Meta's library
for similarity search and clustering of dense vectors. Everything outside `contrib/p_means` and
`benchs/p_means` is upstream Faiss, e-means and o-means. See [INSTALL.md](INSTALL.md) to build it
and the [upstream README](https://github.com/facebookresearch/faiss/blob/main/README.md) for its
documentation, authors and citation. Faiss is MIT-licensed ([LICENSE](LICENSE)), Copyright © Meta
Platforms, Inc.
