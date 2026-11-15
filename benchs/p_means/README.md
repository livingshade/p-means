# p-means on an H100, by e-means' own rules

These scripts time the GPU p-means in [`contrib/p_means/gpu`](../../contrib/p_means/gpu) with
the e-means paper's protocol, on the paper's datasets, and compare it with the paper's stored
numbers instead of re-running the baselines.

## What is reused and what is measured

| Method | Where the number comes from |
| --- | --- |
| Lloyd k-means (Flash-KMeans 0.3.1, 25 iterations) | `../e_means/paper_targets.json`: the paper's per-seed test error and seconds, on an `NVIDIA H100 80GB HBM3` |
| e-means | The paper's Table 3 (`--input cast` column): epochs to reach Flash@25 and the median speedup over it, reproduced in `../e_means/README.md` |
| o-means | Measured here, o-means defaults, same protocol |
| p-means | Measured here, one rule for all datasets, same protocol |

The e-means README says test errors transfer to any hardware and times only to an H100. We
used the same card (`NVIDIA H100 80GB HBM3`). As a check, one e-means run of the paper's
DEEP* 10M setting (seed 10, 2 epochs, CuTe, `--input cast`) took 0.518 s here. The time implied
by Table 3 for that dataset is 0.569 s (Flash's 15.83 s over the 27.8x median speedup), so our
H100 is about 9% faster than the paper's. The speedups below, over Flash-KMeans and e-means,
are therefore about 9% generous to p-means.

Everything else follows `../e_means/paper_xb.py`: the same data files and test split, the
paper's starting rows (`starts`), one untimed identical warm-up call, then the median of three
timed calls when the first is under 2 s, and the exact float32/float64 scorer (`test_error`).
One difference: all budgets of one (dataset, seed) run in one process, one after another, each
with its own warm-up. The paper starts a fresh process per run.

The crossing rule is Table 3's. Budgets go 1.0, 1.2, 1.4, ... epochs. For each seed, the first
budget whose test error is at or below that seed's Flash@25 error is the crossing, and the
seed's speedup is Flash's time over the time at that budget. A cell is the upper median of the
crossings and the median of the per-seed speedups. Seeds 10 to 19 at 1M rows and 10 to 12 at
10M rows (Table 3 uses 10 to 19 everywhere).

## How the p-means rule was picked

By sweeping seed 10 of each dataset against its Flash@25 target and keeping what crossed
cheapest. Seed 10 is also one of the evaluated seeds. The cheapest setting was the same on every
dataset except SPACEV, where `nprobe=8` was faster. The reported results use the common setting
everywhere (`RULE` in `crossing.py`, run with `--rule` for SPACEV), so that one rule covers all
eight datasets. The SPACEV-specific runs stay in `results/crossing.jsonl`.

## Files

| File | What it is |
| --- | --- |
| `pm_xb.py` | Times a list of p-means / o-means configs on one (dataset, seed) with the paper's protocol |
| `crossing.py` | Table 3's crossing sweep, with the p-means rule (`--rule`) and the per-dataset recipes it came from |
| `summarize.py` | Builds the table below from `results/*.jsonl` |
| `sweep.sh` | Prints one line per config (used for the tuning sweeps) |
| `profile_pm.py` | Kernel-time breakdown of one p-means call |
| `micro_coarse.py`, `test_accumulate.py` | Kernel microbenchmarks; the second also checks that the sorted accumulation matches o-means' exactly |
| `results/` | Raw JSON lines: `crossing.jsonl` (p-means, first session), `rule.jsonl` (p-means with the rule on SPACEV, second session), `check.jsonl` (the second session's reference run), `omeans.jsonl` (o-means), `explore/` (tuning sweeps) |

## Running

Set up the e-means paper environment and data first (`../e_means/README.md`, sections 1 and 2,
`--results 10M`), and copy `contrib/e_means` and `contrib/o_means` into the installed
`faiss/contrib`. Then:

```bash
export EMEANS_DATA=/path/to/data EMEANS_CACHE=/path/to/cache
python crossing.py --datasets deep1b-1M spacev-1M fb-ssnpp-1M dino10b-1M-div4096 \
    --seeds 10 11 12 13 14 15 16 17 18 19
python crossing.py --datasets deep1b-10M spacev-10M fb-ssnpp-10M dino10b-10M-div4096 --seeds 10 11 12
python crossing.py --rule --file rule.jsonl --datasets spacev-1M --seeds 10 11 12 13 14 15 16 17 18 19
python crossing.py --rule --file rule.jsonl --datasets spacev-10M --seeds 10 11 12
python crossing.py --method omeans --file omeans.jsonl --datasets deep1b-1M ... --seeds 10 11 12
python summarize.py results/crossing.jsonl
python summarize.py results/rule.jsonl
python summarize.py results/omeans.jsonl
```

## Results

One `NVIDIA H100 80GB HBM3`, PyTorch 2.15.0, Triton 3.9.0, run on 2026-11-15. Each cell is
epochs at the crossing, then the median speedup over the paper's Flash-KMeans@25. Flash@25 is the
median of the paper's 10 seeds. The last two columns divide by e-means' Table 3 speedup.

| Dataset | Flash@25 | e-means (paper) | o-means (here) | p-means (here) | o-means vs e-means | p-means vs e-means |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DEEP* 1M | 0.41 s | 2.0, 12.8x | 1.8, 20.1x | 2.4, 29.2x | 1.57x | 2.28x |
| DINOv3/4096 1M | 1.70 s | 2.0, 12.7x | 1.8, 20.1x | 2.8, 30.5x | 1.58x | 2.40x |
| SPACEV 1M | 0.33 s | 2.4, 9.5x | 2.2, 13.2x | 3.8, 15.1x | 1.39x | 1.59x |
| SimSearchNet++ 1M | 0.48 s | 1.4, 12.9x | 1.2, 21.1x | 1.6, 30.8x | 1.64x | 2.39x |
| DEEP* 10M | 15.84 s | 2.0, 27.8x | 1.6, 48.5x | 3.0, 131.3x | 1.75x | 4.72x |
| DINOv3/4096 10M | 68.00 s | 1.4, 23.3x | 1.0, 66.2x | 1.4, 218.5x | 2.84x | **9.38x** |
| SPACEV 10M | 11.21 s | 2.0, 19.2x | 2.6, 23.7x | 10.2, 23.2x | 1.24x | 1.21x |
| SimSearchNet++ 10M | 21.78 s | 1.6, 23.8x | 2.4, 31.9x | 4.0, 72.5x | 1.34x | 3.05x |

Geometric means over the eight datasets: p-means 2.76x e-means, o-means 1.62x e-means,
p-means 1.71x o-means, p-means 46.0x Flash-KMeans@25.

The p-means rule (all `close_len=0.5`): `nprobe=4, batch_size=65536` at 1M rows and `nprobe=4,
batch_size=262144` at 10M rows. o-means ran its defaults.

The SPACEV rows come from a second session on the same machine. Its reference run (SPACEV 1M, seed
10, `nprobe=8`, 3.0 epochs, `results/check.jsonl`) measured 15.3x, against 16.4x and 17.0x for the
same run in the first session, so the second session was about 8% slower. With the rule, SPACEV
10M needed 8.4 to 10.4 epochs (one seed was still 0.02% above target at 10.0). The
SPACEV-specific setting measured 3.0, 16.2x at 1M rows and 4.0, 42.5x at 10M rows (`nprobe=8`,
`batch_size=524288` at 10M).

Things to keep in mind when reading the table:

- **Different budgets.** p-means needs more epochs than e-means everywhere except DINOv3 10M (3.0
  against 2.0 on DEEP* 10M, 10.2 against 2.0 on SPACEV 10M). It wins on cost per epoch. On
  SPACEV 10M the rule is slower than o-means' defaults.
- **Noise.** With nondeterministic atomics, one seed's error at a fixed budget moved by up to
  about 0.1% between identical runs, and the error is not monotone in the budget. On DEEP* 10M,
  p-means sat 0.1 to 0.3% above the target from 2.0 to 2.8 epochs before crossing at 3.0. At
  1M rows the runs are limited by how fast the CPU launches kernels.
  DEEP* 1M measured 35.3x in an earlier run of the same recipe and 29.2x in the final run.
- **The o-means numbers are not its README's.** Its 4.49x was measured on an RTX 5090 against
  e-means' Triton kernels. Here e-means is the paper's H100 CuTe run, and the o-means README
  already expected a smaller gap on an H100.
- **Seed 10 picked the rule** and is also one of the evaluated seeds.

### What changed in the kernels

Profiling showed four costs. Fixing them made p-means faster than the o-means kernels it starts from:

1. o-means' cell rebuild calls `torch.bincount`, which reads a value back to the host and stalls
   the launch queue at every rebuild (every five steps in the first pass). Counting with
   `index_add_` removed the stall: DEEP* 1M went from 12.2 to 10.9 ms.
2. The coarse kernel processes 32 rows per program, tuned on an RTX 5090. On the H100, 64 rows
   per program is 1.6x faster at 512 cells and 8 probes (`micro_coarse.py`). Removing the
   kernel's per-probe atomics would save only 10-25% at 512 cells, so they stayed.
3. Both scans step through the dimension in slices of 32. At d = 1024, slices of 128 cut a
   DINOv3 1M run from 73 to 61 ms. Past 128 cells, the coarse tile then needs 442 KB of shared
   memory, so wide slices are only used for the fine scan there.
4. At d = 1024 the accumulation of batch sums issued 67 million int64 atomics per step and was
   the largest kernel (31%). Sorting the batch by label and summing runs of equal labels with a
   segmented scan before the atomics halves it (431 to 226 us). The int64 sums are bit-identical
   to o-means' (`test_accumulate.py`). With 3, DINOv3 1M went to 56 ms.

Items 3 and 4 are on by default only for d ≥ 512. At d = 96 to 256 they made no difference.

Tried and dropped: retuning the fine kernel's tile shapes (within 2%), CUDA graphs (at 1M rows
the GPU was already busy 10 of every 12.6 ms), a larger probe count for the closing stretch
(`final_nprobe`, no gain), disabling revival (+2% error), and smaller batches with larger steps.
