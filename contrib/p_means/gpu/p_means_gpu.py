"""p-means on one NVIDIA GPU.

p-means' rule is that a centroid is the ratio of two exponential moving averages, one of the
vector sums and one of the counts of the rows assigned to it. On the GPU this is o-means'
recency-weighted closing mean, with every batch weighted by exp(-(rows seen after it) / (L n)),
used at every budget instead of only below two epochs. Training reuses o-means' Triton kernels
(coarse cells, then the centroids of the probed cells) and e-means' EMA steps and revival, with
every knob exposed so it can be tuned per dataset. That last part is the point.
"""
import math
from dataclasses import replace

import torch
import triton
import triton.language as tl

from faiss.contrib.e_means.base import (
    _CHECK_UNUSED_EVERY,
    _epoch_rows,
    _epoch_seed,
    _resolve_recipe,
    _resolve_unused_threshold,
    _scheduled_alpha,
)
from faiss.contrib.o_means.o_means import _Training, _check_rows
from faiss.contrib.o_means import triton_two_level as tl2
from faiss.contrib.o_means.schedule import (
    Schedule,
    Step,
    num_cells,
    num_probes,
    resolve_batch_law,
)
from faiss.contrib.o_means.triton_kernels import accumulate, ema_update

__all__ = ["Pmeans"]

# Launch shapes of o-means' two-level kernels, retuned for the H100 (o-means tuned them on an
# RTX 5090). Keys: coarse_rows, coarse_warps, coarse_agg, fine_bt, fine_bk, fine_warps,
# fine_stages.
LAUNCH = dict(coarse_rows=None, coarse_warps=None, coarse_agg=None, fine_bt=64, fine_bk=None,
              fine_warps=4, fine_stages=2)

# For wide vectors (measured at d = 1024 on an H100): 128-wide K slices in both scans and
# label-sorted accumulation cut a DINOv3 1M run by 24%. They did nothing at d = 96 to 256.
WIDE_LAUNCH = dict(fine_bd=128, coarse_bd=128, sorted_accumulate=True)


@triton.jit
def _seg_add(f1, v1, f2, v2):
    # Segmented sum: a row that starts a segment (f2) drops what came before it.
    return f1 | f2, tl.where(f2 != 0, v2, v1 + v2)


@triton.jit(do_not_specialize=["R"])
def _accumulate_sorted_kernel(x_ptr, rows_ptr, lab_ptr, sum_ptr, cnt_ptr, R, d, sx, ss,
                              BLOCK_R: tl.constexpr, BLOCK_D: tl.constexpr):
    # Rows arrive sorted by label. Each program sums the runs of equal labels among its
    # BLOCK_R rows with a segmented scan and issues one atomic per run instead of one per row.
    r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    rm = r < R
    rows = tl.load(rows_ptr + r, mask=rm, other=0).to(tl.int64)
    lab = tl.load(lab_ptr + r, mask=rm, other=-1).to(tl.int64)
    prev = tl.load(lab_ptr + r - 1, mask=rm & (r > 0), other=-2).to(tl.int64)
    nxt = tl.load(lab_ptr + r + 1, mask=(r + 1) < R, other=-3).to(tl.int64)
    first = tl.arange(0, BLOCK_R) == 0
    last = tl.arange(0, BLOCK_R) == BLOCK_R - 1
    start = ((lab != prev) | first).to(tl.int32)
    end = rm & ((lab != nxt) | last)
    for d0 in range(0, d, BLOCK_D):
        col = d0 + tl.arange(0, BLOCK_D)
        m = rm[:, None] & (col < d)[None, :]
        v = tl.load(x_ptr + rows[:, None] * sx + col[None, :], mask=m, other=0.0).to(tl.float32)
        v = (v * 16777216.0).to(tl.int64)
        f = tl.broadcast_to(start[:, None], (BLOCK_R, BLOCK_D))
        _, acc = tl.associative_scan((f, v), 0, _seg_add)
        tl.atomic_add(sum_ptr + lab[:, None] * ss + col[None, :], acc,
                      mask=end[:, None] & (col < d)[None, :], sem="relaxed")
    ones = tl.where(rm, 1, 0).to(tl.int32)
    _, c = tl.associative_scan((start, ones), 0, _seg_add)
    tl.atomic_add(cnt_ptr + lab, c, mask=end, sem="relaxed")


def accumulate_sorted(x16, labels, sums, cnt, d, idx=None, block_r=32):
    """Same result as o-means' accumulate (exact int64 sums), with the rows sorted by label
    first so that rows of the same centroid are added up before the atomics."""
    R = int(labels.shape[0])
    if R == 0:
        return
    sl, order = torch.sort(labels)
    rows = idx[order] if idx is not None else order
    block_d = min(128, triton.next_power_of_2(d))
    _accumulate_sorted_kernel[(triton.cdiv(R, block_r),)](
        x16, rows, sl, sums, cnt, R, d, x16.stride(0), sums.stride(0),
        BLOCK_R=block_r, BLOCK_D=block_d, num_warps=4)


def _block_d(dp, want):
    """Largest of want, ..., 32 that divides the padded dimension dp."""
    b = want or 32
    while dp % b:
        b //= 2
    return max(b, 32)


def two_level_assign(x16, c16, cn, cells, ws, p, idx=None, rows=None, launch=LAUNCH):
    """o-means' two_level_assign with its launch shapes exposed."""
    R = int(idx.shape[0]) if idx is not None else int(rows)
    dp = x16.shape[1]
    G, GP = cells.G, cells.GP
    if R == 0:
        return ws.labels[:0]
    xi = idx if idx is not None else x16
    coarse, coarse_rows, coarse_warps, block_k = tl2._launch_params(GP, dp)
    if launch.get("coarse_agg") is not None:
        coarse = tl2._coarse_agg_kernel if launch["coarse_agg"] else tl2._coarse_kernel
    if GP > 128 and launch.get("coarse_rows") is None:
        coarse_rows = 64       # H100: 1.6x faster than o-means' 32 at 512 cells and 8 probes
    coarse_rows = launch.get("coarse_rows") or coarse_rows
    coarse_warps = launch.get("coarse_warps") or coarse_warps
    block_k = launch.get("fine_bk") or block_k
    coarse[(triton.cdiv(R, coarse_rows),)](
        x16, xi, cells.g16, cells.gn, ws.cnt, ws.bucket, R, dp, x16.stride(0),
        cells.g16.stride(0), ws.rows,
        HAS_IDX=idx is not None, P=p, GP=GP, BLOCK_B=coarse_rows,
        # The coarse tile holds all GP cell centres: past 128 cells a 128-wide slice needs
        # more shared memory than an H100 has, so wide slices are for small GP only.
        BLOCK_D=_block_d(dp, launch.get("coarse_bd") if GP <= 128 else 32),
        num_warps=coarse_warps)
    bt = launch.get("fine_bt", 64)
    tl2._fine_kernel[(triton.cdiv(R * p, bt) + G,)](
        x16, xi, c16, cn, cells.off, ws.cnt, ws.bucket, ws.best, ws.lab,
        G, dp, x16.stride(0), c16.stride(0), ws.rows, ws.rows,
        HAS_IDX=idx is not None, P=p, GP=GP, BT=bt, BLOCK_K=block_k,
        BLOCK_D=_block_d(dp, launch.get("fine_bd")), num_warps=launch.get("fine_warps", 4),
        num_stages=launch.get("fine_stages", 2))
    tl2._reduce_child_kernel[(triton.cdiv(R, 128),)](
        ws.best, ws.lab, ws.labels, cells.child, x16, xi, c16, cn, ws.cnt,
        R, G, dp, x16.stride(0), c16.stride(0), ws.rows,
        HAS_IDX=idx is not None, P=p, GP=GP, BLOCK=128, BLOCK_D=32, num_warps=4)
    return ws.labels[:R]


def make_schedule(n, d, k, nepochs, *, batch_law="auto", batch_size=None, alpha0=None,
                  alpha_scale=1.0, nlist=None, nprobe=None, close_len=1.0, close="recency",
                  check_every=_CHECK_UNUSED_EVERY, rebuilds_per_pass=2, unused_threshold=None):
    law = resolve_batch_law(n, batch_law)
    batch, a0 = _resolve_recipe(n, k, max(nepochs, 1.0), law, batch_size, alpha0)
    a0 *= alpha_scale
    rows = tuple(_epoch_rows(n, nepochs))
    tau = _resolve_unused_threshold(unused_threshold, d)
    nlist = min(nlist or num_cells(k), k)
    nprobe = min(nprobe or num_probes(d, k), nlist)
    steps_per_pass = -(-n // batch)
    rebuild_every = max(check_every, -(-steps_per_pass // rebuilds_per_pass))
    total_steps = sum(-(-r // batch) for r in rows)
    total_rows = sum(rows)
    last = len(rows) - 1
    steps, t, done = [], 0, 0
    for epoch, epoch_rows in enumerate(rows):
        for b0 in range(0, epoch_rows, batch):
            b1 = min(b0 + batch, epoch_rows)
            done += b1 - b0
            if close == "recency":
                omega = math.exp(-((total_rows - done) / (close_len * n)))
            else:
                omega = 1.0 if epoch == last else None
            steps.append(Step(epoch, b0, b1, _scheduled_alpha(a0, t, total_steps),
                              tau > 0 and (t + 1) % check_every == 0, omega))
            t += 1
    return Schedule(n=n, d=d, k=k, nepochs=nepochs, batch_law=law, batch_size=batch,
                    alpha0=a0, epoch_rows=rows, steps=tuple(steps), unused_threshold=tau,
                    recency_close=close == "recency", nlist=nlist, nprobe=nprobe,
                    rebuild_every=rebuild_every)


def build_cells(C, k, G, iters, prev=None, generator=None):
    """o-means' build_cells without its host synchronization: torch.bincount reads the
    maximum label back to the host, which stalls the launch queue at every rebuild (every
    five steps in the first pass). Counting with index_add_ keeps the queue full."""
    dev = C.device
    dp = C.shape[1]
    Ck = C[:k]
    if prev is None:
        pick = torch.randperm(k, generator=generator, device=dev)[:G]
        g = Ck[pick].clone()
    else:
        g = prev.g32.clone()
    ones = torch.ones(k, dtype=torch.float32, device=dev)
    for _ in range(iters):
        a = torch.addmm((g * g).sum(1)[None, :], Ck, g.t(), alpha=-2.0).argmin(1)
        sums = torch.zeros((G, dp), dtype=torch.float32, device=dev).index_add_(0, a, Ck)
        cnt = torch.zeros(G, dtype=torch.float32, device=dev).index_add_(0, a, ones)
        g = torch.where((cnt > 0)[:, None], sums / cnt.clamp_min(1)[:, None], g)
    a = torch.addmm((g * g).sum(1)[None, :], Ck, g.t(), alpha=-2.0).argmin(1)
    order = torch.argsort(a, stable=True)
    size = torch.zeros(G, dtype=torch.int64, device=dev).index_add_(
        0, a, torch.ones(k, dtype=torch.int64, device=dev))
    off = torch.zeros(G + 1, dtype=torch.int32, device=dev)
    off[1:] = torch.cumsum(size, 0)
    GP = max(16, triton.next_power_of_2(G))
    g16 = torch.zeros((GP, dp), dtype=torch.float16, device=dev)
    g16[:G] = g
    gn = torch.full((GP,), float("inf"), dtype=torch.float32, device=dev)
    gn[:G] = torch.where(size > 0, g16[:G].float().square().sum(1), tl2.EMPTY_NORM)
    child = torch.full((C.shape[0] + 1,), -1, dtype=torch.int32, device=dev)
    return order, tl2.Cells(G=G, GP=GP, g32=g, g16=g16, gn=gn, off=off, size=size, child=child)


class _PTraining(_Training):
    def __init__(self, x, init, sched, seed, first_pass_rebuild_every, launch,
                 final_nprobe=None, final_frac=0.0):
        if final_nprobe:
            # The workspace must hold the larger probe count of the closing stretch.
            sched = replace(sched, nprobe=max(sched.nprobe, final_nprobe))
        super().__init__(x, init, sched, seed)
        total = sum(sched.epoch_rows)
        self.final_from = total - int(final_frac * sched.n) if final_nprobe else total + 1
        self.first_every = first_pass_rebuild_every
        self.launch = launch

    def rebuild_cells(self, first_pass):
        from faiss.contrib.o_means import o_means as om
        if self.cells is None:
            iters = om._FIRST_BUILD_ITERS
        else:
            iters = om._FIRST_PASS_REBUILD_ITERS if first_pass else om._REBUILD_ITERS
        order, self.cells = build_cells(self.C, self.sched.k, self.sched.nlist, iters,
                                        prev=self.cells, generator=self.cell_rng)
        om.permute_rows(order, self.sched.k, [self.C, self.c16, self.cn, self.usage,
                                              self.close_sums, self.close_cnt])
        self.ids = self.ids[order]

    def train(self, labels_out=None):
        sched = self.sched
        n = sched.n
        probes_low, probes_high = self.nprobe_low, sched.nprobe
        perm, epoch = None, -1
        rows_done, since = 0, 0
        for step in sched.steps:
            if step.epoch != epoch:
                epoch = step.epoch
                g = torch.Generator(device=self.x.device)
                g.manual_seed(_epoch_seed(self.seed, epoch))
                perm = torch.randperm(n, generator=g, device=self.x.device)
            src, idx, R = self.batch_rows(perm, step.b0, step.b1)
            first_pass = rows_done < n
            every = self.first_every if first_pass else sched.rebuild_every
            if self.cells is None or since >= every:
                self.rebuild_cells(first_pass)
                since = 0
            since += 1
            p = probes_high if rows_done + R > self.final_from else probes_low
            labels = two_level_assign(src, self.c16, self.cn, self.cells, self.ws,
                                      p, idx=idx, rows=R, launch=self.launch)
            rows_done += R
            if self.launch.get("sorted_accumulate"):
                accumulate_sorted(src, labels, self.sums, self.cnt, sched.d, idx=idx,
                                  block_r=self.launch.get("acc_rows", 32))
            else:
                accumulate(src, labels, self.sums, self.cnt, sched.d, idx=idx)
            close = step.omega is not None
            ema_update(self.C, self.c16, self.cn, self.usage, self.sums, self.cnt, sched.k,
                       sched.d, step.alpha, self.close_sums if close else None,
                       self.close_cnt if close else None, omega=step.omega if close else 1.0)
            if step.check:
                self.revive()
        return self.closing_mean()


class Pmeans:
    """p-means: train(x, init_centroids) returns the (k, d) float32 centroids."""

    def __init__(self, d, k, nepochs=1.0, seed=1234, *, first_pass_rebuild_every=None,
                 launch=None, final_nprobe=None, final_frac=0.5, **schedule_kw):
        self.final_nprobe, self.final_frac = final_nprobe, float(final_frac)
        self.launch = {**LAUNCH, **(WIDE_LAUNCH if d >= 512 else {}), **(launch or {})}
        self.d, self.k, self.nepochs, self.seed = int(d), int(k), float(nepochs), int(seed)
        self.first_every = first_pass_rebuild_every
        self.schedule_kw = schedule_kw
        self.recipe = None

    def train(self, x, init_centroids):
        _check_rows(x, self.d)
        sched = make_schedule(x.shape[0], self.d, self.k, self.nepochs, **self.schedule_kw)
        first_every = self.first_every or self.schedule_kw.get("check_every",
                                                               _CHECK_UNUSED_EVERY)
        run = _PTraining(x, init_centroids, sched, self.seed, first_every, self.launch,
                         self.final_nprobe, self.final_frac)
        run.nprobe_low = sched.nprobe
        out = run.train()
        self.recipe = dict(batch_size=sched.batch_size, alpha0=sched.alpha0,
                           nlist=sched.nlist, nprobe=sched.nprobe, steps=len(sched.steps),
                           rebuild_every=sched.rebuild_every, first_every=first_every,
                           unused_threshold=sched.unused_threshold)
        return out
