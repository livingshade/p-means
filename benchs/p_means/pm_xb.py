"""Time p-means (and o-means) with the e-means paper's protocol, against the paper's stored
Flash-KMeans@25 targets and e-means Table 3 numbers.

Same data, test split, starting rows (paper_xb.starts), warm-up and timing (paper_xb.timed) and
scorer (paper_xb.test_error) as benchs/e_means/paper_xb.py. To save loading time, all configs of
one (dataset, seed) run in the same process, one after the other, each with its own warm-up.

    python pm_xb.py --dataset deep1b-10M --seed 10 --configs '[{"method": "omeans", "epochs": 1}]'
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EMEANS_BENCH = os.path.join(HERE, "..", "e_means")
sys.path.insert(0, EMEANS_BENCH)
sys.path.insert(0, os.path.join(HERE, "..", "..", "contrib", "p_means", "gpu"))
import paper_data as data  # noqa: E402
import paper_xb as xb  # noqa: E402

from paper_numbers import EMEANS_TABLE3  # noqa: E402


def targets(name, seed):
    t = json.load(open(os.path.join(EMEANS_BENCH, "paper_targets.json")))["table3"][name]
    s = t["per_seed"][str(seed)]
    return dict(flash_mse=s["test_mse"], flash_seconds=s["seconds"])


def run_config(x, init, seed, cfg):
    import torch
    n, d = x.shape
    k = init.shape[0]
    start = torch.from_numpy(init).cuda()
    method = cfg["method"]
    kw = {key: v for key, v in cfg.items() if key not in ("method", "epochs", "tag")}

    if method == "omeans":
        from faiss.contrib.o_means import Omeans

        def call():
            om = Omeans(d, k, nepochs=cfg["epochs"], seed=seed, capture_assignments=False, **kw)
            return om.train(x, init_centroids=start)[0]
    elif method == "pmeans":
        from p_means_gpu import Pmeans

        def call():
            pm = Pmeans(d, k, nepochs=cfg["epochs"], seed=seed, **kw)
            call.recipe = None
            out = pm.train(x, init_centroids=start)
            call.recipe = pm.recipe
            return out
    else:
        raise ValueError(method)
    centroids, seconds, calls = xb.timed(call, warmup=call)
    return centroids, seconds, calls, getattr(call, "recipe", None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--configs", required=True, help="JSON list of configs")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--crossing", action="store_true",
                    help="configs are in increasing budget per tag: skip the rest of a tag "
                         "after its first run at or below Flash@25 (Table 3's rule)")
    ap.add_argument("--file", default="runs.jsonl")
    a = ap.parse_args()
    import torch
    xb.pin_precision()
    t0 = time.time()
    train, test = data.load(a.dataset)
    n, d = train.shape
    k = xb.k_for(n)
    rows = xb.starts(n, k, a.seed)
    order = np.argsort(rows)
    init = np.empty((len(rows), d), dtype=np.float32)
    init[order] = data.rows(train, a.dataset, index=rows[order])
    x = xb.upload(train, a.dataset, torch.float32)
    test = np.ascontiguousarray(test, dtype=np.float32)
    print("loaded %s n=%d d=%d k=%d in %.1fs" % (a.dataset, n, d, k, time.time() - t0),
          file=sys.stderr, flush=True)
    tg = targets(a.dataset, a.seed)
    ep, sp = EMEANS_TABLE3.get(a.dataset, (None, None))
    emeans_seconds = tg["flash_seconds"] / sp if sp else None
    os.makedirs(a.out, exist_ok=True)
    crossed = set()
    for cfg in json.loads(a.configs):
        if a.crossing and cfg.get("tag") in crossed:
            continue
        try:
            c, sec, calls, recipe = run_config(x, init, a.seed, cfg)
        except Exception as e:  # keep going through the sweep
            print(json.dumps(dict(cfg, dataset=a.dataset, error=repr(e)[:500])), flush=True)
            del e
            xb.free()       # a failed call must not leave its memory to the next one
            continue
        mse = xb.test_error(c, test)
        rec = dict(cfg, dataset=a.dataset, seed=a.seed, n=n, k=k, seconds=sec,
                   seconds_calls=calls, test_mse=mse, **tg,
                   mse_vs_flash_pct=100 * (mse / tg["flash_mse"] - 1),
                   speedup_vs_flash=tg["flash_seconds"] / sec,
                   emeans_epochs=ep, emeans_seconds_est=emeans_seconds,
                   speedup_vs_emeans=(emeans_seconds / sec) if emeans_seconds else None,
                   recipe=recipe, gpu=torch.cuda.get_device_name(), torch=torch.__version__)
        if mse <= tg["flash_mse"]:
            crossed.add(cfg.get("tag"))
        line = json.dumps(rec)
        print(line, flush=True)
        with open(os.path.join(a.out, a.file), "a") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
