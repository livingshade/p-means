"""Run Table 3's crossing protocol for p-means, with the one rule (--rule) or per-dataset recipes.

    python crossing.py --datasets deep1b-1M spacev-1M --seeds 10 11 12
"""
import argparse, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
# The p-means recipe for each dataset: chosen by sweeping seed 10 against its Flash@25 target.
RECIPES = {
    "deep1b-1M": [dict(nprobe=4, batch_size=65536)],
    "spacev-1M": [dict(nprobe=8, batch_size=65536)],
    "fb-ssnpp-1M": [dict(nprobe=4, batch_size=65536)],
    "dino10b-1M-div4096": [dict(nprobe=4, batch_size=65536), dict(nprobe=8, batch_size=131072)],
    "deep1b-10M": [dict(nprobe=4, batch_size=262144)],
    "spacev-10M": [dict(nprobe=8, batch_size=524288)],
    "fb-ssnpp-10M": [dict(nprobe=4, batch_size=262144)],
    "dino10b-10M-div4096": [dict(nprobe=4, batch_size=262144), dict(nprobe=8, batch_size=262144)],
}
# One rule for every dataset: the recipe seven of the eight datasets already use.
RULE = {"1M": dict(nprobe=4, batch_size=65536), "10M": dict(nprobe=4, batch_size=262144)}
GRID = [round(1.0 + 0.2 * i, 1) for i in range(26)]          # 1.0 .. 6.0, Table 3 steps

ap = argparse.ArgumentParser()
ap.add_argument("--datasets", nargs="+", required=True)
ap.add_argument("--seeds", nargs="+", type=int, default=[10, 11, 12])
ap.add_argument("--file", default="crossing.jsonl")
ap.add_argument("--method", default="pmeans", choices=["pmeans", "omeans"],
                help="omeans runs o-means' defaults over the same budgets")
ap.add_argument("--rule", action="store_true",
                help="use the one-rule recipe (RULE) instead of the per-dataset RECIPES")
a = ap.parse_args()
for ds in a.datasets:
    for seed in a.seeds:
        if a.method == "omeans":
            cfgs = [dict(method="omeans", tag="omeans", epochs=e) for e in GRID]
        else:
            cfgs = [dict(method="pmeans", tag=i, epochs=e, close_len=0.5, **r)
                    for i, r in enumerate([RULE[ds.split("-")[1]]] if a.rule else RECIPES[ds])
                    for e in GRID]
        subprocess.run([sys.executable, os.path.join(HERE, "pm_xb.py"), "--dataset", ds,
                        "--seed", str(seed), "--configs", json.dumps(cfgs), "--crossing",
                        "--file", a.file], check=False)
