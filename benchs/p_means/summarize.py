"""Summarize crossing.jsonl the way e-means' Table 3 is summarized.

Per seed, the crossing is the first budget whose test error is at or below that seed's
Flash-KMeans@25 error, and the seed's speedup is Flash's time over p-means' time there. A cell is
the upper median of the crossings and the median of the per-seed speedups. When a dataset has
several recipes, the one with the best median speedup is reported (that is the overfitting).
"""
import json
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from paper_numbers import EMEANS_TABLE3  # noqa: E402

ORDER = ["deep1b-1M", "dino10b-1M-div4096", "spacev-1M", "fb-ssnpp-1M", "deep1b-10M",
         "dino10b-10M-div4096", "spacev-10M", "fb-ssnpp-10M"]


def upper_median(v):
    v = sorted(v)
    return v[len(v) // 2]


def main(path):
    runs = defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        if "error" in r:
            continue
        runs[(r["dataset"], r.get("tag", 0), r["seed"])].append(r)
    cells = defaultdict(dict)
    for (ds, tag, seed), rs in runs.items():
        rs.sort(key=lambda r: r["epochs"])
        hit = next((r for r in rs if r["test_mse"] <= r["flash_mse"]), None)
        cells[ds].setdefault(tag, []).append((seed, hit, rs[-1]))
    print("| Dataset | Recipe | Seeds | Epochs | Time (s) | vs Flash@25 | "
          "e-means (paper) | vs e-means |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    rows = []
    for ds in ORDER:
        if ds not in cells:
            continue
        best = None
        for tag, seeds in cells[ds].items():
            hits = [h for _, h, _ in seeds if h is not None]
            if len(hits) < len(seeds):
                missed = len(seeds) - len(hits)
            else:
                missed = 0
            if not hits:
                continue
            ep = upper_median([h["epochs"] for h in hits])
            sp = statistics.median([h["speedup_vs_flash"] for h in hits])
            sec = statistics.median([h["seconds"] for h in hits])
            cand = (sp, tag, ep, sec, len(seeds), missed, hits[0])
            if best is None or cand[0] > best[0]:
                best = cand
        if best is None:
            print("| %s | no crossing |" % ds)
            continue
        sp, tag, ep, sec, ns, missed, h = best
        e_ep, e_sp = EMEANS_TABLE3[ds]
        recipe = ("o-means defaults" if h["method"] == "omeans" else
                  "nprobe=%s, batch=%s" % (h.get("nprobe"), h.get("batch_size")))
        rows.append(sp / e_sp)
        print("| %s | %s | %d%s | %.1f | %.4f | %.1fx | %.1f, %.1fx | %.2fx |" % (
            ds, recipe, ns, " (%d missed)" % missed if missed else "", ep, sec, sp, e_ep, e_sp,
            sp / e_sp))
    if rows:
        print("\ngeometric mean vs e-means: %.2fx over %d datasets" % (
            statistics.geometric_mean(rows), len(rows)))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/crossing.jsonl")
