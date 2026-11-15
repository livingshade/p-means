#!/bin/bash
# usage: sweep.sh DATASET SEED 'JSON configs'
cd "$(dirname "$0")" && . ~/venv/bin/activate
export EMEANS_DATA=$HOME/data EMEANS_CACHE=$HOME/cache TRITON_DEFAULT_FP_FUSION=1
python pm_xb.py --dataset "$1" --seed "$2" --configs "$3" 2>&1 | python3 -c '
import sys,json
for l in sys.stdin:
  try: r=json.loads(l)
  except Exception: print(l.rstrip()[:300]); continue
  cfg={k:v for k,v in r.items() if k not in ("dataset","seed","n","k","seconds","seconds_calls","test_mse","flash_mse","flash_seconds","mse_vs_flash_pct","speedup_vs_flash","emeans_epochs","emeans_seconds_est","speedup_vs_emeans","recipe","gpu","torch","error")}
  if "error" in r: print("ERR", cfg, r["error"][:300]); continue
  print("%-70s t=%.4fs mse%+.3f%% xFlash=%.1f xE=%s %s" % (json.dumps(cfg), r["seconds"], r["mse_vs_flash_pct"], r["speedup_vs_flash"], "%.2f"%r["speedup_vs_emeans"] if r["speedup_vs_emeans"] else "-", (r.get("recipe") or {}).get("steps","")))
'
