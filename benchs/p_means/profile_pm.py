"""Profile one p-means call on a paper dataset: CUDA kernel time by name."""
import json, os, sys
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "e_means"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "contrib", "p_means", "gpu"))
import paper_data as data, paper_xb as xb
from p_means_gpu import Pmeans
name, cfg = sys.argv[1], json.loads(sys.argv[2])
train, test = data.load(name)
n, d = train.shape; k = xb.k_for(n)
rows = xb.starts(n, k, 10); order = np.argsort(rows)
init = np.empty((k, d), np.float32); init[order] = data.rows(train, name, index=rows[order])
x = xb.upload(train, name, torch.float32); start = torch.from_numpy(init).cuda()
ep = cfg.pop("epochs")
call = lambda: Pmeans(d, k, nepochs=ep, seed=10, **cfg).train(x, start)
call(); torch.cuda.synchronize()
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
    call(); torch.cuda.synchronize()
print(p.key_averages().table(sort_by="cuda_time_total", row_limit=25, max_name_column_width=60))
