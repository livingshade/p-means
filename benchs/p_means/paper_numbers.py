"""Numbers quoted from the e-means paper benchmark README (../e_means/README.md, Table 3,
`--input cast` column): epochs for e-means to reach Flash-KMeans@25, and the median speedup."""
EMEANS_TABLE3 = {
    "deep1b-1M": (2.0, 12.8), "dino10b-1M-div4096": (2.0, 12.7), "spacev-1M": (2.4, 9.5),
    "fb-ssnpp-1M": (1.4, 12.9), "deep1b-10M": (2.0, 27.8), "dino10b-10M-div4096": (1.4, 23.3),
    "spacev-10M": (2.0, 19.2), "fb-ssnpp-10M": (1.6, 23.8),
}
