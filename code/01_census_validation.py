"""Section 2.1. Revalidate the census: line counts, factorizations, Korselt criterion, and the
sixteen published values of C(10^k). Writes results/census.npz for the later scripts."""
import os
import time

import numpy as np

from common import PUBLISHED_COUNTS, RESULTS, read_census, save

t0 = time.time()
n, factors = read_census(with_factors=True)
r = np.array([len(f) for f in factors], dtype=np.int8)

bad_product = bad_squarefree = bad_korselt = 0
for m, f in zip(n.tolist(), factors):
    p = 1
    for q in f:
        p *= q
    bad_product += p != m
    bad_squarefree += len(set(f)) != len(f)
    bad_korselt += any((m - 1) % (q - 1) for q in f)
increasing = bool(np.all(np.diff(n) > 0))

checkpoints = {k: int(np.searchsorted(n, 10 ** k)) for k in range(3, 19)}
all_match = all(checkpoints[k] == PUBLISHED_COUNTS[k] for k in checkpoints)

lines = {"carmichael-16": int(np.sum(n < 10 ** 16)),
         "carmichael17": int(np.sum((n >= 10 ** 16) & (n < 10 ** 17))),
         "carmichael18": int(np.sum(n >= 10 ** 17))}
print("line counts", lines, "total", len(n))
print("product mismatches", bad_product, "| not squarefree", bad_squarefree,
      "| Korselt failures", bad_korselt, "| strictly increasing", increasing)
for k in checkpoints:
    print(f"C(10^{k:2d}) published={PUBLISHED_COUNTS[k]:8d} reconstructed={checkpoints[k]:8d}")
print("all sixteen checkpoints match:", all_match)

os.makedirs(RESULTS, exist_ok=True)
np.savez_compressed(os.path.join(RESULTS, "census.npz"), n=n, r=r)
save("01_census_validation.json", dict(total=len(n), lines=lines, bad_product=bad_product,
                                      bad_squarefree=bad_squarefree, bad_korselt=bad_korselt,
                                      increasing=increasing, checkpoints=checkpoints,
                                      all_checkpoints_match=all_match,
                                      r_counts={int(k): int(np.sum(r == k)) for k in np.unique(r)},
                                      elapsed_s=time.time() - t0))
