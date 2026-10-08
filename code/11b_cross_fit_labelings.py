"""Second revision, Section 3. The cross-fitted distance depends on the random labeling that splits
the points into two halves. In the first revision it was the mean over 5 labelings, whose labeling
noise (standard deviation about 0.0004 for one labeling) can move the p-value substantially. Here the
statistic is the mean over K = 40 independent labelings, both on the census and in every realization
of the reference model, so that the labeling noise is reduced by a factor of about 3 relative to K = 5
and enters the census and the null in the same way.

Usage: python code/11b_cross_fit_labelings.py [R] [workers] [K]
"""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark as B
from common import RESULTS, census_cache

R = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
K = int(sys.argv[3]) if len(sys.argv) > 3 else 40
SEED = 20261016


def work(task):
    seed, reps = task
    n, _ = census_cache()
    pos = n.astype(np.float64)
    xk, Lk = B.cumulative_intensity(pos, 20000)
    rng = np.random.default_rng(seed)
    return [B.cross_fit(B.gen_inverse(xk, Lk, len(pos), rng), rng, splits=K) for _ in range(reps)]


def main():
    t0 = time.time()
    n, _ = census_cache()
    g = np.diff(n.astype(np.float64))
    rng = np.random.default_rng(SEED)
    singles = [B.cross_fit(g, rng, splits=1) for _ in range(200)]
    obs = float(np.mean(singles[:K]))
    seq = np.random.SeedSequence(SEED)
    tasks = [(c.generate_state(1)[0], 10) for c in seq.spawn(R // 10)]
    null = []
    with Pool(WORKERS) as pool:
        for v in pool.imap_unordered(work, tasks):
            null += v
    null = np.array(null)
    p = B.mc_pvalue(obs, null, False)
    p_low = float((1 + np.sum(null <= obs)) / (len(null) + 1))
    res = dict(K=K, R=len(null), obs=obs, obs_mean_200_labelings=float(np.mean(singles)),
               obs_sd_single_labeling=float(np.std(singles)), null_mean=float(null.mean()),
               null_q025=float(np.quantile(null, .025)), null_q975=float(np.quantile(null, .975)),
               p_upper=float(p), p_lower=p_low, elapsed_s=time.time() - t0)
    print(json.dumps(res, indent=1))
    with open(os.path.join(RESULTS, "11b_cross_fit_labelings.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
