"""Second revision. Objective rule for the resolution S of the reference intensity at each window W,
chosen from simulations only, without the p-values of the census.

For a window W, the test of D_W is valid if a Poisson process with a smooth intensity is rejected
at the nominal rate when its reference model is built, as in the analysis of the census, from the
moving mean of S gaps of one realization. Two true intensities, both smooth summaries of the
census in which no individual gap enters:
  pchip   monotone cubic interpolation of log C against log x at quarter decades (61 values)
  poly    polynomial of degree 8 in log x fitted by least squares to log C at the same points
For each truth, R realizations give the distribution of D_W; for each S, R realizations of the
reference model estimated from one realization of the truth give the critical value. The actual
level is the rejection rate of the truth at nominal level 0.05.

Rule: for each W, the resolution is the largest S in the grid whose actual level has an exact 95%
binomial interval containing 0.05 under both truths.

Usage: python code/13_resolution_rule.py [R] [workers]
"""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np
from scipy import stats as ss
from scipy.interpolate import PchipInterpolator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark as B
from common import RESULTS, census_cache

R = int(sys.argv[1]) if len(sys.argv) > 1 else 500
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 14
SEED = 20261011
S_GRID = (1000, 2000, 5000, 20000)
TRUTHS = ("pchip", "poly")


def truth_measure(kind):
    n, _ = census_cache()
    grid = 10 ** np.arange(3, 18.01, 0.25)
    Cg = np.searchsorted(n, grid).astype(float)
    t = np.log(grid); y = np.log(np.maximum(Cg, 0.5))
    if kind == "pchip":
        f = PchipInterpolator(t, y)
    else:
        f = np.polynomial.Polynomial.fit(t, y, 8)
    xk = np.exp(np.linspace(np.log(n[0]), np.log(n[-1]), 400001))
    Lk = np.exp(f(np.log(xk)))
    assert np.all(np.diff(Lk) >= 0), f"{kind}: cumulative intensity decreasing"
    return xk, Lk - Lk[0], len(n)


def d_windows(g):
    return np.array([B.sup_dev(g / B.rollmean(g, W), B.exp_cdf) for W in B.WINDOWS])


def work(task):
    truth, S, seed, reps = task
    xk, Lk, N = truth_measure(truth)
    rng = np.random.default_rng(seed)
    if S is None:
        return truth, S, np.array([d_windows(B.gen_inverse(xk, Lk, N, rng)) for _ in range(reps)])
    base = xk[0] + np.concatenate(([0.0], np.cumsum(B.gen_inverse(xk, Lk, N, np.random.default_rng(SEED)))))
    xs, Ls = B.cumulative_intensity(base, S)
    return truth, S, np.array([d_windows(B.gen_inverse(xs, Ls, N, rng)) for _ in range(reps)])


def main():
    t0 = time.time()
    seq = np.random.SeedSequence(SEED)
    tasks = []
    for truth in TRUTHS:
        for S in (None,) + S_GRID:
            for c in seq.spawn(R // 25):
                tasks.append((truth, S, c.generate_state(1)[0], 25))
    acc = {}
    with Pool(WORKERS) as pool:
        for truth, S, arr in pool.imap_unordered(work, tasks):
            acc.setdefault((truth, S), []).append(arr)
    acc = {k: np.vstack(v) for k, v in acc.items()}
    res = {"R": R, "seed": SEED, "levels": {}, "rule": {}}
    for truth in TRUTHS:
        t = acc[(truth, None)]
        for S in S_GRID:
            d = acc[(truth, S)]
            for j, W in enumerate(B.WINDOWS):
                k = int(np.sum(t[:, j] >= np.quantile(d[:, j], 0.95)))
                lo, hi = ss.binomtest(k, len(t)).proportion_ci(method="exact")
                res["levels"][f"{truth}|S={S}|W={W}"] = dict(level=k / len(t), ci=[float(lo), float(hi)])
                print(f"{truth:5s} S={S:6d} W={W:6d} level={k/len(t):.3f} [{lo:.3f}, {hi:.3f}]", flush=True)
    for W in B.WINDOWS:
        ok = [S for S in S_GRID if all(res["levels"][f"{tr}|S={S}|W={W}"]["ci"][0] <= 0.05 <= res["levels"][f"{tr}|S={S}|W={W}"]["ci"][1]
                                       for tr in TRUTHS)]
        res["rule"][str(W)] = max(ok) if ok else None
        print(f"rule: W={W} -> S={res['rule'][str(W)]}", flush=True)
    res["elapsed_s"] = time.time() - t0
    with open(os.path.join(RESULTS, "13_resolution_rule.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"done in {res['elapsed_s']:.0f}s")


if __name__ == "__main__":
    main()
