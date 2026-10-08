"""Second revision, Section 3. Serial dependence of the spacings of the census below 10^18.

1. Profile of the Spearman autocorrelations of the gap ratios at lags 1..50, with the null mean and
   standard deviation at each lag from R realizations of a unit-rate Poisson process of the same
   size (the ratios are scale free). Simultaneous calibration: the null law of max_h |z_h| over
   lags 1..10 and 1..50 gives a simultaneous 95% band. Global statistics fixed in advance of the
   observed pattern: Q_L = sum_{h<=L} z_h^2 for L = 10 and 50, and the directional sum of the
   standardized excesses over lags 1..10.
2. Excess at lags 1 and 2 in each decade from 10^12 to 10^18, against null realizations of the
   decade size, with Monte Carlo standard errors.
3. Lag-1 and lag-2 Spearman autocorrelations of the normalized gaps u^(W) for W = 100, 1,000 and
   5,000 and for the normalization by the counting function alone, each against realizations of
   the reference model passed through the same normalization.

Usage: python code/15_dependence.py [R] [workers]
"""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np
from scipy.interpolate import PchipInterpolator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark as B
from common import RESULTS, census_cache

R = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
SEED = 20261015
LMAX = 50
DECS = range(12, 18)


def setup():
    n, _ = census_cache()
    pos = n.astype(np.float64)
    bounds = [int(np.searchsorted(n, 10 ** k)) for k in list(DECS) + [18]]
    grid = 10 ** np.arange(3, 18.01, 0.25)
    f = PchipInterpolator(np.log(grid), np.log(np.maximum(np.searchsorted(n, grid).astype(float), 0.5)))
    return pos, bounds, f


def u_variants(g, pos_left, f):
    """Normalized gaps for each variant: three windows and the counting-function normalization."""
    out = {f"W={W}": g / B.rollmean(g, W) for W in (100, 1000, 5000)}
    x = pos_left
    sel = x >= 1e6
    with np.errstate(divide="ignore", invalid="ignore"):
        lx = np.log(x[sel])
        mg = np.exp(lx - f(lx)) / f.derivative()(lx)
    v = g[sel] / mg
    out["C0"] = v / v.mean()
    return out


def acf_vector(g, bounds, f, pos_left):
    q = B.ratios(g)
    prof = B.rank_acf(q, LMAX)
    dec = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        dec += list(B.rank_acf(q[a:b - 1], 2))
    uv = [] if f is None else sum((list(B.rank_acf(u, 2)) for u in u_variants(g, pos_left, f).values()), [])
    return np.array(list(prof) + dec + uv)


def work(task):
    kind, seed, reps = task
    pos, bounds, f = setup()
    rng = np.random.default_rng(seed)
    N = len(pos)
    out = []
    if kind == "unit":
        for _ in range(reps):
            out.append(acf_vector(rng.exponential(size=N - 1), bounds, None, None))
    else:
        xk, Lk = B.cumulative_intensity(pos, 20000)
        for _ in range(reps):
            g = B.gen_inverse(xk, Lk, N, rng)
            left = pos[0] + np.concatenate(([0.0], np.cumsum(g)[:-1]))
            out.append(acf_vector(g, bounds, f, left))
    return kind, np.array(out)


def main():
    t0 = time.time()
    pos, bounds, f = setup()
    g = np.diff(pos)
    obs = acf_vector(g, bounds, f, pos[:-1])
    seq = np.random.SeedSequence(SEED)
    tasks = [("unit", c.generate_state(1)[0], 25) for c in seq.spawn(R // 25)]
    tasks += [("ref", c.generate_state(1)[0], 25) for c in seq.spawn(max(1, R // 50))]   # R/2 for the u variants
    acc = {"unit": [], "ref": []}
    with Pool(WORKERS) as pool:
        for kind, arr in pool.imap_unordered(work, tasks):
            acc[kind].append(arr)
    U = np.vstack(acc["unit"]); Rf = np.vstack(acc["ref"])
    np.savez_compressed(os.path.join(RESULTS, "15_dependence.npz"), obs=obs, unit=U, ref=Rf)
    res = {"R_unit": len(U), "R_ref": len(Rf)}
    o, mu, sd = obs[:LMAX], U[:, :LMAX].mean(0), U[:, :LMAX].std(0)
    z, Z = (o - mu) / sd, (U[:, :LMAX] - mu) / sd
    p2 = lambda t, d: float((1 + np.sum(np.abs(d - d.mean()) >= abs(t - d.mean()))) / (len(d) + 1))   # noqa: E731
    pu = lambda t, d: float((1 + np.sum(d >= t)) / (len(d) + 1))                                     # noqa: E731
    res["profile"] = dict(excess=(o - mu).tolist(), z=z.tolist(), null_sd=sd.tolist(),
                          p_individual=[p2(o[h], U[:, h]) for h in range(LMAX)])
    for L in (10, 50):
        mz = np.max(np.abs(Z[:, :L]), axis=1)
        res[f"max_abs_z_{L}"] = dict(obs=float(np.max(np.abs(z[:L]))), crit95=float(np.quantile(mz, .95)),
                                    p=pu(np.max(np.abs(z[:L])), mz), argmax_lag=int(np.argmax(np.abs(z[:L])) + 1))
        Q = np.sum(Z[:, :L] ** 2, axis=1)
        res[f"Q_{L}"] = dict(obs=float(np.sum(z[:L] ** 2)), p=pu(np.sum(z[:L] ** 2), Q))
    Ssum = np.sum(Z[:, :10], axis=1)
    res["S_10"] = dict(obs=float(np.sum(z[:10])), p=p2(np.sum(z[:10]), Ssum))
    res["decades"] = {}
    for i, k in enumerate(DECS):
        j = LMAX + 2 * i
        row = {}
        for h in (1, 2):
            d = U[:, j + h - 1]
            row[f"lag{h}"] = dict(excess=float(obs[j + h - 1] - d.mean()), null_sd=float(d.std()),
                                  mc_se=float(d.std() / np.sqrt(len(d))), z=float((obs[j + h - 1] - d.mean()) / d.std()),
                                  p=p2(obs[j + h - 1], d))
        res["decades"][f"1e{k}-1e{k+1}"] = row
    j0 = LMAX + 2 * len(DECS)
    res["u_variants"] = {}
    for i, name in enumerate(["W=100", "W=1000", "W=5000", "C0"]):
        row = {}
        for h in (1, 2):
            c = j0 + 2 * i + h - 1
            d = Rf[:, c]
            row[f"lag{h}"] = dict(obs=float(obs[c]), null_mean=float(d.mean()), null_sd=float(d.std()),
                                  excess=float(obs[c] - d.mean()), p=p2(obs[c], d))
        res["u_variants"][name] = row
    print(json.dumps({k: v for k, v in res.items() if k != "profile"}, indent=1))
    print("first 10 z:", np.round(z[:10], 2).tolist())
    res["elapsed_s"] = time.time() - t0
    with open(os.path.join(RESULTS, "15_dependence.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
