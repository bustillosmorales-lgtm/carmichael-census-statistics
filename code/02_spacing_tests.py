"""Spacing diagnostics of the Carmichael numbers below 1e18: descriptive table, calibrated
Monte Carlo tests, and power against mild departures from the Poisson law.

Section 3. Usage: python code/02_spacing_tests.py [R_null] [R_window] [R_decade] [R_power]
Writes results/02_spacing_tests.json and prints every figure quoted in Section 3.

Statistics (all computed on the full population of 1,401,644 points):
  D_W    sup deviation from Exp(1) of gaps normalized by a centered moving mean of W gaps
  D_rat  sup deviation of consecutive gap ratios min/max from the Poisson law 2x/(1+x)
         (scale free: no intensity estimate enters)
  m_rat  mean gap ratio, Poisson value 2 log 2 - 1
  D_cf   cross-fitted sup deviation: independent Bernoulli(1/2) thinning, intensity estimated
         from one half only, gaps of the other half normalized by it
  rho_1  lag-1 Spearman correlation of consecutive gap ratios (scale free)
Null: inhomogeneous Poisson process whose intensity is the W=20,000 moving mean of the observed
gaps, passed through the identical pipeline. Monte Carlo p-value (1 + #{T* >= T}) / (R + 1),
two-sided for m_rat and rho_1.
"""
import json, os, sys, time
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.special import ndtr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS, read_census

R_NULL, R_WIN, R_DEC, R_POW = [int(a) for a in (sys.argv[1:5] + ["1000", "200", "200", "100"][len(sys.argv[1:5]):])]
SEED = 20260913


def load():
    n = read_census()
    assert len(n) == 1401644 and np.all(np.diff(n) > 0)
    return n


def rollmean(x, W):
    """Centered moving mean over W terms, truncated at the ends (as in the original scripts)."""
    c = np.concatenate(([0.0], np.cumsum(x)))
    m = len(x); half = W // 2
    lo = np.clip(np.arange(m) - half, 0, m); hi = np.clip(np.arange(m) - half + W, 0, m)
    return (c[hi] - c[lo]) / (hi - lo)


def sup_dev(v, cdf):
    v = np.sort(v); m = len(v); F = cdf(v); k = np.arange(1, m + 1)
    return float(max(np.max(k / m - F), np.max(F - (k - 1) / m)))


exp_cdf = lambda v: 1.0 - np.exp(-v)
ratio_cdf = lambda v: 2.0 * v / (1.0 + v)


def ratios(g):
    return np.minimum(g[:-1], g[1:]) / np.maximum(g[:-1], g[1:])


def spearman_lag1(x):
    rk = np.empty(len(x)); rk[np.argsort(x, kind="stable")] = np.arange(len(x))
    return float(np.corrcoef(rk[:-1], rk[1:])[0, 1])


def cross_fit(pos, rng, W=1000, splits=5):
    """Mean over independent splits of the cross-fitted sup deviation."""
    vals = []
    for _ in range(splits):
        lab = rng.random(len(pos)) < 0.5
        A, B = pos[lab], pos[~lab]
        mA = rollmean(np.diff(A), W); midA = 0.5 * (A[:-1] + A[1:])
        gB = np.diff(B); midB = 0.5 * (B[:-1] + B[1:])
        vals.append(sup_dev(gB / np.interp(midB, midA, mA), exp_cdf))
    return float(np.mean(vals))


def tests(gaps, rng):
    pos = np.concatenate(([0.0], np.cumsum(gaps)))
    q = ratios(gaps)
    return {"D_1000": sup_dev(gaps / rollmean(gaps, 1000), exp_cdf),
            "D_rat": sup_dev(q, ratio_cdf),
            "m_rat": float(q.mean()),
            "D_cf": cross_fit(pos, rng),
            "rho_1": spearman_lag1(q)}


TWO_SIDED = {"m_rat", "rho_1"}


def mc_p(obs, draws, key):
    d = np.asarray(draws); R = len(d)
    if key in TWO_SIDED:
        c = d.mean()
        return (1 + np.sum(np.abs(d - c) >= abs(obs - c))) / (R + 1)
    return (1 + np.sum(d >= obs)) / (R + 1)


def reject(val, draws, key, alpha=0.05):
    d = np.asarray(draws)
    if key in TWO_SIDED:
        c = d.mean(); return abs(val - c) >= np.quantile(np.abs(d - c), 1 - alpha)
    return val >= np.quantile(d, 1 - alpha)


def main():
    rng = np.random.default_rng(SEED)
    out = {"seed": SEED, "R_null": R_NULL, "R_window": R_WIN, "R_decade": R_DEC, "R_power": R_POW}
    t0 = time.time()
    n = load()
    g = np.diff(n).astype(np.float64)
    lam = rollmean(g, 20000)                     # intensity carried by every null draw
    N = len(g)

    # 1. descriptive table by window, with a window-specific null quantile of D_W
    win = {}
    for W in (100, 1000, 5000, 20000):
        u = g / rollmean(g, W)
        null = [sup_dev((e := rng.exponential(size=N) * lam) / rollmean(e, W), exp_cdf) for _ in range(R_WIN)]
        win[W] = dict(mean=float(u.mean()), sd=float(u.std()), median=float(np.median(u)),
                      m2=float((u ** 2).mean()), m3=float((u ** 3).mean()), D=sup_dev(u, exp_cdf),
                      null_q95=float(np.quantile(null, 0.95)), p=float(mc_p(sup_dev(u, exp_cdf), null, "D")))
        print(f"W={W:6d} mean={win[W]['mean']:.4f} sd={win[W]['sd']:.4f} med={win[W]['median']:.4f} "
              f"m2={win[W]['m2']:.3f} m3={win[W]['m3']:.3f} D={win[W]['D']:.5f} q95={win[W]['null_q95']:.5f} p={win[W]['p']:.3f}", flush=True)
    out["windows"] = win

    # 2. decades, W=1000 inside each decade, null of the same size
    dec = {}
    for k in range(12, 18):
        sub = n[(n >= 10 ** k) & (n < 10 ** (k + 1))]
        gs = np.diff(sub).astype(np.float64); u = gs / rollmean(gs, 1000)
        ls = rollmean(gs, min(20000, len(gs) // 2))
        D = sup_dev(u, exp_cdf)
        null = [sup_dev((e := rng.exponential(size=len(gs)) * ls) / rollmean(e, 1000), exp_cdf) for _ in range(R_DEC)]
        dec[k] = dict(count=int(len(sub)), mean=float(u.mean()), sd=float(u.std()), median=float(np.median(u)),
                      D=D, null_q95=float(np.quantile(null, 0.95)), p=float(mc_p(D, null, "D")))
        print(f"1e{k}-1e{k+1}: n={len(sub):7d} mean={u.mean():.4f} sd={u.std():.4f} med={np.median(u):.4f} "
              f"D={D:.5f} q95={dec[k]['null_q95']:.5f} p={dec[k]['p']:.3f}", flush=True)
    out["decades"] = dec

    # 3. calibrated tests on the full population
    obs = tests(g, rng)
    draws = {k: [] for k in obs}
    for r in range(R_NULL):
        s = tests(rng.exponential(size=N) * lam, rng)
        for k in s: draws[k].append(s[k])
        if r in (0, 99, 499): print(f"  null rep {r+1} at {time.time()-t0:.0f}s", flush=True)
    lat = tests(np.clip(rng.normal(1.0, 0.15, size=N), 1e-9, None) * lam, rng)
    out["tests"] = {k: dict(obs=obs[k], null_mean=float(np.mean(draws[k])), null_sd=float(np.std(draws[k])),
                            null_q025=float(np.quantile(draws[k], .025)), null_q975=float(np.quantile(draws[k], .975)),
                            null_q95=float(np.quantile(draws[k], .95)), p=float(mc_p(obs[k], draws[k], k)),
                            lattice=lat[k], lattice_rejects=bool(reject(lat[k], draws[k], k))) for k in obs}
    out["poisson_mean_ratio"] = 2 * np.log(2) - 1
    for k, v in out["tests"].items():
        print(f"{k:7s} obs={v['obs']:.6f} null mean={v['null_mean']:.6f} [{v['null_q025']:.6f}, {v['null_q975']:.6f}] "
              f"p={v['p']:.3f} lattice={v['lattice']:.4f} rejects={v['lattice_rejects']}", flush=True)

    # 4. power against mild departures, level 5 percent from the null draws above
    def gamma_renewal(shape):
        return lambda: rng.gamma(shape, 1.0 / shape, size=N) * lam

    def copula_ar1(rho):
        def f():
            z = np.empty(N); e = rng.standard_normal(N); z[0] = e[0]; s = np.sqrt(1 - rho ** 2)
            # AR(1) Gaussian sequence by recursive filter, exact unit variance
            from scipy.signal import lfilter
            z = lfilter([s], [1.0, -rho], e); z[0] = e[0]
            return -np.log1p(-ndtr(z)) * lam       # Exp(1) margins, lag-1 dependence
        return f

    alts = {"gamma 0.995": gamma_renewal(0.995), "gamma 1.005": gamma_renewal(1.005),
            "gamma 0.99": gamma_renewal(0.99), "gamma 1.01": gamma_renewal(1.01),
            "copula rho 0.002": copula_ar1(0.002), "copula rho 0.005": copula_ar1(0.005)}
    power = {}
    for name, gen in alts.items():
        hits = {k: 0 for k in obs}
        for _ in range(R_POW):
            s = tests(gen(), rng)
            for k in s: hits[k] += bool(reject(s[k], draws[k], k))
        power[name] = {k: hits[k] / R_POW for k in hits}
        print(f"power {name:17s} " + " ".join(f"{k}={power[name][k]:.2f}" for k in hits), flush=True)
    out["power"] = power

    # 5. sensitivity of D_W at W = 1,000 and 20,000 to the resolution S of the intensity carried by the null
    #    (separate generator, so that the figures above do not depend on this block)
    rng2 = np.random.default_rng(20260917)
    obs_w = {W: sup_dev(g / rollmean(g, W), exp_cdf) for W in (1000, 20000)}
    sens = {}
    for S in (2000, 5000, 20000, 100000):
        lamS = rollmean(g, S)
        d = {W: [] for W in obs_w}
        for _ in range(R_WIN):
            e = rng2.exponential(size=N) * lamS
            for W in obs_w:
                d[W].append(sup_dev(e / rollmean(e, W), exp_cdf))
        sens[S] = {W: dict(q95=float(np.quantile(d[W], .95)), p=float(mc_p(obs_w[W], d[W], "D"))) for W in obs_w}
        print(f"intensity smoothed over S={S:6d}: " + "  ".join(f"W={W} q95={sens[S][W]['q95']:.5f} p={sens[S][W]['p']:.3f}" for W in obs_w), flush=True)
    out["intensity_resolution"] = sens

    # 6. intensity taken from the counting function only: mean gap 1/C'(x) from a monotone interpolation
    #    of log C against log x through C at quarter decades, which uses no local gap
    x = n.astype(float)
    grid = 10 ** np.arange(3, 18.01, 0.25)
    Cg = np.searchsorted(n, grid).astype(float)
    keep = Cg > 0
    f = PchipInterpolator(np.log(grid[keep]), np.log(Cg[keep]))
    logx = np.log(0.5 * (x[:-1] + x[1:]))
    sel = n[:-1] >= 10 ** 6
    with np.errstate(divide="ignore"):
        mean_gap = np.exp(logx - f(logx)) / f.derivative()(logx)
    u_glob = g[sel] / mean_gap[sel]
    D_glob = sup_dev(u_glob / u_glob.mean(), exp_cdf)
    dg = []
    for _ in range(R_WIN):
        v = rng2.exponential(size=N)[sel]
        dg.append(sup_dev(v / v.mean(), exp_cdf))
    out["global_intensity"] = dict(checkpoints=int(keep.sum()), n_gaps=int(sel.sum()), mean=float(u_glob.mean()),
                                   sd=float(u_glob.std()), D=D_glob, null_q95=float(np.quantile(dg, .95)),
                                   p=float(mc_p(D_glob, dg, "D")))
    print(f"global intensity from {keep.sum()} values of C(x), n >= 1e6: mean={u_glob.mean():.4f} sd={u_glob.std():.4f} "
          f"D={D_glob:.5f} q95={np.quantile(dg, .95):.5f} p={out['global_intensity']['p']:.3f}", flush=True)

    out["elapsed_s"] = time.time() - t0
    with open(os.path.join(RESULTS, "02_spacing_tests.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"done in {out['elapsed_s']:.0f}s")


if __name__ == "__main__":
    main()
