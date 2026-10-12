"""Second revision. Validation of the implementation against processes with known laws, and of the
calibration procedure against a process whose intensity is known.

Part A, unit tests against independent implementations (SciPy, naive loops).
Part B, homogeneous Poisson process: exponential gaps, ratio law 2x/(1+x) and its mean
        2 log 2 - 1, Poisson interval counts, Kolmogorov null law of the supremum distance.
Part C, inhomogeneous Poisson process with known cumulative intensity Lambda(x) = a x^theta,
        generated twice, by inversion and independently by thinning (Lewis and Shedler):
        Poisson counts with mean Lambda(b) - Lambda(a), independent increments, and exponential
        time-rescaled gaps Lambda(n_{i+1}) - Lambda(n_i).
Part D, size of the procedure: the true process has the smooth mean measure C0 of the census
        (monotone interpolation of log C against log x at quarter decades), which no gap enters.
        The family of statistics is computed on R realizations of the true process, and on R
        realizations of each generator of the reference model (inverse, nested, rank) whose
        intensity is estimated from one realization of the true process, as in the analysis of
        the census. Equality of these distributions means that estimating the intensity does not
        distort the null distribution; the rejection rate of the true process against the 95%
        quantile of each generator is the actual level of the test at nominal level 0.05.

Usage: python code/12_validation.py [R_partD] [workers]
       python code/12_validation.py abc     (parts A-C only, part D kept from results/12_validation.json)
"""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np
from scipy import stats as ss
from scipy.interpolate import PchipInterpolator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark as B
from common import RESULTS, census_cache

ONLY_ABC = len(sys.argv) > 1 and sys.argv[1] == "abc"     # rerun parts A-C and keep the stored part D
R_D = int(sys.argv[1]) if len(sys.argv) > 1 and not ONLY_ABC else 1000
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 14
SEED = 20261010
report = {}


def check(name, ok, detail):
    report[name] = dict(ok=bool(ok), detail=detail)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def part_a(rng):
    x = rng.exponential(size=5000)
    d_ours = B.sup_dev(x, B.exp_cdf)
    d_scipy = ss.kstest(x, "expon").statistic
    check("A1 supremum distance = scipy.stats.kstest", abs(d_ours - d_scipy) < 1e-12, f"{d_ours:.12f} vs {d_scipy:.12f}")
    q = B.ratios(x)
    r_ours = B.rank_acf(q, 3)
    r_scipy = [ss.spearmanr(q[:-h], q[h:]).statistic for h in (1, 2, 3)]
    check("A2 rank autocorrelation ~ scipy.stats.spearmanr", np.max(np.abs(r_ours - r_scipy)) < 2e-3,
          f"max diff {np.max(np.abs(r_ours - r_scipy)):.2e} (differs by O(h/N) edge terms)")
    W = 100
    naive = np.array([x[max(0, i - W // 2): min(len(x), i - W // 2 + W)].mean() for i in range(len(x))])
    check("A3 moving mean = naive loop", np.max(np.abs(B.rollmean(x, W) - naive)) < 1e-12,
          f"max diff {np.max(np.abs(B.rollmean(x, W) - naive)):.1e}")
    g = np.array([3.0, 5.0, 2.0, 2.0])
    check("A4 gap ratio", np.allclose(B.ratios(g), [0.6, 0.4, 1.0]), str(B.ratios(g).tolist()))
    xk = np.array([0.0, 1.0, 3.0]); Lk = np.array([0.0, 2.0, 3.0])
    pts = np.interp(np.array([1.0, 2.5]), Lk, xk)
    check("A5 inversion of a piecewise-linear cumulative intensity", np.allclose(pts, [0.5, 2.0]), str(pts.tolist()))


def part_b(rng):
    N = 10 ** 6
    g = rng.exponential(size=N)
    q = B.ratios(g)
    m = q.mean(); se = q.std() / np.sqrt(len(q))
    check("B1 mean ratio = 2 log 2 - 1", abs(m - (2 * np.log(2) - 1)) < 4 * se, f"{m:.5f} vs {2*np.log(2)-1:.5f} (se {se:.5f})")
    check("B2 ratio law 2x/(1+x)", ss.kstest(q, lambda v: 2 * v / (1 + v)).pvalue > 0.001,
          f"KS p = {ss.kstest(q, lambda v: 2 * v / (1 + v)).pvalue:.3f}")
    pos = np.cumsum(g)
    cnt = np.bincount((pos // 5.0).astype(int))[:-1]
    vm = cnt.var() / cnt.mean()
    check("B3 interval counts: variance/mean = 1", abs(vm - 1) < 0.02, f"{vm:.4f} over {len(cnt)} intervals of length 5")
    # null law of the supremum distance with a known parameter is the Kolmogorov law
    D = [B.sup_dev(rng.exponential(size=2000), B.exp_cdf) for _ in range(2000)]
    p = ss.kstest(D, ss.kstwo(2000).cdf).pvalue
    check("B4 supremum distance follows the Kolmogorov law", p > 0.001, f"KS p = {p:.3f}")
    # the same check over 20,000 replicates, from a dedicated stream so that the draws of part C are unchanged
    r4 = np.random.default_rng(SEED + 4)
    D = [B.sup_dev(r4.exponential(size=2000), B.exp_cdf) for _ in range(20000)]
    p = ss.kstest(D, ss.kstwo(2000).cdf).pvalue
    check("B4b supremum distance follows the Kolmogorov law, 20,000 replicates", p > 0.001, f"KS p = {p:.3f}")


def thinning(a, theta, x0, x1, rng):
    """Independent generator: Lewis-Shedler thinning, decade by decade with a local bound."""
    out = []
    lam = lambda x: a * theta * x ** (theta - 1)          # noqa: E731, decreasing for theta < 1
    edges = np.unique(np.concatenate((10 ** np.arange(np.ceil(np.log10(x0)), np.log10(x1)), [x0, x1])))
    for lo, hi in zip(edges[:-1], edges[1:]):
        bound = lam(lo)
        k = rng.poisson(bound * (hi - lo))
        x = rng.uniform(lo, hi, size=k)
        out.append(np.sort(x[rng.random(k) < lam(x) / bound]))
    return np.concatenate(out)


def part_c(rng):
    a, theta, x0, x1 = 3.0, 0.35, 1e6, 1e12
    Lam = lambda x: a * x ** theta                       # noqa: E731
    expected = Lam(x1) - Lam(x0)
    p_inv = (np.sort(rng.uniform(Lam(x0), Lam(x1), size=rng.poisson(expected))) / a) ** (1 / theta)
    p_thin = thinning(a, theta, x0, x1, rng)
    for name, p in (("inversion", p_inv), ("thinning", p_thin)):
        z = (len(p) - expected) / np.sqrt(expected)
        check(f"C1 [{name}] total count Poisson(Lambda)", abs(z) < 4, f"{len(p)} vs {expected:.0f} (z={z:+.2f})")
        tau = np.diff(Lam(p))
        check(f"C2 [{name}] time-rescaled gaps Exp(1)", ss.kstest(tau, "expon").pvalue > 0.001,
              f"KS p = {ss.kstest(tau, 'expon').pvalue:.3f}")
        e = 10 ** np.linspace(6, 12, 61)
        c = np.histogram(p, e)[0]; mu = np.diff(Lam(e))
        disp = np.sum((c - mu) ** 2 / mu); pd = ss.chi2.sf(disp, len(c))
        check(f"C3 [{name}] counts in 60 intervals Poisson with mean dLambda", pd > 0.001, f"chi2 = {disp:.1f} on {len(c)} df, p = {pd:.3f}")
        u = Lam(p); cu = np.bincount((u - u.min()).astype(int) // 50)[:-1]
        rr = np.corrcoef(cu[:-1], cu[1:])[0, 1]
        check(f"C4 [{name}] independent increments (adjacent counts uncorrelated)", abs(rr) < 4 / np.sqrt(len(cu)),
              f"corr = {rr:+.4f}, bound {4/np.sqrt(len(cu)):.4f}")
    D = ss.ks_2samp(np.diff(Lam(p_inv)), np.diff(Lam(p_thin)))
    check("C5 inversion and thinning give the same rescaled-gap law", D.pvalue > 0.001, f"two-sample KS p = {D.pvalue:.3f}")


# ---------------------------------------------------------------- part D (parallel)
def _true_measure():
    n, _ = census_cache()
    grid = 10 ** np.arange(3, 18.01, 0.25)
    Cg = np.searchsorted(n, grid).astype(float)
    f = PchipInterpolator(np.log(grid), np.log(np.maximum(Cg, 0.5)))
    xk = np.exp(np.linspace(np.log(n[0]), np.log(n[-1]), 400001))
    Lk = np.exp(f(np.log(xk)))
    return xk, Lk - Lk[0], len(n)


def _true_realization(xk, Lk, N, rng):
    return xk[0] + np.concatenate(([0.0], np.cumsum(B.gen_inverse(xk, Lk, N, rng))))


def work_d(task):
    kind, seed, reps = task
    rng = np.random.default_rng(seed)
    xk, Lk, N = _true_measure()
    base = _true_realization(xk, Lk, N, np.random.default_rng(SEED))       # the one 'observed' realization
    xs, Ls = B.cumulative_intensity(base, 20000)
    m = B.rollmean(np.diff(base), 20000)
    out = []
    for _ in range(reps):
        if kind == "true":
            g = B.gen_inverse(xk, Lk, N, rng)
        elif kind == "inverse":
            g = B.gen_inverse(xs, Ls, N, rng)
        elif kind == "nested":
            pilot = base[0] + np.concatenate(([0.0], np.cumsum(B.gen_inverse(xs, Ls, N, rng))))
            x2, L2 = B.cumulative_intensity(pilot, 20000)
            g = B.gen_inverse(x2, L2, N, rng)
        else:
            g = B.gen_rank(m, rng)
        out.append(B.statistics(g, rng))
    return kind, np.array(out)


def part_d():
    seq = np.random.SeedSequence(SEED + 1)
    tasks = []
    for kind in ("true", "inverse", "nested", "rank"):
        for j, c in enumerate(seq.spawn(R_D // 25)):
            tasks.append((kind, c.generate_state(1)[0], 25))
    draws = {k: [] for k in ("true", "inverse", "nested", "rank")}
    with Pool(WORKERS) as pool:
        for kind, arr in pool.imap_unordered(work_d, tasks):
            draws[kind].append(arr)
    draws = {k: np.vstack(v) for k, v in draws.items()}
    np.savez_compressed(os.path.join(RESULTS, "12_validation_partD.npz"), names=np.array(B.STAT_NAMES), **draws)
    res = {}
    for j, name in enumerate(B.STAT_NAMES):
        if name == "D_cf" or name.startswith("rho_") and int(name[4:]) > 2:
            continue
        two = name in B.TWO_SIDED
        t = draws["true"][:, j]
        row = {}
        for k in ("inverse", "nested", "rank"):
            d = draws[k][:, j]
            ks = ss.ks_2samp(t, d)
            if two:
                c = d.mean(); level = float(np.mean(np.abs(t - c) >= np.quantile(np.abs(d - c), 0.95)))
            else:
                level = float(np.mean(t >= np.quantile(d, 0.95)))
            lo, hi = ss.binomtest(int(round(level * len(t))), len(t)).proportion_ci(method="exact")
            row[k] = dict(ks_p=float(ks.pvalue), ks_D=float(ks.statistic), level=level, level_ci=[float(lo), float(hi)],
                          mean_ratio=float(d.mean() / t.mean()) if t.mean() != 0 else None)
        res[name] = row
        print(f"D {name:8s} " + "  ".join(f"{k}: level={row[k]['level']:.3f} [{row[k]['level_ci'][0]:.3f},{row[k]['level_ci'][1]:.3f}] "
                                         f"KS p={row[k]['ks_p']:.3f}" for k in row), flush=True)
    return res


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    part_a(rng); part_b(rng); part_c(rng)
    if ONLY_ABC:
        with open(os.path.join(RESULTS, "12_validation.json")) as fh:
            old = json.load(fh)
        report["partD"], report["R_partD"] = old["partD"], old["R_partD"]
    else:
        report["partD"] = part_d()
        report["R_partD"] = R_D
    report["elapsed_s"] = time.time() - t0
    with open(os.path.join(RESULTS, "12_validation.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    fails = [k for k, v in report.items() if isinstance(v, dict) and v.get("ok") is False]
    print(f"done in {report['elapsed_s']:.0f}s; failures: {fails if fails else 'none'}")


if __name__ == "__main__":
    main()
