"""Second revision, Section 3. Power of the family of spacing statistics against eight families of
departures, each over a grid of its parameter, with 1,000 realizations per setting.

Every alternative is built on the rescaled axis, where the reference model is a unit-rate Poisson
process: a sequence eps_1..eps_{N-1} of unit-mean gaps with the alternative structure is mapped back
to the numerical axis through the inverse of the estimated cumulative intensity Lambda_S of the
census (S = 20,000), exactly as the reference model itself (benchmark.gen_inverse). Under the
reference model eps_i are independent unit exponentials.

Families (parameter, values):
  gamma        renewal, gamma gaps of shape kappa                    clustering (<1) / repulsion (>1)
  ar1          Gaussian AR(1) copula, exponential margins, lag-1 rho  short-range dependence
  fgn          fractional Gaussian noise copula, exponential margins, Hurst H   long-range dependence
  cluster      a fraction pi of gaps split into a pair, the shorter of mean 0.01   rare local clustering
  hardcore     gaps delta + (1 - delta) E                              local repulsion (minimum distance)
  breaks       gamma shape 1 - eps below 10^16 and 1 + eps above       change across decades
  oscillation  Poisson with intensity modulated by 1 + a sin(2 pi i / 2,000)   intensity outside the
               admissible class (varies at the scale of the window)
  antithetic   pairs (E, -log(1 - exp(-E))) with probability p, exponential margins kept
               negative dependence with the marginal law unchanged

Usage: python code/14_power.py [R] [workers]
Writes results/14_power.npz (statistics of every realization) and results/14_power.json.
"""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np
from scipy.special import ndtr
from scipy.signal import lfilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark as B
from common import RESULTS, census_cache

R = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 14
SEED = 20261012
S = 20000

GRID = {
    "gamma": [0.98, 0.99, 0.995, 0.9975, 1.0025, 1.005, 1.01, 1.02],
    "ar1": [0.001, 0.002, 0.003, 0.005, 0.0075, 0.01],
    "fgn": [0.505, 0.51, 0.52, 0.53, 0.55],
    "cluster": [0.0005, 0.001, 0.002, 0.005],
    "hardcore": [0.002, 0.005, 0.01, 0.02],
    "breaks": [0.005, 0.01, 0.02],
    "oscillation": [0.02, 0.05, 0.1],
    "antithetic": [0.005, 0.01, 0.02, 0.05],
}


def fgn(n, H, rng):
    """Fractional Gaussian noise of length n, unit variance, by the Davies-Harte method."""
    k = np.arange(n + 1)
    gam = 0.5 * ((k + 1.0) ** (2 * H) - 2 * k ** (2.0 * H) + np.abs(k - 1.0) ** (2 * H))
    c = np.concatenate((gam, gam[-2:0:-1]))
    lam = np.fft.fft(c).real
    lam = np.clip(lam, 0, None)
    m = len(c)
    z = rng.standard_normal(m) + 1j * rng.standard_normal(m)
    return np.fft.fft(np.sqrt(lam / m) * z).real[:n]


def eps(family, a, n, idx16, rng):
    if family == "gamma":
        return rng.gamma(a, 1.0 / a, size=n)
    if family == "ar1":
        z = lfilter([np.sqrt(1 - a * a)], [1.0, -a], rng.standard_normal(n))
        return -np.log1p(-ndtr(z))
    if family == "fgn":
        return -np.log1p(-ndtr(fgn(n, a, rng)))
    if family == "cluster":
        e = rng.exponential(size=n)
        sel = rng.random(n) < a
        short = np.minimum(rng.exponential(0.01, size=sel.sum()), 0.5 * e[sel])
        e = np.insert(e, np.flatnonzero(sel), short)   # a short gap before each selected gap
        e[np.flatnonzero(sel) + np.arange(1, sel.sum() + 1)] -= short
        return e[:n]
    if family == "hardcore":
        return a + (1 - a) * rng.exponential(size=n)
    if family == "breaks":
        e = np.empty(n)
        e[:idx16] = rng.gamma(1 - a, 1.0 / (1 - a), size=idx16)
        e[idx16:] = rng.gamma(1 + a, 1.0 / (1 + a), size=n - idx16)
        return e
    if family == "oscillation":
        return rng.exponential(size=n) / (1 + a * np.sin(2 * np.pi * np.arange(n) / 2000.0))
    if family == "antithetic":
        e = rng.exponential(size=n)
        sel = np.flatnonzero(rng.random(n // 2) < a) * 2
        e[sel + 1] = -np.log(-np.expm1(-e[sel]))       # U -> 1 - U for the pair partner
        return e
    raise ValueError(family)


def setup():
    n, _ = census_cache()
    pos = n.astype(np.float64)
    xk, Lk = B.cumulative_intensity(pos, S)
    return xk, Lk, len(pos), int(np.searchsorted(n, 10 ** 16))


def to_gaps(e, xk, Lk):
    """Map unit-mean rescaled gaps to the numerical axis through the inverse of Lambda_S."""
    gam = np.concatenate(([0.0], np.cumsum(e)))
    gam = gam / gam[-1] * Lk[-1]
    return np.diff(np.interp(gam, Lk, xk))


def work(task):
    family, a, seed, reps = task
    xk, Lk, N, idx16 = setup()
    rng = np.random.default_rng(seed)
    out = [B.statistics(to_gaps(eps(family, a, N - 1, idx16, rng), xk, Lk), rng) for _ in range(reps)]
    return family, a, np.array(out)


def main():
    """Each setting is saved to results/14_power_parts/ as soon as it is complete, and settings
    already saved are skipped, so that an interrupted run resumes where it stopped. The seeds
    of a setting do not depend on which settings were run before it."""
    t0 = time.time()
    parts = os.path.join(RESULTS, "14_power_parts")
    os.makedirs(parts, exist_ok=True)
    acc = {}
    with Pool(WORKERS) as pool:
        for k, (fam, vals) in enumerate(GRID.items()):
            for j, a in enumerate(vals):
                key = f"{fam}|{a}"
                path = os.path.join(parts, f"{fam}_{a}.npy")
                if os.path.exists(path):
                    acc[key] = np.load(path)
                    continue
                seq = np.random.SeedSequence([SEED, k, j])
                tasks = [(fam, a, c.generate_state(1)[0], 25) for c in seq.spawn(R // 25)]
                arr = np.vstack([x[2] for x in pool.imap_unordered(work, tasks)])
                np.save(path, arr)
                acc[key] = arr
                print(f"  {key} done at {time.time()-t0:.0f}s", flush=True)
    np.savez_compressed(os.path.join(RESULTS, "14_power.npz"), names=np.array(B.STAT_NAMES), **acc)
    with open(os.path.join(RESULTS, "14_power.json"), "w") as fh:
        json.dump(dict(R=R, seed=SEED, S=S, grid=GRID, elapsed_s=time.time() - t0), fh, indent=1)
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
