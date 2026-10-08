"""Reference model, generators and statistics for the second revision (Section 3).

Notation (as in the article):
  n_1 < ... < n_N        observed Carmichael numbers, g_i = n_{i+1} - n_i, i = 1..N-1
  m_S(i)                 centered moving mean of S consecutive observed gaps at gap index i
  lambda_S(x)            estimated intensity: 1 / m_S, placed at the gap midpoints and
                         interpolated linearly in x
  Lambda_S(x)            cumulative intensity, the integral of lambda_S
  reference model        Poisson process with mean measure Lambda_S, conditional on N points
  u_i^(W)                g_i divided by the centered moving mean of W gaps (normalization)

Two generators of the reference model:
  gen_inverse  standard construction: points Lambda_S^{-1}(Gamma_k), Gamma_k the order
               statistics of N uniforms on [0, Lambda_S(n_N)], obtained from normalized
               partial sums of N+1 unit exponentials
  gen_rank     the generator of the first revision: gaps m_S(i) E_i indexed by rank i
"""
import numpy as np
from scipy.special import ndtr  # noqa: F401  (re-exported for alternatives)

WINDOWS = (100, 1000, 5000, 20000)
LAGS = 10


# ----------------------------------------------------------------------------- estimators
def rollmean(x, W):
    """Centered moving mean over W terms, truncated at both ends."""
    c = np.concatenate(([0.0], np.cumsum(x)))
    m = len(x)
    half = W // 2
    idx = np.arange(m)
    lo = np.clip(idx - half, 0, m)
    hi = np.clip(idx - half + W, 0, m)
    return (c[hi] - c[lo]) / (hi - lo)


def cumulative_intensity(pos, S=20000):
    """Knots (x_k, Lambda_k) of the estimated cumulative intensity Lambda_S.

    The intensity 1/m_S is placed at the gap midpoints and integrated by the trapezoid rule;
    the first and last knots are the first and last observed points."""
    g = np.diff(pos)
    lam = 1.0 / rollmean(g, S)
    mid = 0.5 * (pos[:-1] + pos[1:])
    x = np.concatenate(([pos[0]], mid, [pos[-1]]))
    lv = np.concatenate(([lam[0]], lam, [lam[-1]]))
    L = np.concatenate(([0.0], np.cumsum(0.5 * (lv[1:] + lv[:-1]) * np.diff(x))))
    return x, L


def gen_inverse(xk, Lk, N, rng):
    """N points of the Poisson process with cumulative intensity (xk, Lk), conditional on N,
    by inversion of the cumulative intensity. Returns the N-1 gaps."""
    e = rng.exponential(size=N + 1)
    gam = np.cumsum(e)[:-1] / e.sum() * Lk[-1]
    pts = np.interp(gam, Lk, xk)
    return np.diff(pts)


def gen_rank(m, rng):
    """Rank-indexed exponential gaps m_i E_i (generator of the first revision)."""
    return rng.exponential(size=len(m)) * m


def slow_variation(pos, S=20000, W=1000):
    """Relative change of the estimated mean gap m_S across a window of W gaps,
    |m_S(i + W/2) - m_S(i - W/2)| / m_S(i), at every interior gap index i.
    This is the quantity that the condition 'slowly varying at window W' bounds."""
    m = rollmean(np.diff(pos), S)
    h = W // 2
    return np.abs(m[2 * h:] - m[:-2 * h]) / m[h:-h]


# ----------------------------------------------------------------------------- statistics
exp_cdf = lambda v: 1.0 - np.exp(-v)              # noqa: E731
ratio_cdf = lambda v: 2.0 * v / (1.0 + v)         # noqa: E731


def sup_dev(v, cdf):
    """Supremum distance between the empirical distribution of v and the continuous cdf."""
    v = np.sort(v)
    m = len(v)
    F = cdf(v)
    k = np.arange(1, m + 1)
    return float(max(np.max(k / m - F), np.max(F - (k - 1) / m)))


def ratios(g):
    return np.minimum(g[:-1], g[1:]) / np.maximum(g[:-1], g[1:])


def rank_acf(x, L=LAGS):
    """Spearman autocorrelations at lags 1..L (ranks standardized once)."""
    rk = np.empty(len(x))
    rk[np.argsort(x, kind="stable")] = np.arange(len(x), dtype=float)
    rk = (rk - rk.mean()) / rk.std()
    return np.array([np.mean(rk[:-h] * rk[h:]) for h in range(1, L + 1)])


def cross_fit(g, rng, W=1000, splits=5):
    """Mean over independent Bernoulli(1/2) labelings of the cross-fitted supremum distance."""
    pos = np.concatenate(([0.0], np.cumsum(g)))
    vals = []
    for _ in range(splits):
        lab = rng.random(len(pos)) < 0.5
        A, B = pos[lab], pos[~lab]
        mA = rollmean(np.diff(A), W)
        midA = 0.5 * (A[:-1] + A[1:])
        gB = np.diff(B)
        midB = 0.5 * (B[:-1] + B[1:])
        vals.append(sup_dev(gB / np.interp(midB, midA, mA), exp_cdf))
    return float(np.mean(vals))


STAT_NAMES = ([f"D_{W}" for W in WINDOWS] + ["D_ratio", "m_ratio", "D_cf"]
              + [f"rho_{h}" for h in range(1, LAGS + 1)] + [f"urho_{h}" for h in range(1, 3)])


def statistics(g, rng, cf=True):
    """Vector of every statistic of the family, in the order of STAT_NAMES.

    urho_h is the lag-h Spearman autocorrelation of the normalized gaps u^(1000), a second
    dependence measure that does not share a gap between consecutive terms."""
    out = []
    u1000 = None
    for W in WINDOWS:
        u = g / rollmean(g, W)
        if W == 1000:
            u1000 = u
        out.append(sup_dev(u, exp_cdf))
    q = ratios(g)
    out += [sup_dev(q, ratio_cdf), float(q.mean())]
    out.append(cross_fit(g, rng) if cf else np.nan)
    out += list(rank_acf(q))
    out += list(rank_acf(u1000, 2))
    return np.array(out)


TWO_SIDED = {"m_ratio"} | {f"rho_{h}" for h in range(1, LAGS + 1)} | {"urho_1", "urho_2"}


def mc_pvalue(t, draws, two_sided):
    """Monte Carlo p-value (1 + #{T* at least as extreme}) / (R + 1)."""
    d = np.asarray(draws)
    R = len(d)
    if two_sided:
        c = d.mean()
        return (1 + np.sum(np.abs(d - c) >= abs(t - c))) / (R + 1)
    return (1 + np.sum(d >= t)) / (R + 1)


def mc_se(p, R):
    """Monte Carlo standard error of an estimated p-value."""
    return float(np.sqrt(p * (1 - p) / R))
