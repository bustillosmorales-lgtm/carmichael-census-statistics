"""External validation, secondary analysis: leading digits of the Carmichael numbers from 10^18 to
10^24 (prereg/analysis_plan_1e24.md), with the calibration of script 16 applied to that range.

Piecewise prediction: within decade k (k = 18..23) the leading digit follows P_{theta_k}, with
theta_k = log10(C(10^(k+1)) / C(10^k)) from the counts of the census itself; the chi-square is
calibrated by the parametric bootstrap of script 16 (Poisson decade counts with the observed means,
multinomial digits, exponents re-estimated in every replicate).
Smooth prediction (sensitivity, as in script 16): the expected count of digit d in decade k is
C0((d+1) 10^k) - C0(d 10^k), with C0 the monotone interpolation of log C against log x through the
counts at quarter decades from 10^18 to 10^24, calibrated by the Poisson process with mean measure C0
with C0 re-estimated in every replicate.

Input: d.npy (leading digit), hi.npy, lo.npy (n = hi * 10^12 + lo) written by script 18.
Usage: python code/22_digits_1e24.py [R] DATA_DIR
"""
import json, os, sys, time
from decimal import Decimal, getcontext
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.stats import chi2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS, generalized_benford

DATA = sys.argv[-1]
R = int(sys.argv[1]) if len(sys.argv) > 2 else 100000
SEED = 20261011
SPLIT = 10 ** 12
K = list(range(18, 24))
getcontext().prec = 40


def P(theta):
    if abs(theta) < 1e-12:
        d = np.arange(1, 10.0)
        return np.log10(1 + 1 / d)
    return generalized_benford(theta)


def count_below(hi, lo, x):
    """Number of n < x (x a positive integer) in the sorted census n = hi * 10^12 + lo."""
    xh, xl = divmod(x, SPLIT)
    a = int(np.searchsorted(hi, np.uint64(xh), side="left"))
    b = int(np.searchsorted(hi, np.uint64(xh), side="right"))
    return a + int(np.sum(np.asarray(lo[a:b]) < xl))


def chisq(o, p):
    e = p * o.sum()
    return float(((o - e) ** 2 / e).sum())


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    d = np.load(os.path.join(DATA, "d.npy"), mmap_mode="r")
    hi = np.load(os.path.join(DATA, "hi.npy"), mmap_mode="r")
    lo = np.load(os.path.join(DATA, "lo.npy"), mmap_mode="r")
    bounds = [count_below(hi, lo, 10 ** k) for k in K + [24]]
    obs_kd = np.array([np.bincount(np.asarray(d[bounds[j]:bounds[j + 1]]), minlength=10)[1:] for j in range(len(K))])
    obs = obs_kd.sum(0); cnt = obs_kd.sum(1); C18 = bounds[0]

    def prediction(dec_counts):
        cum = C18 + np.concatenate(([0], np.cumsum(dec_counts)))
        th = np.log10(cum[1:] / cum[:-1])
        return sum(c * P(t) for c, t in zip(dec_counts, th)) / dec_counts.sum(), th

    p0, th0 = prediction(cnt)
    x2 = chisq(obs, p0)
    tv = 0.5 * float(np.abs(obs / obs.sum() - p0).sum())
    w = float(np.sqrt(x2 / obs.sum()))
    boot = np.empty(R)
    for b in range(R):
        c = rng.poisson(cnt)
        o = np.array([rng.multinomial(ci, P(t)) for ci, t in zip(c, th0)])
        boot[b] = chisq(o.sum(0), prediction(c)[0])
    p_boot = (1 + np.sum(boot >= x2)) / (R + 1)
    print(f"N={obs.sum()} chi-square {x2:.2f}; TV {tv:.6f}; Cohen w {w:.6f}; boot mean {boot.mean():.3f}, "
          f"q95 {np.quantile(boot, .95):.2f}, p = {p_boot:.6f}", flush=True)

    # smooth prediction through the counts at quarter decades
    qexp = [Decimal(18) + Decimal(j) / 4 for j in range(25)]
    qint = [int((Decimal(10) ** e).to_integral_value(rounding="ROUND_CEILING")) for e in qexp]
    qgrid = np.array([float(Decimal(10) ** e) for e in qexp])
    Cq = np.array([count_below(hi, lo, x) for x in qint], dtype=float)
    dcell = np.array([dd * 10.0 ** k for k in K for dd in range(1, 10)] + [1e24])
    brk = np.unique(np.concatenate((qgrid, dcell)))

    def smooth_pred(cum_at_q):
        f = PchipInterpolator(np.log(qgrid), np.log(np.maximum(cum_at_q, 0.5)))
        cells = np.diff(np.exp(f(np.log(dcell)))).reshape(len(K), 9)
        return cells.sum(0) / cells.sum()

    ps = smooth_pred(Cq)
    x2s = chisq(obs, ps)
    f0 = PchipInterpolator(np.log(qgrid), np.log(Cq))
    mu = np.diff(np.exp(f0(np.log(brk))))
    qi = np.searchsorted(brk, qgrid)
    cell_of = np.searchsorted(dcell, brk[:-1], side="right") - 1
    Rs = R
    boot_s = np.empty(Rs)
    for b in range(Rs):
        c = rng.poisson(mu)
        cum = C18 + np.concatenate(([0], np.cumsum(c)))
        o = np.bincount(cell_of, weights=c, minlength=len(dcell) - 1)[: len(dcell) - 1].reshape(len(K), 9).sum(0)
        boot_s[b] = chisq(o, smooth_pred(cum[qi].astype(float)))
    p_boot_s = (1 + np.sum(boot_s >= x2s)) / (Rs + 1)
    tvs = 0.5 * float(np.abs(obs / obs.sum() - ps).sum())
    print(f"smooth: chi-square {x2s:.2f}; TV {tvs:.6f}; boot mean {boot_s.mean():.3f}, q95 {np.quantile(boot_s, .95):.2f}, "
          f"p = {p_boot_s:.6f} (R={Rs})", flush=True)
    print("observed  :", " ".join(f"{v:.5f}" for v in obs / obs.sum()))
    print("piecewise :", " ".join(f"{v:.5f}" for v in p0))
    print("smooth    :", " ".join(f"{v:.5f}" for v in ps))
    out = dict(seed=SEED, R=R, R_smooth=Rs, N=int(obs.sum()), decade_counts=cnt.tolist(), theta=th0.tolist(),
               observed=(obs / obs.sum()).tolist(), piecewise_prediction=p0.tolist(), chi_square=x2,
               p_asymptotic_8df=float(chi2.sf(x2, 8)), total_variation=tv, cohen_w=w,
               boot_mean=float(boot.mean()), boot_q95=float(np.quantile(boot, .95)), p_boot=float(p_boot),
               smooth_prediction=ps.tolist(), chi_square_smooth=x2s, total_variation_smooth=tvs,
               cohen_w_smooth=float(np.sqrt(x2s / obs.sum())), boot_smooth_mean=float(boot_s.mean()),
               boot_smooth_q95=float(np.quantile(boot_s, .95)), p_boot_smooth=float(p_boot_s),
               quarter_decade_counts=Cq.tolist(), elapsed_s=time.time() - t0)
    with open(os.path.join(RESULTS, "22_digits_1e24.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"done in {out['elapsed_s']:.0f}s")


if __name__ == "__main__":
    main()
