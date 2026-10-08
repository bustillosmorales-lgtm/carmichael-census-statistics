"""Second revision, Section 4. Calibration of the leading-digit chi-square when the exponents of
the generalized Benford law are estimated from the same census.

Null model H0: within decade k (10^k <= n < 10^(k+1)) the numbers have density proportional to
x^(theta_k - 1), so that their leading digit has law P_{theta_k}, and the decade counts are
Poisson. The parameters theta_k are estimated as in the article, theta_k = log10(C(10^(k+1))/C(10^k)).

Parametric bootstrap: in each replicate the decade counts are drawn as Poisson with the observed
means, the digits of decade k are multinomial with law P_{theta_k}, the exponents theta_k* are
re-estimated from the replicate counts, the prediction is recomputed with them, and the chi-square
is recomputed. The replicates therefore carry the estimation of the exponents and the dependence
between the fitted probabilities and the digit counts.

A second prediction, used as a sensitivity check, replaces the piecewise power law by the smooth
counting function: the expected count of digit d in decade k is C0((d+1) 10^k) - C0(d 10^k), with C0
the monotone interpolation of log C against log x through the counts at quarter decades.
"""
import os, sys
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.stats import chi2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import PUBLISHED_COUNTS as C, census_cache, generalized_benford, leading_digit, save


def P(theta):
    """Generalized Benford law, with its limit (the Benford law) at theta = 0."""
    if abs(theta) < 1e-12:
        d = np.arange(1, 10.0)
        return np.log10(1 + 1 / d)
    return generalized_benford(theta)

SEED = 20261009
R = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
rng = np.random.default_rng(SEED)
n, r = census_cache()
ld = leading_digit(n)
K = range(3, 18)
obs_kd = np.array([np.bincount(ld[(n >= 10 ** k) & (n < 10 ** (k + 1))], minlength=10)[1:] for k in K])
obs = obs_kd.sum(0)
cnt = obs_kd.sum(1)
C3 = int(np.sum(n < 10 ** 3))                    # 1 (the number 561)


def prediction(dec_counts):
    cum = C3 + np.concatenate(([0], np.cumsum(dec_counts)))
    th = np.log10(cum[1:] / cum[:-1])
    p = sum(c * P(t) for c, t in zip(dec_counts, th)) / dec_counts.sum()
    return p, th


def chisq(o, p):
    e = p * o.sum()
    return float(((o - e) ** 2 / e).sum())


p0, th0 = prediction(cnt)
x2 = chisq(obs, p0)
tv = 0.5 * float(np.abs(obs / obs.sum() - p0).sum())
w = float(np.sqrt(x2 / obs.sum()))
print(f"chi-square {x2:.2f}; asymptotic p on 8 df {chi2.sf(x2, 8):.4f}; TV {tv:.5f}; Cohen w {w:.5f}")

boot = np.empty(R)
for b in range(R):
    c = rng.poisson(cnt)
    o = np.array([rng.multinomial(ci, P(t)) for ci, t in zip(c, th0)])
    pb, _ = prediction(c)
    boot[b] = chisq(o.sum(0), pb)
p_boot = (1 + np.sum(boot >= x2)) / (R + 1)
print(f"bootstrap: mean {boot.mean():.3f} (effective df), var {boot.var():.3f}, 95% quantile {np.quantile(boot, .95):.2f}, "
      f"p = {p_boot:.5f} (MC se {np.sqrt(p_boot * (1 - p_boot) / R):.5f})")

# second prediction: smooth counting function C0 (monotone interpolation of log C against log x
# through the counts at quarter decades). Its null is the Poisson process with mean measure C0,
# simulated on the common refinement of the digit cells and the quarter-decade grid, with C0
# re-estimated from every replicate.
qgrid = 10 ** np.arange(3, 18.01, 0.25)
dcell = np.array([d * 10.0 ** k for k in K for d in range(1, 10)] + [1e18])
brk = np.unique(np.concatenate((qgrid, dcell)))


def smooth_pred(cum_at_q):
    f = PchipInterpolator(np.log(qgrid), np.log(np.maximum(cum_at_q, 0.5)))
    c0 = np.exp(f(np.log(dcell)))
    cells = np.diff(c0).reshape(len(K), 9)      # decade k, digit d
    return cells.sum(0) / cells.sum()


Cq = C3 + np.searchsorted(n[n >= 10 ** 3], qgrid).astype(float)
ps = smooth_pred(Cq)
x2s = chisq(obs, ps)
f0 = PchipInterpolator(np.log(qgrid), np.log(Cq))
mu = np.diff(np.exp(f0(np.log(brk))))           # expected count in each elementary interval
qi = np.searchsorted(brk, qgrid)                # quarter-decade points as indices into brk
cell_of = np.searchsorted(dcell, brk[:-1], side="right") - 1
boot_s = np.empty(R)
for b in range(R):
    c = rng.poisson(mu)
    cum = C3 + np.concatenate(([0], np.cumsum(c)))
    o = np.bincount(cell_of, weights=c, minlength=len(dcell) - 1)[: len(dcell) - 1].reshape(len(K), 9).sum(0)
    boot_s[b] = chisq(o, smooth_pred(cum[qi].astype(float)))
p_boot_s = (1 + np.sum(boot_s >= x2s)) / (R + 1)
print("smooth-C prediction:", " ".join(f"{v:.4f}" for v in ps), f"chi-square {x2s:.2f}")
print(f"smooth-C bootstrap: mean {boot_s.mean():.3f}, 95% quantile {np.quantile(boot_s, .95):.2f}, p = {p_boot_s:.5f}")
print("piecewise prediction:", " ".join(f"{v:.4f}" for v in p0))
print("observed            :", " ".join(f"{v:.4f}" for v in obs / obs.sum()))

save("16_digits_bootstrap.json", dict(seed=SEED, R=R, chi_square=x2, p_asymptotic_8df=float(chi2.sf(x2, 8)),
     total_variation=tv, cohen_w=w, boot_mean=float(boot.mean()), boot_var=float(boot.var()),
     boot_q95=float(np.quantile(boot, .95)), p_boot=float(p_boot),
     p_boot_se=float(np.sqrt(p_boot * (1 - p_boot) / R)), theta=th0.tolist(),
     smooth_prediction=ps.tolist(), chi_square_smooth=x2s, boot_smooth_mean=float(boot_s.mean()),
     boot_smooth_q95=float(np.quantile(boot_s, .95)), p_boot_smooth=float(p_boot_s), observed=(obs / obs.sum()).tolist(),
     piecewise_prediction=p0.tolist()))
