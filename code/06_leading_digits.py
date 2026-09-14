"""Section 4. Leading digits: observed distribution against the growth-curve prediction, the
chi-square statistic, the best-fitting exponent by number of prime factors inside the decade
1e17-1e18 with multinomial bootstrap intervals, and the aggregate fit used in Section 7, item (i)."""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import chi2

from common import PUBLISHED_COUNTS as C, census_cache, generalized_benford as P, leading_digit, save

SEED = 20260916
B = 2000
rng = np.random.default_rng(SEED)
n, r = census_cache()
ld = leading_digit(n)
obs = np.bincount(ld, minlength=10)[1:]
p_obs = obs / obs.sum()

# prediction: decade k (10^k..10^(k+1)) carries the law with theta_k = log10(C(10^(k+1))/C(10^k)),
# weighted by its occupancy; decades from 10^3 upward hold every number except 561
pred = np.zeros(9)
tot = 0
wsum = 0.0
for k in range(3, 18):
    cnt = C[k + 1] - C[k]
    th = np.log10(C[k + 1] / C[k])
    pred += cnt * P(th)
    tot += cnt
    wsum += cnt * th
pred /= tot
theta_bar = wsum / tot
E = pred * obs.sum()
chi_sq = float(((obs - E) ** 2 / E).sum())
crit = float(chi2.ppf(0.95, 8))
p_value = float(chi2.sf(chi_sq, 8))
dev = p_obs - pred
print("observed ", " ".join(f"{v:.4f}" for v in p_obs))
print("predicted", " ".join(f"{v:.4f}" for v in pred))
print(f"max |obs-pred| = {np.max(np.abs(dev)):.5f}, max relative = {100 * np.max(np.abs(dev) / pred):.2f}%")
print(f"chi-square = {chi_sq:.2f} on 8 df, 5% critical value {crit:.2f}, p = {p_value:.4f}")


def fit(counts):
    p = counts / counts.sum()
    return minimize_scalar(lambda a: ((p - P(a)) ** 2).sum(), bounds=(0.01, 3.0), method="bounded").x


dec = (n >= 10 ** 17) & (n < 10 ** 18)
by_r = {}
for k in sorted(set(r[dec].tolist())):
    c = np.bincount(ld[dec & (r == k)], minlength=10)[1:]
    a = fit(c)
    boot = [fit(rng.multinomial(c.sum(), c / c.sum())) for _ in range(B)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    by_r[int(k)] = dict(count=int(c.sum()), alpha=float(a), ci95=[float(lo), float(hi)])
    print(f"r={k:2d} count={c.sum():7d} alpha={a:.3f} 95% CI [{lo:.3f}, {hi:.3f}]")

alpha_all = fit(obs)
low_r = (r >= 3) & (r <= 6)
alpha_low_r = fit(np.bincount(ld[low_r], minlength=10)[1:])
print(f"aggregate best alpha {alpha_all:.4f}, occupancy-weighted theta {theta_bar:.4f}, gap {alpha_all - theta_bar:+.4f}")
print(f"restricted to r = 3..6 ({low_r.sum()} numbers): best alpha {alpha_low_r:.4f}, gap {alpha_low_r - theta_bar:+.4f}")

save("06_leading_digits.json", dict(
    seed=SEED, bootstrap=B, counts=obs.tolist(), observed=p_obs.tolist(), predicted=pred.tolist(),
    max_abs_dev=float(np.max(np.abs(dev))), max_rel_dev=float(np.max(np.abs(dev) / pred)),
    chi_square=chi_sq, df=8, critical_5pct=crit, p_value=p_value, n_predicted=int(tot),
    alpha_by_r_decade17=by_r, decade17_total=int(dec.sum()),
    alpha_aggregate=float(alpha_all), theta_bar=float(theta_bar), gap_aggregate=float(alpha_all - theta_bar),
    count_r3to6=int(low_r.sum()), alpha_r3to6=float(alpha_low_r), gap_r3to6=float(alpha_low_r - theta_bar)))
