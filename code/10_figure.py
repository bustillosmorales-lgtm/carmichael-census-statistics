"""Figure 1. (a) Survival function of the normalized spacings (W = 1,000) against the exponential law,
with the pointwise 95% band of 200 inhomogeneous-Poisson draws passed through the same pipeline and
the near-lattice negative control. (b) Best-fitting exponent of the generalized Benford law by number
of prime factors in the decade 1e17-1e18, with the bootstrap intervals of results/06_leading_digits.json."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import RESULTS, census_cache, rollmean

SEED = 20260918
rng = np.random.default_rng(SEED)
n, _ = census_cache()
g = np.diff(n).astype(np.float64)
N = len(g)
lam = rollmean(g, 20000)
grid = np.linspace(0, 7, 141)


def survival(u):
    s = np.sort(u)
    return 1.0 - np.searchsorted(s, grid, side="right") / len(s)


u = g / rollmean(g, 1000)
band = np.array([survival((e := rng.exponential(size=N) * lam) / rollmean(e, 1000)) for _ in range(200)])
lo, hi = np.quantile(band, [0.025, 0.975], axis=0)
lat = np.clip(rng.normal(1.0, 0.15, size=N), 1e-9, None) * lam
s_lat = survival(lat / rollmean(lat, 1000))

with open(os.path.join(RESULTS, "06_leading_digits.json")) as fh:
    digits = json.load(fh)
rs = [int(k) for k in digits["alpha_by_r_decade17"] if 3 <= int(k) <= 10]
alpha = [digits["alpha_by_r_decade17"][str(k)]["alpha"] for k in rs]
ci = np.array([digits["alpha_by_r_decade17"][str(k)]["ci95"] for k in rs])

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
ax[0].fill_between(grid, np.maximum(lo, 1e-5), np.maximum(hi, 1e-5), color="0.8", label="Poisson null, 95% band")
ax[0].semilogy(grid, np.exp(-grid), "k--", lw=1, label=r"exponential $e^{-t}$")
ax[0].semilogy(grid, np.maximum(survival(u), 1e-5), color="#1f5fa6", lw=1.4, label="Carmichael numbers")
ax[0].semilogy(grid, np.maximum(s_lat, 1e-5), ":", color="#b03a2e", lw=1.4, label="near-lattice control")
ax[0].set_ylim(1e-4, 1.5)
ax[0].set_xlabel("normalized gap $t$")
ax[0].set_ylabel(r"$P(u>t)$")
ax[0].set_title("(a)", loc="left")
ax[0].legend(frameon=False, fontsize=7)
ax[1].errorbar(rs, alpha, yerr=[np.array(alpha) - ci[:, 0], ci[:, 1] - np.array(alpha)], fmt="o-",
               color="#1f5fa6", ms=4, capsize=2, lw=1.2)
ax[1].axhline(0, color="#2e7d32", ls="--", lw=1)
ax[1].axhline(1, color="#b03a2e", ls="--", lw=1)
ax[1].text(3, 0.04, "Benford limit", color="#2e7d32", fontsize=7)
ax[1].text(3, 0.93, "uniform", color="#b03a2e", fontsize=7)
ax[1].set_ylim(-0.05, 1.05)
ax[1].set_xlabel("number of prime factors $r$")
ax[1].set_ylabel(r"best-fitting exponent $\alpha$")
ax[1].set_title("(b)", loc="left")
fig.tight_layout()
os.makedirs(os.path.join(RESULTS, "figures"), exist_ok=True)
fig.savefig(os.path.join(RESULTS, "figures", "fig_spacings_digits.pdf"))
fig.savefig(os.path.join(RESULTS, "figures", "fig_spacings_digits.png"), dpi=600)
print("band at t = 5:", lo[100], hi[100], "data:", survival(u)[100])
