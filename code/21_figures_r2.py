"""Second revision, figures of Section 3.

  fig_power.pdf    rejection rate at the 5% level against the parameter of each of the eight families
                   of departures (script 14, tabulated by script 14b), one curve per primary statistic,
                   with exact 95% binomial intervals; the dotted line is the nominal level 0.05.
  fig_profile.pdf  standardized excess Spearman autocorrelation z_h of the gap ratios at lags 1 to 50
                   (script 15), with the simultaneous 95% bands given by the null distribution of
                   max |z_h| over lags 1-10 and 1-50, and the pointwise band +-1.96.

Usage: python code/21_figures_r2.py
"""
import json, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS

OUT = os.path.join(RESULTS, "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})

# ----------------------------------------------------------------------------- power curves
with open(os.path.join(RESULTS, "14b_power_table.json")) as fh:
    tab = json.load(fh)["table"]
with open(os.path.join(RESULTS, "14_power.json")) as fh:
    grid = json.load(fh)["grid"]
TITLES = {"gamma": r"(i) gamma renewal, shape $\kappa$", "ar1": r"(ii) AR(1) copula, $\rho$",
          "fgn": r"(iii) fGn copula, Hurst $H$", "cluster": r"(iv) rare clustering, $\pi$",
          "hardcore": r"(v) repulsion, $\delta$", "breaks": r"(vi) change at $10^{16}$, $\eta$",
          "oscillation": r"(vii) modulated intensity, $a$", "antithetic": r"(viii) antithetic pairs, $p$"}
STATS = [("D_1000", r"$D_{1000}$", "#1f5fa6", "o"), ("D_ratio", r"$D_{\mathrm{ratio}}$", "#b03a2e", "s"),
         ("m_ratio", r"$\bar q$", "#e08a1e", "^"), ("D_cf", r"$D_{\mathrm{cf}}$", "#2e7d32", "D"),
         ("Q_10", r"$Q_{10}$", "#6a3d9a", "v")]
fig, axes = plt.subplots(2, 4, figsize=(7.4, 4.0), sharey=True)
for ax, fam in zip(axes.ravel(), grid):
    xs = grid[fam]
    pos = np.arange(len(xs))
    for j, (key, lab, col, mk) in enumerate(STATS):
        r = np.array([tab[f"{fam}|{a}"][key]["rate"] for a in xs])
        ci = np.array([tab[f"{fam}|{a}"][key]["ci"] for a in xs])
        off = (j - 2) * 0.06
        ax.errorbar(pos + off, r, yerr=[r - ci[:, 0], ci[:, 1] - r], color=col, marker=mk, ms=2.5, lw=0.9,
                    elinewidth=0.6, capsize=0, label=lab)
    ax.axhline(0.05, color="0.4", ls=":", lw=0.8)
    ax.set_xticks(pos)
    ax.set_xticklabels([f"{a:g}" for a in xs], rotation=60, fontsize=6)
    ax.set_title(TITLES[fam], fontsize=7, loc="left")
    ax.set_ylim(-0.03, 1.03)
for ax in axes[:, 0]:
    ax.set_ylabel("rejection rate")
axes[0, 0].legend(frameon=False, fontsize=6, loc="center left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_power.pdf"))
fig.savefig(os.path.join(OUT, "fig_power.png"), dpi=600)

# ----------------------------------------------------------------------------- dependence profile
with open(os.path.join(RESULTS, "15_dependence.json")) as fh:
    dep = json.load(fh)
z = np.array(dep["profile"]["z"])
h = np.arange(1, len(z) + 1)
c10, c50 = dep["max_abs_z_10"]["crit95"], dep["max_abs_z_50"]["crit95"]
fig, ax = plt.subplots(figsize=(7.2, 2.4))
ax.axhspan(-1.96, 1.96, color="0.88", lw=0, label="pointwise 95%")
ax.hlines([-c10, c10], 0.5, 10.5, colors="#b03a2e", lw=1, label=f"simultaneous 95%, lags 1-10 ($\\pm${c10:.2f})")
ax.hlines([-c50, c50], 0.5, 50.5, colors="#1f5fa6", lw=1, ls="--", label=f"simultaneous 95%, lags 1-50 ($\\pm${c50:.2f})")
ax.vlines(h, 0, z, color="k", lw=1)
ax.plot(h, z, "ko", ms=2.5)
ax.axhline(0, color="k", lw=0.5)
ax.set_xlim(0.5, 50.5)
ax.set_xlabel("lag $h$")
ax.set_ylabel("$z_h$")
ax.legend(frameon=False, fontsize=6, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_profile.pdf"))
fig.savefig(os.path.join(OUT, "fig_profile.png"), dpi=600)
print("figures written to", OUT)
