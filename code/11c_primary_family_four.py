"""Second revision, Section 3. Primary family of four statistics, with D_cf reported descriptively.

Part E of script 12 showed that the Monte Carlo test of D_cf (40 labelings) rejects a true Poisson
process with known smooth intensity in 26 to 31% of realizations at nominal level 0.05, under all
three generators: the reference model underestimates the null mean of D_cf (0.00118 against
0.00127). D_cf is therefore removed from the tested family and kept as a descriptive measure.

This script reuses the joint null distribution saved by script 11b (no new simulation) and
recomputes, over D_1000, D_ratio, mean ratio and Q_10, the Monte Carlo p-values, Holm's step-down
adjustment and the single-step min-p global p-value of Westfall and Young.

Usage: python code/11c_primary_family_four.py
"""
import importlib, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark as B
from common import RESULTS

pvals = importlib.import_module("11b_primary_family").pvals


def main():
    z = np.load(os.path.join(RESULTS, "11b_primary_family.npz"))
    obs, D = z["obs"], z["null"]
    mu, sd = D[:, 4:].mean(0), D[:, 4:].std(0)
    Q = lambda x: np.sum(((x - mu) / sd) ** 2, axis=-1)                     # noqa: E731
    labels = ["D_1000", "D_ratio", "m_ratio", "Q_10"]
    fam_t = np.array([obs[0], obs[1], obs[2], Q(obs[4:])])
    fam_D = np.column_stack([D[:, 0], D[:, 1], D[:, 2], Q(D[:, 4:])])
    two = [False, False, True, False]
    p_t, p_D = pvals(fam_t, fam_D, two)
    m = len(labels)
    order = np.argsort(p_t); holm = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * p_t[i])); holm[i] = run
    p_global = float((1 + np.sum(p_D.min(1) <= p_t.min())) / (len(D) + 1))
    d = D[:, 3]
    res = dict(R=len(D),
               family={lab: dict(obs=float(fam_t[i]), p=float(p_t[i]), p_se=B.mc_se(p_t[i], len(D)),
                                 p_holm=float(holm[i])) for i, lab in enumerate(labels)},
               global_minp=dict(min_p=float(p_t.min()), p_global=p_global),
               D_cf_descriptive=dict(obs=float(obs[3]), null_mean=float(d.mean()),
                                     q025=float(np.quantile(d, .025)), q975=float(np.quantile(d, .975))))
    print(json.dumps(res, indent=1))
    with open(os.path.join(RESULTS, "11c_primary_family_four.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
