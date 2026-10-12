"""Second revision, Section 3. Rejection rates of the power study (script 14) at the 5% level, with
exact 95% binomial (Clopper-Pearson) intervals.

Critical values come from the null distributions of the primary analysis: D_1000, D_ratio, the mean
ratio and Q_10 from the 2,000 realizations of script 11b; D_cf from the 2,000 realizations by inversion
of script 11, because the power study averages D_cf over 5 labelings, as script 11 does (script 11b
uses 40). The mean ratio is two-sided around the simulated null mean.

Usage: python code/14b_power_table.py
"""
import json, os, sys
import numpy as np
from scipy.stats import beta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS


def cp(k, n):
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return float(lo), float(hi)


def main():
    pw = np.load(os.path.join(RESULTS, "14_power.npz"))
    names = list(pw["names"])
    nb = np.load(os.path.join(RESULTS, "11b_primary_family.npz"))["null"]
    ref = np.load(os.path.join(RESULTS, "11_reference_model.npz"))
    inv = ref["inverse"]; rnames = list(ref["names"])
    mu, sd = nb[:, 4:].mean(0), nb[:, 4:].std(0)
    Qn = np.sum(((nb[:, 4:] - mu) / sd) ** 2, axis=1)
    c = dict(D_1000=np.quantile(nb[:, 0], .95), D_ratio=np.quantile(nb[:, 1], .95),
             D_cf=np.quantile(inv[:, rnames.index("D_cf")], .95), Q_10=np.quantile(Qn, .95))
    mr = nb[:, 2]; mrc = mr.mean(); mr_crit = np.quantile(np.abs(mr - mrc), .95)
    ir = [names.index(f"rho_{h}") for h in range(1, 11)]
    out = {}
    for key in [k for k in pw.files if k != "names"]:
        A = pw[key]; n = len(A)
        rej = dict(D_1000=A[:, names.index("D_1000")] > c["D_1000"],
                   D_ratio=A[:, names.index("D_ratio")] > c["D_ratio"],
                   m_ratio=np.abs(A[:, names.index("m_ratio")] - mrc) > mr_crit,
                   D_cf=A[:, names.index("D_cf")] > c["D_cf"],
                   Q_10=np.sum(((A[:, ir] - mu) / sd) ** 2, axis=1) > c["Q_10"])
        out[key] = {s: dict(rate=float(v.mean()), ci=cp(int(v.sum()), n)) for s, v in rej.items()}
        out[key]["R"] = n
        print(f"{key:18s} " + " ".join(f"{s}={out[key][s]['rate']:.3f}" for s in rej), flush=True)
    with open(os.path.join(RESULTS, "14b_power_table.json"), "w") as fh:
        json.dump(dict(critical={k: float(v) for k, v in c.items()}, m_ratio_center=float(mrc),
                       m_ratio_halfwidth=float(mr_crit), table=out), fh, indent=1)


if __name__ == "__main__":
    main()
