"""Second revision. Slow-variation condition of the article at W = 1,000 for the census from 10^18
to 10^24, for the full range, by number of prime factors and by decade within each class.

Reads the lambda_knots.npz files written by 19_census_1e24_analysis.py prepare, which store the
estimated cumulative intensity at one knot per 100 gaps, so that +-500 gaps is +-5 knots. The
quantity is |m_S(i + W/2) - m_S(i - W/2)| / m_S(i), as in benchmark.slow_variation.

Usage: python code/23_slow_variation_1e24.py DATA_DIR
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS

DATA = sys.argv[-1]
CLASSES = range(3, 13)


def mean_gap(folder):
    kn = np.load(os.path.join(DATA, folder, "lambda_knots.npz"))
    dens = np.diff(kn["L"]) / np.diff(kn["x"])      # local intensity between knots 100 gaps apart
    return 1.0 / dens[1:-1]                          # drop the padded end knots


def slow_variation(m):
    return np.abs(m[10:] - m[:-10]) / m[5:-5]


out = {}
for r in ["all"] + list(CLASSES):
    sv = slow_variation(mean_gap("analysis" if r == "all" else f"analysis_r{r}"))
    q = np.quantile(sv, [0.5, 0.9, 0.99]).tolist()
    out[str(r)] = dict(median=q[0], q90=q[1], q99=q[2], share_above_005=float(np.mean(sv > 0.05)), knots=len(sv) + 10)
    print(r, {k: round(v, 5) for k, v in out[str(r)].items()}, flush=True)
with open(os.path.join(RESULTS, "23_slow_variation_1e24_by_r.json"), "w") as fh:
    json.dump(out, fh, indent=1)

res = {}
for r in CLASSES:
    folder = f"analysis_r{r}"
    sv = slow_variation(mean_gap(folder))
    with open(os.path.join(DATA, folder, "meta.json")) as fh:
        db = json.load(fh)["decade_bounds"]
    gi = (np.arange(len(sv)) + 5) * 100              # gap index of the centre knot
    row = []
    for j in range(6):
        s = sv[(gi >= db[j]) & (gi < db[j + 1])]
        row.append((db[j + 1] - db[j], round(float(np.median(s)), 4) if len(s) else None,
                    round(float(np.mean(s > 0.05)), 3) if len(s) else None))
    res[str(r)] = row
    print(r, row, flush=True)
with open(os.path.join(RESULTS, "23_slow_variation_1e24_by_r_decade.json"), "w") as fh:
    json.dump(res, fh, indent=1)
