"""Section 3.3. Joint serial-dependence test on consecutive gap ratios (scale free), lags 1..L.
S = sum_L (r_L - E0 r_L)            directional: common positive dependence
Q = sum_L ((r_L - E0 r_L)/sd0_L)^2   portmanteau: dependence of any sign
Null: unit-rate Poisson (ratios are scale free). Monte Carlo p-values with R draws.
Also: the same statistics on the decade 1e17-1e18 alone, and on alternating halves of the index
range, to see whether a signal is stable across the census."""
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS, read_census

R = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
L = 10
rng = np.random.default_rng(20260915)
n = read_census()


def lagcorr(g):
    q = np.minimum(g[:-1], g[1:]) / np.maximum(g[:-1], g[1:])
    rk = np.empty(len(q)); rk[np.argsort(q, kind="stable")] = np.arange(len(q))
    rk = (rk - rk.mean()) / rk.std()
    return np.array([np.mean(rk[:-k] * rk[k:]) for k in range(1, L + 1)])


sets = {"all": np.diff(n).astype(float),
        "1e17-1e18": np.diff(n[n >= 10**17]).astype(float),
        "below 1e17": np.diff(n[n < 10**17]).astype(float)}
obs = {k: lagcorr(v) for k, v in sets.items()}
null = {k: [] for k in sets}
t = time.time()
for r in range(R):
    e = rng.exponential(size=len(sets["all"]))
    for k, v in sets.items():
        null[k].append(lagcorr(e[: len(v)]))
    if r == 0: print("rep s", round(time.time() - t, 2), flush=True)
res = {}
for k in sets:
    a = np.array(null[k]); mu = a.mean(0); sd = a.std(0)
    S0 = (a - mu).sum(1); Q0 = (((a - mu) / sd) ** 2).sum(1)
    So = (obs[k] - mu).sum(); Qo = (((obs[k] - mu) / sd) ** 2).sum()
    pS = (1 + np.sum(np.abs(S0) >= abs(So))) / (R + 1); pQ = (1 + np.sum(Q0 >= Qo)) / (R + 1)
    z = (obs[k] - mu) / sd
    res[k] = dict(N=len(sets[k]), excess=(obs[k] - mu).tolist(), z=z.tolist(), S=float(So), pS=float(pS), Q=float(Qo), pQ=float(pQ))
    print(f"{k:10s} N={len(sets[k])} z by lag: " + " ".join(f"{x:+.2f}" for x in z))
    print(f"{'':10s} excess lag1..3: " + " ".join(f"{x:+.5f}" for x in (obs[k] - mu)[:3]) + f"   S={So:+.5f} p={pS:.3f}   Q={Qo:.2f} p={pQ:.3f}")
json.dump(res, open(os.path.join(RESULTS, "03_serial_dependence.json"), "w"), indent=1)
