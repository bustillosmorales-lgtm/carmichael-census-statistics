"""Second revision, Section 3. Spacing analysis stratified by the number of prime factors r.

The census is a superposition of the subsequences with r = 3, 4, ..., 11 prime factors, whose
proportions change with scale. A superposition of many sparse non-Poisson streams can look Poisson
(Palm-Khintchine), so each stratum is analyzed as a point process of its own: its gaps, its own
estimated intensity Lambda_S with S = min(20,000, N_r / 2), its own reference model (inversion),
and the same statistics. Within each stratum the statistics are also computed on the gaps whose
left point lies in each of the decades 10^15-10^16, 10^16-10^17 and 10^17-10^18 (r x decade
interaction), against the same replicates. The local intensity of a stratum decreases
monotonically with x, so the decade split is also a split by intensity level.

Usage: python code/17_strata_factor_count.py [R] [workers]
"""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark as B
from common import RESULTS, census_cache

R = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 12
SEED = 20261014
MIN_N = 10000
DECADES = (15, 16, 17)
NAMES = ["D_1000", "D_ratio", "m_ratio"] + [f"rho_{h}" for h in range(1, B.LAGS + 1)]


def stats(g, left):
    """Statistics overall and per decade of the left point (left = positions of the left points)."""
    u = g / B.rollmean(g, 1000)
    q = B.ratios(g)

    def row(sel_u, sel_q):
        return [B.sup_dev(u[sel_u], B.exp_cdf), B.sup_dev(q[sel_q], B.ratio_cdf), float(q[sel_q].mean())] + list(B.rank_acf(q[sel_q]))

    out = row(slice(None), slice(None))
    m12 = left >= 1e12
    out += row(m12, m12[:-1])
    for k in DECADES:
        m = (left >= 10.0 ** k) & (left < 10.0 ** (k + 1))
        out += row(m, m[:-1]) if m.sum() > 2000 else [np.nan] * len(NAMES)
    return np.array(out)


def stratum(rr):
    n, r = census_cache()
    return n[r == rr].astype(np.float64)


def work(task):
    rr, seed, reps = task
    pos = stratum(rr)
    S = int(min(20000, len(pos) // 2))
    xk, Lk = B.cumulative_intensity(pos, S)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(reps):
        g = B.gen_inverse(xk, Lk, len(pos), rng)
        left = pos[0] + np.concatenate(([0.0], np.cumsum(g)[:-1]))
        out.append(stats(g, left))
    return rr, np.array(out)


def main():
    t0 = time.time()
    n, r = census_cache()
    strata = [int(k) for k in np.unique(r) if np.sum(r == k) >= MIN_N]
    obs = {}
    for rr in strata:
        pos = stratum(rr)
        obs[rr] = stats(np.diff(pos), pos[:-1])
    seq = np.random.SeedSequence(SEED)
    tasks = [(rr, c.generate_state(1)[0], 25) for rr in strata for c in seq.spawn(R // 25)]
    acc = {}
    with Pool(WORKERS) as pool:
        for rr, arr in pool.imap_unordered(work, tasks):
            acc.setdefault(rr, []).append(arr)
    res = {"R": R, "seed": SEED, "strata": {}}
    labels = ["all", "ge1e12"] + [f"1e{k}-1e{k+1}" for k in DECADES]
    for rr in strata:
        d = np.vstack(acc[rr]); o = obs[rr]
        posr = stratum(rr)
        sv = B.slow_variation(posr, int(min(20000, len(posr) // 2)), 1000)
        res["strata"][rr] = {"N": int(np.sum(r == rr)), "S": int(min(20000, np.sum(r == rr) // 2)),
                             "slow_variation_W1000": dict(zip(["50", "90", "99", "max"], np.quantile(sv, [.5, .9, .99, 1]).tolist())),
                             "share_gaps_slow_variation_above_0.05": float(np.mean(sv > 0.05))}
        print(f"r={rr}: slow variation W=1000 median {np.median(sv):.4f}, 99% {np.quantile(sv, .99):.4f}, share>0.05 {np.mean(sv > 0.05):.3f}", flush=True)
        for j, lab in enumerate(labels):
            sl = slice(len(NAMES) * j, len(NAMES) * (j + 1))
            oj, dj = o[sl], d[:, sl]
            if np.isnan(oj[0]):
                continue
            mu, sd = dj[:, 3:].mean(0), dj[:, 3:].std(0)
            Q = lambda x: np.sum(((x - mu) / sd) ** 2, axis=-1)          # noqa: E731
            row = {}
            for i, nm in enumerate(NAMES[:5]):
                row[nm] = dict(obs=float(oj[i]), null_mean=float(dj[:, i].mean()),
                               p=float(B.mc_pvalue(oj[i], dj[:, i], nm in B.TWO_SIDED or nm == "m_ratio")))
            row["Q_10"] = dict(obs=float(Q(oj[3:])), p=float(B.mc_pvalue(Q(oj[3:]), Q(dj[:, 3:]), False)))
            left = n[r == rr][:-1].astype(float)
            row["n_gaps"] = (int(np.sum(r == rr) - 1) if lab == "all" else int(np.sum(left >= 1e12)) if lab == "ge1e12"
                             else int(np.sum((left >= 10.0 ** int(lab[2:4])) & (left < 10.0 ** (int(lab[2:4]) + 1)))))
            res["strata"][rr][lab] = row
            print(f"r={rr:2d} {lab:10s} n={row['n_gaps']:7d} " + " ".join(f"{k}: p={v['p']:.3f}" for k, v in row.items() if isinstance(v, dict)), flush=True)
    res["elapsed_s"] = time.time() - t0
    with open(os.path.join(RESULTS, "17_strata_factor_count.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"done in {res['elapsed_s']:.0f}s")


if __name__ == "__main__":
    main()
