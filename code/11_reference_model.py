"""Second revision, Section 3. Null distributions of the full family of spacing statistics under
three generators of the reference model, all passed through the identical computation:

  inverse  Poisson process with the estimated cumulative intensity Lambda_S (S = 20,000),
           simulated by inversion, conditional on N (primary generator)
  nested   as inverse, but Lambda is re-estimated in every replicate from a pilot realization
           of the reference model, so that the uncertainty of the intensity estimate enters the
           null distribution (unconditional calibration)
  rank     rank-indexed exponential gaps m_S(i) E_i (generator of the first revision)

Usage: python code/11_reference_model.py [R_inverse] [R_nested] [R_rank] [workers]
Writes results/11_reference_model.npz (all draws) and results/11_reference_model.json.
"""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark as B
from common import RESULTS, census_cache

ARGS = [int(a) for a in sys.argv[1:5]] + [2000, 2000, 1000, 16][len(sys.argv[1:5]):]
R_INV, R_NEST, R_RANK, WORKERS = ARGS
SEED = 20261008
S = 20000


def _setup():
    n, _ = census_cache()
    pos = n.astype(np.float64)
    xk, Lk = B.cumulative_intensity(pos, S)
    m = B.rollmean(np.diff(pos), S)
    return pos, xk, Lk, m


def work(task):
    kind, seed, reps = task
    pos, xk, Lk, m = _setup()
    N = len(pos)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(reps):
        if kind == "inverse":
            g = B.gen_inverse(xk, Lk, N, rng)
        elif kind == "nested":
            pilot = pos[0] + np.concatenate(([0.0], np.cumsum(B.gen_inverse(xk, Lk, N, rng))))
            xs, Ls = B.cumulative_intensity(pilot, S)
            g = B.gen_inverse(xs, Ls, N, rng)
        else:
            g = B.gen_rank(m, rng)
        out.append(B.statistics(g, rng))
    return kind, np.array(out)


def main():
    t0 = time.time()
    pos, xk, Lk, m = _setup()
    rng = np.random.default_rng(SEED)
    obs = B.statistics(np.diff(pos), rng)
    lattice = B.statistics(np.clip(rng.normal(1.0, 0.15, size=len(m)), 1e-9, None) * m, rng)

    ss = np.random.SeedSequence(SEED)
    tasks = []
    for kind, R in (("inverse", R_INV), ("nested", R_NEST), ("rank", R_RANK)):
        chunk = 25
        for j, c in enumerate(ss.spawn((R + chunk - 1) // chunk)):
            tasks.append((kind, c.generate_state(1)[0], min(chunk, R - j * chunk)))
    draws = {"inverse": [], "nested": [], "rank": []}
    with Pool(WORKERS) as pool:
        for i, (kind, arr) in enumerate(pool.imap_unordered(work, tasks)):
            draws[kind].append(arr)
            if i % 20 == 0:
                print(f"  {i+1}/{len(tasks)} chunks at {time.time()-t0:.0f}s", flush=True)
    draws = {k: np.vstack(v) for k, v in draws.items()}
    np.savez_compressed(os.path.join(RESULTS, "11_reference_model.npz"), names=np.array(B.STAT_NAMES),
                        obs=obs, lattice=lattice, **draws)

    sv = B.slow_variation(pos, S, 1000)
    res = {"seed": SEED, "S": S, "R": {k: len(v) for k, v in draws.items()},
           "slow_variation_W1000_quantiles": dict(zip(["50", "90", "99", "99.9", "max"],
                                                      np.quantile(sv, [.5, .9, .99, .999, 1]).tolist())),
           "stats": {}}
    for j, name in enumerate(B.STAT_NAMES):
        two = name in B.TWO_SIDED
        row = {"obs": float(obs[j]), "lattice": float(lattice[j])}
        for k, d in draws.items():
            p = B.mc_pvalue(obs[j], d[:, j], two)
            row[k] = dict(mean=float(d[:, j].mean()), sd=float(d[:, j].std()),
                          q025=float(np.quantile(d[:, j], .025)), q975=float(np.quantile(d[:, j], .975)),
                          q95=float(np.quantile(d[:, j], .95)), p=float(p), p_se=B.mc_se(p, len(d)))
        res["stats"][name] = row
        print(f"{name:8s} obs={obs[j]:.6f} " + "  ".join(
            f"{k}: mean={row[k]['mean']:.6f} p={row[k]['p']:.4f}" for k in draws), flush=True)
    res["elapsed_s"] = time.time() - t0
    with open(os.path.join(RESULTS, "11_reference_model.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"done in {res['elapsed_s']:.0f}s")


if __name__ == "__main__":
    main()
