"""External validation on the Carmichael numbers from 10^18 to 10^24, following
prereg/analysis_plan_1e24.md (pushed on 8 October 2026, before the data were analyzed).

Implementation notes (memory: 306.9 million gaps, so every pass works in blocks):
  * gaps are stored in single precision (relative error 6e-8); every float operation is in
    double precision inside a block;
  * the supremum distances are evaluated after the probability-integral transform of the
    reference law (1 - exp(-u) for the normalized gaps, 2q/(1+q) for the ratios), from histograms
    with 2^22 bins on [0, 1]; the discretization error is below 1e-6, against distances of order
    5e-5 (per decade: 2^20 bins);
  * the Spearman autocorrelations use as ranks the empirical distribution of the ratios evaluated
    from the same histogram, with linear interpolation inside a bin (ties within a bin of width
    2.4e-7 are broken by interpolation);
  * the gaps of decade k are those whose left point is the j-th number with
    C(10^k) <= j < C(10^(k+1)), in every replicate (rank-based assignment).
These are declared deviations of implementation, not of procedure.

Modes:
  prepare   observed statistics, Lambda_S knots (S = 20,000, one knot per 100 gaps)
  ref R     R realizations of the reference model (inversion of Lambda_S, conditional on N)
  unit R    R realizations of a unit-rate Poisson process of the same size (ratio statistics)
  report    p-values, Holm adjustment, lag-1 and lag-2 excess, decades
Usage: python code/19_census_1e24_analysis.py MODE [R] [workers] DATA_DIR
"""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS

MODE = sys.argv[1]
DATA = sys.argv[-1]
# Optional argument r=K (secondary analysis of the plan): restrict the analysis to the numbers with K
# prime factors in the range; every procedure is unchanged and the outputs carry the suffix _rK.
STRAT = next((int(x[2:]) for x in sys.argv[1:-1] if x.startswith("r=")), None)
SUFFIX = "" if STRAT is None else f"_r{STRAT}"
SEED = 20261013
SPLIT = 10 ** 12
W, S, LAGS = 1000, 20000, 10
BLK = 1 << 24
NB_FULL, NB_DEC = 1 << 22, 1 << 20
OUT = os.path.join(DATA, "analysis" if STRAT is None else f"analysis_r{STRAT}")
os.makedirs(OUT, exist_ok=True)


# ----------------------------------------------------------------------------- data
def load_stratum(hi, lo, a):
    """Gaps between consecutive numbers with STRAT prime factors, n >= 10^18, built block by block."""
    rr = np.load(os.path.join(DATA, "r.npy"), mmap_mode="r")
    Gs, prev, first = [], None, None
    thr = [10 ** (k - 12) for k in range(18, 25)]
    dec = [0] * len(thr)
    for s in range(a, len(hi), BLK):
        e = min(s + BLK, len(hi))
        m = np.asarray(rr[s:e]) == STRAT
        h = np.asarray(hi[s:e])[m].astype(np.int64)
        l = np.asarray(lo[s:e])[m].astype(np.int64)
        if len(h) == 0:
            continue
        for j, t in enumerate(thr):
            dec[j] += int(np.sum(h < t))
        if first is None:
            first = float(h[0]) * SPLIT + float(l[0])
        if prev is not None:
            h = np.concatenate(([prev[0]], h)); l = np.concatenate(([prev[1]], l))
        # in double precision: a gap of a sparse class can exceed the int64 range once scaled by 10^12
        Gs.append((np.diff(h).astype(np.float64) * SPLIT + np.diff(l)).astype(np.float32))
        prev = (h[-1], l[-1])
    return np.concatenate(Gs), first, dec


def load_observed():
    hi = np.load(os.path.join(DATA, "hi.npy"), mmap_mode="r")
    lo = np.load(os.path.join(DATA, "lo.npy"), mmap_mode="r")
    a = int(np.searchsorted(hi, np.uint64(10 ** 6)))          # first n >= 10^18
    if STRAT is not None:
        return load_stratum(hi, lo, a)
    N = len(hi) - a
    G = np.empty(N - 1, np.float32)
    for s in range(0, N - 1, BLK):
        e = min(s + BLK, N - 1)
        h = np.asarray(hi[a + s: a + e + 1], dtype=np.int64)
        l = np.asarray(lo[a + s: a + e + 1], dtype=np.int64)
        G[s:e] = (np.diff(h) * SPLIT + np.diff(l)).astype(np.float64)
    first = float(hi[a]) * SPLIT + float(lo[a])
    dec = [int(np.searchsorted(hi, np.uint64(10 ** (k - 12)))) for k in range(18, 25)]
    return G, first, dec


def decade_bounds():
    with open(os.path.join(OUT, "meta.json")) as fh:
        return json.load(fh)["decade_bounds"]


# ----------------------------------------------------------------------------- statistics
def local_mean(G, a, b):
    """Centered moving mean of W gaps for indices a..b-1, truncated at the global ends."""
    M = len(G)
    h = W // 2
    s0, s1 = max(0, a - h), min(M, b - h + W)
    c = np.concatenate(([0.0], np.cumsum(G[s0:s1], dtype=np.float64)))
    idx = np.arange(a, b)
    lo = np.clip(idx - h, 0, M) - s0
    hi = np.clip(idx - h + W, 0, M) - s0
    return (c[hi] - c[lo]) / (hi - lo)


def hist_bin(t, nb):
    return np.minimum((t * nb).astype(np.int64), nb - 1)


def sup_from_hist(h):
    """Supremum distance between the empirical law of the transformed values and U(0,1)."""
    N = h.sum()
    F = np.cumsum(h) / N
    edges = np.arange(1, len(h) + 1) / len(h)
    Fprev = np.concatenate(([0.0], F[:-1]))
    return float(max(np.max(np.abs(F - edges)), np.max(np.abs(Fprev - (edges - 1.0 / len(h))))))


def statistics(G, dec, with_u=True):
    """D_1000, D_ratio, mean ratio, rho_1..rho_10 (Spearman, ratios), overall and per decade."""
    M = len(G)
    nd = len(dec) - 1
    hu = np.zeros(NB_FULL, np.int64); hq = np.zeros(NB_FULL, np.int64)
    hud = np.zeros((nd, NB_DEC), np.int64); hqd = np.zeros((nd, NB_DEC), np.int64)
    sq = 0.0; sqd = np.zeros(nd); nqd = np.zeros(nd)

    def segs(a, b):
        """(decade, start, stop) for the contiguous pieces of the index range [a, b)."""
        for j in range(nd):
            s_, e_ = max(a, dec[j]), min(b, dec[j + 1])
            if j == 0:
                s_ = a if a < dec[1] else s_
            if j == nd - 1:
                e_ = b if b > dec[-2] else e_
            if e_ > s_:
                yield j, s_, e_

    # pass 1: histograms
    for a in range(0, M, BLK):
        b = min(a + BLK, M)
        if with_u:
            g = G[a:b].astype(np.float64)
            tu = -np.expm1(-g / local_mean(G, a, b))
            hu += np.bincount(hist_bin(tu, NB_FULL), minlength=NB_FULL)
            for j, s_, e_ in segs(a, b):
                hud[j] += np.bincount(hist_bin(tu[s_ - a:e_ - a], NB_DEC), minlength=NB_DEC)
        bq = min(b, M - 1)
        if bq > a:
            g2 = G[a:bq + 1].astype(np.float64)
            q = np.minimum(g2[:-1], g2[1:]) / np.maximum(g2[:-1], g2[1:])
            tq = 2 * q / (1 + q)
            hq += np.bincount(hist_bin(tq, NB_FULL), minlength=NB_FULL)
            sq += q.sum()
            for j, s_, e_ in segs(a, bq):
                hqd[j] += np.bincount(hist_bin(tq[s_ - a:e_ - a], NB_DEC), minlength=NB_DEC)
                sqd[j] += q[s_ - a:e_ - a].sum(); nqd[j] += e_ - s_
    Mq = M - 1
    # pass 2: rank autocorrelations from the empirical law of the ratios
    cdf = np.concatenate(([0.0], np.cumsum(hq) / Mq))

    def z_of(i0, i1):
        g2 = G[i0:i1 + 1].astype(np.float64)
        q = np.minimum(g2[:-1], g2[1:]) / np.maximum(g2[:-1], g2[1:])
        x = 2 * q / (1 + q) * NB_FULL
        j = np.minimum(x.astype(np.int64), NB_FULL - 1)
        return cdf[j] + (x - j) * (cdf[j + 1] - cdf[j])
    s1 = 0.0; s2 = 0.0
    prod = np.zeros(LAGS); prodd = np.zeros((nd, LAGS)); cntd = np.zeros((nd, LAGS))
    s1d = np.zeros(nd); s2d = np.zeros(nd)
    for a in range(0, Mq, BLK):
        b = min(a + BLK, Mq)
        v = z_of(a, min(b + LAGS, Mq))
        vb = v[: b - a]
        s1 += vb.sum(); s2 += (vb ** 2).sum()
        for j, s_, e_ in segs(a, b):
            w = vb[s_ - a:e_ - a]
            s1d[j] += w.sum(); s2d[j] += (w ** 2).sum()
        for h in range(1, LAGS + 1):
            n_ok = min(b, Mq - h) - a
            if n_ok <= 0:
                continue
            p = v[:n_ok] * v[h:h + n_ok]
            prod[h - 1] += p.sum()
            for j, s_, e_ in segs(a, a + n_ok):
                prodd[j, h - 1] += p[s_ - a:e_ - a].sum(); cntd[j, h - 1] += e_ - s_
    mu = s1 / Mq; var = s2 / Mq - mu ** 2
    rho = [(prod[h - 1] / (Mq - h) - mu ** 2) / var for h in range(1, LAGS + 1)]
    sup_u = (lambda h: sup_from_hist(h)) if with_u else (lambda h: float("nan"))
    res = dict(D_1000=sup_u(hu), D_ratio=sup_from_hist(hq), m_ratio=sq / Mq, rho=rho, decades=[])
    for j in range(nd):
        n_j = hqd[j].sum()
        mu_j = s1d[j] / n_j; var_j = s2d[j] / n_j - mu_j ** 2
        res["decades"].append(dict(D_1000=sup_u(hud[j]), D_ratio=sup_from_hist(hqd[j]), m_ratio=sqd[j] / nqd[j],
                                   rho=[(prodd[j, h] / cntd[j, h] - mu_j ** 2) / var_j for h in range(LAGS)]))
    return res


def flat(res):
    """Vector form: overall then each decade, each as [D_1000, D_ratio, m_ratio, rho_1..rho_10]."""
    row = lambda r: [r["D_1000"], r["D_ratio"], r["m_ratio"]] + list(r["rho"])   # noqa: E731
    return np.array(row(res) + sum((row(d) for d in res["decades"]), []))


# ----------------------------------------------------------------------------- generators
def ref_replicate(seed, M, first):
    """Gaps of the reference model: inversion of Lambda_S, conditional on N = M + 1 points."""
    kn = np.load(os.path.join(OUT, "lambda_knots.npz"))
    xk, Lk = kn["x"], kn["L"]
    n_exp = M + 2                                 # N + 1 exponentials give N interior points
    nblk = (n_exp + BLK - 1) // BLK
    tot = 0.0
    for bi in range(nblk):
        tot += np.random.default_rng([seed, bi]).exponential(size=min(BLK, n_exp - bi * BLK)).sum()
    G = np.empty(M, np.float32)
    run = 0.0; prev = None; filled = 0
    for bi in range(nblk):
        e = np.random.default_rng([seed, bi]).exponential(size=min(BLK, n_exp - bi * BLK))
        gam = run + np.cumsum(e); run = gam[-1]
        if bi == nblk - 1:
            gam = gam[:-1]                        # the last exponential closes the interval
        pts = np.interp(gam / tot * Lk[-1], Lk, xk)
        if prev is not None:
            pts = np.concatenate(([prev], pts))
        d = np.diff(pts)
        G[filled:filled + len(d)] = d; filled += len(d)
        prev = pts[-1]
    assert filled == M, (filled, M)
    return G


class LazyGaps:
    """The gaps of one realization, generated block by block on demand instead of stored.

    Block bi of random numbers always comes from default_rng([seed, bi]), so any range of gaps
    can be regenerated; the values are identical to those of ref_replicate and of the stored
    unit-rate gaps (same seeds, same operations, same single-precision rounding)."""

    def __init__(self, mode, seed, M):
        self.mode, self.seed, self.M = mode, seed, M
        self.cache = {}
        if mode == "ref":
            kn = np.load(os.path.join(OUT, "lambda_knots.npz"))
            self.xk, self.Lk = kn["x"], kn["L"]
            self.n_exp = M + 2
            self.nblk = (self.n_exp + BLK - 1) // BLK
            tot = 0.0; run = 0.0; self.offset = []
            for bi in range(self.nblk):
                e = self._exp(bi)
                tot += e.sum()
                self.offset.append(run)
                run = (run + np.cumsum(e))[-1]
            self.tot = tot

    def __len__(self):
        return self.M

    def _exp(self, bi):
        size = min(BLK, (self.n_exp if self.mode == "ref" else self.M) - bi * BLK)
        return np.random.default_rng([self.seed, bi]).exponential(size=size)

    def _block(self, bi):
        """Gaps whose index lies in [bi*BLK, (bi+1)*BLK) for unit; for ref, the points of block bi."""
        if bi in self.cache:
            return self.cache[bi]
        if len(self.cache) >= 3:
            self.cache.pop(next(iter(self.cache)))
        if self.mode == "unit":
            v = self._exp(bi).astype(np.float32)
        else:
            gam = self.offset[bi] + np.cumsum(self._exp(bi))
            if bi == self.nblk - 1:
                gam = gam[:-1]
            v = np.interp(gam / self.tot * self.Lk[-1], self.Lk, self.xk)
        self.cache[bi] = v
        return v

    def _points(self, a, b):
        """Points with exponential index a..b-1 (ref mode)."""
        out = []
        for bi in range(a // BLK, (b - 1) // BLK + 1):
            v = self._block(bi)
            out.append(v[max(a - bi * BLK, 0): min(b - bi * BLK, len(v))])
        return np.concatenate(out)

    def __getitem__(self, sl):
        a, b = sl.start, min(sl.stop, self.M)
        if self.mode == "unit":
            out = []
            for bi in range(a // BLK, (b - 1) // BLK + 1):
                v = self._block(bi)
                out.append(v[max(a - bi * BLK, 0): min(b - bi * BLK, len(v))])
            return np.concatenate(out)
        return np.diff(self._points(a, b + 1)).astype(np.float32)


def work(task):
    mode, seed = task
    with open(os.path.join(OUT, "meta.json")) as fh:
        meta = json.load(fh)
    M, dec = meta["M"], meta["decade_bounds"]
    return flat(statistics(LazyGaps(mode, seed, M), dec, with_u=(mode == "ref")))


def run_mc(mode, R, workers):
    path = os.path.join(OUT, f"{mode}_draws.npy")
    done = np.load(path) if os.path.exists(path) else np.empty((0, 13 * 7))
    seeds = [(mode, int(s)) for s in np.random.SeedSequence([SEED, 1 if mode == "ref" else 2]).generate_state(R)][len(done):]
    t0 = time.time()
    with Pool(workers) as pool:
        for i, v in enumerate(pool.imap(work, seeds)):
            done = np.vstack((done, v))
            np.save(path, done)
            print(f"  {mode} {len(done)}/{R} at {time.time()-t0:.0f}s", flush=True)


def prepare():
    t0 = time.time()
    G, first, dec_abs = load_observed()
    M = len(G)
    a0 = dec_abs[0]
    dec = [d - a0 for d in dec_abs]
    # Lambda_S: intensity 1/m_S at gap midpoints, trapezoid integral, one knot per 100 gaps
    xs, ms = [], []
    pos_run = 0.0
    for a in range(0, M, BLK):
        b = min(a + BLK, M)
        h = S // 2
        s0, s1 = max(0, a - h), min(M, b - h + S)
        c = np.concatenate(([0.0], np.cumsum(G[s0:s1], dtype=np.float64)))
        idx = np.arange(a, b, 100)
        lo = np.clip(idx - h, 0, M) - s0; hi = np.clip(idx - h + S, 0, M) - s0
        ms.append((c[hi] - c[lo]) / (hi - lo))
        cg = np.concatenate(([0.0], np.cumsum(G[a:b], dtype=np.float64)))
        xs.append(pos_run + cg[idx - a] + 0.5 * G[idx].astype(np.float64))
        pos_run += cg[-1]
    x = np.concatenate(xs); lam = 1.0 / np.concatenate(ms)
    x = np.concatenate(([0.0], x, [pos_run])); lam = np.concatenate(([lam[0]], lam, [lam[-1]]))
    L = np.concatenate(([0.0], np.cumsum(0.5 * (lam[1:] + lam[:-1]) * np.diff(x))))
    np.savez(os.path.join(OUT, "lambda_knots.npz"), x=x, L=L)
    with open(os.path.join(OUT, "meta.json"), "w") as fh:
        json.dump(dict(M=M, N=M + 1, first=first, decade_bounds=dec, span=pos_run, Lambda_total=float(L[-1])), fh)
    obs = statistics(G, dec)
    np.save(os.path.join(OUT, "observed.npy"), flat(obs))
    with open(os.path.join(RESULTS, f"19_census_1e24_observed{SUFFIX}.json"), "w") as fh:
        json.dump(dict(N=M + 1, decade_bounds=dec, observed=obs, elapsed_s=time.time() - t0), fh, indent=1)
    print(json.dumps(dict(N=M + 1, D_1000=obs["D_1000"], D_ratio=obs["D_ratio"], m_ratio=obs["m_ratio"],
                          rho_1_to_3=obs["rho"][:3]), indent=1), f"\n{time.time()-t0:.0f}s")


def report():
    obs = np.load(os.path.join(OUT, "observed.npy"))
    ref = np.load(os.path.join(OUT, "ref_draws.npy"))
    unit = np.load(os.path.join(OUT, "unit_draws.npy"))
    names = ["D_1000", "D_ratio", "m_ratio"] + [f"rho_{h}" for h in range(1, LAGS + 1)]
    labels = ["all"] + [f"1e{k}-1e{k+1}" for k in range(18, 24)]

    def block(v, j):
        return v[..., 13 * j: 13 * (j + 1)]

    def pval(t, d, two):
        if two:
            c = d.mean(); return (1 + np.sum(np.abs(d - c) >= abs(t - c))) / (len(d) + 1)
        return (1 + np.sum(d >= t)) / (len(d) + 1)

    out = {"R_ref": len(ref), "R_unit": len(unit), "by_range": {}}
    for j, lab in enumerate(labels):
        o, r, u = block(obs, j), block(ref, j), block(unit, j)
        mu, sd = u[:, 3:].mean(0), u[:, 3:].std(0)
        Q = lambda x: np.sum(((x - mu) / sd) ** 2, axis=-1)                 # noqa: E731
        stats = {
            "D_1000": (o[0], pval(o[0], r[:, 0], False), len(r)),
            "D_ratio": (o[1], pval(o[1], r[:, 1], False), len(r)),
            "m_ratio": (o[2], pval(o[2], r[:, 2], True), len(r)),
            "Q_10": (float(Q(o[3:])), pval(Q(o[3:]), Q(u[:, 3:]), False), len(u)),
            "rho_1_excess": (o[3] - mu[0], pval(o[3], u[:, 3], True), len(u)),
        }
        ps = np.array([v[1] for v in stats.values()])
        order = np.argsort(ps); holm = np.empty(5); run = 0.0
        for rank, i in enumerate(order):
            run = max(run, min(1.0, (5 - rank) * ps[i])); holm[i] = run
        out["by_range"][lab] = dict(
            stats={k: dict(value=float(v[0]), p=float(v[1]), R=v[2], p_holm=float(holm[i])) for i, (k, v) in enumerate(stats.items())},
            excess={f"lag{h}": dict(value=float(o[2 + h] - mu[h - 1]), null_sd=float(sd[h - 1]),
                                    mc_se_of_null_mean=float(sd[h - 1] / np.sqrt(len(u))), z=float((o[2 + h] - mu[h - 1]) / sd[h - 1]))
                    for h in range(1, LAGS + 1)},
            null_rho_mean=mu.tolist(), null_rho_sd=sd.tolist())
        print(lab, {k: (round(v['value'], 6), round(v['p'], 4), round(v['p_holm'], 4)) for k, v in out["by_range"][lab]["stats"].items()})
        print("   excess lag1 %.5f (z %.1f)  lag2 %.5f (z %.1f)" % (out["by_range"][lab]["excess"]["lag1"]["value"], out["by_range"][lab]["excess"]["lag1"]["z"],
                                                                  out["by_range"][lab]["excess"]["lag2"]["value"], out["by_range"][lab]["excess"]["lag2"]["z"]))
    with open(os.path.join(RESULTS, f"19_census_1e24_report{SUFFIX}.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    if MODE == "prepare":
        prepare()
    elif MODE in ("ref", "unit"):
        run_mc(MODE, int(sys.argv[2]), int(sys.argv[3]))
    elif MODE == "report":
        report()
