"""Shared helpers: data location, census loader, estimators and result writer."""
import gzip
import json
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")

CENSUS_FILES = ["carmichael-16.gz", "carmichael17.gz", "carmichael18.gz"]

# C(10^k) for k = 3..18, as published by Pinch (cartable.html)
PUBLISHED_COUNTS = {3: 1, 4: 7, 5: 16, 6: 43, 7: 105, 8: 255, 9: 646, 10: 1547, 11: 3605,
                    12: 8241, 13: 19279, 14: 44706, 15: 105212, 16: 246683, 17: 585355,
                    18: 1401644}


def read_census(with_factors=False):
    """Return the Carmichael numbers below 1e18 (int64, increasing) and, optionally, their factors."""
    ns, fs = [], []
    for name in CENSUS_FILES:
        with gzip.open(os.path.join(DATA, name), "rt") as fh:
            for line in fh:
                parts = line.split()
                if not parts:
                    continue
                ns.append(int(parts[0]))
                if with_factors:
                    fs.append(tuple(int(x) for x in parts[1:]))
    n = np.array(ns, dtype=np.int64)
    return (n, fs) if with_factors else n


def census_cache():
    """Numbers and number of prime factors, cached in results/census.npz by script 01."""
    path = os.path.join(RESULTS, "census.npz")
    if not os.path.exists(path):
        raise SystemExit("run code/01_census_validation.py first")
    d = np.load(path)
    return d["n"], d["r"]


def rollmean(x, W):
    """Centered moving mean of W consecutive terms, truncated at both ends."""
    c = np.concatenate(([0.0], np.cumsum(x)))
    m = len(x)
    half = W // 2
    lo = np.clip(np.arange(m) - half, 0, m)
    hi = np.clip(np.arange(m) - half + W, 0, m)
    return (c[hi] - c[lo]) / (hi - lo)


def sup_deviation(values, cdf):
    """Kolmogorov-Smirnov distance between the empirical distribution of values and cdf."""
    v = np.sort(values)
    m = len(v)
    F = cdf(v)
    k = np.arange(1, m + 1)
    return float(max(np.max(k / m - F), np.max(F - (k - 1) / m)))


def exp_cdf(v):
    return 1.0 - np.exp(-v)


def generalized_benford(alpha):
    """Leading-digit law P_alpha(d) = ((d+1)^alpha - d^alpha) / (10^alpha - 1), d = 1..9."""
    d = np.arange(1, 10.0)
    return ((d + 1) ** alpha - d ** alpha) / (10 ** alpha - 1)


def leading_digit(n):
    """Leading decimal digit of each positive integer in an int64 array."""
    e = np.floor(np.log10(n.astype(np.float64))).astype(np.int64)
    d = n // (10 ** e)
    # correct floating-point misplacement of exact powers of ten
    d = np.where(d >= 10, d // 10, d)
    d = np.where(d == 0, (n // 10 ** (e - 1)) // 10, d)
    return d.astype(np.int64)


def save(name, obj):
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, name), "w") as fh:
        json.dump(obj, fh, indent=1, default=float)
