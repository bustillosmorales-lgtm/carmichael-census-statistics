"""Section 7. Figures behind the withdrawn measurements.
Item (ii): counts in intervals of fixed length in n over [0, 1e18), data against a homogeneous and
an inhomogeneous Poisson process (five draws each).
Item (iii): the space of three-entry tuples with LCM at most 120, split into tuples for which the
Korselt congruence has no solution in g and admissible tuples, and how many of the latter occur."""
import pickle
from math import gcd, lcm

import numpy as np

from common import census_cache, read_census, save

n, _ = census_cache()
x = n.astype(np.float64)
N = len(x)
TOP = 1e18
idx = np.arange(N, dtype=np.float64)


def vmr(pts, width):
    nb = int(TOP / width)
    c, _ = np.histogram(pts, bins=np.linspace(0.0, TOP, nb + 1))
    return float(c.var() / c.mean())


def inhomogeneous(seed):
    t = np.cumsum(np.random.default_rng(seed).exponential(1.0, N))
    return np.interp(t[t < N - 1], idx, x)


def homogeneous(seed):
    return np.sort(np.random.default_rng(1000 + seed).uniform(0.0, TOP, N))


item2 = {}
for w in (5e10, 1.1e11, 1e12):
    inh = [vmr(inhomogeneous(s), w) for s in range(5)]
    hom = [vmr(homogeneous(s), w) for s in range(5)]
    item2[f"{w:.1e}"] = dict(data=vmr(x, w), inhomogeneous_mean=float(np.mean(inh)), inhomogeneous_min=min(inh),
                             inhomogeneous_max=max(inh), homogeneous_mean=float(np.mean(hom)))
    print(f"length {w:.1e}: data {item2[f'{w:.1e}']['data']:.2f}, inhomogeneous {np.mean(inh):.2f} "
          f"({min(inh):.2f} to {max(inh):.2f}), homogeneous {np.mean(hom):.2f}")

_, factors = read_census(with_factors=True)
occurring = set()
for f in factors:
    if len(f) == 3:
        g = gcd(gcd(f[0] - 1, f[1] - 1), f[2] - 1)
        occurring.add(tuple(sorted((q - 1) // g for q in f)))


def esym(a):
    e = [1] + [0] * len(a)
    for v in a:
        for j in range(len(a), 0, -1):
            e[j] += e[j - 1] * v
    return e[1:]


def classes(a):
    e = esym(a)
    return sum(all(sum(pow(g0, j, ai) * ej for j, ej in enumerate(e)) % ai == 0 for ai in a)
               for g0 in range(lcm(*a)))


CAP = 120
total = empty = admissible = occur = numax = 0
for a1 in range(1, CAP + 1):
    for a2 in range(a1 + 1, CAP + 1):
        for a3 in range(a2 + 1, CAP + 1):
            if gcd(gcd(a1, a2), a3) != 1 or lcm(a1, a2, a3) > CAP:
                continue
            v = classes((a1, a2, a3))
            numax = max(numax, v)
            total += 1
            empty += v == 0
            if v:
                admissible += 1
                occur += (a1, a2, a3) in occurring
print(f"three-entry tuples with LCM <= {CAP}: {total}; no solution in g: {empty}; admissible: {admissible}, "
      f"of which occur in the census: {occur}; largest number of classes: {numax}; empty fraction {100 * empty / total:.1f}%")
save("09_withdrawn_measurements.json", dict(item2=item2, item3=dict(total=total, empty=empty, admissible=admissible,
                                                                   occurring=occur, max_classes=numax,
                                                                   empty_fraction=empty / total)))
