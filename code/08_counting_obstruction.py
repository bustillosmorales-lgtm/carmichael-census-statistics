"""Section 6. The chronology of the shifted-prime exponent with the Carmichael exponent it yields,
the subset-product equidistribution table, the index-two subgroups, the Davenport guarantee
against the supply of primes, the observed exponent at 1e24, and the function-field check."""
import math
from itertools import product
from math import gcd, lcm

import numpy as np
from sympy import divisors, factorint, isprime, totient

from common import save

# theta values as given by Lichtman (2022), Table in Sect. 1; the Carmichael exponent
# 0.4736 (1 - theta) is a lower bound and is therefore truncated, not rounded, to four decimals
CHRONOLOGY = [(1979, "Wooldridge", 2 * (math.sqrt(2) - 1)), (1980, "Pomerance", 625 / (512 * math.e)),
              (1983, "Balog", 0.35), (1986, "Fouvry-Grupp", 3 / (7 * math.exp(0.3))),
              (1989, "Friedlander", 1 / (2 * math.sqrt(math.e))), (1998, "Baker-Harman", 0.2961),
              (2022, "Lichtman", 15 / (32 * math.sqrt(math.e)))]
trunc4 = lambda x: math.floor(x * 1e4 + 1e-9) / 1e4   # the offset absorbs binary round-off of exact decimals
chron = []
for year, who, th in CHRONOLOGY:
    e = 0.4736 * (1 - th)
    chron.append(dict(year=year, author=who, theta=th, theta_4=trunc4(th), exponent=e, exponent_4=trunc4(e)))
    print(f"{year} {who:13s} theta={th:.6f} -> {trunc4(th):.4f}   exponent={e:.6f} -> {trunc4(e):.4f}")
observed_1e24 = math.log10(308279939) / 24
print(f"observed exponent at 1e24: {observed_1e24:.4f}")


def carmichael_lambda(L):
    out = 1
    for p, e in factorint(L).items():
        out = lcm(out, 2 ** (e - 2) if (p == 2 and e >= 3) else (p - 1) * p ** (e - 1))
    return out


equid = {}
for L in (5040, 55440, 720720, 4324320):
    primes = [d + 1 for d in divisors(L) if isprime(d + 1) and L % (d + 1)]
    phi = int(totient(L))
    lam = carmichael_lambda(L)
    units = np.array([x for x in range(L) if gcd(x, L) == 1], dtype=np.int64)
    idx = np.full(L, -1, dtype=np.int64)
    idx[units] = np.arange(phi)
    v = np.zeros(phi)
    v[idx[1]] = 1.0
    for p in primes:                      # distribution of the product of a uniform random subset
        w = np.zeros(phi)
        w[idx[(units * (p % L)) % L]] = v
        v = 0.5 * (v + w)
    dev = float(np.max(np.abs(v * phi - 1)))
    rank2 = sum((0 if e == 1 else 1 if e == 2 else 2) if p == 2 else 1 for p, e in factorint(L).items())
    davenport_sq = lam ** 2
    davenport_classical = lam * (1 + math.log(phi / lam))
    equid[L] = dict(r=len(primes), phi=phi, lam=lam, max_rel_dev=dev, index_two_subgroups=2 ** rank2 - 1,
                    lam_sq=davenport_sq, classical=davenport_classical, shortfall_sq=davenport_sq / len(primes))
    print(f"L={L:8d} r={len(primes):4d} phi={phi:7d} lambda={lam:4d} dev={dev:.1e} index-2={2 ** rank2 - 1:4d} "
          f"lambda^2={davenport_sq} classical={davenport_classical:.0f} ratio={davenport_sq / len(primes):.0f}")

# function fields over F_3: for squarefree composite monic n of degree <= 5, absolute Fermat behaviour
# (b^(q^deg n) = b mod n for all b) holds exactly when every irreducible factor degree divides deg n
q = 3


def pmul(a, b):
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] = (out[i + j] + x * y) % q
    return out


def pmod(a, m):
    a = a[:]
    while len(a) >= len(m):
        c = a[-1] * pow(m[-1], -1, q) % q
        s = len(a) - len(m)
        for i, y in enumerate(m):
            a[s + i] = (a[s + i] - c * y) % q
        while a and a[-1] == 0:
            a.pop()
    return a


def ppow(b, e, m):
    r, b = [1], pmod(b, m)
    while e:
        if e & 1:
            r = pmod(pmul(r, b), m) or [0]
        b = pmod(pmul(b, b), m) or [0]
        e >>= 1
    return r


def monic(deg):
    for coeffs in product(range(q), repeat=deg):
        yield list(coeffs) + [1]


irreducible = {}
for d in range(1, 6):
    irreducible[d] = []
    for m in monic(d):
        if all(pmod(m, f) != [] for dd in range(1, d // 2 + 1) for f in irreducible[dd]):
            irreducible[d].append(m)


def factor_degrees(m):
    degs = []
    for d in range(1, 6):
        for f in irreducible[d]:
            while len(m) - 1 >= d and pmod(m, f) == []:
                degs.append(d)
                # divide m by f
                quo, rem = [0] * (len(m) - len(f) + 1), m[:]
                for i in range(len(quo) - 1, -1, -1):
                    c = rem[i + len(f) - 1] * pow(f[-1], -1, q) % q
                    quo[i] = c
                    for j, y in enumerate(f):
                        rem[i + j] = (rem[i + j] - c * y) % q
                m = quo
    return degs


cases = agree = 0
for deg in range(2, 6):
    for m in monic(deg):
        degs = factor_degrees(m)
        if len(degs) < 2 or sum(degs) != deg:
            continue
        # squarefree: no repeated irreducible factor (check via gcd with derivative is avoided: test repeats)
        fs = []
        mm = m[:]
        square = False
        for d in sorted(set(degs)):
            for f in irreducible[d]:
                k = 0
                while pmod(mm, f) == []:
                    k += 1
                    quo, rem = [0] * (len(mm) - len(f) + 1), mm[:]
                    for i in range(len(quo) - 1, -1, -1):
                        c = rem[i + len(f) - 1] * pow(f[-1], -1, q) % q
                        quo[i] = c
                        for j, y in enumerate(f):
                            rem[i + j] = (rem[i + j] - c * y) % q
                    mm = quo
                square |= k > 1
        if square:
            continue
        cases += 1
        # every residue b of degree < deg, the zero polynomial included, must satisfy b^(q^deg) = b mod n
        residues = [list(c) for c in product(range(q), repeat=deg)]
        norm = lambda p: [x for x in p] if any(p) else [0]
        trim = lambda p: (lambda s: s if s else [0])(p[:max((i + 1 for i, x in enumerate(p) if x), default=0)])
        fermat = all(trim(ppow(trim(b), q ** deg, m)) == trim(b) for b in residues if any(b))
        criterion = all(deg % d == 0 for d in degs)
        agree += fermat == criterion
print(f"function fields over F_3, squarefree composites of degree 2..5: {cases}, criterion agrees on {agree}")

save("08_counting_obstruction.json", dict(chronology=chron, observed_exponent_1e24=observed_1e24,
                                          equid={int(k): v for k, v in equid.items()},
                                          function_field_cases=cases, function_field_agree=agree))
