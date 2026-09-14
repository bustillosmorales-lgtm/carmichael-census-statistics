"""Section 5. Factor-tuple decomposition of every Carmichael number with at most six prime factors,
the sieve for a prescribed tuple validated against the census, the run to 1e24 for (1,2,3), the
closed form of Chernick for three factors, and the number of admissible classes modulo the LCM."""
import time
from collections import Counter
from math import gcd, lcm

from sympy import isprime

from common import read_census, save

_, factors = read_census(with_factors=True)
tuples = {k: Counter() for k in (3, 4, 5, 6)}
for f in factors:
    k = len(f)
    if k > 6:
        continue
    g = 0
    for q in f:
        g = gcd(g, q - 1)
    tuples[k][tuple(sorted((q - 1) // g for q in f))] += 1

table = {}
for k in (3, 4, 5, 6):
    top = tuples[k].most_common(6)
    table[k] = dict(numbers=sum(tuples[k].values()), distinct=len(tuples[k]),
                    top=[[list(a), c] for a, c in top])
    print(f"k={k}: numbers={table[k]['numbers']} distinct={table[k]['distinct']} top={top[:2]}")
total_le6 = sum(table[k]["numbers"] for k in table)
print("numbers with at most six prime factors:", total_le6)


def esym(a):
    e = [1] + [0] * len(a)
    for x in a:
        for j in range(len(a), 0, -1):
            e[j] += e[j - 1] * x
    return e[1:]


def admissible(a):
    """Residues g0 modulo lcm(a) with a_i | e_1 + g e_2 + ... + g^(k-1) e_k for every i."""
    e = esym(a)
    return [g0 for g0 in range(lcm(*a))
            if all(sum(pow(g0, j, ai) * ej for j, ej in enumerate(e)) % ai == 0 for ai in a)]


def sieve(a, bound):
    """All Carmichael numbers below bound whose tuple, in the sense of eq. (gatuple), is exactly a."""
    L = lcm(*a)
    classes = admissible(a)
    out = []
    base = 0
    while True:
        smallest = 1
        for ai in a:                               # product at g = base, a lower bound for this block
            smallest *= max(base, 1) * ai + 1
        if smallest >= bound:
            break
        for g0 in classes:
            g = base + g0
            if g < 1:
                continue
            ps = [g * ai + 1 for ai in a]
            prod = 1
            for p in ps:
                prod *= p
            if prod >= bound:
                continue
            if all(isprime(p) for p in ps) and gcd(*[p - 1 for p in ps]) == g:
                out.append(prod)
        base += L
    return out


present = {a: c for k in tuples for a, c in tuples[k].items()}
small = sorted((a for a in present if lcm(*a) <= 44), key=lambda a: (lcm(*a), a))
agree = [len(sieve(a, 10 ** 18)) == present[a] for a in small]
three = [a for a in small if len(a) == 3]
print(f"tuples in the census with LCM <= 44: {len(small)} ({len(three)} with three factors); sieve agrees on {sum(agree)}")
extract = {str(a): [present[a], len(sieve(a, 10 ** 18))] for a in
           [(1, 2, 3), (1, 2, 5), (1, 3, 4), (1, 3, 5), (1, 3, 8), (1, 2, 15), (2, 3, 5), (1, 6, 7)]}

t = time.time()
s18 = sieve((1, 2, 3), 10 ** 18)
t18 = time.time() - t
t = time.time()
s24 = sieve((1, 2, 3), 10 ** 24)
t24 = time.time() - t
print(f"(1,2,3): {len(s18)} below 1e18 in {t18:.2f}s, {len(s24)} below 1e24 in {t24:.1f}s")

# closed form of Chernick (Kellner 2022, Sect. 4): delta = -e1 e2^{-1} mod e3 for pairwise coprime tuples
cop = [(a, b, c) for a in range(1, 40) for b in range(a + 1, 40) for c in range(b + 1, 40)
       if gcd(a, b) == gcd(a, c) == gcd(b, c) == 1]
closed_ok = 0
for a in cop:
    e1, e2, e3 = esym(a)
    d = (-e1 * pow(e2, -1, e3)) % e3 if gcd(e2, e3) == 1 else None
    closed_ok += d is not None and admissible(a) == [d]
print(f"closed form equals the enumerated residue on {closed_ok} of {len(cop)} pairwise coprime tuples")

classes = {}
for k in (3, 4, 5, 6):
    c = Counter(len(admissible(a)) for a in tuples[k] if lcm(*a) <= 3000)
    classes[k] = {int(v): int(m) for v, m in sorted(c.items())}
    print(f"k={k}: census tuples with LCM <= 3000 = {sum(c.values())}, admissible-class counts {classes[k]}")

save("07_tuples_and_sieve.json", dict(
    table={int(k): v for k, v in table.items()}, total_at_most_six=total_le6,
    lcm44_tuples=len(small), lcm44_three_factor=len(three), lcm44_agree=int(sum(agree)), extract=extract,
    sieve_123_1e18=len(s18), seconds_1e18=t18, sieve_123_1e24=len(s24), seconds_1e24=t24,
    chernick_closed_form_ok=int(closed_ok), chernick_tuples=len(cop), admissible_classes_lcm3000=classes))
