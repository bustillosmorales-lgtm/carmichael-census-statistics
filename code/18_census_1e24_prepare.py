"""External validation, step 1: read the tabulation of Shallue and Webster to 10^24, check its
integrity as stated in prereg/analysis_plan_1e24.md, and store compact arrays.

Input: the file new_table.txt (each line: n followed by its prime factors), either whole or as the
byte-range parts part0..part{K-1} written by a parallel download, in a directory given as the
first argument. Output in that directory:
  hi.npy, lo.npy   uint64, n = hi * 10^12 + lo (n < 10^24 does not fit in 64 bits)
  r.npy            uint8, number of prime factors
  d.npy            uint8, leading decimal digit
and results/18_census_1e24_integrity.json.

Usage: python code/18_census_1e24_prepare.py DATA_DIR [workers]
"""
import glob, hashlib, json, os, sys, time
from array import array
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS

DIR = sys.argv[1]
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
SPLIT = 10 ** 12
OEIS = {18: 1401644, 19: 3381806, 20: 8220777, 21: 20138200, 22: 49679870, 23: 123381982, 24: 308279939}


def parts():
    p = sorted(glob.glob(os.path.join(DIR, "part*")), key=lambda s: int(s.rsplit("part", 1)[1]))
    return p if p else [os.path.join(DIR, "new_table.txt")]


def work(i):
    ps = parts()
    hi = array("Q"); lo = array("Q"); r = array("B"); d = array("B")
    bad_prod = bad_rep = bad_kor = bad_order = 0
    prev = -1
    first = last = None

    def lines():
        with open(ps[i], "rb") as fh:
            if i > 0:                            # the first partial line belongs to part i-1
                fh.readline()
            tail = b""
            for ln in fh:
                if ln.endswith(b"\n"):
                    yield ln
                else:
                    tail = ln
        if tail:                                 # complete the last line from the next part
            with open(ps[i + 1], "rb") as fh:
                yield tail + fh.readline()

    for ln in lines():
        f = ln.split()
        if not f:
            continue
        n = int(f[0])
        ps_ = [int(x) for x in f[1:]]
        prod = 1
        for p in ps_:
            prod *= p
        if prod != n:
            bad_prod += 1
        if len(set(ps_)) != len(ps_):
            bad_rep += 1
        n1 = n - 1
        for p in ps_:
            if n1 % (p - 1):
                bad_kor += 1
                break
        if n <= prev:
            bad_order += 1
        prev = n
        if first is None:
            first = n
        q, rem = divmod(n, SPLIT)
        hi.append(q); lo.append(rem); r.append(len(ps_)); d.append(f[0][0] - 48)
    last = prev
    for name, arr, dt in (("hi", hi, np.uint64), ("lo", lo, np.uint64), ("r", r, np.uint8), ("d", d, np.uint8)):
        np.save(os.path.join(DIR, f"_{name}{i}.npy"), np.frombuffer(arr, dtype=dt))
    return i, dict(lines=len(hi), bad_product=bad_prod, bad_repeat=bad_rep, bad_korselt=bad_kor,
                   bad_order=bad_order, first=first, last=last)


def main():
    t0 = time.time()
    ps = parts()
    h = hashlib.sha256()
    for p in ps:
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                h.update(chunk)
    size = sum(os.path.getsize(p) for p in ps)
    print(f"sha256 {h.hexdigest()} size {size} ({time.time()-t0:.0f}s)", flush=True)
    with Pool(min(WORKERS, len(ps))) as pool:
        out = dict(pool.imap_unordered(work, range(len(ps))))
    info = [out[i] for i in range(len(ps))]
    order_across = all(info[i]["last"] < info[i + 1]["first"] for i in range(len(info) - 1))
    for name in ("hi", "lo", "r", "d"):
        np.save(os.path.join(DIR, f"{name}.npy"), np.concatenate([np.load(os.path.join(DIR, f"_{name}{i}.npy")) for i in range(len(ps))]))
        for i in range(len(ps)):
            os.remove(os.path.join(DIR, f"_{name}{i}.npy"))
    hi = np.load(os.path.join(DIR, "hi.npy"), mmap_mode="r")
    counts = {}
    for k in range(18, 25):
        # n < 10^k  <=>  hi < 10^(k-12)  (since lo < 10^12)
        counts[k] = int(np.searchsorted(hi, np.uint64(10 ** (k - 12))))
    res = dict(sha256=h.hexdigest(), size=size, lines=int(sum(x["lines"] for x in info)),
               bad_product=sum(x["bad_product"] for x in info), bad_repeat=sum(x["bad_repeat"] for x in info),
               bad_korselt=sum(x["bad_korselt"] for x in info),
               bad_order=sum(x["bad_order"] for x in info) + (0 if order_across else 1),
               counts=counts, counts_match_oeis={k: counts[k] == OEIS[k] for k in OEIS},
               elapsed_s=time.time() - t0)
    print(json.dumps(res, indent=1))
    with open(os.path.join(RESULTS, "18_census_1e24_integrity.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
