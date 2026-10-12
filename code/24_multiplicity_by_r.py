"""Global multiplicity over the secondary analysis by number of prime factors, 10^18-10^24.
Reads results/19_r{r}_report.txt (r = 3..12) and results/23_slow_variation_1e24_by_r_decade.json.
Reports the count of p < 0.05 and p <= 0.01 against expectation and the smallest Benjamini-Hochberg
value, over all cells and over the cells with at least 100,000 numbers (the class threshold of the plan)."""
import ast
import json
import os
import re

RES = os.path.join(os.path.dirname(__file__), "..", "results")
LABS = ["1e18-1e19", "1e19-1e20", "1e20-1e21", "1e21-1e22", "1e22-1e23", "1e23-1e24"]
sv = json.load(open(os.path.join(RES, "23_slow_variation_1e24_by_r_decade.json")))


def summary(rows):
    n = len(rows)
    ps = sorted(rows, key=lambda x: x[3])
    bh = [min(1.0, x[3] * n / (i + 1)) for i, x in enumerate(ps)]
    for i in range(n - 2, -1, -1):
        bh[i] = min(bh[i], bh[i + 1])
    return dict(tests=n, below_005=sum(x[3] < 0.05 for x in rows), expected_005=0.05 * n,
                at_most_001=sum(x[3] <= 0.01 for x in rows), expected_001=0.01 * n,
                min_bh=bh[0], smallest=[list(x) for x in ps[:5]])


rows_all, rows_big = [], []
for r in range(3, 13):
    for line in open(os.path.join(RES, f"19_r{r}_report.txt")):
        m = re.match(r"^(all|1e\S+) (\{.*\})$", line.strip())
        if not m:
            continue
        lab = m.group(1)
        big = lab == "all" or sv[str(r)][LABS.index(lab)][0] + 1 >= 100000
        for k, (_, p, _) in ast.literal_eval(m.group(2)).items():
            rows_all.append((r, lab, k, p))
            if big:
                rows_big.append((r, lab, k, p))

out = dict(all_cells=summary(rows_all), cells_ge_100000=summary(rows_big))
json.dump(out, open(os.path.join(RES, "24_multiplicity_by_r.json"), "w"), indent=1)
for k, v in out.items():
    print(k, {a: b for a, b in v.items() if a != "smallest"})
