"""Section 3.2. Variance-to-mean ratio of interval counts in the scale defined by eq. (norm), for
four windows and five interval lengths, against an inhomogeneous Poisson process that carries the
empirical intensity (moving mean over 1,000 gaps), averaged over five draws with seeds 0..4 and
passed through the identical pipeline. The table calibrates the instrument; it is not a test."""
import numpy as np

from common import census_cache, rollmean, save

WINDOWS = (100, 1000, 5000, 20000)
LENGTHS = (1, 2, 5, 20, 100)


def vmr(gaps, W, lam):
    t = np.cumsum(gaps / rollmean(gaps, W))
    nb = int(t[-1] // lam)
    c, _ = np.histogram(t, bins=np.arange(nb + 1) * lam)
    return float(c.var() / c.mean())


n, _ = census_cache()
g = np.diff(n).astype(np.float64)
intensity = rollmean(g, 1000)
cells = {}
for lam in LENGTHS:
    for W in WINDOWS:
        data = vmr(g, W, lam)
        control = float(np.mean([vmr(np.random.default_rng(s).exponential(1.0, len(g)) * intensity, W, lam)
                                 for s in range(5)]))
        cells[f"{lam},{W}"] = dict(data=data, control=control, ratio=data / control)
    print(f"lambda={lam:3d} " + "  ".join(f"W={W}: {cells[f'{lam},{W}']['data']:.4f}/{cells[f'{lam},{W}']['control']:.4f}"
                                         for W in WINDOWS))
ratios = [c["ratio"] for c in cells.values()]
datas = [c["data"] for c in cells.values()]
print(f"data/control ratio in [{min(ratios):.3f}, {max(ratios):.3f}]; data in [{min(datas):.2f}, {max(datas):.2f}]")
save("05_interval_counts.json", dict(cells=cells, ratio_min=min(ratios), ratio_max=max(ratios),
                                     data_min=min(datas), data_max=max(datas)))
