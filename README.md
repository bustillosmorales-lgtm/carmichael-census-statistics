# Statistical structure of Carmichael numbers from a complete census

Code and results for the article *Statistical Structure of Carmichael Numbers from a Complete
Census: Spacings, Leading Digits, and Factor-Tuple Analysis*, by Francisco Bustillos and Víctor Leiva.

Every table and figure of the article is produced by one script in `code/`. The captured output of
each run is in `results/`, and `verify_all.py` compares every number quoted in the article with
those outputs.

## Data

The analysis uses the complete lists of Carmichael numbers below 10^18 published by R. G. E. Pinch at
<http://www.s369624816.websitehome.co.uk/rgep/cartable.html>. The files are not redistributed here.
`code/00_download_data.py` downloads them into `data/` and checks their SHA-256 checksums:

| file | content | lines | SHA-256 |
|---|---|---|---|
| `carmichael-16.gz` | n < 10^16, with factorization | 246,683 | `f6aff981…182b4` |
| `carmichael17.gz` | 10^16 ≤ n < 10^17 | 338,672 | `1e3fcc81…85ea55b` |
| `carmichael18.gz` | 10^17 ≤ n < 10^18 | 816,289 | `21a9e3a7…fbde1c` |
| `car3-18.gz` | three prime factors, n < 10^18 | 35,586 | `c7226104…bb375c` |

The full checksums are in `code/00_download_data.py`. The files were retrieved on 20 August 2026 and
again on 13 September 2026 with identical checksums.

## Requirements

Python 3.12 with the versions in `requirements.txt` (NumPy 2.4.6, SciPy 1.17.1, SymPy 1.14.0,
Matplotlib 3.10.9). Every computation runs on a single core.

## How to run

From the repository root:

```
python code/00_download_data.py
python code/01_census_validation.py      # writes results/census.npz, used by the other scripts
python code/02_spacing_tests.py          # about one hour
python code/03_serial_dependence.py      # about 25 minutes
python code/05_interval_counts.py
python code/06_leading_digits.py
python code/07_tuples_and_sieve.py       # includes the sieve run to 10^24
python code/08_counting_obstruction.py
python code/09_withdrawn_measurements.py
python code/10_figure.py                 # needs the output of 06
python verify_all.py
```

## Map from scripts to the article

| script | article | what it produces |
|---|---|---|
| `01_census_validation.py` | Section 2.1 | line counts, factorization and Korselt checks, the 16 values of C(10^k) |
| `02_spacing_tests.py` | Section 3, Tables 1, 2, 4 and 5 | normalized spacings by window and decade with calibrated quantiles, the Monte Carlo tests, the power study, the sensitivity to the resolution of the null intensity, and the test with the intensity taken from the counting function |
| `03_serial_dependence.py` | Section 3.3 | joint serial-dependence tests over lags 1 to 10 |
| `05_interval_counts.py` | Section 3.2, Table 3 | variance-to-mean ratios of interval counts against the inhomogeneous benchmark |
| `06_leading_digits.py` | Section 4, Tables 6 and 7 | observed and predicted digit laws, chi-square, exponent by number of prime factors with bootstrap intervals, aggregate and restricted fits |
| `07_tuples_and_sieve.py` | Section 5, Tables 8 and 9 | tuple decomposition, sieve validation on 105 tuples, the run to 10^24, Chernick closed form, admissible classes |
| `08_counting_obstruction.py` | Section 6, Tables 10 and 11 | chronology of the exponent, subset-product equidistribution, index-two subgroups, Davenport bounds, function-field check |
| `09_withdrawn_measurements.py` | Section 7 | items (ii) and (iii) |
| `10_figure.py` | Figure 1 | survival function with the null band, exponent by number of prime factors |

## Random seeds

All simulations use NumPy `default_rng` with fixed seeds written at the top of each script:
20260913 (`02`), 20260917 (`02`, sensitivity and counting-function blocks), 20260915 (`03`),
seeds 0 to 4 (`05`, following the original analysis), 20260916 (`06`), 20260918 (`10`).

## License

MIT for the code. The census data belong to their author and are downloaded from the original source.
