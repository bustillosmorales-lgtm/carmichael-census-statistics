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

The external validation of the second revision uses the tabulation of Shallue and Webster to 10^24,
distributed by J. Webster as `new_table.txt` at <https://blue.butler.edu/~jewebste/new_table.txt>
(18,358,310,309 bytes, 308,279,939 lines, each a Carmichael number followed by its prime factors;
SHA-256 `c934af2a208832689586d12bbfe9e9cfc4ccb6447375baa39308d2e3373e6170`). The file is not
redistributed. Download it into a directory `DATA_DIR` outside the repository; the scripts 18, 19, 22
and 23 take that directory as their last argument.

## Analysis plan

`prereg/analysis_plan_1e24.md` is the analysis plan for the range 10^18 to 10^24. It was committed
(commit `edcb136`, 8 October 2026) before any statistic of that range was computed.

## Requirements

Python 3.12 with the versions in `requirements.txt` (NumPy 2.4.6, SciPy 1.17.1, SymPy 1.14.0,
Matplotlib 3.10.9). The scripts of the first revision run on a single core. The Monte Carlo scripts
of the second revision take the number of worker processes as an argument and run for several
hours each; the analysis of the range to 10^24 needs about 16 GB of memory.

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

Second revision, with the arguments used for the article (the last number before `DATA_DIR`, or the
last argument otherwise, is the number of worker processes):

```
python code/11_reference_model.py 2000 2000 1000 3
python code/11b_primary_family.py 2000 3 40
python code/05b_interval_counts_inversion.py 200 3 50 20
python code/12_validation.py 1000 3 1000
python code/13_resolution_rule.py 400 3
python code/14_power.py 1000 3
python code/14b_power_table.py
python code/15_dependence.py 1000 3
python code/16_digits_bootstrap.py
python code/17_strata_factor_count.py 1000 3
python code/20_secondary_inversion.py 2000 3
python code/25_sensitivity_inversion.py 2000 3
python code/21_figures_r2.py
python code/18_census_1e24_prepare.py DATA_DIR
python code/19_census_1e24_analysis.py prepare DATA_DIR
python code/19_census_1e24_analysis.py unit 1000 2 DATA_DIR
python code/19_census_1e24_analysis.py ref 200 2 DATA_DIR
python code/19_census_1e24_analysis.py report DATA_DIR   # the confirmatory analysis with R = 200, as in the plan
python code/19_census_1e24_analysis.py ref 1000 2 DATA_DIR   # extends the same seed sequence to R = 1,000
python code/19_census_1e24_analysis.py report DATA_DIR
#   the same four steps with r=K before DATA_DIR, for K = 3..12, give the analysis by number of prime factors
python code/22_digits_1e24.py 100000 DATA_DIR
python code/23_slow_variation_1e24.py DATA_DIR
python code/24_multiplicity_by_r.py
python verify_all.py
```

`verify_all.py` checks the numbers of the first revision and, through `verify_r2.py`, the numbers of
the second. The quoted table cells of the second revision are copied from the manuscript source by
`tools/extract_quoted_r2.py` into `verify_r2_quoted.json`, and each is compared with the output rounded
to the decimals printed.

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

The table above follows the numbering of the first revision. For the second revision:

| script | second revision | what it produces |
|---|---|---|
| `benchmark.py` | Sections 2.2 to 2.6 | generators of the reference model (inversion, nested, rank-indexed), the statistics, the slow-variation quantity |
| `11_reference_model.py` | Sections 2.2 and 3.4, Table 7 | the three generators compared on the census below 10^18 |
| `11b_primary_family.py` | Section 3.3, Table 6 | primary family, Holm and Westfall–Young adjustments, D_cf with 40 labelings |
| `05b_interval_counts_inversion.py` | Section 3.2, Table 5 | interval counts against the reference model by inversion, calibrated on known Poisson processes |
| `12_validation.py` | Sections 2.5 and 3.1, Table 2 | unit tests, known homogeneous and inhomogeneous processes, actual level of each test, D_cf with 40 labelings |
| `13_resolution_rule.py` | Section 3.1, Table 3 | actual level of D_W over the grid of resolutions S, and the window rule |
| `14_power.py`, `14b_power_table.py` | Sections 2.6 and 3.6, Figure 2, Table A1 | power over eight families with exact binomial intervals |
| `15_dependence.py` | Section 3.5, Figure 1 | profile of excess autocorrelations at lags 1 to 50 with simultaneous bands, Q_10, Q_50 |
| `16_digits_bootstrap.py` | Section 4.1, Table 12 | bootstrap calibration of the leading-digit test, piecewise and smooth predictions |
| `17_strata_factor_count.py` | Section 3.5, Table 9 | analysis by number of prime factors below 10^18 |
| `18_census_1e24_prepare.py` | Section 2.1 | integrity of the tabulation to 10^24 and compact arrays |
| `19_census_1e24_analysis.py` | Section 3.7, Tables 10 and 11 | confirmatory analysis of 10^18 to 10^24, by decade and by number of prime factors |
| `20_secondary_inversion.py` | Sections 3.2, 3.3 and 3.5, Tables 4 and 8 | multiscale analysis, decades, near-lattice control |
| `21_figures_r2.py` | Figures 1 and 2 | dependence profile and power curves |
| `22_digits_1e24.py` | Section 4.1, Table 13 | leading digits from 10^18 to 10^24 |
| `23_slow_variation_1e24.py` | Section 3.7, Table 11 | slow-variation condition by class and decade from 10^18 to 10^24 |
| `24_multiplicity_by_r.py` | Section 3.7 | multiplicity over the classes and their decades from 10^18 to 10^24 |
| `25_sensitivity_inversion.py` | Section 3.4 | sensitivity to the resolution S and normalization by the counting function, with the generator by inversion |

## Random seeds

All simulations use NumPy `default_rng` with fixed seeds written at the top of each script:
20260913 (`02`), 20260917 (`02`, sensitivity and counting-function blocks), 20260915 (`03`),
seeds 0 to 4 (`05`, following the original analysis), 20260916 (`06`), 20260918 (`10`).
Second revision: 20261008 (`11`), 20261016 (`11b`), 20261010 and 20261014 for the 20,000-replicate
check (`12`), 20261011 (`13`, `20`, `22`), 20261012 (`14`), 20261015 (`15`), 20261009 (`16`),
20261014 (`17`), 20261013 (`19`), 20261017 (`25`), 20261018 and 20261019 (`05b`), 20261015 for part E (`12`).

## License

MIT for the code. The census data belong to their author and are downloaded from the original source.
