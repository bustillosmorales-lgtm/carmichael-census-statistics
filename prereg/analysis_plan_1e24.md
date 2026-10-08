# Analysis plan for the external validation on the census from 10^18 to 10^24

Written on 8 October 2026, before any statistic of the data described below was computed.
At the time of writing, the only part of the file that had been read was its first 36 lines
(the Carmichael numbers from 561 to 658801), to check the format. No count, gap, ratio,
digit or correlation of the range 10^18 to 10^24 had been computed.

## Data

- File `new_table.txt` published by J. Webster at <https://blue.butler.edu/~jewebste/new_table.txt>
  (Last-Modified 17 October 2025, 18,358,310,309 bytes). Each line has a Carmichael number
  followed by its prime factors in increasing order. The tabulation is that of Shallue and
  Webster (arXiv:2506.09903v3), with 308,279,939 Carmichael numbers below 10^24.
- The SHA-256 checksum of the file is recorded in `results/18_census_1e24.json` when it is first read.
- Integrity checks, carried out before any statistic: the numbers are strictly increasing;
  each product of the listed factors equals the number; no factor repeats; the Korselt
  criterion holds; the counts C(10^k) for k = 18,...,24 agree with OEIS A055553; the records
  below 10^18 coincide with the census of Pinch used in the article.

## Range

The confirmatory range is 10^18 <= n < 10^24, which is disjoint from the census on which the
methods were developed. Results below 10^18 are not part of the confirmatory analysis.

## Procedures

Every procedure is the one used in the article for the census below 10^18, with no change:

- Normalization by the centered moving mean of W = 1,000 gaps, truncated at both ends.
- Reference model: Poisson process whose mean measure is the integral of the intensity
  estimated by the centered moving mean of S = 20,000 gaps, simulated by inversion of the
  cumulative intensity, conditional on the observed number of points.
- Monte Carlo p-value (1 + #{T* >= t}) / (R + 1), two-sided around the simulated mean for
  the mean ratio and the correlations.

## Confirmatory hypotheses and statistics

Five statistics, each with its p-value, and Holm's step-down adjustment over the five at
family-wise level 0.05:

1. D_1000, the supremum distance between the normalized gaps and the unit exponential law.
2. D_ratio, the supremum distance between the ratios q_i = min(g_i, g_{i+1}) / max(g_i, g_{i+1})
   and the law 2x/(1+x).
3. The mean ratio, against its simulated null mean.
4. Q_10, the sum over lags 1 to 10 of the squared standardized excess Spearman autocorrelations
   of the ratios, standardized by the null mean and standard deviation at each lag.
5. The excess lag-one autocorrelation of the ratios.

Directional prediction recorded in advance, from the census below 10^18: the excess
autocorrelations of the ratios at lags 1 and 2 are positive, of about 0.0014 and 0.0020. The
validation reports the estimated excess at lags 1 and 2 with its Monte Carlo standard error and
states whether the sign and size are reproduced.

## Monte Carlo resolution

R = 200 realizations for the full-range statistics 1 to 3. The smallest attainable p-value is
then 1/201 = 0.005. If a p-value falls between 0.01 and 0.10, R is increased to 1,000 for that
statistic. The null mean and standard deviation of the ratio autocorrelations, which do not
depend on the intensity, are taken from 1,000 realizations of a unit-rate Poisson process of the
same size.

## Secondary analyses (reported, not part of the confirmatory family)

- The same five statistics in each decade from 10^18 to 10^24.
- The same statistics for each number of prime factors with at least 100,000 members in the range.
- The leading-digit chi-square against the decade-weighted generalized Benford law, calibrated by
  the parametric bootstrap described in `code/16_digits_bootstrap.py`.

## Deviations

Any deviation from this plan will be reported in the article and in the results file, with its
reason.
