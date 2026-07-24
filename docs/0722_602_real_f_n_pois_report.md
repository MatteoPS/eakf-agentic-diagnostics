# Diagnostic Report: 0722_602_real_f_n_pois

**Turns used:** 2  
**Checks run:** ensemble_spread_collapse_alpha, ensemble_spread_collapse_beta, kalman_update_activity

**Flagged:** kalman_update_activity

---

## Flagged: kalman_update_activity
Locations 58, 61, 68, 94 exceed 3.0 SD in post/prior update ratio (mean 0.244).

## Evidence gathered
- per_location_kalman_ratios: flagged locs (NWT 0.61, Nunavut 0.58, PEI 0.59, Yukon 0.62) roughly 2x overall mean.
- per_location_collapse_ratios: same 4 locs show alpha/beta collapse ratios ~0.75–1.06, far above typical 0.3–0.6 range.

## Most likely explanation [Confidence: medium]
- Locs 58/61/68/94 = smallest-population Canadian territories (NWT, Nunavut, PEI, Yukon).
- Sparse/noisy case counts → wide prior spread relative to obs → large EAKF corrections each step.
- High collapse ratios confirm ensembles barely narrowing → consistent with under-constrained, noisy small-population fits.

## Alternatives considered
- Model misspecification for these locs: possible but collapse ratio pattern strongly tracks population size, favoring noise explanation.
- Data reporting artifacts (sparse/zero-inflated case counts): plausible co-driver, not distinguishable from noise mechanism here.

## What would resolve the ambiguity
- Compare raw case-count time series variance for these 4 locs vs mid-size locs.
- Check kalman_ratio_timeseries per flagged location for spikes vs sustained elevation.
- Test if lowering observation noise floor for small pops reduces ratio outliers.