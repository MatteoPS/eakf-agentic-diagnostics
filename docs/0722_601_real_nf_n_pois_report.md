# Diagnostic Report: 0722_601_real_nf_n_pois

**Turns used:** 2  
**Checks run:** ensemble_spread_collapse_alpha, ensemble_spread_collapse_beta, kalman_update_activity

**Flagged:** kalman_update_activity

---

## Flagged: kalman_update_activity
Locations 58, 61, 68, 94 mean ratios (0.60, 0.59, 0.58, 0.63) >3 SD above overall mean 0.246.

## Evidence gathered
- per_location_kalman_ratios: outliers = NWT, Nunavut, PEI, Yukon — all small-population jurisdictions.
- kalman_ratio_timeseries (loc 58): ratio persistently elevated (~0.5–0.8) across nearly entire series, not a transient spike.
- Absolute values still well below under_updating_threshold (0.98) — flagged only via relative z-score, not absolute failure.

## Most likely explanation [Confidence: medium]
Small population size → sparse/noisy case counts → high observation variance relative to prior → weaker variance reduction (higher post/prior ratio) at every assimilation step. Persistent (not episodic) elevation across full time series supports structural population-size effect rather than a transient ensemble collapse or data glitch.

## Alternatives considered
- Transient outbreak-driven observation noise: ruled out by sustained (not spiky) elevation across ~350 days.
- Ensemble collapse/instability: unlikely, ratio staying <1 and stable, no runaway trend.
- Reporting artifact (zero-inflated case data) in these low-count regions: plausible but can't distinguish from population-size effect alone.

## What would resolve the ambiguity
- Compare location population sizes vs ratio across all 96 locs (check correlation).
- Inspect raw case-count time series for these 4 locations for zero-inflation/reporting gaps.
-