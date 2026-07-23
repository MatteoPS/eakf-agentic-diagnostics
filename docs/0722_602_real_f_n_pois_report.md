# Diagnostic Report: 0722_602_real_f_n_pois

**Turns used:** 2  
**Checks run:** ensemble_spread_collapse_alpha, ensemble_spread_collapse_beta, kalman_update_activity

**Flagged:** kalman_update_activity

---

## Flagged: kalman_update_activity
Four locations (58, 61, 68, 94) show mean posterior/prior update ratios >3 SD from the cross-location mean (0.244), all clustering around 0.58–0.62 instead of the typical 0.09–0.36 range.

## Evidence gathered
- per_location_kalman_ratios: The four outliers are Northwest Territories, Nunavut, Prince Edward Island, and Yukon — all small-population Canadian territories/provinces, each with ratios 2-2.5x the global mean.
- kalman_ratio_timeseries (loc 58, NWT): Ratio is persistently elevated (~0.4–0.8) across nearly the entire time series, not a transient spike, and stays well below the under_updating (0.98) and over_aggressive floor (0.05) thresholds throughout.

## Most likely explanation [Confidence: medium]
These are the smallest-population locations in the metapopulation (NWT, Nunavut, PEI, Yukon), where observed case counts are low and dominated by Poisson observation noise relative to signal. In a Poisson-likelihood EAKF, small denominators produce comparatively larger relative innovations, so the filter's posterior/prior spread ratio is systematically higher even under correct assimilation — this looks like a structural feature of small-population locations rather than a pathological update failure, especially since the ratio stays well inside the valid operating band (0.05–0.98) and is stable over time rather than diverging.

## Alternatives considered
- Localized model misspecification (e.g., wrong population size or reporting delay) for these four locations specifically, which could also inflate the ratio — cannot be fully ruled out without checking population/obs-error parameters.
- Data quality issue (sparse/noisy case reports in these low-count regions) driving inflated updates independent of population size per se.

## What would resolve the ambiguity
- Compare mean_post_prior_ratio against each location's population size or mean case count to confirm the small-population correlation.
- Check per_location_collapse_ratios for these same four locations to see if ensemble spread is also collapsing, which would indicate a deeper issue beyond noise-dri