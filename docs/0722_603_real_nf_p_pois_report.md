# Diagnostic Report: 0722_603_real_nf_p_pois

**Turns used:** 2  
**Checks run:** ensemble_spread_collapse_alpha, ensemble_spread_collapse_beta, kalman_update_activity

**Flagged:** kalman_update_activity

---

## Flagged: kalman_update_activity
Locations 58, 68, 94 show mean post/prior ratio >3 SD above overall mean (0.254).

## Evidence gathered
- per_location_kalman_ratios: outliers = Northwest Territories, PEI, Yukon (0.64, 0.59, 0.64); Nunavut close (0.57) but not flagged.
- kalman_ratio_timeseries (loc 58): ratio sustained ~0.4–0.9 across nearly entire series, not a single spike.

## Most likely explanation [Confidence: medium]
- Small-population Canadian territories → sparse/low case counts → high observation noise relative to signal.
- EAKF gain scales inversely w/ signal-to-noise → persistently large posterior/prior updates.
- Sustained (not transient) elevation → structural, tied to population size, not a data glitch.

## Alternatives considered
- Data reporting anomaly (spikes/zeros) in these regions → ratio would show intermittent extremes, not sustained plateau — less likely.
- Prior variance misspecified only for territories → possible but consistent w/ small-pop noise explanation, hard to distinguish.

## What would resolve the ambiguity
- Compare case-count magnitude/variance for locs 58, 68, 94, 61 vs rest.
- Check if observation error variance scaling accounts for population size.
- Confirm Nunavut (61) shares same population-driven pattern despite not crossing 3 SD.