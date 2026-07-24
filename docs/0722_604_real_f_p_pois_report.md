# Diagnostic Report: 0722_604_real_f_p_pois

**Turns used:** 3  
**Checks run:** ensemble_spread_collapse_alpha, ensemble_spread_collapse_beta, kalman_update_activity

**Flagged:** kalman_update_activity

---

## Flagged: kalman_update_activity
Locations 58 (NW Territories), 68 (PEI), 94 (Yukon) show post/prior ratios >3 SD above the 0.26 location mean.

## Evidence gathered
- per_location_kalman_ratios: 3 flagged locs all in 0.55–0.62 range vs ~0.26 typical → ~2x elevated.
- kalman_ratio_timeseries (loc 58): ratio persistently 0.5–0.8 across nearly entire trajectory, not a transient spike.
- Cross-check: Nunavut (0.55) and Quintana Roo (0.048) also extreme but fell just inside 3 SD cutoff.

## Most likely explanation [Confidence: medium]
- Flagged locs → smallest population/case-count territories (NWT, PEI, Yukon).
- Sparse Poisson case counts → high relative observation noise per report.
- High obs noise relative to prior spread → EAKF gain stays small → posterior variance barely shrinks → persistently elevated ratio.
- Pattern sustained across full time series (not day-specific) → supports structural/population-driven cause, not a transient data glitch.

## Alternatives considered
- Localized data anomaly/reporting spike: ruled less likely, ratio elevated ~whole series