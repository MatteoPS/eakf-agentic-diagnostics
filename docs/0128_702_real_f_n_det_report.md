# Diagnostic Report: 0128_702_real_f_n_det

**Turns used:** 2  
**Checks run:** ensemble_spread_collapse_alpha, ensemble_spread_collapse_beta, kalman_update_activity

**Flagged:** ensemble_spread_collapse_alpha, kalman_update_activity

---

## Flagged: ensemble_spread_collapse_alpha, kalman_update_activity
Alpha spread collapsed to 8.1% of early-run value at location 30 (warn=10%); location 81 kalman ratio deviates >3 SD from location mean (0.138 vs ~0.43 overall).

## Evidence gathered
- alpha_trajectory_full (loc 30): std drops from ~0.11–0.15 (days 0–60) to ~0.01–0.04 (day 65+), never recovers.
- alpha_trajectory_full (loc 30): mean alpha also drifts steadily downward over same window → consistent convergence, not noise artifact.
- kalman_ratio_timeseries (loc 81): highly bimodal — long stretches near-zero (~0.0005–0.002) punctuated by sharp spikes to 0.3–0.9.

## Most likely explanation [Confidence: medium]
Loc 30 → early strong/consistent observation signal → EAKF repeatedly pulls alpha same direction → ensemble variance collapses and stays low (genuine convergence, not divergence artifact). Loc 81 → sparse/noisy case reporting → most days produce near-zero innovation (no update), occasional large-case-count days trigger big Kalman gain → average ratio pulled down, flagged as outlier vs smoother locations.

## Alternatives considered
- Loc 30 collapse from model misspecification (alpha stuck at bound): mean trend still evolving, not pinned at 0/1, argues against hard bound artifact.
- Loc 81 outlier as generic under-updating: overall frac_valid_days (0.986) similar to others, so not