# Diagnostic Report: 0128_704_real_f_p_det

**Turns used:** 2  
**Checks run:** ensemble_spread_collapse_alpha, ensemble_spread_collapse_beta, kalman_update_activity

**Flagged:** kalman_update_activity

---

## Flagged: kalman_update_activity
Location 81 (Tamaulipas, MX) mean update ratio 0.146 vs pool mean 0.428, >3 SD outlier.

## Evidence gathered
- kalman_ratio_timeseries (loc 81): long stretches (~150+ days) near 0.0005–0.03, interspersed with episodic bursts to 0.5–0.8.
- per_location_collapse_ratios (loc 81): alpha (0.487) and beta (0.349) collapse ratios both mid-pack, not extreme vs other locations.

## Most likely explanation [Confidence: medium]
- Intermittent/sparse case reporting in Tamaulipas → many days with near-zero innovation → EAKF has nothing to correct → ratio ≈0.
- Occasional reporting bursts (batched or delayed data) → large innovation → brief high-ratio spikes.
- Net effect: low mean ratio driven by data sparsity/intermittency, not ensemble variance collapse (collapse ratios normal).

## Alternatives considered
- Ensemble variance collapse (over-aggressive filtering): ruled out — alpha/beta collapse ratios unremarkable, similar to other locations.
- Persistent under-updating due to model misspecification: possible but timeseries shows episodic strong updates, inconsistent with pure under-updating.

## What would resolve the ambiguity
- Check raw observed case-count series for loc 81 → confirm zero/near-zero reporting stretches align with near-zero ratio periods.
- Compare reporting cadence (weekly/batched) for Tamaulipas vs other locations.
- Check coverage_by_forecast_week for loc 81 to see if bursts correlate with data revision dates.