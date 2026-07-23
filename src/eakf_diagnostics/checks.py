"""
checks.py

Deterministic, non-LLM checks for EAKF ensemble PROCESS health.

These answer "did the estimation process behave correctly?" -- not "did
the model get the right answer?". Run on real-data runs (no ground truth
needed), operating on ensemble internals only.

Note: originally had a parameter-bound clipping check here instead of
check_kalman_update_activity(). Turned out checkbound_para.m resamples
out-of-bound members to interior values rather than clipping to the
boundary, so fraction-at-bound is always 0.0% no matter what -- useless.
Swapped it for the Kalman activity check below, which uses
prior_var_rec/post_var_rec already saved to the .mat file.

Each check returns a CheckResult rather than a bare bool since the agent
downstream needs actual numbers to reason about severity, not just a
verdict.

Collapse thresholds are informed by spread ratios across the 8 real-data
runs I have (601-604 Poisson, 701-704 deterministic dev) -- range 10-26%.
Kalman activity thresholds are still placeholders, pending a closer look
at real prior_var_rec/post_var_rec values. Calibration notes per function.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from enum import Enum


class Severity(str, Enum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


@dataclass
class CheckResult:
    check_name: str
    severity: Severity
    summary: str
    details: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────
# Check 1: Ensemble spread collapse
# ─────────────────────────────────────────────────────────────────────────

def check_ensemble_spread_collapse(
    param_trajectories: np.ndarray,
    param_label: str = "param",
    warn_ratio: float = 0.10,
    fail_ratio: float = 0.05,
    burn_in_days: int = 5,
) -> CheckResult:
    """
    Flags premature collapse of ensemble spread — the EAKF becoming
    overconfident, where members converge to near-identical values and
    stop representing uncertainty.

    Args:
        param_trajectories: (n_days, n_ensemble, n_locations) — confirmed
            axis order from real 601 file (para_post: day/ensemble/param,
            sliced to alpha or beta locations via alphamaps/betamap).
        param_label: "alpha" or "beta", used in check_name for disambiguation.
        warn_ratio: if (late std / early std) < this, flag WARN.
        fail_ratio: same, but for FAIL.
        burn_in_days: days to skip at start (initial spread is wide by
            design; comparing against day 0 is not meaningful).

    warn_ratio=0.10 comes from real data -- lowest collapse ratio I've seen
    across 8 runs was 10.6% (run 701, alpha). fail_ratio=0.05 is below
    anything observed so far, just there to catch severe outliers. Revisit
    both if the reference set grows.
    """
    n_days, n_ensemble, n_locations = param_trajectories.shape
    check_name = f"ensemble_spread_collapse_{param_label}"

    if n_days <= burn_in_days:
        return CheckResult(
            check_name, Severity.WARN,
            f"Run too short ({n_days} days) to evaluate collapse "
            f"(burn_in_days={burn_in_days}).",
            {"n_days": n_days},
        )

    std_over_time = param_trajectories.std(axis=1)  # (n_days, n_locations)
    early_std = std_over_time[burn_in_days:burn_in_days + 5, :].mean(axis=0)
    late_std  = std_over_time[-5:, :].mean(axis=0)

    ratio = np.divide(
        late_std, early_std,
        out=np.full_like(early_std, np.nan),
        where=early_std > 1e-10,
    )

    worst_idx   = int(np.nanargmin(ratio)) if not np.all(np.isnan(ratio)) else None
    worst_ratio = float(ratio[worst_idx])  if worst_idx is not None else None

    if worst_ratio is None:
        sev     = Severity.WARN
        summary = "Could not compute collapse ratio (all early-std ~0)."
    elif worst_ratio < fail_ratio:
        sev     = Severity.FAIL
        summary = (
            f"Severe {param_label} spread collapse: location {worst_idx} "
            f"shrank to {worst_ratio:.1%} of early-run spread."
        )
    elif worst_ratio < warn_ratio:
        sev     = Severity.WARN
        summary = (
            f"Possible {param_label} spread collapse: location {worst_idx} "
            f"shrank to {worst_ratio:.1%} of early-run spread."
        )
    else:
        sev     = Severity.OK
        summary = f"No {param_label} spread collapse (min ratio {worst_ratio:.1%})."

    return CheckResult(
        check_name, sev, summary,
        {
            "param_label": param_label,
            "ratio_per_location": ratio.tolist(),
            "worst_location_idx": worst_idx,
            "worst_ratio": worst_ratio,
            "warn_ratio": warn_ratio,
            "fail_ratio": fail_ratio,
        },
    )


# ─────────────────────────────────────────────────────────────────────────
# Check 2: Kalman update activity (replaces clipping check)
# ─────────────────────────────────────────────────────────────────────────

def check_kalman_update_activity(
    prior_var_rec: np.ndarray,
    post_var_rec: np.ndarray,
    over_aggressive_threshold: float = 0.05,
    under_updating_threshold: float  = 0.98,
    location_outlier_z: float        = 3.0,
) -> CheckResult:
    """
    Checks whether the EAKF is updating the ensemble by a healthy amount
    at each assimilation step, using the ratio post_var / prior_var recorded
    per day per location.

    Two failure modes:
      OVER-AGGRESSIVE (ratio << 1): filter collapses variance too fast at
          each step; ensemble loses diversity early, forecast uncertainty
          understated. Complements the spread-collapse check, but operates
          per-step rather than across the whole run.
      UNDER-UPDATING (ratio near 1): filter barely responds to observations;
          ensemble not being informed by data.

    Also flags per-location outliers: a single location whose mean ratio
    deviates strongly from the others suggests a location-specific problem
    (bad observations, unusual epidemic dynamics, boundary condition issue).

    Args:
        prior_var_rec: (n_days, n_locations) ensemble variance before update
        post_var_rec:  (n_days, n_locations) ensemble variance after update
        over_aggressive_threshold: mean ratio below this → WARN/FAIL
        under_updating_threshold:  mean ratio above this → WARN
        location_outlier_z: z-score above this → flag that location

    Thresholds below are placeholders -- haven't looked closely at
    prior_var_rec/post_var_rec across the reference set (601-604, 701-704)
    yet. Swap in real numbers once that's done.
    """
    # days with zero prior variance mean no assimilation happened there
    # (no observation that day) -- not a failure, just skip them
    valid = prior_var_rec > 1e-12
    ratio = np.where(
        valid,
        np.divide(post_var_rec, prior_var_rec,
                  out=np.ones_like(prior_var_rec),
                  where=valid),
        np.nan,
    )

    frac_valid = valid.mean()
    if frac_valid < 0.1:
        return CheckResult(
            "kalman_update_activity", Severity.WARN,
            f"Only {frac_valid:.1%} of days have non-zero prior variance; "
            f"cannot meaningfully assess Kalman update activity.",
            {"frac_valid_days": float(frac_valid)},
        )

    # Per-location mean ratio (ignoring NaN days)
    mean_ratio_per_loc = np.nanmean(ratio, axis=0)   # (n_locations,)
    overall_mean       = float(np.nanmean(ratio))

    # Per-location outlier detection
    loc_mean  = float(np.nanmean(mean_ratio_per_loc))
    loc_std   = float(np.nanstd(mean_ratio_per_loc))
    # require both a z-score hit AND a real absolute gap (>0.05) -- otherwise
    # when all locations are nearly identical, tiny loc_std blows up the
    # z-scores and flags noise
    if loc_std > 1e-10:
        zscores = (mean_ratio_per_loc - loc_mean) / loc_std
        abs_dev = np.abs(mean_ratio_per_loc - loc_mean)
        outlier_locs = np.where(
            (np.abs(zscores) > location_outlier_z) & (abs_dev > 0.05)
        )[0].tolist()
    else:
        outlier_locs = []

    findings = []
    severity = Severity.OK

    if overall_mean < over_aggressive_threshold:
        severity = Severity.FAIL
        findings.append(
            f"Over-aggressive updating: mean post/prior variance ratio "
            f"{overall_mean:.3f} (threshold {over_aggressive_threshold})."
        )
    elif overall_mean < over_aggressive_threshold * 2:
        severity = Severity.WARN
        findings.append(
            f"Possibly over-aggressive updating: mean ratio {overall_mean:.3f}."
        )

    if overall_mean > under_updating_threshold:
        severity = max(severity, Severity.WARN,
                       key=lambda s: ["ok", "warn", "fail"].index(s.value))
        findings.append(
            f"Under-updating: mean post/prior variance ratio "
            f"{overall_mean:.3f} (threshold {under_updating_threshold}). "
            f"Filter may not be responding to observations."
        )

    if outlier_locs:
        severity = max(severity, Severity.WARN,
                       key=lambda s: ["ok", "warn", "fail"].index(s.value))
        findings.append(
            f"Location-specific outlier(s) in update activity: "
            f"locations {outlier_locs} deviate >{location_outlier_z:.1f} SD "
            f"from the location mean ratio."
        )

    if not findings:
        summary = f"Kalman update activity normal (mean ratio {overall_mean:.3f})."
    else:
        summary = " | ".join(findings)

    return CheckResult(
        "kalman_update_activity", severity, summary,
        {
            "overall_mean_ratio": overall_mean,
            "mean_ratio_per_location": mean_ratio_per_loc.tolist(),
            "outlier_locations": outlier_locs,
            "frac_valid_days": float(frac_valid),
            "over_aggressive_threshold": over_aggressive_threshold,
            "under_updating_threshold":  under_updating_threshold,
            "location_outlier_z": location_outlier_z,
        },
    )


# ─────────────────────────────────────────────────────────────────────────
# Check 3: Forecast coverage miscalibration
# ─────────────────────────────────────────────────────────────────────────

def check_coverage_miscalibration(
    coverage_observed: float,
    nominal_coverage: float = 0.95,
    warn_delta: float = 0.10,
    fail_delta: float = 0.20,
) -> CheckResult:
    """
    Flags miscalibrated forecast uncertainty: if the 95% PI doesn't
    actually contain truth ~95% of the time, the ensemble's stated
    uncertainty is untrustworthy.

    Args:
        coverage_observed: empirical coverage from Forecasts/ metrics
            (already computed by make_forecast_metrics_real.m — this check
            does NOT recompute it, just evaluates it).
        nominal_coverage: target coverage level (0.95 for 95% PI).
        warn_delta / fail_delta: |observed - nominal| thresholds.

    Thresholds are placeholders -- haven't loaded any Forecasts/ data yet
    to pick real values.
    """
    delta     = abs(coverage_observed - nominal_coverage)
    direction = (
        "over-confident (PI too narrow)"
        if coverage_observed < nominal_coverage
        else "under-confident (PI too wide)"
    )

    if delta >= fail_delta:
        sev     = Severity.FAIL
        summary = (
            f"Severe miscalibration: {direction}. "
            f"Observed {coverage_observed:.1%} vs nominal {nominal_coverage:.1%}."
        )
    elif delta >= warn_delta:
        sev     = Severity.WARN
        summary = (
            f"Possible miscalibration: {direction}. "
            f"Observed {coverage_observed:.1%} vs nominal {nominal_coverage:.1%}."
        )
    else:
        sev     = Severity.OK
        summary = (
            f"Coverage well-calibrated "
            f"({coverage_observed:.1%} vs nominal {nominal_coverage:.1%})."
        )

    return CheckResult(
        "coverage_miscalibration", sev, summary,
        {
            "coverage_observed": coverage_observed,
            "nominal_coverage":  nominal_coverage,
            "delta":             delta,
            "warn_delta":        warn_delta,
            "fail_delta":        fail_delta,
        },
    )


# ─────────────────────────────────────────────────────────────────────────
# Convenience wrapper
# ─────────────────────────────────────────────────────────────────────────

def run_all_checks(model_run, coverage_observed: float | None = None) -> list[CheckResult]:
    """
    Run all checks against a loaded ModelRun and return results as a list.
    Coverage check only runs if coverage_observed is provided (requires
    Forecasts/ data, not just Model_Runs/).
    """
    results = [
        check_ensemble_spread_collapse(model_run.alpha_trajectories, param_label="alpha"),
        check_ensemble_spread_collapse(model_run.beta_trajectories,  param_label="beta"),
        check_kalman_update_activity(model_run.prior_var_rec, model_run.post_var_rec),
    ]
    if coverage_observed is not None:
        results.append(check_coverage_miscalibration(coverage_observed))
    return results
