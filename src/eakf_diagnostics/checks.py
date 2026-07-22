"""
checks.py

Deterministic, non-LLM checks for EAKF ensemble PROCESS health.

Scope reminder (see project memory): these checks answer "did the
estimation process behave correctly?" -- NOT "did the model get the right
answer?". They run on real-data runs (no ground truth needed) and operate
on ensemble internals: parameter trajectories and, where available,
forecast coverage stats.

Three checks, each returns a CheckResult (not a bare bool) because the
LLM agent downstream needs the actual numbers, not just a verdict, to
reason about severity and possible causes.

THRESHOLDS ARE PLACEHOLDERS. They're set to plausible-sounding defaults
so the logic is exercisable now, on dummy data. Once real Model_Runs/
files are available, these need to be calibrated against the known-good
and known-bad reference runs (per project build order step 1) -- do not
trust these numbers for real diagnosis until that calibration happens.
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
# Check 1: Ensemble variance collapse
# ─────────────────────────────────────────────────────────────────────────

def check_ensemble_collapse(
    param_trajectories: np.ndarray,
    warn_ratio: float = 0.1,
    fail_ratio: float = 0.02,
    burn_in_days: int = 5,
) -> CheckResult:
    """
    Flags premature collapse of ensemble spread -- the EAKF becoming
    overconfident, where members converge to near-identical values and
    stop representing uncertainty.

    Args:
        param_trajectories: (n_days, n_ensemble, n_locations) array -- this
            is the axis order ModelRun.alpha_trajectories/.beta_trajectories
            actually produce (confirmed against real 601 file: para_post's
            axes are day/ensemble/param, not param/ensemble/day as first
            assumed before any real data was available).
        warn_ratio: if late-run ensemble std / early-run ensemble std
            drops below this, flag WARN.
        fail_ratio: same, but for FAIL.
        burn_in_days: days to skip at the start (initial spread is often
            wide by design; not meaningful to compare against day 0).

    PLACEHOLDER: warn_ratio/fail_ratio still need calibration against real
    known-good/known-bad runs -- see module docstring. Axis order itself
    IS now confirmed, just not the thresholds.
    """
    n_days, n_ensemble, n_locations = param_trajectories.shape
    if n_days <= burn_in_days:
        return CheckResult(
            "ensemble_collapse", Severity.WARN,
            f"Run too short ({n_days} days) to evaluate collapse with "
            f"burn_in_days={burn_in_days}.",
            {"n_days": n_days},
        )

    std_over_time = param_trajectories.std(axis=1)  # (n_days, n_locations)
    early_std = std_over_time[burn_in_days:burn_in_days + 5, :].mean(axis=0)
    late_std = std_over_time[-5:, :].mean(axis=0)

    # avoid divide-by-zero for locations with zero early spread
    ratio = np.divide(
        late_std, early_std,
        out=np.full_like(early_std, np.nan),
        where=early_std > 1e-10,
    )

    worst_idx = np.nanargmin(ratio) if not np.all(np.isnan(ratio)) else None
    worst_ratio = ratio[worst_idx] if worst_idx is not None else None

    if worst_ratio is None:
        severity = Severity.WARN
        summary = "Could not compute collapse ratio (all early-std ~0)."
    elif worst_ratio < fail_ratio:
        severity = Severity.FAIL
        summary = (
            f"Severe ensemble collapse: location {worst_idx} spread shrank "
            f"to {worst_ratio:.1%} of early-run spread."
        )
    elif worst_ratio < warn_ratio:
        severity = Severity.WARN
        summary = (
            f"Possible ensemble collapse: location {worst_idx} spread shrank "
            f"to {worst_ratio:.1%} of early-run spread."
        )
    else:
        severity = Severity.OK
        summary = f"No collapse detected (min ratio {worst_ratio:.1%})."

    return CheckResult(
        "ensemble_collapse", severity, summary,
        {
            "ratio_per_location": ratio.tolist(),
            "worst_location_idx": int(worst_idx) if worst_idx is not None else None,
            "worst_ratio": float(worst_ratio) if worst_ratio is not None else None,
            "warn_ratio": warn_ratio,
            "fail_ratio": fail_ratio,
        },
    )


# ─────────────────────────────────────────────────────────────────────────
# Check 2: Parameter bound clipping
# ─────────────────────────────────────────────────────────────────────────

def check_parameter_clipping(
    param_trajectories: np.ndarray,
    param_min: np.ndarray | float,
    param_max: np.ndarray | float,
    warn_frac: float = 0.10,
    fail_frac: float = 0.30,
    tol: float = 1e-6,
) -> CheckResult:
    """
    Flags parameters that spend a large fraction of the run pinned at
    their bounds (paramin/paramax) instead of settling to an interior
    value -- suggests the bounds are misspecified or the model is being
    forced to compensate for something else.

    Args:
        param_trajectories: (n_days, n_ensemble, n_locations) -- see
            check_ensemble_collapse docstring for axis order confirmation.
        param_min, param_max: bounds for this parameter. Real paramin/
            paramax fields are PER-PARAMETER arrays (shape (1, 196) in the
            confirmed 601 file), not single shared scalars -- pass either
            a scalar (broadcasts to all locations) or a 1D array of length
            n_locations matching param_trajectories' last axis (i.e.
            ModelRun.alpha_bounds / .beta_bounds, already sliced to the
            right params via alphamaps/betamap).
        warn_frac / fail_frac: fraction of (day, ensemble-member, location)
            points sitting within `tol` of a bound, above which to flag.

    PLACEHOLDER: warn_frac/fail_frac still need calibration against real
    known-good/known-bad runs.
    """
    param_min = np.asarray(param_min)
    param_max = np.asarray(param_max)
    # broadcast (n_locations,) bounds against (n_days, n_ensemble, n_locations)
    at_min = np.abs(param_trajectories - param_min) < tol
    at_max = np.abs(param_trajectories - param_max) < tol
    at_bound = at_min | at_max

    frac_clipped = at_bound.mean()
    frac_at_min = at_min.mean()
    frac_at_max = at_max.mean()

    if frac_clipped >= fail_frac:
        severity = Severity.FAIL
        summary = f"Heavy bound-clipping: {frac_clipped:.1%} of ensemble-days at a bound."
    elif frac_clipped >= warn_frac:
        severity = Severity.WARN
        summary = f"Some bound-clipping: {frac_clipped:.1%} of ensemble-days at a bound."
    else:
        severity = Severity.OK
        summary = f"No significant clipping ({frac_clipped:.1%} of ensemble-days at a bound)."

    return CheckResult(
        "parameter_clipping", severity, summary,
        {
            "frac_clipped": float(frac_clipped),
            "frac_at_min": float(frac_at_min),
            "frac_at_max": float(frac_at_max),
            "warn_frac": warn_frac,
            "fail_frac": fail_frac,
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
    Flags miscalibrated forecast uncertainty: if the nominal_coverage
    (e.g. 95%) prediction interval doesn't actually contain truth close
    to that fraction of the time, the ensemble's stated uncertainty is
    untrustworthy (either over- or under-confident).

    Args:
        coverage_observed: empirical coverage rate, e.g. from
            Forecasts/*_fore_res_group.mat coverage field, already
            computed by make_forecast_metrics_real.m -- this check does
            NOT recompute it, just evaluates it.
        nominal_coverage: the target coverage (0.95 for a 95% PI)
        warn_delta / fail_delta: |observed - nominal| threshold

    PLACEHOLDER: warn_delta/fail_delta need calibration against real runs.
    """
    delta = abs(coverage_observed - nominal_coverage)
    direction = "over-confident (too narrow)" if coverage_observed < nominal_coverage \
        else "under-confident (too wide)"

    if delta >= fail_delta:
        severity = Severity.FAIL
        summary = (
            f"Severe miscalibration: {direction}. Observed coverage "
            f"{coverage_observed:.1%} vs nominal {nominal_coverage:.1%}."
        )
    elif delta >= warn_delta:
        severity = Severity.WARN
        summary = (
            f"Possible miscalibration: {direction}. Observed coverage "
            f"{coverage_observed:.1%} vs nominal {nominal_coverage:.1%}."
        )
    else:
        severity = Severity.OK
        summary = f"Coverage well-calibrated ({coverage_observed:.1%} vs nominal {nominal_coverage:.1%})."

    return CheckResult(
        "coverage_miscalibration", severity, summary,
        {
            "coverage_observed": coverage_observed,
            "nominal_coverage": nominal_coverage,
            "delta": delta,
            "warn_delta": warn_delta,
            "fail_delta": fail_delta,
        },
    )


def run_all_checks(model_run, coverage_observed: float | None = None) -> list[CheckResult]:
    """
    Convenience wrapper: run all checks against a loaded ModelRun and
    return results as a list.

    Args:
        model_run: an eakf_diagnostics.extract.ModelRun instance.
        coverage_observed: optional, from Forecasts/ metrics.
    """
    alpha_min, alpha_max = model_run.alpha_bounds
    beta_min, beta_max = model_run.beta_bounds

    results = [
        check_ensemble_collapse(model_run.alpha_trajectories),
        check_ensemble_collapse(model_run.beta_trajectories),
        check_parameter_clipping(model_run.alpha_trajectories, alpha_min, alpha_max),
        check_parameter_clipping(model_run.beta_trajectories, beta_min, beta_max),
    ]
    if coverage_observed is not None:
        results.append(check_coverage_miscalibration(coverage_observed))
    return results
