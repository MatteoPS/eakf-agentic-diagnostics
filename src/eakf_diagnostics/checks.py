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
        param_trajectories: (n_locations, n_ensemble, n_days) array, e.g.
            ModelRun.beta_trajectories or .alpha_trajectories.
        warn_ratio: if late-run ensemble std / early-run ensemble std
            drops below this, flag WARN.
        fail_ratio: same, but for FAIL.
        burn_in_days: days to skip at the start (initial spread is often
            wide by design; not meaningful to compare against day 0).

    PLACEHOLDER: warn_ratio/fail_ratio need calibration against real
    known-good/known-bad runs -- see module docstring.
    """
    n_locations, n_ensemble, n_days = param_trajectories.shape
    if n_days <= burn_in_days:
        return CheckResult(
            "ensemble_collapse", Severity.WARN,
            f"Run too short ({n_days} days) to evaluate collapse with "
            f"burn_in_days={burn_in_days}.",
            {"n_days": n_days},
        )

    std_over_time = param_trajectories.std(axis=1)  # (n_locations, n_days)
    early_std = std_over_time[:, burn_in_days:burn_in_days + 5].mean(axis=1)
    late_std = std_over_time[:, -5:].mean(axis=1)

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
    param_min: float,
    param_max: float,
    warn_frac: float = 0.10,
    fail_frac: float = 0.30,
    tol: float = 1e-6,
) -> CheckResult:
    """
    Flags parameters that spend a large fraction of the run pinned at
    their bounds (paramin/paramax from parafit_vars.mat) instead of
    settling to an interior value -- suggests the bounds are misspecified
    or the model is being forced to compensate for something else.

    Args:
        param_trajectories: (n_locations, n_ensemble, n_days)
        param_min, param_max: the bounds this parameter was constrained to
            (checkbound_para.m enforces these after each EAKF update)
        warn_frac / fail_frac: fraction of (location, ensemble-member, day)
            points sitting within `tol` of a bound, above which to flag.

    PLACEHOLDER: warn_frac/fail_frac need calibration against real runs.
    """
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
            "param_min": param_min,
            "param_max": param_max,
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


def run_all_checks(
    beta_trajectories: np.ndarray,
    alpha_trajectories: np.ndarray,
    beta_bounds: tuple[float, float],
    alpha_bounds: tuple[float, float],
    coverage_observed: float | None = None,
) -> list[CheckResult]:
    """Convenience wrapper: run all three checks and return results as a list."""
    results = [
        check_ensemble_collapse(beta_trajectories),
        check_ensemble_collapse(alpha_trajectories),
        check_parameter_clipping(beta_trajectories, *beta_bounds),
        check_parameter_clipping(alpha_trajectories, *alpha_bounds),
    ]
    if coverage_observed is not None:
        results.append(check_coverage_miscalibration(coverage_observed))
    return results
