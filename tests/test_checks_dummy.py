"""
test_checks_dummy.py

Exercises checks.py logic using synthetic dummy arrays — NOT real Model_Runs
data. Verifies that each check produces sane-direction verdicts on obviously-
good and obviously-bad synthetic inputs.

This is mechanics testing, not calibration. Thresholds in checks.py are
still placeholders for the Kalman activity check.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eakf_diagnostics.checks import (
    check_ensemble_spread_collapse,
    check_kalman_update_activity,
    check_coverage_miscalibration,
    Severity,
)

rng = np.random.default_rng(42)
N_DAYS, N_ENS, N_LOC = 437, 150, 96


def make_healthy_trajectory():
    """Spread stays roughly constant — no collapse."""
    base = rng.normal(1.5, 0.3, size=(N_DAYS, N_ENS, N_LOC))
    return base


def make_collapsing_trajectory():
    """Spread shrinks to ~1% of initial by end of run."""
    traj = np.zeros((N_DAYS, N_ENS, N_LOC))
    for d in range(N_DAYS):
        decay = max(0.01, 1 - d / N_DAYS)
        traj[d] = 1.5 + rng.normal(0, 0.3 * decay, size=(N_ENS, N_LOC))
    return traj


def make_healthy_var_rec():
    """post_var ~ 0.6 * prior_var — moderate, healthy reduction, consistent
    across locations so no location-outlier fires by random chance."""
    prior = np.abs(rng.normal(1.0, 0.2, size=(N_DAYS, N_LOC))) + 0.1
    # Use a small, location-uniform noise so all 96 locations stay close
    ratio = 0.60 + rng.normal(0, 0.005, size=(N_DAYS, N_LOC))
    post  = prior * ratio
    return prior, post


def make_over_aggressive_var_rec():
    """post_var ~ 0.02 * prior_var — filter collapsing variance at every step."""
    prior = np.abs(rng.normal(1.0, 0.2, size=(N_DAYS, N_LOC))) + 0.1
    post  = prior * rng.uniform(0.01, 0.03, size=(N_DAYS, N_LOC))
    return prior, post


def make_under_updating_var_rec():
    """post_var ~ 0.99 * prior_var — filter barely responding to observations."""
    prior = np.abs(rng.normal(1.0, 0.2, size=(N_DAYS, N_LOC))) + 0.1
    post  = prior * rng.uniform(0.985, 0.995, size=(N_DAYS, N_LOC))
    return prior, post


# ── Collapse check ─────────────────────────────────────────────────────────

def test_collapse_healthy_is_ok():
    result = check_ensemble_spread_collapse(make_healthy_trajectory(), param_label="beta")
    print(f"[collapse healthy]     {result.severity}  {result.summary}")
    assert result.severity == Severity.OK


def test_collapse_collapsing_is_flagged():
    result = check_ensemble_spread_collapse(make_collapsing_trajectory(), param_label="beta")
    print(f"[collapse collapsing]  {result.severity}  {result.summary}")
    assert result.severity in (Severity.WARN, Severity.FAIL)


# ── Kalman update activity check ───────────────────────────────────────────

def test_kalman_healthy_is_ok():
    prior, post = make_healthy_var_rec()
    result = check_kalman_update_activity(prior, post)
    print(f"[kalman healthy]       {result.severity}  {result.summary}")
    assert result.severity == Severity.OK


def test_kalman_over_aggressive_is_flagged():
    prior, post = make_over_aggressive_var_rec()
    result = check_kalman_update_activity(prior, post)
    print(f"[kalman over-agg]      {result.severity}  {result.summary}")
    assert result.severity in (Severity.WARN, Severity.FAIL)


def test_kalman_under_updating_is_flagged():
    prior, post = make_under_updating_var_rec()
    result = check_kalman_update_activity(prior, post)
    print(f"[kalman under-upd]     {result.severity}  {result.summary}")
    assert result.severity == Severity.WARN


# ── Coverage check ─────────────────────────────────────────────────────────

def test_coverage_good_is_ok():
    result = check_coverage_miscalibration(0.94)
    print(f"[coverage ok]          {result.severity}  {result.summary}")
    assert result.severity == Severity.OK


def test_coverage_bad_is_flagged():
    result = check_coverage_miscalibration(0.60)
    print(f"[coverage bad]         {result.severity}  {result.summary}")
    assert result.severity == Severity.FAIL


if __name__ == "__main__":
    test_collapse_healthy_is_ok()
    test_collapse_collapsing_is_flagged()
    test_kalman_healthy_is_ok()
    test_kalman_over_aggressive_is_flagged()
    test_kalman_under_updating_is_flagged()
    test_coverage_good_is_ok()
    test_coverage_bad_is_flagged()
    print("\nAll dummy-data mechanics tests passed.")
