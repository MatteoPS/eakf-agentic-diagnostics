"""
test_checks_dummy.py

Exercises checks.py logic using synthetic dummy arrays -- NOT real
Model_Runs data (not available yet). Purpose: verify the check functions
run without error and produce sane-direction verdicts on obviously-good
and obviously-bad synthetic inputs, before real .mat files arrive.

This is NOT calibration. Thresholds are still placeholders. This just
confirms the mechanics work.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eakf_diagnostics.checks import (
    check_ensemble_collapse,
    check_parameter_clipping,
    check_coverage_miscalibration,
    Severity,
)

rng = np.random.default_rng(42)


def make_healthy_trajectory(n_locations=3, n_ensemble=100, n_days=60, center=1.5, spread=0.3):
    """Ensemble spread stays roughly constant over time -- no collapse."""
    traj = np.zeros((n_locations, n_ensemble, n_days))
    for d in range(n_days):
        traj[:, :, d] = center + rng.normal(0, spread, size=(n_locations, n_ensemble))
    return traj


def make_collapsing_trajectory(n_locations=3, n_ensemble=100, n_days=60, center=1.5, spread=0.3):
    """Ensemble spread shrinks to near-zero by the end -- simulates collapse."""
    traj = np.zeros((n_locations, n_ensemble, n_days))
    for d in range(n_days):
        decay = max(0.01, 1 - d / n_days)  # spread shrinks linearly to ~1%
        traj[:, :, d] = center + rng.normal(0, spread * decay, size=(n_locations, n_ensemble))
    return traj


def make_clipped_trajectory(n_locations=2, n_ensemble=100, n_days=60, pmin=0.0, pmax=5.0, frac_at_bound=0.4):
    """A chunk of ensemble-days pinned exactly at pmax."""
    traj = pmin + rng.uniform(0, pmax - pmin, size=(n_locations, n_ensemble, n_days))
    mask = rng.random(traj.shape) < frac_at_bound
    traj[mask] = pmax
    return traj


def test_collapse_check_flags_healthy_as_ok():
    traj = make_healthy_trajectory()
    result = check_ensemble_collapse(traj)
    print(f"[healthy]    severity={result.severity}  {result.summary}")
    assert result.severity == Severity.OK


def test_collapse_check_flags_collapsing_as_fail_or_warn():
    traj = make_collapsing_trajectory()
    result = check_ensemble_collapse(traj)
    print(f"[collapsing] severity={result.severity}  {result.summary}")
    assert result.severity in (Severity.WARN, Severity.FAIL)


def test_clipping_check_flags_healthy_as_ok():
    traj = pmin_healthy = 0.0 + rng.uniform(1.0, 4.0, size=(2, 100, 60))  # well within [0,5]
    result = check_parameter_clipping(traj, param_min=0.0, param_max=5.0)
    print(f"[healthy clip]  severity={result.severity}  {result.summary}")
    assert result.severity == Severity.OK


def test_clipping_check_flags_clipped_as_fail():
    traj = make_clipped_trajectory(frac_at_bound=0.4)
    result = check_parameter_clipping(traj, param_min=0.0, param_max=5.0)
    print(f"[clipped]       severity={result.severity}  {result.summary}")
    assert result.severity == Severity.FAIL


def test_coverage_check_flags_good_calibration():
    result = check_coverage_miscalibration(coverage_observed=0.94, nominal_coverage=0.95)
    print(f"[cov ok]    severity={result.severity}  {result.summary}")
    assert result.severity == Severity.OK


def test_coverage_check_flags_overconfident():
    result = check_coverage_miscalibration(coverage_observed=0.60, nominal_coverage=0.95)
    print(f"[cov bad]   severity={result.severity}  {result.summary}")
    assert result.severity == Severity.FAIL


if __name__ == "__main__":
    test_collapse_check_flags_healthy_as_ok()
    test_collapse_check_flags_collapsing_as_fail_or_warn()
    test_clipping_check_flags_healthy_as_ok()
    test_clipping_check_flags_clipped_as_fail()
    test_coverage_check_flags_good_calibration()
    test_coverage_check_flags_overconfident()
    print("\nAll dummy-data mechanics tests passed.")
