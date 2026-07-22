#!/usr/bin/env python3
"""
run_diagnostics.py

Entry point: point this at a Model_Runs/*.mat file, it runs deterministic
checks, and only calls the LLM agent if something was flagged.

STATUS: end-to-end skeleton. extract.load_model_run() has not yet been
tested against a real file (none available at time of writing -- user is
re-running real-data runs 601-604). Do not trust this script's output
until that validation happens; see project build order step 1/2.

Usage:
    python run_diagnostics.py path/to/Model_Runs/some_run.mat --beta-bounds 0 5 --alpha-bounds 0 0.6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from eakf_diagnostics.extract import load_model_run, inspect_file_schema
from eakf_diagnostics.checks import run_all_checks, Severity
from eakf_diagnostics.agent import run_diagnostic_agent


def main():
    parser = argparse.ArgumentParser(description="Run EAKF ensemble process-health diagnostics.")
    parser.add_argument("mat_path", help="Path to a Model_Runs/*.mat file")
    parser.add_argument("--beta-bounds", nargs=2, type=float, default=[0.0, 5.0],
                         help="paramin/paramax for beta (from parafit_vars.mat)")
    parser.add_argument("--alpha-bounds", nargs=2, type=float, default=[0.0, 0.6],
                         help="paramin/paramax for alpha")
    parser.add_argument("--coverage", type=float, default=None,
                         help="Observed forecast coverage, if known (from Forecasts/ metrics)")
    parser.add_argument("--inspect-only", action="store_true",
                         help="Just dump the file's HDF5 schema and exit (use this FIRST on a new file)")
    parser.add_argument("--skip-agent", action="store_true",
                         help="Run checks only, skip the LLM investigation step (no API cost)")
    args = parser.parse_args()

    mat_path = Path(args.mat_path)

    if args.inspect_only:
        schema = inspect_file_schema(mat_path)
        print(json.dumps(schema, indent=2, default=str))
        return

    run = load_model_run(mat_path)
    print(f"Loaded run: {run.run_path.name}")
    print(f"  n_ensemble={run.n_ensemble}  n_days={run.n_days}")

    results = run_all_checks(
        beta_trajectories=run.beta_trajectories,
        alpha_trajectories=run.alpha_trajectories,
        beta_bounds=tuple(args.beta_bounds),
        alpha_bounds=tuple(args.alpha_bounds),
        coverage_observed=args.coverage,
    )

    print("\n--- Deterministic check results ---")
    for r in results:
        print(f"[{r.severity.value.upper():5s}] {r.check_name}: {r.summary}")

    flagged = [r for r in results if r.severity != Severity.OK]
    if not flagged:
        print("\nNo issues flagged. Clean run -- agent not invoked (saves API cost).")
        return

    if args.skip_agent:
        print(f"\n{len(flagged)} issue(s) flagged. Skipping agent (--skip-agent set).")
        return

    print(f"\n{len(flagged)} issue(s) flagged. Invoking diagnostic agent...")
    run_id = run.run_path.stem
    report = run_diagnostic_agent(results, run_id=run_id)
    print("\n--- Diagnostic report ---")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
