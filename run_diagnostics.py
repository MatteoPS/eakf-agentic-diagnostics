#!/usr/bin/env python3
"""
run_diagnostics.py

Entry point: point this at a Model_Runs/*.mat file, it runs deterministic
checks, and only calls the LLM agent if something was flagged.

STATUS: extraction schema confirmed against a real file (601). Check
thresholds are still placeholders pending calibration against a known-good
/ known-bad pair (build order step 1). Agent tool-wiring (fetch_more_detail)
still stubbed.

Usage:
    python run_diagnostics.py path/to/Model_Runs/some_run.mat
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
    print(f"  n_days={run.n_days}  n_ensemble={run.n_ensemble}  n_params={run.n_params}")

    results = run_all_checks(run, coverage_observed=args.coverage)

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
