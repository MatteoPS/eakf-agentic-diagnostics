# eakf-agentic-diagnostics

An LLM-powered diagnostic agent for the [`NA_SEIR-EAKF_forecast`](https://github.com/matteoperini/NA_SEIR-EAKF_forecast) pipeline — a continental-scale metapopulation SEIR model with Ensemble Adjustment Kalman Filter (EAKF) data assimilation for epidemic forecasting across North America.

---

## Overview

This tool reads the output of completed EAKF model runs, runs deterministic numeric checks on ensemble internals, and — when anomalies are flagged — invokes a Claude API tool-use agent to investigate further and write a structured diagnostic report.

The target question is: **did the estimation process behave correctly?** Not whether the forecast was accurate (that is already computed by the parent pipeline's `make_forecast_metrics_real.m`), but whether the EAKF ensemble dynamics were mechanistically healthy: adequate spread, well-calibrated updates, no filter pathologies.

---

## Why this exists

The parent pipeline produces 437 days × 96 locations × 150 ensemble members × 196 parameters of posterior state for each run. After each batch of runs, the current practice is to visually scan plots (ensemble mean ± 95% CI for α, β, daily incidence, susceptibles) to catch anomalies: collapse, parameter drift, weird location-specific behavior. This is:

- **Slow** at scale — the MATLAB plotting script generates dozens of panels per run batch.
- **Inconsistent** — what one person flags as "suspicious spread" another eyeballs past.
- **Silent about mechanism** — a plot showing narrow CIs doesn't distinguish "filter correctly became confident" from "filter collapsed under observation pressure and got lucky."

This agent reads the ensemble internals directly (not rendered plots) and flags anomalies with stated reasoning and calibrated uncertainty, replacing the manual visual scan.

---

## Architecture

```
Model_Runs/*.mat
       │
       ▼
 extract.py          ← h5py loader; confirmed schema against real 601-604 files
       │
       ▼
 checks.py           ← deterministic checks (no LLM); returns CheckResult list
  ├─ ensemble_spread_collapse (alpha, beta)
  ├─ kalman_update_activity   (prior_var_rec / post_var_rec ratio)
  └─ coverage_miscalibration  (optional; requires Forecasts/ data)
       │
       ├── all OK → exit (no API cost)
       │
       └── any WARN/FAIL
              │
              ▼
         agent.py            ← raw Claude API tool-use loop (claude-sonnet-5)
          while loop:
            ├─ tool: fetch_more_detail
            │    └─ per-location trajectory slices, day-by-day ratios, etc.
            └─ stop_reason == "end_turn" → structured diagnostic report
                   └─ required fields: confidence level, alternative
                      explanations, resolving evidence
```

The agent is invoked via the raw Anthropic Python SDK (`client.messages.create` with `tools=` and a `while` loop over `tool_use` / `tool_result` messages). No agent framework (LangChain, LangGraph, CrewAI) is used — the tool-use loop is plain Python so the full reasoning trajectory is inspectable in the message history and not abstracted away.

---

## Why an LLM

The deterministic checks produce structured numeric findings, but the *interpretation* of those findings requires domain context that is not straightforwardly encodable in rules:

1. **Contextual judgment.** Is a 16% late/early spread ratio pathological for this pipeline? It depends on epidemic phase, location-specific data quality, assimilation window, and how this run compares to others. A rule can threshold but cannot reason about alternatives.

2. **Multi-check synthesis.** When spread collapse and low Kalman update activity co-occur at the same location, the joint interpretation differs from either flag alone. Writing exhaustive rules for all combinations across 96 locations × multiple checks scales poorly.

3. **Calibrated uncertainty, not confident verdicts.** The most actionable diagnostic output is not "FAIL" but "this looks like filter overconfidence at location 30, confidence medium — an alternative explanation is that observations at this location were sparse in this period, which would produce the same signature; inspecting `obs_var_rec` at that location would resolve this." This requires reasoning under uncertainty in a way a rule-based system's binary severity codes cannot express. The system prompt enforces this structure on every agent response.

---

## Usage

### Prerequisites

```bash
pip install -r requirements.txt
# requires ANTHROPIC_API_KEY in environment for agent mode
```

### Inspect a new file's schema before running (recommended first time)

```bash
python run_diagnostics.py path/to/Model_Runs/0722_601_real_nf_n_pois.mat --inspect-only
```

### Run checks only (no API cost)

```bash
python run_diagnostics.py path/to/Model_Runs/0722_601_real_nf_n_pois.mat --skip-agent
```

### Full diagnostic with agent investigation

```bash
export ANTHROPIC_API_KEY=sk-...
python run_diagnostics.py path/to/Model_Runs/0722_601_real_nf_n_pois.mat
```

Optional: add `--coverage 0.83` if you have empirical coverage from `Forecasts/` data.

---

## Example output

### Clean run (checks only, no agent invoked)

```
Loaded run: 0722_601_real_nf_n_pois.mat
  n_days=437  n_ensemble=150  n_params=196

--- Deterministic check results ---
[OK   ] ensemble_spread_collapse_alpha: No alpha spread collapse (min ratio 20.5%).
[OK   ] ensemble_spread_collapse_beta: No beta spread collapse (min ratio 16.8%).
[OK   ] kalman_update_activity: Kalman update activity normal (mean ratio 0.XXX).
No issues flagged. Clean run -- agent not invoked (saves API cost).
```

### Flagged run (from dev deterministic runs, run 702)

```
Loaded run: 0128_702_real_f_n_det.mat
  n_days=437  n_ensemble=150  n_params=196

--- Deterministic check results ---
[WARN ] ensemble_spread_collapse_alpha: Possible alpha spread collapse:
        location 30 shrank to 8.1% of early-run spread.
[OK   ] ensemble_spread_collapse_beta: No beta spread collapse (min ratio 15.6%).
[OK   ] kalman_update_activity: Kalman update activity normal (mean ratio 0.XXX).
1 issue(s) flagged. Invoking diagnostic agent...

--- Diagnostic report ---
<!-- TODO: paste real agent report here once fetch_more_detail is wired to live data -->
```

*Note: `kalman_update_activity` mean ratio values shown as `0.XXX` — real values pending inspection of `prior_var_rec`/`post_var_rec` across the reference run set. This section will be updated once the agent tool-wiring is complete.*

---

## Evaluation

**What has been validated:**

- Extraction layer (`extract.py`) runs without error on all 8 available real-data runs: 601–604 (Poisson, production model) and 701–704 (deterministic, dev model, dropped variant). Schema confirmed against the 601 file via `inspect_file_schema()`.
- All 8 runs produce check output without errors. Collapse ratios are internally consistent: 10–26% range across runs, with run 702 (location 30, alpha) as the only outlier below the `warn_ratio=0.10` threshold.
- Deterministic check mechanics verified against synthetic dummy arrays (see `tests/test_checks_dummy.py`) — each check produces the correct severity direction on obviously healthy and obviously pathological inputs.

**What has NOT been validated:**

- `fetch_more_detail` tool in `agent.py` currently returns stub JSON — agent reasoning has not yet been evaluated on real flagged runs.
- `check_kalman_update_activity` thresholds are placeholders. Real `prior_var_rec`/`post_var_rec` values have not yet been inspected to inform what "over-aggressive" or "under-updating" looks like for this specific pipeline.
- No formally "known-bad" run exists in the production Poisson set (601–604): all four runs cluster tightly in behavior. The 702 WARN is from a dropped model variant and is used as a development test case only.

This is an honest, narrow evaluation scope — not a benchmark. The tool is in active development.

---

## Known limitations

**1. Kalman activity thresholds are uncalibrated.**
`over_aggressive_threshold=0.05` and `under_updating_threshold=0.98` in `check_kalman_update_activity` are placeholder values set to make the check logic exercisable. They have no empirical basis yet. Do not interpret OK/WARN on this check as meaningful until calibration against real `prior_var_rec`/`post_var_rec` values is done.

**2. Agent tool (`fetch_more_detail`) is stubbed.**
The agent can call `fetch_more_detail` to pull per-location trajectory slices or day-by-day values, but the tool currently returns a stub response rather than real data. Agent reports generated in the current state are based only on the check summary text, not on deeper investigation of the run internals.

**3. Coverage check requires manual input.**
`check_coverage_miscalibration` takes a scalar empirical coverage value from `Forecasts/` data. It does not yet read `Forecasts/*_fore_res_group.mat` directly — that wiring is planned for the next development step.

**4. Schema confirmed against one file (601).**
`EXPECTED_FIELDS` in `extract.py` was validated against `0722_601_real_nf_n_pois.mat`. Files 602–604 and 701–704 run without error, which confirms field names are shared, but axis order and dtype assumptions beyond the six confirmed fields have not been independently verified for each run.

---

## Failure cases

### Clipping check: zero-information by design

An initial check, `check_parameter_clipping`, measured the fraction of ensemble-days where a parameter value sat exactly at its `paramin`/`paramax` bound — under the assumption that `checkbound_para.m` enforced bounds by clipping violating members to the boundary value.

After running this check against all 8 real-data runs and observing exactly `0.0%` clipping in every case, the MATLAB source (`checkbound_para.m`) was read directly. The function does not clip — it **resamples** out-of-bound members to a fresh random value strictly inside `[mina, maxa]`, computed from the current ensemble spread. By construction, no value can equal `paramin` or `paramax` exactly after a resample event, making the exact-bound check structurally incapable of firing.

**Fix:** the check was replaced with `check_kalman_update_activity`, which uses `prior_var_rec`/`post_var_rec` fields already written to the `.mat` file by the pipeline, measuring how aggressively the filter reduces ensemble variance per update step. This required no changes to the MATLAB model code.

The original check was removed in commit `[TODO: add commit hash]`.

---

## Relationship to parent repo

This repo contains no model code from `NA_SEIR-EAKF_forecast`. It reads only the output `.mat` files that the parent pipeline writes to `Model_Runs/`. The parent pipeline's MATLAB scripts (`model_forecast_run.m`, `checkbound_para.m`, `checkbound_yesterday.m`, etc.) are not copied here and are not a dependency.

The diagnostic agent is a standalone post-processing tool, not a component of the assimilation loop. Running it does not modify any parent pipeline output files.

Parent repo: [`NA_SEIR-EAKF_forecast`](https://github.com/matteoperini/NA_SEIR-EAKF_forecast) — see that repo's README for full pipeline documentation, data sources, and MATLAB dependencies.
