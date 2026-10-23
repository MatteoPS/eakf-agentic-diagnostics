# eakf-agentic-diagnostics

[![CI](https://github.com/MatteoPS/eakf-agentic-diagnostics/actions/workflows/ci.yml/badge.svg)](https://github.com/MatteoPS/eakf-agentic-diagnostics/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/MatteoPS/eakf-agentic-diagnostics/graph/badge.svg)](https://codecov.io/gh/MatteoPS/eakf-agentic-diagnostics)
[![Release](https://img.shields.io/github/v/release/MatteoPS/eakf-agentic-diagnostics)](https://github.com/MatteoPS/eakf-agentic-diagnostics/releases)
[![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Claude API](https://img.shields.io/badge/Claude%20API-tool%20use-D97757)](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview)

> Cheap deterministic checks decide *whether* something is wrong; a Claude tool-use agent investigates *why*, pulling real ensemble data before it commits to a calibrated diagnosis.

A Claude-powered diagnostic agent for [`NA_SEIR-EAKF_forecast`](https://github.com/MatteoPS/NA_SEIR-EAKF_forecast), a continental-scale SEIR epidemic model with Ensemble Adjustment Kalman Filter (EAKF) data assimilation across 96 North American locations.

---

## What it does

Each model run produces 437 days × 150 ensemble members × 196 parameters of posterior state, written to a `.mat` file. After a batch of runs, the current practice is to visually scan plots of ensemble spread and parameter trajectories across 96 locations to catch anomalies — slow, inconsistent, and silent about mechanism.

This tool reads the ensemble internals directly and checks whether the **estimation process** behaved correctly — not whether the forecast was accurate (the parent pipeline already computes MAE/WIS), but whether the EAKF dynamics were healthy: adequate ensemble spread, well-calibrated per-step updates, no filter pathologies. Deterministic numeric checks run first at zero API cost; Claude is only invoked when something is flagged, and it investigates using real run data rather than just rephrasing the check output. Reports are saved to `docs/` as markdown.

### Example output

Excerpt from a real agent report on run 602 ([full report](docs/0722_602_real_f_n_pois_report.md)):

```markdown
## Flagged: kalman_update_activity
Locations 58, 61, 68, 94 exceed 3.0 SD in post/prior update ratio (mean 0.244).

## Evidence gathered
- per_location_kalman_ratios: flagged locs (NWT 0.61, Nunavut 0.58, PEI 0.59, Yukon 0.62) roughly 2x overall mean.
- per_location_collapse_ratios: same 4 locs show alpha/beta collapse ratios ~0.75–1.06, far above typical 0.3–0.6 range.

## Most likely explanation [Confidence: medium]
- Locs 58/61/68/94 = smallest-population Canadian territories (NWT, Nunavut, PEI, Yukon).
- Sparse/noisy case counts → wide prior spread relative to obs → large EAKF corrections each step.
```

---

## Repo structure

```
run_diagnostics.py            entry point / CLI
src/eakf_diagnostics/
  extract.py                  .mat (HDF5/v7.3) loader → numpy arrays
  checks.py                   deterministic, non-LLM health checks
  agent.py                    Claude API tool-use loop
  __init__.py
tests/
  test_checks_dummy.py        checks.py mechanics verified on synthetic arrays
  test_agent_loop.py          agent loop + tool executor, scripted fake client
docs/
  *_report.md                 agent reports from real flagged runs
.github/workflows/            CI (ruff + pytest) and tag-triggered release
pyproject.toml / requirements.txt
CHANGELOG.md
```

---

## Architecture

```
Model_Runs/*.mat
       │
       ▼
 extract.py          ← h5py loader; schema confirmed against real 601-604 files
       │
       ▼
 checks.py           ← deterministic checks (no LLM); returns CheckResult list
  ├─ ensemble_spread_collapse_alpha   late/early std ratio per location
  ├─ ensemble_spread_collapse_beta    same for transmission rate
  ├─ kalman_update_activity           post_var/prior_var ratio per location/day
  └─ coverage_miscalibration          optional; requires Forecasts/ data
       │
       ├── all OK → exit cleanly (zero API cost)
       │
       └── any WARN/FAIL
              │
              ▼
         agent.py            ← raw Claude API tool-use loop (claude-sonnet-5)
          while loop:
            ├─ tool: fetch_more_detail → real numpy slices from ModelRun
            │    fields: beta/alpha trajectories, Kalman ratio timeseries,
            │             per-location collapse/update summaries
            └─ stop → structured markdown report saved to docs/
```

---

## Why an LLM

The deterministic checks flag anomalies, but interpreting them requires context that rules alone can't encode:

1. **Judgment over thresholds.** Is a 16% late/early spread ratio pathological? It depends on epidemic phase, location-specific data quality, and how this run compares to others. A rule can threshold but cannot reason about alternatives.

2. **Multi-check synthesis.** When spread collapse and low Kalman update activity co-occur at the same locations, the joint interpretation differs from either flag alone. Writing exhaustive rules for all combinations across 96 locations scales poorly.

3. **Calibrated uncertainty, not verdicts.** The most actionable output is not "FAIL" but: *"this looks like filter overconfidence at Nunavut, confidence medium — an alternative is data sparsity from low COVID case counts; checking coverage by location would resolve this."* This requires reasoning under uncertainty that binary severity codes cannot express. The system prompt enforces this structure on every report.

---

## Claude API usage

`agent.py` calls `client.messages.create(model="claude-sonnet-5", tools=..., ...)` inside a plain `while` loop over `tool_use` / `tool_result` messages. No agent framework (LangChain, LangGraph, CrewAI) is used — the full reasoning trajectory is visible in the message history and not abstracted away.

The tool `fetch_more_detail` is wired to real run data — it lets Claude pull trajectory slices and per-location Kalman ratio timeseries from the actual ensemble before writing its report. See [`docs/`](docs/) for real agent output from a flagged run.

---

## Agent engineering

What keeps the agent cheap, bounded and auditable:

- **Deterministic gate.** No LLM call unless a numeric check flags WARN/FAIL. Clean runs cost $0.
- **Bounded loop.** `max_turns` caps tool-use rounds; the result says `max_turns_exceeded` instead of looping or silently returning nothing.
- **Truncation detection.** A response that stops on `max_tokens` is reported as `truncated`, not passed off as a finished report.
- **Tools return summaries, not arrays.** `fetch_more_detail` does the numpy work and hands Claude structured numbers per location/day, keeping context small and numbers exact.
- **Enum-constrained tool input.** The `field` parameter is a closed enum, and tool errors come back as JSON the agent can read and recover from.
- **Enforced output contract.** The system prompt fixes the report sections, including confidence, alternatives, and what evidence would resolve ambiguity.
- **Full transcript.** Every prompt, tool call and tool result is saved to `docs/{run_id}_transcript.json`, so any conclusion can be traced back to the data it saw.
- **Tested without the API.** `tests/test_agent_loop.py` drives the loop with a scripted fake client (via the `client=` parameter) covering tool round-trips, truncation, turn limits and the no-flag short-circuit.

---

## Usage

```bash
pip install -e .            # or: pip install -r requirements.txt

# .env file with ANTHROPIC_API_KEY=sk-ant-...
# (key never goes in shell history)

# inspect a new file's schema before trusting the loader
python run_diagnostics.py path/to/run.mat --inspect-only

# deterministic checks only, no API cost
python run_diagnostics.py path/to/run.mat --skip-agent

# full run: checks + agent investigation if anything is flagged
python run_diagnostics.py path/to/run.mat \
  --statecodes path/to/statecodes.csv

# report saved to docs/{run_id}_report.md, transcript to docs/{run_id}_transcript.json

# gate a batch pipeline: exit 1 if any check reaches FAIL (or use warn)
python run_diagnostics.py path/to/run.mat --skip-agent --fail-on fail
```

---

## Design decisions

**Why the parameter clipping check was removed.**
An initial check in `checks.py` measured what fraction of ensemble members were pinned exactly at their `paramin`/`paramax` bounds — on the assumption that `checkbound_para.m` enforced bounds by clipping violating members to the boundary. After running it on all 8 available runs and observing exactly `0.0%` clipping in every case, the MATLAB source was read directly: `checkbound_para.m` *resamples* out-of-bound members to a fresh random value in the interior, making an exact-at-boundary observation structurally impossible. The check was removed and replaced with `check_kalman_update_activity()`, which uses `prior_var_rec`/`post_var_rec` fields already written to the `.mat` output — no MATLAB changes needed.

This is documented here rather than in code comments because it directly shapes what the tool does and does not measure.

---

## Evaluation

**Validated:**
- Extraction layer runs without error on 8 real-data runs (601–604 Poisson production, 701–704 deterministic dev). Schema confirmed against the 601 file via `inspect_file_schema()`.
- `check_ensemble_spread_collapse` thresholds informed by empirical range across these 8 runs: 10–26% late/early spread ratio. Run 702 (location 30, alpha, 8.1%) is the only outlier below the `warn_ratio=0.10` threshold.
- `check_kalman_update_activity` flagged four small Canadian territories (Northwest Territories, Nunavut, PEI, Yukon) in run 602 with update ratios 2.5× the cross-location mean. The agent investigated, cross-checked the day-by-day ratio timeseries at the worst location, and produced a medium-confidence explanation — see [`docs/0722_602_real_f_n_pois_report.md`](docs/0722_602_real_f_n_pois_report.md).

**Not yet validated:**
- `check_kalman_update_activity` thresholds are placeholders calibrated to not fire on healthy runs, but the failure thresholds have not been confirmed against a known-pathological run.
- `coverage_miscalibration` requires `Forecasts/` data not yet integrated.
- Agent `fetch_more_detail` has been tested on one real flagged case (run 602); generalization to other failure modes is untested.

---

## Relationship to parent repo

This repo contains no model code. It reads only the `.mat` output files that `NA_SEIR-EAKF_forecast` writes to `Model_Runs/`. Running this tool does not modify any parent pipeline output.

Parent repo: [`NA_SEIR-EAKF_forecast`](https://github.com/MatteoPS/NA_SEIR-EAKF_forecast) — see that repo for full pipeline documentation, MATLAB dependencies, and data sources.

---

## Development

```bash
pip install -e ".[dev]"
ruff check .
pytest -v          # no API key or .mat data needed
```

CI runs both on Python 3.10–3.13 for every push and PR. Pushing a `v*` tag runs the tests, builds the package and publishes a GitHub Release with notes from [`CHANGELOG.md`](CHANGELOG.md).

## License

[MIT](LICENSE)
