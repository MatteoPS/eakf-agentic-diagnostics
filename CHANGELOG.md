# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-23

First tagged release.

### Added
- HDF5/v7.3 `.mat` loader for `NA_SEIR-EAKF_forecast` model runs, with schema validation (`extract.py`).
- Deterministic health checks: ensemble spread collapse (alpha, beta), Kalman update activity, coverage miscalibration (`checks.py`).
- Claude tool-use agent invoked only when a check is flagged; `fetch_more_detail` tool serves real run data (`agent.py`).
- Truncation detection when a response hits `--max-tokens`.
- Full agent transcript (prompts, tool calls, tool results) returned and saved as `docs/{run_id}_transcript.json`.
- `--fail-on {never,warn,fail}` CLI flag for non-zero exit codes in batch pipelines.
- Agent-loop tests with a scripted fake client (no network or API key needed).
- `pyproject.toml` packaging, MIT license, GitHub Actions CI (ruff + pytest, Python 3.10–3.13) and tag-triggered release workflow.
- Example agent reports for runs 601–604 and 702/704 in `docs/`.

### Removed
- Parameter clipping check: the MATLAB pipeline resamples out-of-bound members rather than clipping, so it could never fire (see README, *Design decisions*).

[0.1.0]: https://github.com/MatteoPS/eakf-agentic-diagnostics/releases/tag/v0.1.0
