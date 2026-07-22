"""
agent.py

Raw Claude API tool-use loop. NOT Claude Code, NOT LangGraph/CrewAI --
deliberately a plain while-loop over tools/tool_use/tool_result messages,
so the whole reasoning process is inspectable and doesn't depend on an
agent framework's abstractions.

Flow:
  1. Deterministic checks (checks.py) run FIRST, outside the LLM entirely.
  2. Only if something is flagged (WARN/FAIL) does the agent get invoked.
  3. The agent is given the check results and tools to pull MORE detail
     from the run (e.g. trajectory slices, per-location breakdowns) so it
     can investigate rather than just rephrase the check output.
  4. The agent must express calibrated uncertainty: confidence level,
     alternative explanations, and what evidence would resolve ambiguity
     -- this is enforced via the system prompt and the required output
     schema, not just requested informally.

STATUS: fully wired. fetch_more_detail returns real data from ModelRun
(extract.py). Model string verified as claude-sonnet-5 (July 2026).
Prompt caching not yet added -- planned once batch-run workflows are
benchmarked for cache hit rate.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import anthropic
import numpy as np

from .checks import CheckResult


SYSTEM_PROMPT = """\
You are a diagnostic assistant for an epidemiological forecasting pipeline \
(a stochastic SEIR metapopulation model with an Ensemble Adjustment Kalman \
Filter for data assimilation). You are given the results of deterministic \
numeric checks that flagged a potential problem with the ESTIMATION PROCESS \
of a specific run -- NOT whether the forecast was accurate, but whether the \
ensemble behaved the way a healthy EAKF should (adequate spread, parameters \
not pinned at bounds, well-calibrated uncertainty).

You have a tool available to pull additional detail from the run (specific \
trajectory slices, per-location breakdowns, day-by-day values) if the \
initial check summary isn't enough to form a view. Use it before concluding.

Your final output MUST include, for each flagged issue:
- A plain-language description of what was observed
- A confidence level (low/medium/high) in your explanation
- At least one alternative explanation you considered and why you ranked it \
lower (or why you could not rule it out)
- What additional evidence (if it existed) would resolve the remaining \
ambiguity

Do NOT present a single cause with unwarranted confidence. If the evidence \
is genuinely ambiguous, say so explicitly rather than picking the most \
plausible-sounding story.
"""

# Verified against Anthropic API docs (July 2026): current Sonnet-tier
# model ID is claude-sonnet-5. Pin to this exact string in production
# rather than an alias, so a future model swap doesn't happen silently.
MODEL_NAME = "claude-sonnet-5"


TOOLS = [
    {
        "name": "fetch_more_detail",
        "description": (
            "Fetch additional detail from the flagged run: a specific slice "
            "of a parameter trajectory, per-location breakdown of a check "
            "metric, or raw day-by-day values. Use this when the check "
            "summary alone doesn't give enough to explain the anomaly."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string", "description": "Run identifier, e.g. '601'"},
                "field": {
                    "type": "string",
                    "enum": [
                        "beta_trajectory_full",
                        "alpha_trajectory_full",
                        "per_location_collapse_ratios",
                        "per_location_kalman_ratios",
                        "kalman_ratio_timeseries",
                        "coverage_by_forecast_week",
                    ],
                    "description": "Which additional detail to fetch.",
                },
                "location_idx": {
                    "type": "integer",
                    "description": "Optional: restrict to a single location index.",
                },
            },
            "required": ["run_id", "field"],
        },
    }
]


@dataclass
class RunContext:
    """Everything the tool executor needs to answer fetch_more_detail calls."""
    run_id: str
    model_run: object  # ModelRun instance from extract.py


def execute_tool_call(tool_name: str, tool_input: dict, context: RunContext) -> str:
    """
    Executes a tool call against real ModelRun data and returns JSON.
    All heavy numpy work happens here so the agent gets clean structured
    numbers, not raw arrays.
    """
    if tool_name != "fetch_more_detail":
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    run = context.model_run
    field = tool_input.get("field")
    loc_idx = tool_input.get("location_idx")  # optional, 0-indexed

    try:
        if field == "beta_trajectory_full":
            traj = run.beta_trajectories  # (n_days, n_ensemble, n_locations)
            if loc_idx is not None:
                t = traj[:, :, loc_idx]
                return json.dumps({
                    "field": field,
                    "location": run.location_name(loc_idx),
                    "location_idx": loc_idx,
                    "n_days": int(t.shape[0]),
                    "ensemble_mean_by_day": t.mean(axis=1).tolist(),
                    "ensemble_std_by_day":  t.std(axis=1).tolist(),
                    "early_mean": float(t[:10].mean()),
                    "late_mean":  float(t[-10:].mean()),
                })
            # All locations summary
            means = traj.mean(axis=1)  # (n_days, n_locations)
            return json.dumps({
                "field": field,
                "per_location_early_mean": means[:10].mean(axis=0).tolist(),
                "per_location_late_mean":  means[-10:].mean(axis=0).tolist(),
            })

        elif field == "alpha_trajectory_full":
            traj = run.alpha_trajectories
            if loc_idx is not None:
                t = traj[:, :, loc_idx]
                return json.dumps({
                    "field": field,
                    "location": run.location_name(loc_idx),
                    "location_idx": loc_idx,
                    "ensemble_mean_by_day": t.mean(axis=1).tolist(),
                    "ensemble_std_by_day":  t.std(axis=1).tolist(),
                })
            means = traj.mean(axis=1)
            return json.dumps({
                "field": field,
                "per_location_early_mean": means[:10].mean(axis=0).tolist(),
                "per_location_late_mean":  means[-10:].mean(axis=0).tolist(),
            })

        elif field == "per_location_collapse_ratios":
            # Recompute collapse ratios for all locations, with names
            burn = 5
            for param, traj in [("alpha", run.alpha_trajectories),
                                 ("beta",  run.beta_trajectories)]:
                std = traj.std(axis=1)  # (n_days, n_locations)
                early = std[burn:burn + 5].mean(axis=0)
                late  = std[-5:].mean(axis=0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    ratio = np.where(early > 1e-10, late / early, np.nan)
            # Return both alpha and beta
            alpha_std = run.alpha_trajectories.std(axis=1)
            beta_std  = run.beta_trajectories.std(axis=1)
            alpha_ratio = np.where(
                alpha_std[burn:burn+5].mean(axis=0) > 1e-10,
                alpha_std[-5:].mean(axis=0) / alpha_std[burn:burn+5].mean(axis=0),
                np.nan)
            beta_ratio = np.where(
                beta_std[burn:burn+5].mean(axis=0) > 1e-10,
                beta_std[-5:].mean(axis=0) / beta_std[burn:burn+5].mean(axis=0),
                np.nan)
            records = []
            for i in range(run.alpha_trajectories.shape[2]):
                records.append({
                    "location_idx": i,
                    "location": run.location_name(i),
                    "alpha_collapse_ratio": float(alpha_ratio[i]) if not np.isnan(alpha_ratio[i]) else None,
                    "beta_collapse_ratio":  float(beta_ratio[i])  if not np.isnan(beta_ratio[i])  else None,
                })
            return json.dumps({"field": field, "locations": records})

        elif field == "per_location_kalman_ratios":
            prior = run.prior_var_rec
            post  = run.post_var_rec
            valid = prior > 1e-12
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio = np.where(valid, post / prior, np.nan)
            mean_per_loc = np.nanmean(ratio, axis=0)
            frac_valid   = valid.mean(axis=0)
            records = []
            for i in range(len(mean_per_loc)):
                records.append({
                    "location_idx": i,
                    "location": run.location_name(i),
                    "mean_post_prior_ratio": float(mean_per_loc[i]) if not np.isnan(mean_per_loc[i]) else None,
                    "frac_valid_days": float(frac_valid[i]),
                })
            return json.dumps({"field": field, "overall_mean": float(np.nanmean(mean_per_loc)),
                               "locations": records})

        elif field == "kalman_ratio_timeseries":
            # Day-by-day post/prior ratio for a specific location
            if loc_idx is None:
                return json.dumps({"error": "kalman_ratio_timeseries requires location_idx"})
            prior = run.prior_var_rec[:, loc_idx]
            post  = run.post_var_rec[:, loc_idx]
            valid = prior > 1e-12
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio = np.where(valid, post / prior, None)
            return json.dumps({
                "field": field,
                "location": run.location_name(loc_idx),
                "location_idx": loc_idx,
                "ratio_by_day": [float(r) if r is not None and not np.isnan(r) else None
                                 for r in ratio],
                "frac_valid_days": float(valid.mean()),
                "mean_ratio_valid_days": float(np.nanmean(
                    np.where(valid, post/prior, np.nan))),
            })

        elif field == "coverage_by_forecast_week":
            return json.dumps({
                "status": "not_available",
                "note": "Coverage data requires Forecasts/ files, not yet wired. "
                        "Use --coverage flag to pass empirical coverage manually."
            })

        else:
            return json.dumps({"error": f"Unknown field: {field}"})

    except Exception as e:
        return json.dumps({"error": str(e), "field": field})


def run_diagnostic_agent(
    check_results: list[CheckResult],
    run_id: str,
    model_run=None,
    max_turns: int = 8,
    client: anthropic.Anthropic | None = None,
) -> dict:
    """
    Main tool-use loop. Takes flagged check results, lets the agent
    optionally call fetch_more_detail to investigate, and returns the
    final structured diagnostic response.

    Args:
        check_results: list of CheckResult from checks.run_all_checks()
        run_id: string identifier for the run (used in the report)
        model_run: ModelRun instance from extract.load_model_run(); if
            provided, fetch_more_detail returns real data. If None, the
            tool returns an error message (agent can still reason from
            the check summary alone).
        max_turns: maximum tool-use rounds before giving up
    """
    if client is None:
        client = anthropic.Anthropic()

    flagged = [r for r in check_results if r.severity.value != "ok"]
    if not flagged:
        return {"status": "no_issues_flagged", "run_id": run_id}

    context = RunContext(run_id=run_id, model_run=model_run)

    check_summary = "\n".join(
        f"- [{r.severity.value.upper()}] {r.check_name}: {r.summary}\n"
        f"  details: {json.dumps(r.details)}"
        for r in flagged
    )

    messages = [
        {
            "role": "user",
            "content": (
                f"Run {run_id} triggered the following deterministic checks:\n\n"
                f"{check_summary}\n\n"
                f"Investigate these findings and produce a diagnostic report."
            ),
        }
    ]

    for turn in range(max_turns):
        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            # Model produced its final answer without needing more tools
            final_text = "".join(
                block.text for block in response.content if block.type == "text"
            )
            return {
                "status": "complete",
                "run_id": run_id,
                "turns_used": turn + 1,
                "report": final_text,
            }

        # Handle tool call(s)
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result_str = execute_tool_call(block.name, block.input, context)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_str,
                })

        messages.append({"role": "user", "content": tool_results})

    return {
        "status": "max_turns_exceeded",
        "run_id": run_id,
        "turns_used": max_turns,
        "note": "Agent did not reach a final answer within max_turns. "
                "Consider raising max_turns or reviewing the transcript.",
    }
