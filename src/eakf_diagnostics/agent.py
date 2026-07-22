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

STATUS: skeleton only. The `fetch_more_detail` tool currently returns a
stub -- it needs to be wired to a live ModelRun object (from extract.py)
once real .mat files are available to test against. Model string is
verified (claude-sonnet-5, checked July 2026); max_tokens and
prompt-caching cache_control blocks are still placeholders -- prompt
caching wiring is a TODO once real check-result payloads (which get
reused across a batch of runs) are available to cache against.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import anthropic

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
                        "per_location_clipping_fractions",
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
    """
    Everything the tool executor needs to actually answer fetch_more_detail
    calls for a given run. STUB: currently holds nothing real. Once
    extract.py can load real files, this should wrap a ModelRun instance
    (or the relevant precomputed arrays) so execute_tool_call can slice
    into real data instead of returning a placeholder.
    """
    run_id: str
    # model_run: ModelRun  # TODO: wire once real data available


def execute_tool_call(tool_name: str, tool_input: dict, context: RunContext) -> str:
    """
    Executes a tool call and returns the result as a string (JSON-encoded
    for structured data). STUB implementation -- wire to real ModelRun
    data once extract.py is validated against actual Model_Runs/*.mat files.
    """
    if tool_name == "fetch_more_detail":
        field = tool_input.get("field")
        # STUB: real implementation should index into context.model_run
        return json.dumps({
            "status": "stub",
            "note": (
                f"fetch_more_detail(field='{field}') not yet wired to real data. "
                f"This is a placeholder response for pipeline development before "
                f"real Model_Runs/*.mat files are available."
            ),
            "run_id": context.run_id,
        })

    return json.dumps({"error": f"Unknown tool: {tool_name}"})


def run_diagnostic_agent(
    check_results: list[CheckResult],
    run_id: str,
    max_turns: int = 6,
    client: anthropic.Anthropic | None = None,
) -> dict:
    """
    Main tool-use loop. Takes flagged check results, lets the agent
    optionally call fetch_more_detail to investigate, and returns the
    final structured diagnostic response.

    Only call this for runs where at least one check returned WARN/FAIL --
    don't waste API calls (and money, see project spending cap) on clean
    runs with nothing to investigate. Caller (main pipeline script) is
    responsible for that gating.
    """
    if client is None:
        client = anthropic.Anthropic()  # picks up ANTHROPIC_API_KEY from env

    flagged = [r for r in check_results if r.severity.value != "ok"]
    if not flagged:
        return {"status": "no_issues_flagged", "run_id": run_id}

    context = RunContext(run_id=run_id)

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
