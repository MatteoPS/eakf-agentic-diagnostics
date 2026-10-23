"""
test_agent_loop.py

Exercises the tool-use loop in agent.py with a scripted fake Anthropic
client -- no network, no API key. The fake returns canned responses in
order, so each test pins down one control-flow path: tool call then final
answer, truncation, running out of turns, and the no-flags short-circuit.

The tool executor runs against a real ModelRun built from small synthetic
arrays, so tool results are genuine numbers, not mocks.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from anthropic.types import TextBlock, ToolUseBlock

from eakf_diagnostics.agent import execute_tool_call, run_diagnostic_agent, RunContext
from eakf_diagnostics.checks import CheckResult, Severity
from eakf_diagnostics.extract import ModelRun

N_DAYS, N_ENS, N_LOC = 30, 10, 4


# ── Fakes ──────────────────────────────────────────────────────────────────

# real SDK block types, so transcript serialisation is tested against the
# same pydantic objects the live client returns
def text_block(text):
    return TextBlock(type="text", text=text)


def tool_use_block(block_id, field, location_idx=None):
    tool_input = {"run_id": "test", "field": field}
    if location_idx is not None:
        tool_input["location_idx"] = location_idx
    return ToolUseBlock(type="tool_use", id=block_id, name="fetch_more_detail", input=tool_input)


def response(stop_reason, *blocks):
    return SimpleNamespace(stop_reason=stop_reason, content=list(blocks))


class FakeClient:
    """Stands in for anthropic.Anthropic(); replays scripted responses and
    records every messages.create() call for inspection."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        # snapshot: the agent mutates the same list across turns
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return self._responses.pop(0)


def make_run():
    rng = np.random.default_rng(0)
    n_params = 2 * N_LOC
    para_post = rng.normal(1.0, 0.2, size=(N_DAYS, N_ENS, n_params))
    prior = np.abs(rng.normal(1.0, 0.1, size=(N_DAYS, N_LOC))) + 0.1
    return ModelRun(
        run_path=Path("test_run.mat"),
        para_post=para_post,
        alphamaps=np.arange(1, N_LOC + 1)[None, :],
        betamap=np.arange(N_LOC + 1, n_params + 1)[None, :],
        dailyIr_post_rec=np.zeros((N_DAYS, N_ENS, N_LOC)),
        S_post=np.zeros((N_ENS, N_DAYS, N_LOC)),
        paramin=np.zeros((1, n_params)),
        paramax=np.full((1, n_params), 10.0),
        prior_var_rec=prior,
        post_var_rec=prior * 0.5,
    )


FLAGGED = [CheckResult(check_name="kalman_update_activity", severity=Severity.WARN,
                       summary="2 locations are outliers", details={"outliers": [1, 3]})]


# ── Loop control flow ──────────────────────────────────────────────────────

def test_tool_call_then_final_answer():
    client = FakeClient([
        response("tool_use", text_block("Checking per-location ratios."),
                 tool_use_block("toolu_1", "per_location_kalman_ratios")),
        response("end_turn", text_block("## Flagged: kalman_update_activity\nreport body")),
    ])
    result = run_diagnostic_agent(FLAGGED, run_id="test", model_run=make_run(), client=client)

    assert result["status"] == "complete"
    assert result["turns_used"] == 2
    assert result["report"].startswith("## Flagged: kalman_update_activity")

    # second call must carry the tool result back, matched by id, with real data
    tool_msg = client.calls[1]["messages"][-1]
    assert tool_msg["role"] == "user"
    [tool_result] = tool_msg["content"]
    assert tool_result["tool_use_id"] == "toolu_1"
    payload = json.loads(tool_result["content"])
    assert len(payload["locations"]) == N_LOC
    assert payload["overall_mean"] == pytest.approx(0.5)

    # transcript: initial prompt, tool call, tool result, final answer
    transcript = result["transcript"]
    assert [m["role"] for m in transcript] == ["user", "assistant", "user", "assistant"]
    json.dumps(transcript)  # must be serialisable as-is


def test_truncated_response_is_reported():
    client = FakeClient([response("max_tokens", text_block("## Flagged: kalman_update_act"))])
    result = run_diagnostic_agent(FLAGGED, run_id="test", model_run=make_run(), client=client, max_tokens=16)

    assert result["status"] == "truncated"
    assert "max_tokens" in result["note"]
    assert client.calls[0]["max_tokens"] == 16


def test_max_turns_exceeded():
    client = FakeClient([
        response("tool_use", tool_use_block(f"toolu_{i}", "per_location_collapse_ratios"))
        for i in range(3)
    ])
    result = run_diagnostic_agent(FLAGGED, run_id="test", model_run=make_run(), client=client, max_turns=3)

    assert result["status"] == "max_turns_exceeded"
    assert len(client.calls) == 3
    assert len(result["transcript"]) == 7  # prompt + 3 x (tool call, tool result)


def test_no_flags_skips_llm():
    ok = [CheckResult(check_name="ensemble_spread_collapse_beta", severity=Severity.OK, summary="fine")]
    client = FakeClient([])
    result = run_diagnostic_agent(ok, run_id="test", client=client)

    assert result["status"] == "no_issues_flagged"
    assert client.calls == []


def test_invalid_max_tokens_rejected():
    with pytest.raises(ValueError):
        run_diagnostic_agent(FLAGGED, run_id="test", client=FakeClient([]), max_tokens=0)


# ── Tool executor ──────────────────────────────────────────────────────────

def test_timeseries_requires_location():
    ctx = RunContext(run_id="test", model_run=make_run())
    out = json.loads(execute_tool_call("fetch_more_detail", {"field": "kalman_ratio_timeseries"}, ctx))
    assert "error" in out


def test_timeseries_for_location():
    ctx = RunContext(run_id="test", model_run=make_run())
    out = json.loads(execute_tool_call(
        "fetch_more_detail", {"field": "kalman_ratio_timeseries", "location_idx": 2}, ctx))
    assert len(out["ratio_by_day"]) == N_DAYS
    assert out["mean_ratio_valid_days"] == pytest.approx(0.5)


def test_unknown_field_and_tool():
    ctx = RunContext(run_id="test", model_run=make_run())
    assert "error" in json.loads(execute_tool_call("fetch_more_detail", {"field": "nope"}, ctx))
    assert "error" in json.loads(execute_tool_call("other_tool", {}, ctx))
