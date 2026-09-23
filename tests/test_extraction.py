"""Provider-agnostic response extraction + cross-boundary cause propagation.

These lock down the two pieces most likely to break silently:
  * pulling output tokens / tool-call ids out of OpenAI-, Anthropic-, and
    Gemini-shaped responses, and
  * carrying the ambient cause across a process/queue boundary via inject/extract.
"""
from __future__ import annotations

import dsmeter as ds


def test_openai_shaped_response_extraction():
    """dict shaped like an OpenAI chat completion -> out_tokens + tool_call id."""
    ds.init(run_id="openai")
    resp = {
        "model": "gpt-4o",
        "usage": {"completion_tokens": 42, "prompt_tokens": 10},
        "choices": [
            {"message": {"content": "hi", "tool_calls": [{"id": "call_abc", "function": {"name": "search"}}]}}
        ],
    }
    with ds.generation(agent="a") as g:
        g.record(resp)
    ev = ds.recorder().events[-1]
    assert ev["out_tokens"] == 42
    assert ev["in_tokens"] == 10
    assert ev["model"] == "gpt-4o"
    assert "call_abc" in ev["issued_tool_calls"]
    assert ev["tokens_estimated"] is False


def test_anthropic_shaped_response_extraction():
    """dict shaped like an Anthropic message -> out_tokens + tool_use id."""
    ds.init(run_id="anthropic")
    resp = {
        "model": "claude-sonnet-5",
        "usage": {"output_tokens": 77, "input_tokens": 20},
        "content": [
            {"type": "text", "text": "let me check"},
            {"type": "tool_use", "id": "toolu_1", "name": "lookup"},
        ],
    }
    with ds.generation(agent="a") as g:
        g.record(resp)
    ev = ds.recorder().events[-1]
    assert ev["out_tokens"] == 77
    assert ev["in_tokens"] == 20
    assert "toolu_1" in ev["issued_tool_calls"]


def test_token_estimation_fallback_when_provider_gives_nothing():
    """No usage block -> fall back to a ~4-chars-per-token estimate, flagged as estimated."""
    ds.init(run_id="estimate")
    with ds.generation(agent="a") as g:
        g.record(text="a" * 40)          # 40 chars -> ~10 tokens
    ev = ds.recorder().events[-1]
    assert ev["out_tokens"] == 10
    assert ev["tokens_estimated"] is True


def test_inject_extract_round_trip_carries_cause_across_boundary():
    """A worker that only receives an envelope still links back to the issuing gen."""
    ds.init(run_id="xproc")
    with ds.generation(agent="orch") as o:
        o.record(out_tokens=10)
    carrier = ds.inject()                 # serialize the ambient cause into headers
    assert carrier.get("cpot-cause") == o.id

    parent = ds.extract(carrier)          # ... on the far side of a queue/process
    with ds.generation(agent="worker", caused_by=parent) as w:
        w.record(out_tokens=5, is_action=True)

    ev = ds.recorder().events[-1]
    assert ev["parent_gen_ids"] == [o.id]
    assert ev["edge_source"][o.id] == "envelope"


def test_action_tool_marks_sink():
    """Declaring an action tool turns the generation that issues it into a DAG sink."""
    ds.init(run_id="action")
    ds.set_action_tools({"submit_decision"})
    with ds.generation(agent="a") as g:
        g.record(out_tokens=15, tool_calls=["submit_decision"])
    ev = ds.recorder().events[-1]
    assert ev["is_action"] is True
