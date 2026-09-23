"""Offline test of the Google ADK adapter.

Drives ``CpotPlugin``'s before/after-model callbacks with fake ``LlmResponse``
objects (a two-tool loop followed by a final answer) and checks the recorded CPOT
trace. This proves the adapter wiring is correct **without** needing ``google-adk``
installed or a Gemini API key.

Run with:  ``pytest``
"""
from __future__ import annotations

import asyncio

import pytest

import dsmeter as ds
from dsmeter.integrations.adk import CpotPlugin


# --- fake google.genai.types-shaped objects ------------------------------- #
class UM:
    def __init__(self, out, inp=0, thoughts=0):
        self.candidates_token_count = out
        self.prompt_token_count = inp
        self.thoughts_token_count = thoughts


class FC:
    def __init__(self, name):
        self.name = name
        self.id = None
        self.args = {}


class Part:
    def __init__(self, text=None, function_call=None):
        self.text = text
        self.function_call = function_call


class Content:
    def __init__(self, parts):
        self.parts = parts
        self.role = "model"


class LlmResponse:
    def __init__(self, out, tool=None):
        self.usage_metadata = UM(out)
        self.model = "gemini-2.0-flash"
        self.content = Content(
            [Part(function_call=FC(tool))] if tool else [Part(text="DELAY: storm + 40m traffic.")]
        )


class CBCtx:
    def __init__(self, invocation_id, agent_name):
        self.invocation_id = invocation_id
        self.agent_name = agent_name


async def _drive_two_tool_loop():
    ds.init(run_id="adk-sim")
    plugin = CpotPlugin()
    ctx = CBCtx("inv1", "ops_agent")

    # call 1: model asks for get_weather
    await plugin.before_model_callback(callback_context=ctx, llm_request=None)
    await plugin.after_model_callback(callback_context=ctx, llm_response=LlmResponse(60, tool="get_weather"))
    # call 2: model asks for get_traffic
    await plugin.before_model_callback(callback_context=ctx, llm_request=None)
    await plugin.after_model_callback(callback_context=ctx, llm_response=LlmResponse(50, tool="get_traffic"))
    # call 3: final answer (no tool) -> action sink
    await plugin.before_model_callback(callback_context=ctx, llm_request=None)
    await plugin.after_model_callback(callback_context=ctx, llm_response=LlmResponse(40))

    report = ds.analyze(list(ds.recorder().events), alpha=0.0)
    gens = [e for e in ds.recorder().events if e.get("kind") != "tool"]
    return report, gens


def test_adk_adapter_serial_tool_loop():
    report, gens = asyncio.run(_drive_two_tool_loop())

    assert report.T == pytest.approx(150), "CPOT should be 60+50+40=150"
    assert report.n_steps == 1, "one invocation must map to one CPOT step"

    action_sinks = [g for g in gens if g["is_action"]]
    assert len(action_sinks) == 1, "exactly one action sink (the final tool-free answer)"
    assert action_sinks[0]["out_tokens"] == 40, "the final answer must be the sole action sink"

    # the tool loop must form a serial ambient chain: gen2<-gen1, gen3<-gen2
    assert gens[1]["parent_gen_ids"] == [gens[0]["gen_id"]], "ambient chain broken at call 2"
    assert gens[2]["parent_gen_ids"] == [gens[1]["gen_id"]], "ambient chain broken at call 3"
