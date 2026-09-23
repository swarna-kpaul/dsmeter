"""Measure CPOT in a Google ADK agent.

The ENTIRE integration is: register one plugin on the Runner. Everything else is a normal
ADK agent. The plugin records every model call as a CPOT generation, chains a tool loop into
a serial DAG via the ambient cause, and treats the final tool-free answer as the action sink.

Requirements:
    pip install "dsmeter[adk]"       # installs dsmeter + google-adk
    export GOOGLE_API_KEY=...         # or configure Vertex AI (GOOGLE_GENAI_USE_VERTEXAI=TRUE)

Run:
    python examples/adk_example.py
"""
from __future__ import annotations

import asyncio

import dsmeter as ds
from dsmeter.integrations.adk import CpotPlugin

from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.genai import types


# --- two tools, so the agent runs a multi-call tool loop (a real decision chain) ---
def get_weather(city: str) -> dict:
    """Return current weather for a city."""
    return {"city": city, "temp_c": 12, "condition": "thunderstorm"}


def get_traffic(city: str) -> dict:
    """Return current airport traffic delay in minutes for a city."""
    return {"city": city, "delay_min": 40}


root_agent = Agent(
    name="ops_agent",
    model="gemini-2.0-flash",
    instruction=(
        "You decide whether to delay a flight. First call get_weather and get_traffic, "
        "then answer with exactly 'DELAY' or 'GO' and one sentence of justification."
    ),
    tools=[get_weather, get_traffic],
)

APP = "cpot_demo"


async def main():
    # 1) start CPOT recording
    ds.init(run_id="adk-demo", out="adk_trace.jsonl")
    # (optional) if your agent commits via a specific tool, name it so it becomes the sink:
    # ds.set_action_tools({"submit_decision"})

    # 2) the whole integration: one plugin on the Runner
    runner = InMemoryRunner(agent=root_agent, app_name=APP, plugins=[CpotPlugin()])

    # 3) run one query (= one CPOT step)
    uid, sid = "u1", "s1"
    await runner.session_service.create_session(app_name=APP, user_id=uid, session_id=sid)
    msg = types.Content(
        role="user",
        parts=[types.Part(text="Should we delay the 6pm flight out of Boston?")],
    )
    async for event in runner.run_async(user_id=uid, session_id=sid, new_message=msg):
        if event.is_final_response() and event.content and event.content.parts:
            print("ANSWER:", event.content.parts[0].text)

    ds.recorder().close()

    # 4) analyze
    report = ds.analyze("adk_trace.jsonl", alpha=0.0)
    print("\n=== CPOT ===")
    print(report.summary())
    for si, s in report.per_step.items():
        print(f"  step {si}: CPOT={s['cpot']:.0f} out-tokens over {s['n']} generations "
              f"(wall {s['wall']:.2f}s)")


if __name__ == "__main__":
    asyncio.run(main())
