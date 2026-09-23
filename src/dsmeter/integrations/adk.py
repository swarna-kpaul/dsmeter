"""Google ADK adapter — one Plugin that records a CPOT generation per model call.

Usage:
    import dsmeter as ds
    from dsmeter.integrations.adk import CpotPlugin
    ds.init(run_id="demo", out="trace.jsonl")
    runner = InMemoryRunner(agent=root_agent, app_name="app", plugins=[CpotPlugin()])

How parent ids are traced (no model output is read):
- Parent = the previous generation in the same invocation, via the ambient contextvar
  (ADK runs an agent's model calls in one async flow, so the cause propagates). A tool loop
  therefore forms a serial chain, and the final tool-free answer is the action sink.
- One ADK invocation (one user turn) = one CPOT step by default.
- For cross-agent transfers, the sub-agent's calls inherit the cause through the same
  contextvar; if a transfer crosses a task/process boundary, wrap the hand-off with
  ds.inject()/ds.extract().
"""
from __future__ import annotations

from .. import capture as _cap
from .. import context as _ctx

try:                                            # real ADK
    from google.adk.plugins.base_plugin import BasePlugin
except Exception:                               # pragma: no cover
    try:
        from google.adk.plugins import BasePlugin
    except Exception:
        class BasePlugin:                       # shim so the module imports without ADK installed
            def __init__(self, name: str = "cpot"):
                self.name = name


class CpotPlugin(BasePlugin):
    def __init__(self, name: str = "cpot", action_tools=None, step_per_invocation: bool = True):
        super().__init__(name=name)
        self._pending: dict = {}                # invocation key -> [GenHandle, ...]
        self._steps: dict = {}                  # invocation key -> step index
        self._step_seq = 0
        self._step_per_invocation = step_per_invocation
        if action_tools:
            _cap.set_action_tools(action_tools)

    @staticmethod
    def _key(cbctx):
        return getattr(cbctx, "invocation_id", None) or id(cbctx)

    async def before_model_callback(self, *, callback_context, llm_request):
        key = self._key(callback_context)
        if self._step_per_invocation and key not in self._steps:
            self._step_seq += 1
            self._steps[key] = self._step_seq
        step = self._steps.get(key)
        agent = getattr(callback_context, "agent_name", None)
        h = _cap.open_generation(agent=agent, step=step, phase="main")
        self._pending.setdefault(key, []).append(h)
        return None                             # proceed with the real LLM call

    async def after_model_callback(self, *, callback_context, llm_response):
        stack = self._pending.get(self._key(callback_context))
        if not stack:
            return None
        h = stack.pop()
        h.record(response=llm_response)         # pull usage_metadata + function calls
        # action sink = a response that issued NO further tool/function calls (the final answer)
        if not h.event.issued_tool_calls:
            h.event.is_action = True
        _cap.close_generation(h)                # emit + set ambient cause
        return None
