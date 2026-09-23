"""dsmeter — measure Decision Span (DS) in any agentic framework.

Quick start:
    import dsmeter as ds
    ds.init(run_id="demo", out="trace.jsonl")
    with ds.generation(agent="a") as g:
        resp = client.chat.completions.create(...)
        g.record(resp)
    print(ds.analyze("trace.jsonl").summary())
"""
from __future__ import annotations

from contextlib import contextmanager

try:                                    # populated from package metadata when installed
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("dsmeter")
except Exception:                        # pragma: no cover - source checkout / not installed
    __version__ = "0.1.0"

from . import context as _ctx
from .context import inject, extract, current_cause, new_id
from .store import init as _store_init, recorder
from .capture import (
    generation, open_generation, close_generation, GenHandle,
    set_action_tools, mark_action,
)
from .engine import analyze, Report, load_events, span_step
from .schema import GenerationEvent, ToolEvent

__all__ = [
    "__version__",
    "init", "generation", "open_generation", "close_generation", "GenHandle",
    "cause", "step", "step_barrier", "agent", "current_cause", "inject", "extract",
    "set_action_tools", "mark_action", "analyze", "Report", "load_events", "span_step",
    "GenerationEvent", "ToolEvent", "new_id", "recorder", "wrap_openai", "wrap_anthropic",
]


def init(run_id: str = "run", out: str | None = None):
    """Start a CPOT recording session. `out` = JSONL path (None = in-memory only)."""
    r = _store_init(run_id, out)
    _ctx.set_run(run_id)
    _ctx._cause.set(None)          # fresh causal chain per run
    return r


@contextmanager
def cause(gen_id):
    """Explicitly set the ambient cause for an enclosed region."""
    tok = _ctx.set_cause(gen_id)
    try:
        yield
    finally:
        _ctx.reset_cause(tok)


@contextmanager
def step(index):
    """Group the generations inside this block into CPOT step `index`."""
    tok = _ctx._step.set(index)
    try:
        yield
    finally:
        try:
            _ctx._step.reset(tok)
        except Exception:
            pass


_step_counter = {"i": 0}


def step_barrier():
    """Advance to the next step (call at each env.advance()). Returns the new step index."""
    _step_counter["i"] += 1
    _ctx._step.set(_step_counter["i"])
    return _step_counter["i"]


@contextmanager
def agent(name):
    """Tag the generations inside this block with an agent id."""
    tok = _ctx._agent.set(name)
    try:
        yield
    finally:
        try:
            _ctx._agent.reset(tok)
        except Exception:
            pass


def wrap_openai(client):
    """Auto-trace an OpenAI client: every chat.completions.create becomes a generation."""
    orig = client.chat.completions.create

    def create(*a, **k):
        with generation() as g:
            resp = orig(*a, **k)
            g.record(resp)
            return resp

    client.chat.completions.create = create
    return client


def wrap_anthropic(client):
    """Auto-trace an Anthropic client: every messages.create becomes a generation."""
    orig = client.messages.create

    def create(*a, **k):
        with generation() as g:
            resp = orig(*a, **k)
            g.record(resp)
            return resp

    client.messages.create = create
    return client
