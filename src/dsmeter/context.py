"""Context propagation: the ambient 'cause' contextvar + cross-boundary inject/extract.

This is the plumbing that lets a child LLM call discover the id of the call that caused it
WITHOUT reading any model output. See cpot_library_plan.md section 3.
"""
from __future__ import annotations

import contextvars
import uuid

# The ambient "cause" = id of the last generation completed in this logical flow.
# contextvars propagate into asyncio tasks at creation time, so in-process spawns
# (orchestrator -> worker) inherit the cause for free.
_cause: contextvars.ContextVar = contextvars.ContextVar("cpot_cause", default=None)
_run: contextvars.ContextVar = contextvars.ContextVar("cpot_run", default=None)
_step: contextvars.ContextVar = contextvars.ContextVar("cpot_step", default=None)
_agent: contextvars.ContextVar = contextvars.ContextVar("cpot_agent", default=None)

CAUSE_KEY = "cpot-cause"   # header key used by inject/extract across boundaries
RUN_KEY = "cpot-run"


def new_id(prefix: str = "gen") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# --- ambient cause ---
def current_cause():
    return _cause.get()


def set_cause(gid):
    return _cause.set(gid)            # returns a Token


def reset_cause(token):
    try:
        _cause.reset(token)
    except Exception:
        pass


# --- run / step / agent ---
def current_run():
    return _run.get()


def set_run(rid):
    return _run.set(rid)


def current_step():
    return _step.get()


def current_agent():
    return _agent.get()


# --- cross-process propagation (source 3) ---
def inject(carrier: dict | None = None) -> dict:
    """Write the current cause + run id into a message envelope / headers dict."""
    carrier = {} if carrier is None else carrier
    c = _cause.get()
    if c is not None:
        carrier[CAUSE_KEY] = c
    r = _run.get()
    if r is not None:
        carrier[RUN_KEY] = r
    return carrier


def extract(carrier: dict | None):
    """Recover a cause gen-id from a received envelope. Pass the result to generation(caused_by=...)."""
    if not carrier:
        return None
    return carrier.get(CAUSE_KEY)
