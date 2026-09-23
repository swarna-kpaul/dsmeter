"""Generation capture: the generation() context manager, open/close primitives, and
provider-agnostic extraction of output tokens / tool calls from a response object.

Nothing here reads model output to infer edges — edges come only from captured ids.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterable

from . import context as _ctx
from .schema import GenerationEvent, ToolEvent
from .store import recorder

_ACTION_TOOLS: set[str] = set()


def set_action_tools(names: Iterable[str]) -> None:
    """Declare which tool names commit an environment-mutating action (=> DAG sinks)."""
    _ACTION_TOOLS.update(names)


class _Tagged(str):
    """A str carrying provenance, so plain-text results work with prompt_from=."""
    def __new__(cls, value, gen_id):
        s = super().__new__(cls, value)
        s.__cpot_gen__ = gen_id
        return s


class GenHandle:
    def __init__(self, event: GenerationEvent):
        self.id = event.gen_id
        self.event = event
        self._closed = False
        self.output_text = None

    def record(self, response: Any = None, *, out_tokens=None, in_tokens=None,
               model=None, tool_calls=None, text=None, is_action=None, phase=None):
        """Fill the event from a provider response and/or explicit values. Does NOT emit."""
        ev = self.event
        if response is not None:
            u = _extract_usage(response)
            if u:
                if u.get("out") is not None:
                    ev.out_tokens = u["out"]
                if u.get("in") is not None:
                    ev.in_tokens = u["in"]
                ev.tokens_estimated = u.get("estimated", False)
            m = _extract_model(response)
            if m:
                ev.model = m
            for t in _extract_tool_calls(response):
                if t not in ev.issued_tool_calls:
                    ev.issued_tool_calls.append(t)
            txt = _extract_text(response)
            if txt is not None:
                self.output_text = txt
        if out_tokens is not None:
            ev.out_tokens = out_tokens
            ev.tokens_estimated = False
        if in_tokens is not None:
            ev.in_tokens = in_tokens
        if model is not None:
            ev.model = model
        if tool_calls is not None:
            for t in tool_calls:
                if t not in ev.issued_tool_calls:
                    ev.issued_tool_calls.append(t)
        if text is not None:
            self.output_text = text
        if is_action is not None:
            ev.is_action = is_action
        if phase is not None:
            ev.phase = phase
        # token fallback only if the provider gave us nothing
        if ev.out_tokens == 0 and self.output_text:
            ev.out_tokens = max(1, len(self.output_text) // 4)
            ev.tokens_estimated = True
        return self

    def as_result(self, obj):
        """Tag a result object with this generation's id, for downstream prompt_from=."""
        if isinstance(obj, str):
            return _Tagged(obj, self.id)
        try:
            setattr(obj, "__cpot_gen__", self.id)
        except Exception:
            pass
        return obj


def open_generation(*, agent=None, step=None, parents=None, caused_by=None,
                    caused_by_tool=None, prompt_from=None, phase="main",
                    is_action=False, model=None) -> GenHandle:
    """Mint a generation and resolve its parents by priority. Returns a handle; not yet emitted."""
    gid = _ctx.new_id()
    edge_source: dict = {}
    pgs: list = []
    ptools: list = []

    # priority: explicit > object-provenance > envelope > (tool below) > ambient.
    # parents=[] explicitly means "no parent" (a root off a step input) and skips ambient.
    if parents is not None:
        for p in parents:
            if p and p not in pgs:
                pgs.append(p)
                edge_source[p] = "explicit"
    elif prompt_from:
        for o in prompt_from:
            src = getattr(o, "__cpot_gen__", None)
            if src and src not in pgs:
                pgs.append(src)
                edge_source[src] = "object"
    elif caused_by:
        pgs = [caused_by]
        edge_source[caused_by] = "envelope"
    else:
        amb = _ctx.current_cause()
        if amb:
            pgs = [amb]
            edge_source[amb] = "ambient"

    if caused_by_tool:
        tl = caused_by_tool if isinstance(caused_by_tool, (list, tuple)) else [caused_by_tool]
        for t in tl:
            if t:
                ptools.append(t)
                edge_source[f"tool:{t}"] = "tool"

    ev = GenerationEvent(
        gen_id=gid,
        run_id=_ctx.current_run() or recorder().run_id,
        parent_gen_ids=pgs,
        parent_tool_ids=ptools,
        agent_id=agent if agent is not None else _ctx.current_agent(),
        step_index=step if step is not None else _ctx.current_step(),
        t_start=time.time(),
        phase=phase,
        is_action=is_action,
        model=model,
        edge_source=edge_source,
    )
    return GenHandle(ev)


def close_generation(handle: GenHandle, response=None, **record_kw) -> GenHandle:
    """Finalize, emit the event (+ ToolEvents for issued tools), and set ambient cause."""
    if handle._closed:
        return handle
    if response is not None or record_kw:
        handle.record(response, **record_kw)
    ev = handle.event
    ev.t_end = time.time()
    if not ev.is_action and _ACTION_TOOLS:
        if any(t in _ACTION_TOOLS for t in ev.issued_tool_calls):
            ev.is_action = True
    rec = recorder()
    for tid in ev.issued_tool_calls:
        rec.emit(ToolEvent(tool_id=tid, issued_by_gen=ev.gen_id,
                           t_start=ev.t_start, t_end=ev.t_end))
    rec.emit(ev)
    _ctx.set_cause(ev.gen_id)      # become the cause for the continuation
    handle._closed = True
    return handle


@contextmanager
def generation(**kw):
    """Context manager form. Call handle.record(resp) inside; parents resolved at entry."""
    h = open_generation(**kw)
    try:
        yield h
    finally:
        close_generation(h)


def mark_action(handle: GenHandle) -> GenHandle:
    handle.event.is_action = True
    return handle


# --------------------------------------------------------------------------- #
# provider-agnostic extraction (OpenAI / Anthropic / Gemini-ADK / dict)
# --------------------------------------------------------------------------- #
def _g(obj, *names):
    for n in names:
        if isinstance(obj, dict):
            if n in obj and obj[n] is not None:
                return obj[n]
        else:
            v = getattr(obj, n, None)
            if v is not None:
                return v
    return None


def _extract_usage(resp):
    # Gemini / ADK: usage_metadata (thoughts count as decode output)
    um = _g(resp, "usage_metadata")
    if um is not None:
        out = _g(um, "candidates_token_count") or 0
        thoughts = _g(um, "thoughts_token_count") or 0
        inp = _g(um, "prompt_token_count") or 0
        return {"out": int(out) + int(thoughts), "in": int(inp), "estimated": False}
    # OpenAI / Anthropic: usage
    u = _g(resp, "usage")
    if u is not None:
        out = _g(u, "output_tokens", "completion_tokens")
        inp = _g(u, "input_tokens", "prompt_tokens")
        if out is not None or inp is not None:
            return {"out": int(out or 0), "in": int(inp or 0), "estimated": False}
    return None


def _extract_model(resp):
    return _g(resp, "model", "model_version")


def _extract_text(resp):
    # OpenAI chat
    ch = _g(resp, "choices")
    if ch:
        msg = _g(ch[0], "message")
        if msg is not None:
            t = _g(msg, "content")
            if isinstance(t, str):
                return t
    # Anthropic: content = list of blocks with .text
    content = _g(resp, "content")
    if isinstance(content, list):
        parts = [_g(b, "text") for b in content]
        parts = [p for p in parts if p]
        if parts:
            return "".join(parts)
    # Gemini / ADK: content.parts[].text
    if content is not None and not isinstance(content, (list, str)):
        parts = _g(content, "parts")
        if parts:
            txts = [_g(p, "text") for p in parts]
            txts = [t for t in txts if t]
            if txts:
                return "".join(txts)
    if isinstance(resp, str):
        return resp
    return None


def _extract_tool_calls(resp):
    ids: list = []
    # OpenAI
    ch = _g(resp, "choices")
    if ch:
        msg = _g(ch[0], "message")
        tcs = _g(msg, "tool_calls") if msg is not None else None
        if tcs:
            for tc in tcs:
                tid = _g(tc, "id") or _g(_g(tc, "function") or {}, "name")
                if tid:
                    ids.append(str(tid))
            return ids
    # Anthropic tool_use blocks
    content = _g(resp, "content")
    if isinstance(content, list):
        for b in content:
            if _g(b, "type") == "tool_use":
                tid = _g(b, "id") or _g(b, "name")
                if tid:
                    ids.append(str(tid))
        if ids:
            return ids
    # Gemini / ADK function calls
    gfc = getattr(resp, "get_function_calls", None)
    fcs = None
    if callable(gfc):
        try:
            fcs = gfc()
        except Exception:
            fcs = None
    if not fcs:
        fcs = []
        if content is not None and not isinstance(content, (list, str)):
            parts = _g(content, "parts")
            if parts:
                for p in parts:
                    fc = _g(p, "function_call")
                    if fc is not None:
                        fcs.append(fc)
    for fc in (fcs or []):
        tid = _g(fc, "id") or _g(fc, "name")
        if tid:
            ids.append(str(tid))
    return ids
