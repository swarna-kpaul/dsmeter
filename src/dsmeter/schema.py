"""Canonical event schema for CPOT. Everything downstream is a pure function of these."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class GenerationEvent:
    """One LLM generation = one DAG node, weighted by out_tokens."""
    gen_id: str
    run_id: str = "run"
    parent_gen_ids: list = field(default_factory=list)   # LLM causes (fan-in => many)
    parent_tool_ids: list = field(default_factory=list)  # tool-result causes (resolved to issuing gen)
    agent_id: str | None = None
    step_index: int | None = None                        # env step -> per-step CPOT
    t_start: float = 0.0
    t_end: float = 0.0
    in_tokens: int = 0
    out_tokens: int = 0                                   # <-- NODE WEIGHT
    tokens_estimated: bool = False
    model: str | None = None
    issued_tool_calls: list = field(default_factory=list)  # tool_ids this gen emitted
    is_action: bool = False                               # committed action / final answer -> sink
    phase: str = "main"                                   # main | belief | role | reflection | summarization
    edge_source: dict = field(default_factory=dict)       # parent_id -> "explicit"|"object"|"envelope"|"tool"|"ambient"|"fuzzy"
    kind: str = "generation"

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class ToolEvent:
    """Optional record linking a tool_id back to the generation that issued it."""
    tool_id: str
    issued_by_gen: str | None = None
    t_start: float = 0.0
    t_end: float = 0.0
    result_ref: str | None = None
    kind: str = "tool"

    def to_json(self) -> dict:
        return asdict(self)
