"""CPOT computation: build the per-step DAG, take the node-weighted longest path to the
action sinks, aggregate T / W / P, and validate the reconstruction.

CPOT is a node-weighted longest path (weight = output tokens), computed PER STEP and summed.
"""
from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass, field


def load_events(path):
    gens, tools = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            (tools if r.get("kind") == "tool" else gens).append(r)
    return gens, tools


def cost(g, alpha=0.0):
    return float(g.get("out_tokens", 0)) + alpha * float(g.get("in_tokens", 0))


def _wall(g):
    a, b = float(g.get("t_start", 0.0)), float(g.get("t_end", 0.0))
    return max(0.0, b - a)


def _index(gens, tools):
    by_id = {g["gen_id"]: g for g in gens}
    tool_issuer = {}
    for t in tools:
        if t.get("issued_by_gen"):
            tool_issuer[t["tool_id"]] = t["issued_by_gen"]
    return by_id, tool_issuer


def _preds(g, by_id, tool_issuer, within):
    ps = set()
    for p in g.get("parent_gen_ids", []):
        if p in by_id and p in within:
            ps.add(p)
    for tid in g.get("parent_tool_ids", []):
        src = tool_issuer.get(tid)
        if src in by_id and src in within:
            ps.add(src)
    return ps


def _topo(ids, preds):
    indeg = {i: 0 for i in ids}
    succ = defaultdict(list)
    for i in ids:
        for p in preds[i]:
            succ[p].append(i)
            indeg[i] += 1
    q = deque([i for i in ids if indeg[i] == 0])
    order = []
    while q:
        n = q.popleft()
        order.append(n)
        for m in succ[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                q.append(m)
    if len(order) != len(ids):        # cycle guard (shouldn't happen for a DAG)
        order += [i for i in ids if i not in set(order)]
    return order


def span_step(step_gens, by_id, tool_issuer, alpha=0.0, weight="tokens"):
    """Longest node-weighted path to the action sinks within one step."""
    ids = {g["gen_id"] for g in step_gens}
    preds = {g["gen_id"]: _preds(g, by_id, tool_issuer, ids) for g in step_gens}
    w = (lambda g: _wall(g)) if weight == "wall" else (lambda g: cost(g, alpha))
    finish = {}
    for gid in _topo(ids, preds):
        base = max((finish[p] for p in preds[gid] if p in finish), default=0.0)
        finish[gid] = w(by_id[gid]) + base
    sinks = [g["gen_id"] for g in step_gens if g.get("is_action")]
    if not sinks:
        sinks = [g["gen_id"] for g in step_gens]
    return max((finish[s] for s in sinks), default=0.0)


@dataclass
class Report:
    T: float
    W: float
    parallelism: float
    T_wall: float
    per_step: dict
    W_main: float
    W_learning: float
    n_gens: int
    n_steps: int
    edge_sources: dict
    warnings: list = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"CPOT decision time   T = {self.T:.0f} out-tokens   ({self.n_steps} steps, {self.n_gens} generations)",
            f"compute work         W = {self.W:.0f}   (main={self.W_main:.0f}, learning={self.W_learning:.0f})",
            f"parallelism        W/T = {self.parallelism:.2f}",
            f"wall span       T_wall = {self.T_wall:.3f} s",
            f"edge sources         = {self.edge_sources}",
        ]
        if self.warnings:
            lines.append("warnings:")
            lines += [f"  - {w}" for w in self.warnings]
        return "\n".join(lines)


def analyze(path_or_events, alpha=0.0) -> Report:
    if isinstance(path_or_events, str):
        gens, tools = load_events(path_or_events)
    else:
        evs = list(path_or_events)
        gens = [e for e in evs if e.get("kind") != "tool"]
        tools = [e for e in evs if e.get("kind") == "tool"]

    by_id, tool_issuer = _index(gens, tools)
    all_ids = set(by_id)

    steps = defaultdict(list)
    for g in gens:
        steps[g.get("step_index")].append(g)

    T = T_wall = 0.0
    per_step = {}
    for si in sorted(steps.keys(), key=lambda x: (x is None, x)):
        sg = steps[si]
        s_tok = span_step(sg, by_id, tool_issuer, alpha, "tokens")
        s_wall = span_step(sg, by_id, tool_issuer, alpha, "wall")
        per_step[si] = {"cpot": s_tok, "wall": s_wall, "n": len(sg)}
        T += s_tok
        T_wall += s_wall

    W = sum(cost(g, alpha) for g in gens)
    W_main = sum(cost(g, alpha) for g in gens if g.get("phase", "main") == "main")

    # edge-source histogram + validation
    srcs = defaultdict(int)
    orphans = 0
    for g in gens:
        for _, s in (g.get("edge_source") or {}).items():
            srcs[s] += 1
        if not g.get("parent_gen_ids") and not g.get("parent_tool_ids"):
            orphans += 1

    warnings = []
    fuzzy = srcs.get("fuzzy", 0)
    exact = sum(v for k, v in srcs.items() if k != "fuzzy")
    if (exact + fuzzy) and fuzzy / (exact + fuzzy) > 0.2:
        warnings.append(f"{fuzzy}/{exact + fuzzy} edges from fuzzy text-matching — low confidence.")
    if orphans > 1:
        warnings.append(f"{orphans} generations have no captured parent (roots / lost propagation).")

    # causality: a parent must finish before its child starts (only where timestamps exist)
    viol = 0
    for g in gens:
        for p in _preds(g, by_id, tool_issuer, all_ids):
            pe, cs = by_id[p].get("t_end", 0), g.get("t_start", 0)
            if pe and cs and pe > cs + 1e-6:
                viol += 1
    if viol:
        warnings.append(f"{viol} edges violate causality (parent ends after child starts) — provenance suspect.")

    P = (W / T) if T else float("nan")
    return Report(T=T, W=W, parallelism=P, T_wall=T_wall, per_step=per_step,
                  W_main=W_main, W_learning=W - W_main, n_gens=len(gens),
                  n_steps=len(steps), edge_sources=dict(srcs), warnings=warnings)
