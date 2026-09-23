# dsmeter

**Measure the Decision Span (DS) of any LLM agent system — the inherent,
hardware-independent *decision time* — by capturing the causal id of every LLM
call and taking the longest output-token-weighted path through the resulting DAG.**

[![CI](https://github.com/swarna-kpaul/dsmeter/actions/workflows/ci.yml/badge.svg)](https://github.com/swarna-kpaul/dsmeter/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

No wall-clock. No reading model output. `dsmeter` answers a question raw latency
can't: *how much serial "thinking" does your agent actually do to reach a
decision* — independent of GPUs, batching, or network jitter.

---

## Table of contents

- [Why Decision Span?](#why-decision-span)
- [Install](#install)
- [Quick start](#quick-start)
- [Core concepts](#core-concepts)
- [Usage guide](#usage-guide)
  - [1. Start a recording](#1-start-a-recording)
  - [2. Record generations](#2-record-generations)
  - [3. Group work into steps](#3-group-work-into-steps)
  - [4. Mark the action sink](#4-mark-the-action-sink)
  - [5. Tell dsmeter who caused what](#5-tell-dsmeter-who-caused-what)
  - [6. Cross-process / queue boundaries](#6-cross-process--queue-boundaries)
  - [7. Auto-trace a client](#7-auto-trace-a-client)
- [Framework integrations](#framework-integrations)
- [Analyzing a trace](#analyzing-a-trace)
- [The `Report` object](#the-report-object)
- [Trace file format](#trace-file-format)
- [API reference](#api-reference)
- [Development & testing](#development--testing)
- [Project layout](#project-layout)
- [License](#license)

---

## Why Decision Span?

Wall-clock latency of an agent conflates things you care about (how much serial
reasoning was required) with things you don't (which GPU ran it, how full the
batch was, network round-trips). **Decision Span (DS)** isolates the first.

`dsmeter` models an agent run as a DAG where:

- **each node is one LLM generation, weighted by its output tokens** —
  autoregressive decoding is irreducibly serial, so output tokens are the true
  unit of unavoidable "thinking time"; and
- **each edge `B ← A` means A's captured id was propagated to cause B** — a
  tool-call id, an ambient context variable, a message envelope, or object
  provenance — **never inferred from text.**

The **Decision Span** is the longest output-token-weighted path through that DAG
to the committed action, computed per step and summed. Parallel work that isn't
on the critical path doesn't inflate it — which is exactly what you want when
comparing agent designs.

---

## Install

```bash
pip install dsmeter                 # core (zero dependencies)

pip install "dsmeter[adk]"          # + Google ADK adapter
pip install "dsmeter[openai]"       # + OpenAI SDK
pip install "dsmeter[anthropic]"    # + Anthropic SDK
pip install "dsmeter[dev]"          # + pytest / coverage for development
```

From a local checkout:

```bash
pip install -e ".[dev]"
```

`dsmeter` requires **Python 3.9+** and has **no required runtime dependencies** —
the LLM SDKs are optional extras used only by the adapters.

---

## Quick start

```python
import dsmeter as ds

ds.init(run_id="demo", out="trace.jsonl")

with ds.step(0):
    with ds.generation(agent="planner") as g:
        resp = client.chat.completions.create(...)   # your normal LLM call
        g.record(resp)                               # pulls output tokens + tool calls
    # the next generation in this flow auto-links to g via the ambient cause
    with ds.generation(agent="planner") as g2:
        resp2 = client.chat.completions.create(...)
        g2.record(resp2, is_action=True)             # the committed answer = the sink

report = ds.analyze("trace.jsonl")
print(report.summary())
```

```
CPOT decision time   T = 180 out-tokens   (1 steps, 2 generations)
compute work         W = 180   (main=180, learning=0)
parallelism        W/T = 1.00
wall span       T_wall = 0.412 s
edge sources         = {'ambient': 1}
```

`g.record()` reads only the response's **usage metadata and tool-call ids** — it
never parses the generated text to decide anything.

---

## Core concepts

| Term | Symbol | Meaning |
|------|--------|---------|
| **Node weight** | — | A generation's **output tokens** (the serial decode cost). |
| **Edge `B ← A`** | — | A's captured id was propagated to cause B. Never text-inferred. |
| **CPOT / Decision Span** | `T` | Longest node-weighted path to the action sink, **per step, summed**. |
| **Work** | `W` | Total output tokens across *all* generations (on- and off-path). |
| **Parallelism** | `P = W/T` | How much compute ran concurrently off the critical path. `1.0` = fully serial. |
| **Wall span** | `T_wall` | Same longest-path computation but weighted by measured wall-clock, for comparison. |

A generation can be tagged with a **phase** (`main`, `belief`, `role`,
`reflection`, `summarization`). Off-path learning (e.g. a big `belief` update
that isn't on the path to the action) counts toward `W` but **not** toward the
decision span `T` — so background reasoning doesn't masquerade as decision time.

---

## Usage guide

### 1. Start a recording

```python
import dsmeter as ds

rec = ds.init(run_id="my-run", out="trace.jsonl")   # out=None => in-memory only
```

- `out` is a JSONL path; each event is appended and flushed as it happens.
- Pass `out=None` to keep everything in memory (analyze `ds.recorder().events`
  directly — handy for tests and notebooks).
- Call `ds.recorder().close()` when done to flush and close the file.

### 2. Record generations

The `generation()` context manager mints a node, resolves its parents **at
entry**, and emits the event on exit:

```python
with ds.generation(agent="researcher") as g:
    resp = client.chat.completions.create(model="gpt-4o", messages=[...])
    g.record(resp)          # provider-agnostic: OpenAI / Anthropic / Gemini-ADK / dict
```

`record()` accepts a provider response and/or explicit overrides:

```python
g.record(
    resp,                   # a provider response object or dict (optional)
    out_tokens=128,         # override / supply the node weight directly
    in_tokens=512,
    model="gpt-4o",
    tool_calls=["call_abc"],# ids this generation issued
    text="...",             # only used to estimate tokens if usage is missing
    is_action=True,         # this generation commits the action (a DAG sink)
    phase="belief",         # main | belief | role | reflection | summarization
)
```

If the provider returns no usage block, `dsmeter` falls back to a
`~4-chars-per-token` estimate from `text` and flags the event
`tokens_estimated=True`.

> Prefer explicit primitives? Use `open_generation(...)` / `close_generation(h)`
> instead of the context manager — same arguments, manual lifetime.

### 3. Group work into steps

CPOT is computed **per step** and summed. A "step" is one decision cycle (one
user turn, one environment tick):

```python
with ds.step(0):
    ...        # generations here belong to step 0

with ds.step(1):
    ...        # step 1
```

For environment loops, call `ds.step_barrier()` at each `env.advance()` to get a
fresh auto-incrementing step index:

```python
while not done:
    ds.step_barrier()
    ... # agent acts
```

### 4. Mark the action sink

The decision span ends at the **committed action**. Mark it one of three ways:

```python
# a) inline on the final answer
g.record(resp, is_action=True)

# b) after the fact
ds.mark_action(handle)

# c) declare which tools commit an action — any generation issuing one becomes a sink
ds.set_action_tools({"submit_decision", "place_order"})
```

If no sink is marked in a step, every generation in that step is treated as a
potential sink (the longest path wins).

### 5. Tell dsmeter who caused what

Edges come **only** from captured ids, resolved by this priority:

```python
# (1) explicit — you know the parents
with ds.generation(agent="coord", parents=[w1.id, w2.id]) as c:
    c.record(out_tokens=60, is_action=True)

# parents=[] means "root off a step input" (no parent, skip the ambient cause)
with ds.generation(agent="worker", parents=[]) as w:
    ...

# (2) object provenance — a result carries its origin
tagged = producer.as_result(some_object)          # stamp origin
with ds.generation(prompt_from=[tagged]) as g:    # link to whatever produced it
    ...

# (3) envelope — across a boundary (see §6)
with ds.generation(caused_by=parent_id) as g:
    ...

# (4) tool result — a generation caused by a specific tool call
with ds.generation(caused_by_tool="call_abc") as g:
    ...

# (5) ambient (default) — parent = last generation completed in this flow.
# contextvars propagate into asyncio tasks, so orchestrator→worker links form for free.
with ds.generation() as g:
    ...
```

You can also pin the ambient cause for a region explicitly:

```python
with ds.cause(some_gen_id):
    with ds.generation() as g:      # parent = some_gen_id
        ...
```

### 6. Cross-process / queue boundaries

When a hand-off crosses a process, thread pool, or message queue, carry the cause
in the envelope:

```python
# producer side
headers = ds.inject()               # -> {"cpot-cause": "...", "cpot-run": "..."}
queue.put({"headers": headers, "payload": task})

# consumer side (different process)
msg = queue.get()
parent = ds.extract(msg["headers"])
with ds.generation(caused_by=parent) as g:
    ...
```

`inject()` writes into an existing dict if you pass one, so it composes with HTTP
headers or Kafka record headers.

### 7. Auto-trace a client

Wrap a client once and every call becomes a generation automatically:

```python
client = ds.wrap_openai(client)         # every chat.completions.create is traced
client = ds.wrap_anthropic(client)      # every messages.create is traced
```

---

## Framework integrations

### Google ADK (one line)

The entire integration is registering one plugin on the `Runner`:

```python
import dsmeter as ds
from dsmeter.integrations.adk import CpotPlugin

ds.init(run_id="demo", out="trace.jsonl")

runner = InMemoryRunner(agent=root_agent, app_name="app", plugins=[CpotPlugin()])
```

The plugin records every model call as a generation, chains a tool loop into a
serial DAG via the ambient cause, treats the final tool-free answer as the action
sink, and maps one ADK invocation (one user turn) to one CPOT step.

```python
CpotPlugin(
    action_tools={"submit_decision"},   # optional: tools that commit an action
    step_per_invocation=True,           # one invocation = one CPOT step (default)
)
```

A full runnable example lives in [`examples/adk_example.py`](examples/adk_example.py).

*Roadmap: LangGraph, OpenAI-Agents, OpenTelemetry, and proxy adapters.*

---

## Analyzing a trace

```python
report = ds.analyze("trace.jsonl", alpha=0.0)     # from a JSONL file
report = ds.analyze(ds.recorder().events)          # or from in-memory events
print(report.summary())

for step, s in report.per_step.items():
    print(f"step {step}: CPOT={s['cpot']:.0f} tokens over {s['n']} gens (wall {s['wall']:.2f}s)")

for w in report.warnings:
    print("WARNING:", w)
```

`alpha` optionally adds a fraction of **input** tokens to each node weight
(`weight = out_tokens + alpha * in_tokens`); the default `0.0` counts decode only.

`analyze()` also validates the reconstruction and surfaces warnings when the
provenance looks suspect — a high share of fuzzy text-matched edges, orphan
generations with no captured parent, or edges that violate causality (a parent
that ends after its child starts).

---

## The `Report` object

| Field | Type | Meaning |
|-------|------|---------|
| `T` | `float` | **Decision Span** — Σ per-step CPOT, in output tokens. |
| `W` | `float` | Total work — output tokens across all generations. |
| `parallelism` | `float` | `W / T`. `1.0` = fully serial; higher = more off-path concurrency. |
| `T_wall` | `float` | Longest-path span weighted by wall-clock seconds. |
| `per_step` | `dict` | `step_index -> {"cpot", "wall", "n"}`. |
| `W_main` | `float` | Work in `phase="main"` generations. |
| `W_learning` | `float` | Work in off-path learning phases (`W - W_main`). |
| `n_gens` | `int` | Number of generations. |
| `n_steps` | `int` | Number of steps. |
| `edge_sources` | `dict` | Histogram of how edges were resolved (`ambient`, `explicit`, `tool`, …). |
| `warnings` | `list[str]` | Provenance / causality validation warnings. |
| `.summary()` | `-> str` | Human-readable one-block summary. |

---

## Trace file format

Traces are newline-delimited JSON (JSONL). Two record kinds:

**Generation** (`kind: "generation"`):

```json
{
  "gen_id": "gen_1a2b3c4d5e6f", "run_id": "demo",
  "parent_gen_ids": ["gen_0000…"], "parent_tool_ids": [],
  "agent_id": "planner", "step_index": 0,
  "t_start": 1750.0, "t_end": 1750.4,
  "in_tokens": 512, "out_tokens": 128, "tokens_estimated": false,
  "model": "gpt-4o", "issued_tool_calls": ["call_abc"],
  "is_action": false, "phase": "main",
  "edge_source": {"gen_0000…": "ambient"}, "kind": "generation"
}
```

**Tool** (`kind: "tool"`) — links a `tool_id` back to the generation that issued it:

```json
{"tool_id": "call_abc", "issued_by_gen": "gen_1a2b…", "t_start": 1750.0, "t_end": 1750.4, "kind": "tool"}
```

Everything downstream (`analyze`, `span_step`, the `Report`) is a **pure function
of these events** — you can post-process traces offline, in another language, or
long after the run.

---

## API reference

**Session**
- `init(run_id="run", out=None) -> Recorder` — start a recording session.
- `recorder() -> Recorder` — the process-global recorder (`.events`, `.close()`).

**Capture**
- `generation(**kw)` — context manager yielding a `GenHandle`.
- `open_generation(**kw) -> GenHandle` / `close_generation(handle, ...)` — manual form.
- `GenHandle.record(response=None, *, out_tokens, in_tokens, model, tool_calls, text, is_action, phase)`.
- `GenHandle.as_result(obj)` — stamp provenance onto a result for `prompt_from=`.
- `set_action_tools(names)` — declare action-committing tool names.
- `mark_action(handle)` — mark a generation as the action sink.

`generation()` / `open_generation()` keyword arguments: `agent`, `step`,
`parents`, `caused_by`, `caused_by_tool`, `prompt_from`, `phase`, `is_action`,
`model`.

**Context / grouping**
- `step(index)` — context manager grouping generations into a CPOT step.
- `step_barrier() -> int` — advance to the next auto-incremented step.
- `agent(name)` — context manager tagging an agent id.
- `cause(gen_id)` — context manager pinning the ambient cause.
- `current_cause()`, `inject(carrier=None)`, `extract(carrier)`, `new_id(prefix="gen")`.

**Client wrappers**
- `wrap_openai(client)`, `wrap_anthropic(client)`.

**Analysis**
- `analyze(path_or_events, alpha=0.0) -> Report`.
- `load_events(path) -> (gens, tools)`, `span_step(...)`.
- `Report`, `GenerationEvent`, `ToolEvent`.

---

## Development & testing

```bash
git clone https://github.com/swarna-kpaul/dsmeter
cd dsmeter
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

pytest                                   # full suite (offline, no API keys)
pytest --cov=dsmeter --cov-report=term-missing
```

The test suite (`tests/`) pins CPOT against **five hand-computed agentic
topologies** — serial tool loop, orchestrator fan-out/in, parallel-off-snapshot,
deep cascade, and off-path learning — plus the ADK adapter driven offline with
fake responses. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines.

---

## Project layout

```
dsmeter/
├── src/dsmeter/
│   ├── __init__.py            public API (init, generation, step, analyze, …)
│   ├── schema.py              GenerationEvent / ToolEvent dataclasses
│   ├── context.py             ambient cause contextvars + inject/extract
│   ├── store.py               append-only JSONL recorder
│   ├── capture.py             generation capture + provider-agnostic extraction
│   ├── engine.py              CPOT longest-path computation + Report
│   └── integrations/
│       └── adk.py             Google ADK plugin
├── tests/                     golden-trace conformance + adapter tests
├── examples/
│   └── adk_example.py         runnable ADK demo
├── pyproject.toml
├── CHANGELOG.md
└── CONTRIBUTING.md
```

---

## License

[MIT](LICENSE) © Swarna Kamal Paul
