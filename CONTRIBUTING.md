# Contributing to dsmeter

Thanks for your interest in improving `dsmeter`. This project measures the
**Decision Span (DS)** of agentic LLM systems; correctness of the CPOT
computation is the top priority, so most contributions come with a test.

## Development setup

```bash
git clone https://github.com/swarna-kpaul/dsmeter
cd dsmeter
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -e ".[dev]"             # editable install + pytest/coverage
```

Optional adapter extras:

```bash
pip install -e ".[dev,adk]"         # to work on the Google ADK integration
```

## Running the tests

```bash
pytest                              # full suite
pytest -v                           # verbose, per-case
pytest --cov=dsmeter --cov-report=term-missing
```

The suite is fully offline — no API keys or network calls. The ADK adapter is
exercised with fake response objects, so it runs even without `google-adk`
installed.

## Guidelines

- **Every behavioral change needs a test.** New agentic topologies should be
  added to `tests/test_conformance.py` with hand-computed expected CPOT/`W`.
- **Never infer causal edges from model text.** Edges must come from captured
  ids (tool-call id, contextvar, envelope, or explicit `parents=`).
- Keep the core (`schema`, `context`, `store`, `capture`, `engine`) free of
  hard dependencies on any LLM SDK — those belong in `integrations/` and in the
  optional-dependency extras.
- Match the existing style: `from __future__ import annotations`, small pure
  functions, and docstrings that state the invariant being maintained.

## Releasing

1. Update `version` in `pyproject.toml` and add a section to `CHANGELOG.md`.
2. Tag `vX.Y.Z` and push; CI runs the test matrix.
3. Build and publish: `python -m build && twine upload dist/*`.
