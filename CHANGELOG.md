# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-09-23

### Added
- Initial public release of `dsmeter`.
- Core CPOT engine: node-weighted (output-token) longest-path decision-span
  computation, per step and aggregated (`T`, `W`, `W/T`, per-step breakdown).
- Provider-agnostic generation capture for OpenAI-, Anthropic-, and
  Gemini/ADK-shaped responses (output tokens, tool-call ids, model).
- Causal-edge resolution by priority: explicit parents, object provenance,
  cross-boundary envelope (`inject`/`extract`), tool-call ids, ambient contextvar.
- Ambient-cause context propagation across asyncio tasks and process boundaries.
- Google ADK adapter (`dsmeter.integrations.adk.CpotPlugin`) — one plugin on the
  `Runner`, no other code changes.
- `wrap_openai` / `wrap_anthropic` client auto-tracing helpers.
- Append-only JSONL trace store plus in-memory mirror for offline analysis.
- Analyzer validation: fuzzy-edge, orphan, and causality-violation warnings.
- Conformance test suite covering five canonical agentic topologies and the
  ADK adapter (offline, no API key required).

[Unreleased]: https://github.com/swarna-kpaul/dsmeter/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/swarna-kpaul/dsmeter/releases/tag/v0.1.0
