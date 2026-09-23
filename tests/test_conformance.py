"""Golden-trace conformance tests for the CPOT engine.

Each test constructs a canonical agentic topology with hand-computed output-token
weights, records it through the public ``dsmeter`` API, and asserts that the
analyzer reproduces the CPOT (decision span) and total work ``W`` we computed by
hand. If any of these break, the longest-path computation or the edge-resolution
priority has regressed.

Run with:  ``pytest``   (or ``pytest tests/test_conformance.py -v``)
"""
from __future__ import annotations

import pytest

import dsmeter as ds
from dsmeter.engine import analyze


def _events():
    return list(ds.recorder().events)


# --------------------------------------------------------------------------- #
# Each builder returns (events, expected_cpot). Comments show the hand math.
# --------------------------------------------------------------------------- #
def style_serial():
    """Single-agent tool loop A->B->C (ambient chain). CPOT = 100+80+50 = 230."""
    ds.init(run_id="serial")
    with ds.step(0):
        with ds.generation(agent="a") as a:
            a.record(out_tokens=100, tool_calls=["t1"])
        with ds.generation(agent="a") as b:          # ambient parent = A
            b.record(out_tokens=80, tool_calls=["t2"])
        with ds.generation(agent="a") as c:          # ambient parent = B
            c.record(out_tokens=50, is_action=True)
    return _events(), 230


def style_orchestrator():
    """Orchestrator O -> {w1,w2} -> coord. CPOT = 40 + max(100,90) + 60 = 200."""
    ds.init(run_id="orch")
    with ds.step(0):
        with ds.generation(agent="orch") as o:
            o.record(out_tokens=40, tool_calls=["spawn"])
        with ds.generation(agent="w1", parents=[o.id]) as w1:
            w1.record(out_tokens=100)
        with ds.generation(agent="w2", parents=[o.id]) as w2:
            w2.record(out_tokens=90)
        with ds.generation(agent="coord", parents=[w1.id, w2.id]) as c:
            c.record(out_tokens=60, is_action=True)
    return _events(), 200


def style_parallel():
    """Parallel agents off one snapshot (roots) -> coord. CPOT = max(100,90) + 60 = 160."""
    ds.init(run_id="parallel")
    with ds.step(0):
        with ds.generation(agent="w1", parents=[]) as w1:   # root off a step input
            w1.record(out_tokens=100)
        with ds.generation(agent="w2", parents=[]) as w2:   # root off a step input
            w2.record(out_tokens=90)
        with ds.generation(agent="coord", parents=[w1.id, w2.id]) as c:
            c.record(out_tokens=60, is_action=True)
    return _events(), 160


def style_cascade():
    """Deep delegation A->B->C->D (ambient chain), each 30. CPOT = 120."""
    ds.init(run_id="cascade")
    with ds.step(0):
        with ds.generation() as a:
            a.record(out_tokens=30, tool_calls=["t"])
        with ds.generation() as b:
            b.record(out_tokens=30, tool_calls=["t"])
        with ds.generation() as c:
            c.record(out_tokens=30, tool_calls=["t"])
        with ds.generation() as d:
            d.record(out_tokens=30, is_action=True)
    return _events(), 120


def style_offpath():
    """Off-path learning: A(100) -> {belief(500) leaf, C(50) action}. CPOT = 150, W = 650."""
    ds.init(run_id="offpath")
    with ds.step(0):
        with ds.generation(agent="a") as a:
            a.record(out_tokens=100, tool_calls=["t"])
        with ds.generation(agent="a", parents=[a.id], phase="belief") as bel:
            bel.record(out_tokens=500)                 # big, but not on the action path
        with ds.generation(agent="a", parents=[a.id]) as c:
            c.record(out_tokens=50, is_action=True)
    return _events(), 150


GOLDEN_STYLES = [
    ("serial_tool_loop", style_serial),
    ("orchestrator_fan_out_in", style_orchestrator),
    ("parallel_off_snapshot", style_parallel),
    ("deep_cascade", style_cascade),
    ("off_path_learning", style_offpath),
]


@pytest.mark.parametrize("name,builder", GOLDEN_STYLES, ids=[n for n, _ in GOLDEN_STYLES])
def test_cpot_matches_hand_computed(name, builder):
    events, expected = builder()
    report = analyze(events, alpha=0.0)
    assert report.T == pytest.approx(expected), f"{name}: CPOT {report.T} != expected {expected}"


def test_off_path_learning_excluded_from_cpot_but_counted_in_work():
    """Off-path 'belief' generation must inflate W but never enter the decision span."""
    events, _ = style_offpath()
    report = analyze(events, alpha=0.0)
    assert report.W == pytest.approx(650), "total work must include the off-path belief"
    assert report.W_learning == pytest.approx(500), "belief phase is learning work"
    assert report.W_main == pytest.approx(150), "main-phase work is only the action path"
    assert report.T == pytest.approx(150), "decision span excludes off-path learning"


def test_orchestrator_reports_parallelism_credit():
    """Concurrent workers must give W/T > 1 (work done in parallel, off the span)."""
    events, _ = style_orchestrator()
    report = analyze(events, alpha=0.0)
    assert report.parallelism > 1.0, "orchestrator fan-out should show concurrency credit"
    assert report.W == pytest.approx(290)
    assert report.T == pytest.approx(200)
