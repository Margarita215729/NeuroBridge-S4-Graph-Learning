"""Tests for Phase 8 envelope reporting (synthetic data only)."""

from __future__ import annotations

import re

import pandas as pd

from neurobridge_graph.reference_envelope import (
    build_envelope_from_summary_table,
    score_node_deltas_against_envelope,
    score_graph_deltas_against_envelope,
    score_hazard_deltas_against_envelope,
    build_phase8_envelope_summary,
)
from neurobridge_graph.envelope_reporting import (
    generate_envelope_interpretation,
    generate_phase8_report,
)

# Forbidden phrasing that must never appear without negation.
_FORBIDDEN = [
    "diagnosis", "risk score", "abnormal", "disease",
    "exposure measurement", "population norm", "treatment",
]


def _assert_no_forbidden(text: str):
    low = text.lower()
    for phrase in _FORBIDDEN:
        idx = 0
        while True:
            pos = low.find(phrase, idx)
            if pos == -1:
                break
            # Look back across the clause for an explicit negation.
            prefix = low[max(0, pos - 30):pos]
            negated = any(tok in prefix for tok in ("not ", "no ", "without ", "never "))
            assert negated, (
                f"Found '{phrase}' without negation: ...{low[max(0,pos-30):pos+15]}..."
            )
            idx = pos + len(phrase)


def _envelope() -> pd.DataFrame:
    summary = pd.DataFrame({
        "feature": ["Cardiovascular regulation", "mean_node_activation", "space_radiation"],
        "feature_type": ["node", "graph", "hazard"],
        "median_delta": [0.0, 0.0, 0.0],
        "mad_delta": [0.05, 0.03, 0.05],
        "lower_bound": [-0.12, -0.08, -0.12],
        "upper_bound": [0.12, 0.08, 0.12],
        "n_reference": [30, 30, 30],
        "data_type": ["schema_demonstration_not_scientific_evidence"] * 3,
    })
    return build_envelope_from_summary_table(summary)


def _node_scores():
    nd = pd.DataFrame([
        {"subject_id": "S1", "timepoint": "T2", "mission_phase": "inflight",
         "domain": "Cardiovascular regulation", "delta_activation": 0.4},
    ])
    return score_node_deltas_against_envelope(nd, _envelope())


def _graph_scores():
    gd = pd.DataFrame([
        {"subject_id": "S1", "timepoint": "T2", "mission_phase": "inflight",
         "metric": "mean_node_activation", "delta_value": 0.3},
    ])
    return score_graph_deltas_against_envelope(gd, _envelope())


def _hazard_scores():
    hd = pd.DataFrame([
        {"subject_id": "S1", "timepoint": "T2", "mission_phase": "inflight",
         "hazard": "space_radiation", "delta_hazard_relevance": 0.3},
    ])
    return score_hazard_deltas_against_envelope(hd, _envelope())


def test_interpretation_outside_has_guardrail():
    scores = _node_scores()
    text = generate_envelope_interpretation(scores.iloc[0], layer="node")
    assert "outside the expected variability envelope" in text
    assert "does not mean disease, danger, or health risk" in text.lower()
    _assert_no_forbidden(text)


def test_report_runs_with_all_layers_and_marks_demo():
    ns, gs, hs = _node_scores(), _graph_scores(), _hazard_scores()
    summary = build_phase8_envelope_summary(ns, gs, hs)
    report = generate_phase8_report(summary, ns, gs, hs, envelope_df=_envelope())
    assert "Phase 8" in report
    assert "EXAMPLE" in report  # demo provenance detected
    assert "LIMITATIONS" in report
    assert "does not define whether a person is healthy or unhealthy" in report
    _assert_no_forbidden(report)


def test_report_runs_with_optional_hazard_missing():
    ns, gs = _node_scores(), _graph_scores()
    summary = build_phase8_envelope_summary(ns, gs, None)
    report = generate_phase8_report(summary, ns, gs, None, envelope_df=_envelope())
    assert "Hazard-context delta scoring: unavailable" in report
    _assert_no_forbidden(report)


def test_report_provenance_note_override():
    ns, gs = _node_scores(), _graph_scores()
    summary = build_phase8_envelope_summary(ns, gs, None)
    report = generate_phase8_report(
        summary, ns, gs, None, envelope_df=_envelope(),
        data_provenance_note="Calibration source: provided analog data.")
    assert "provided analog data" in report
    _assert_no_forbidden(report)
