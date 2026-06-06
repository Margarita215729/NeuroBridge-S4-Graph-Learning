"""Tests for Phase 7 explanation generator (synthetic data only)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from neurobridge_graph.trajectory_attribution import (
    compute_node_attribution,
    compute_graph_metric_attribution,
    compute_subgraph_attribution_from_node_deltas,
    compute_hazard_context_attribution,
    compute_recovery_attribution,
    build_phase7_attribution_summary,
)
from neurobridge_graph.explanation_generator import (
    generate_subject_timepoint_explanation,
    generate_phase7_report,
)

# Forbidden phrasing that must never appear in generated language.
_FORBIDDEN = [
    "diagnosis",
    "treatment",
    "risk score",
    "causal attribution",
    "radiation effect",
    "exposure attribution",
    "predicts astronaut health",
    "disease attribution",
]


def _node_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"subject_id": "S1", "timepoint": "T2", "mission_phase": "inflight",
         "domain": "Cardiovascular regulation", "baseline_activation": 0.8,
         "current_activation": 1.1, "delta_activation": 0.3,
         "absolute_delta_activation": 0.3, "direction": "increase"},
        {"subject_id": "S1", "timepoint": "T2", "mission_phase": "inflight",
         "domain": "Metabolic regulation", "baseline_activation": 0.9,
         "current_activation": 1.0, "delta_activation": 0.1,
         "absolute_delta_activation": 0.1, "direction": "increase"},
        {"subject_id": "S1", "timepoint": "T0", "mission_phase": "baseline",
         "domain": "Cardiovascular regulation", "baseline_activation": 0.8,
         "current_activation": 0.8, "delta_activation": 0.0,
         "absolute_delta_activation": 0.0, "direction": "stable"},
        {"subject_id": "S1", "timepoint": "T0", "mission_phase": "baseline",
         "domain": "Metabolic regulation", "baseline_activation": 0.9,
         "current_activation": 0.9, "delta_activation": 0.0,
         "absolute_delta_activation": 0.0, "direction": "stable"},
    ])


def _graph_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"subject_id": "S1", "timepoint": "T2", "mission_phase": "inflight",
         "metric": "mean_node_activation", "baseline_value": 0.85,
         "current_value": 1.05, "delta_value": 0.2, "absolute_delta_value": 0.2},
    ])


def _hazard_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"subject_id": "S1", "timepoint": "T2", "mission_phase": "inflight",
         "hazard": "gravity_fields", "baseline_hazard_relevance": 0.8,
         "current_hazard_relevance": 1.0, "delta_hazard_relevance": 0.2,
         "coverage_fraction": 0.8},
    ])


def _recovery_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"subject_id": "S1", "metric": "graph_density", "baseline_value": 0.3,
         "peak_value": 0.6, "final_value": 0.58, "peak_delta_from_baseline": 0.3,
         "final_delta_from_baseline": 0.28, "recovery_fraction": 0.1},
    ])


def _assert_no_forbidden(text: str):
    low = text.lower()
    for phrase in _FORBIDDEN:
        # "not diagnosis" / "not treatment guidance" are allowed (negation).
        idx = 0
        while True:
            pos = low.find(phrase, idx)
            if pos == -1:
                break
            prefix = low[max(0, pos - 5):pos]
            assert "not " in prefix, f"Found forbidden phrase '{phrase}' without negation: ...{low[max(0,pos-20):pos+20]}..."
            idx = pos + len(phrase)


def test_explanation_contains_guardrail_and_drivers():
    na = compute_node_attribution(_node_df())
    ga = compute_graph_metric_attribution(_graph_df())
    sa = compute_subgraph_attribution_from_node_deltas(na)
    ha = compute_hazard_context_attribution(_hazard_df())
    ra = compute_recovery_attribution(_recovery_df())
    text = generate_subject_timepoint_explanation("S1", "T2", na, ga, sa, ha, ra)
    assert "Cardiovascular regulation" in text
    assert "baseline-relative" in text
    assert "not diagnosis" in text.lower()
    assert "not exposure" in text.lower() or "not hazard exposure" in text.lower()
    _assert_no_forbidden(text)


def test_explanation_baseline_no_shift():
    na = compute_node_attribution(_node_df())
    ga = compute_graph_metric_attribution(_graph_df())
    text = generate_subject_timepoint_explanation("S1", "T0", na, ga)
    assert "baseline" in text.lower()
    _assert_no_forbidden(text)


def test_explanation_handles_optional_layers_missing():
    na = compute_node_attribution(_node_df())
    ga = compute_graph_metric_attribution(_graph_df())
    text = generate_subject_timepoint_explanation("S1", "T2", na, ga)
    assert "Cardiovascular regulation" in text
    _assert_no_forbidden(text)


def test_explanation_unknown_subject():
    na = compute_node_attribution(_node_df())
    ga = compute_graph_metric_attribution(_graph_df())
    text = generate_subject_timepoint_explanation("UNKNOWN", "T2", na, ga)
    assert "no node-level attribution" in text.lower()


def test_report_runs_with_all_layers():
    na = compute_node_attribution(_node_df())
    ga = compute_graph_metric_attribution(_graph_df())
    sa = compute_subgraph_attribution_from_node_deltas(na)
    ha = compute_hazard_context_attribution(_hazard_df())
    ra = compute_recovery_attribution(_recovery_df())
    summary = build_phase7_attribution_summary(na, ga, sa, ha, ra)
    report = generate_phase7_report(summary, na, ga, sa, ha, ra,
                                    data_provenance_note="Schema demonstration only.")
    assert "Phase 7" in report
    assert "LIMITATIONS" in report
    assert "Schema demonstration only." in report
    _assert_no_forbidden(report)


def test_report_runs_with_optional_layers_missing():
    na = compute_node_attribution(_node_df())
    ga = compute_graph_metric_attribution(_graph_df())
    summary = build_phase7_attribution_summary(na, ga)
    report = generate_phase7_report(summary, na, ga)
    assert "unavailable" in report.lower()
    assert "Phase 7" in report
    _assert_no_forbidden(report)
