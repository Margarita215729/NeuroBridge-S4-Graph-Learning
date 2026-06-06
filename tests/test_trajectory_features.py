"""Tests for Phase 6 trajectory_features.py."""

import numpy as np
import pandas as pd
import pytest

from neurobridge_graph.trajectory_features import (
    compute_recovery_slope,
    compute_time_to_baseline_like_state,
    compute_recovery_fraction,
    compute_hazard_context_delta,
    compute_recovery_metrics_table,
    identify_dominant_trajectory_shift,
)


def _trajectory_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"subject_id": "S1", "timepoint": "T0", "mission_phase": "baseline",
         "time_index": 0, "mean_node_activation": 0.8, "max_node_activation": 0.9},
        {"subject_id": "S1", "timepoint": "T1", "mission_phase": "inflight",
         "time_index": 1, "mean_node_activation": 1.3, "max_node_activation": 1.5},
        {"subject_id": "S1", "timepoint": "T2", "mission_phase": "postflight",
         "time_index": 2, "mean_node_activation": 1.4, "max_node_activation": 1.6},
        {"subject_id": "S1", "timepoint": "T3", "mission_phase": "recovery",
         "time_index": 3, "mean_node_activation": 1.0, "max_node_activation": 1.1},
        {"subject_id": "S1", "timepoint": "T4", "mission_phase": "recovery",
         "time_index": 4, "mean_node_activation": 0.85, "max_node_activation": 0.95},
    ])


def _hazard_longitudinal() -> pd.DataFrame:
    rows = []
    for tp, phase, tidx, scores in [
        ("T0", "baseline", 0, {"space_radiation": 0.7, "gravity_fields": 0.8}),
        ("T1", "inflight", 1, {"space_radiation": 1.0, "gravity_fields": 1.1}),
        ("T2", "recovery", 2, {"space_radiation": 0.75, "gravity_fields": 0.85}),
    ]:
        for hz, score in scores.items():
            rows.append({
                "subject_id": "S1", "timepoint": tp, "mission_phase": phase,
                "time_index": tidx, "hazard": hz,
                "hazard_relevance_score": score, "coverage_fraction": 0.8,
            })
    return pd.DataFrame(rows)


def test_recovery_slope_negative_during_recovery():
    slope = compute_recovery_slope(_trajectory_df(), "mean_node_activation")
    # Recovery phase goes from 1.0 to 0.85 — negative slope.
    assert slope is not None
    assert slope < 0


def test_time_to_baseline_like_state():
    t = compute_time_to_baseline_like_state(
        _trajectory_df(), "mean_node_activation", tolerance=0.2,
    )
    assert t is not None
    assert t == 3  # T3 at time_index 3 (value 1.0) first returns within tolerance 0.2 of baseline 0.8


def test_recovery_fraction_full_recovery():
    frac = compute_recovery_fraction(baseline_value=0.8, peak_value=1.4, final_value=0.8)
    assert frac == pytest.approx(1.0)


def test_recovery_fraction_partial():
    frac = compute_recovery_fraction(baseline_value=0.8, peak_value=1.4, final_value=1.0)
    assert frac is not None
    assert 0 < frac < 1


def test_recovery_fraction_no_deviation_returns_none():
    assert compute_recovery_fraction(0.8, 0.8, 0.8) is None


def test_hazard_context_delta_from_baseline():
    delta = compute_hazard_context_delta(_hazard_longitudinal())
    assert not delta.empty
    assert "delta_hazard_relevance" in delta.columns
    inflight = delta[(delta["timepoint"] == "T1") & (delta["hazard"] == "space_radiation")].iloc[0]
    assert inflight["delta_hazard_relevance"] == pytest.approx(0.3)
    assert "not exposure" in inflight["interpretation"].lower()


def test_recovery_metrics_table():
    rec = compute_recovery_metrics_table(_trajectory_df())
    assert not rec.empty
    assert "recovery_fraction" in rec.columns
    assert "interpretation" in rec.columns
    row = rec[rec["metric"] == "mean_node_activation"].iloc[0]
    assert row["peak_delta_from_baseline"] > 0


def test_identify_dominant_trajectory_shift():
    node_delta = pd.DataFrame([
        {"subject_id": "S1", "timepoint": "T1", "mission_phase": "inflight",
         "domain": "Cardiovascular regulation", "delta_activation": 0.5,
         "absolute_delta_activation": 0.5, "direction": "increase"},
        {"subject_id": "S1", "timepoint": "T1", "mission_phase": "inflight",
         "domain": "Metabolic regulation", "delta_activation": 0.1,
         "absolute_delta_activation": 0.1, "direction": "increase"},
    ])
    graph_delta = pd.DataFrame([
        {"subject_id": "S1", "timepoint": "T1", "mission_phase": "inflight",
         "metric": "mean_node_activation", "delta_value": 0.5,
         "absolute_delta_value": 0.5},
    ])
    hazard_delta = pd.DataFrame([
        {"subject_id": "S1", "timepoint": "T1", "mission_phase": "inflight",
         "hazard": "gravity_fields", "delta_hazard_relevance": 0.3},
    ])
    result = identify_dominant_trajectory_shift(node_delta, graph_delta, hazard_delta)
    assert len(result) == 1
    assert result.iloc[0]["dominant_domain"] == "Cardiovascular regulation"
    assert result.iloc[0]["dominant_hazard"] == "gravity_fields"
