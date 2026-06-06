"""Tests for Phase 8 reference-calibrated envelope (synthetic data only)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neurobridge_graph.reference_envelope import (
    SCHEMA_DEMO_DATA_TYPE,
    WITHIN,
    NEAR,
    OUTSIDE,
    INSUFFICIENT,
    create_example_reference_envelope,
    build_envelope_from_reference_deltas,
    build_envelope_from_summary_table,
    compute_robust_z_score,
    classify_envelope_position,
    score_node_deltas_against_envelope,
    score_graph_deltas_against_envelope,
    score_hazard_deltas_against_envelope,
    build_phase8_envelope_summary,
)


# --------------------------------------------------------------------------
# Robust z-score
# --------------------------------------------------------------------------

def test_robust_z_score_basic():
    # value == median -> 0
    assert compute_robust_z_score(0.0, 0.0, 0.1) == pytest.approx(0.0)
    # known: (0.2 - 0.0) / (1.4826 * 0.1) ~ 1.349
    z = compute_robust_z_score(0.2, 0.0, 0.1)
    assert z == pytest.approx(0.2 / (1.4826 * 0.1), rel=1e-3)


def test_robust_z_score_handles_nan():
    assert np.isnan(compute_robust_z_score(np.nan, 0.0, 0.1))
    assert np.isnan(compute_robust_z_score(0.1, np.nan, 0.1))


def test_robust_z_score_zero_mad_no_div_error():
    # epsilon prevents division by zero
    z = compute_robust_z_score(0.5, 0.0, 0.0)
    assert np.isfinite(z)


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------

def test_classify_within():
    assert classify_envelope_position(0.0, -0.12, 0.12, robust_z=0.0) == WITHIN


def test_classify_outside_by_bounds():
    assert classify_envelope_position(0.5, -0.12, 0.12, robust_z=0.5) == OUTSIDE


def test_classify_outside_by_z():
    assert classify_envelope_position(0.05, -1.0, 1.0, robust_z=2.5) == OUTSIDE


def test_classify_near_by_z():
    assert classify_envelope_position(0.05, -1.0, 1.0, robust_z=1.7) == NEAR


def test_classify_insufficient_when_bounds_and_z_missing():
    assert classify_envelope_position(0.1, np.nan, np.nan, robust_z=None) == INSUFFICIENT


# --------------------------------------------------------------------------
# Envelope construction
# --------------------------------------------------------------------------

def _reference_deltas() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for feat, scale in [("Cardiovascular regulation", 0.05), ("Metabolic regulation", 0.05)]:
        for v in rng.normal(0.0, scale, size=40):
            rows.append({"feature": feat, "feature_type": "node", "delta_value": v})
    return pd.DataFrame(rows)


def test_build_envelope_from_reference_deltas():
    env = build_envelope_from_reference_deltas(_reference_deltas())
    assert set(["feature", "median_delta", "mad_delta", "lower_bound", "upper_bound",
                "n_reference", "envelope_method"]).issubset(env.columns)
    assert (env["n_reference"] == 40).all()
    # bounds ordered
    assert (env["lower_bound"] < env["upper_bound"]).all()
    # median near zero for symmetric noise
    assert env["median_delta"].abs().max() < 0.05


def test_build_envelope_from_reference_deltas_missing_col():
    with pytest.raises(ValueError):
        build_envelope_from_reference_deltas(pd.DataFrame({"x": [1, 2]}))


def test_build_envelope_from_summary_table_normalizes_aliases():
    summary = pd.DataFrame({
        "domain": ["A", "B"],
        "median": [0.0, 0.1],
        "mad": [0.05, 0.05],
        "p05": [-0.1, -0.05],
        "p95": [0.1, 0.25],
        "n": [30, 30],
    })
    env = build_envelope_from_summary_table(summary)
    assert list(env["feature"]) == ["A", "B"]
    assert env["lower_bound"].iloc[0] == -0.1
    assert env["upper_bound"].iloc[1] == 0.25
    assert (env["n_reference"] == 30).all()


def test_build_envelope_from_summary_derives_bounds_from_mad():
    summary = pd.DataFrame({"feature": ["A"], "median_delta": [0.0], "mad_delta": [0.1]})
    env = build_envelope_from_summary_table(summary)
    # lower/upper derived from median +/- 2 * 1.4826 * mad
    assert env["lower_bound"].iloc[0] < 0 < env["upper_bound"].iloc[0]


def test_build_envelope_empty():
    assert build_envelope_from_summary_table(pd.DataFrame()).empty


# --------------------------------------------------------------------------
# Example envelope
# --------------------------------------------------------------------------

def test_example_envelope_marked_schema_demo(tmp_path):
    out = tmp_path / "example_reference_variability_envelope.csv"
    df = create_example_reference_envelope(out)
    assert out.exists()
    assert (df["data_type"] == SCHEMA_DEMO_DATA_TYPE).all()
    # covers all three feature types
    assert set(df["feature_type"]) == {"node", "graph", "hazard"}


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def _envelope() -> pd.DataFrame:
    summary = pd.DataFrame({
        "feature": ["Cardiovascular regulation", "Metabolic regulation",
                    "mean_node_activation", "space_radiation"],
        "feature_type": ["node", "node", "graph", "hazard"],
        "median_delta": [0.0, 0.0, 0.0, 0.0],
        "mad_delta": [0.05, 0.05, 0.03, 0.05],
        "lower_bound": [-0.12, -0.12, -0.08, -0.12],
        "upper_bound": [0.12, 0.12, 0.08, 0.12],
        "n_reference": [30, 30, 30, 30],
    })
    return build_envelope_from_summary_table(summary)


def _node_deltas() -> pd.DataFrame:
    return pd.DataFrame([
        {"subject_id": "S1", "timepoint": "T2", "mission_phase": "inflight",
         "domain": "Cardiovascular regulation", "delta_activation": 0.4},   # outside
        {"subject_id": "S1", "timepoint": "T2", "mission_phase": "inflight",
         "domain": "Metabolic regulation", "delta_activation": 0.02},        # within
        {"subject_id": "S1", "timepoint": "T2", "mission_phase": "inflight",
         "domain": "Unknown domain", "delta_activation": 0.9},               # insufficient
    ])


def test_score_node_deltas():
    scores = score_node_deltas_against_envelope(_node_deltas(), _envelope())
    pos = dict(zip(scores["domain"], scores["envelope_position"]))
    assert pos["Cardiovascular regulation"] == OUTSIDE
    assert pos["Metabolic regulation"] == WITHIN
    assert pos["Unknown domain"] == INSUFFICIENT
    # outside row carries guardrail-friendly interpretation
    outside_interp = scores[scores["domain"] == "Cardiovascular regulation"]["interpretation"].iloc[0]
    assert "not diagnosis" in outside_interp.lower()


def test_score_graph_deltas():
    gd = pd.DataFrame([
        {"subject_id": "S1", "timepoint": "T2", "mission_phase": "inflight",
         "metric": "mean_node_activation", "delta_value": 0.3},
    ])
    scores = score_graph_deltas_against_envelope(gd, _envelope())
    assert scores.iloc[0]["envelope_position"] == OUTSIDE
    assert "delta_value" in scores.columns


def test_score_hazard_deltas_missing_table():
    empty = score_hazard_deltas_against_envelope(None, _envelope())
    assert empty.empty
    empty2 = score_hazard_deltas_against_envelope(pd.DataFrame(), _envelope())
    assert empty2.empty
    assert "delta_hazard_relevance" in empty.columns


def test_score_hazard_deltas_values():
    hd = pd.DataFrame([
        {"subject_id": "S1", "timepoint": "T2", "mission_phase": "inflight",
         "hazard": "space_radiation", "delta_hazard_relevance": 0.3},
    ])
    scores = score_hazard_deltas_against_envelope(hd, _envelope())
    assert scores.iloc[0]["envelope_position"] == OUTSIDE


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------

def test_build_phase8_summary():
    ns = score_node_deltas_against_envelope(_node_deltas(), _envelope())
    gd = pd.DataFrame([
        {"subject_id": "S1", "timepoint": "T2", "mission_phase": "inflight",
         "metric": "mean_node_activation", "delta_value": 0.3},
    ])
    gs = score_graph_deltas_against_envelope(gd, _envelope())
    summary = build_phase8_envelope_summary(ns, gs, None)
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["overall_envelope_flag"] == OUTSIDE
    assert row["n_outside_node_envelope"] == 1
    assert row["n_outside_graph_envelope"] == 1
    assert row["top_outside_domain"] == "Cardiovascular regulation"


def test_build_phase8_summary_empty():
    assert build_phase8_envelope_summary(pd.DataFrame(), pd.DataFrame(), None).empty
