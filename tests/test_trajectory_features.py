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
    derive_longitudinal_hazard_deltas,
    ensure_longitudinal_hazard_deltas,
    LONGITUDINAL_HAZARD_DELTA_COLUMNS,
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


# ---------------------------------------------------------------------------
# Longitudinal hazard-context delta generation / fallback
# ---------------------------------------------------------------------------

def _node_deltas_with_levels() -> pd.DataFrame:
    """Two domains that map onto gravity_fields, across two timepoints."""
    return pd.DataFrame([
        # baseline (delta 0)
        {"subject_id": "S1", "timepoint": "T0_baseline", "mission_phase": "baseline",
         "domain": "Cardiovascular regulation", "baseline_activation": 0.8,
         "current_activation": 0.8, "delta_activation": 0.0},
        {"subject_id": "S1", "timepoint": "T0_baseline", "mission_phase": "baseline",
         "domain": "Body composition / physical status", "baseline_activation": 0.6,
         "current_activation": 0.6, "delta_activation": 0.0},
        # inflight (increase)
        {"subject_id": "S1", "timepoint": "T2_inflight", "mission_phase": "inflight",
         "domain": "Cardiovascular regulation", "baseline_activation": 0.8,
         "current_activation": 1.1, "delta_activation": 0.3},
        {"subject_id": "S1", "timepoint": "T2_inflight", "mission_phase": "inflight",
         "domain": "Body composition / physical status", "baseline_activation": 0.6,
         "current_activation": 0.7, "delta_activation": 0.1},
    ])


def test_derive_has_expected_columns_and_guardrail():
    out = derive_longitudinal_hazard_deltas(_node_deltas_with_levels())
    assert list(out.columns) == LONGITUDINAL_HAZARD_DELTA_COLUMNS
    assert not out.empty
    # every hazard appears per timepoint
    assert set(out["hazard"]).issuperset({"gravity_fields"})
    # guardrail language present, forbidden language absent
    joined = " ".join(out["interpretation"].tolist()).lower()
    assert "not exposure measurement" in joined
    assert "not risk scoring" in joined
    assert "not diagnosis" in joined
    for bad in ("hazard risk score", "exposure detected", "radiation effect", "health risk"):
        assert bad not in joined


def test_derive_levels_baseline_and_current():
    out = derive_longitudinal_hazard_deltas(_node_deltas_with_levels())
    gf = out[(out["timepoint"] == "T2_inflight") & (out["hazard"] == "gravity_fields")].iloc[0]
    # baseline computed from baseline_activation (not assumed 0)
    assert gf["baseline_hazard_relevance"] > 0
    assert gf["delta_hazard_relevance"] > 0
    assert gf["top_contributing_domain"] != "n/a"


def test_derive_delta_only_assumes_zero_baseline():
    node_deltas = _node_deltas_with_levels().drop(
        columns=["baseline_activation", "current_activation"])
    out = derive_longitudinal_hazard_deltas(node_deltas)
    gf = out[(out["timepoint"] == "T2_inflight") & (out["hazard"] == "gravity_fields")].iloc[0]
    assert gf["baseline_hazard_relevance"] == 0.0
    assert gf["delta_hazard_relevance"] > 0
    assert "assumed to be 0.0" in gf["interpretation"]


def test_derive_empty_inputs_safe():
    out = derive_longitudinal_hazard_deltas(pd.DataFrame())
    assert out.empty
    assert list(out.columns) == LONGITUDINAL_HAZARD_DELTA_COLUMNS


def test_derive_missing_required_columns_safe():
    bad = pd.DataFrame([{"subject_id": "S1", "foo": 1}])
    out = derive_longitudinal_hazard_deltas(bad)
    assert out.empty
    assert list(out.columns) == LONGITUDINAL_HAZARD_DELTA_COLUMNS


def test_ensure_loads_existing_file(tmp_path):
    tdir = tmp_path / "tables"
    tdir.mkdir()
    existing = pd.DataFrame([{
        "subject_id": "S1", "timepoint": "T0", "mission_phase": "baseline",
        "hazard": "gravity_fields", "baseline_hazard_relevance": 0.5,
        "current_hazard_relevance": 0.5, "delta_hazard_relevance": 0.0,
        "coverage_fraction": 1.0, "interpretation": "existing row",
    }])
    existing.to_csv(tdir / "longitudinal_hazard_deltas.csv", index=False)
    out = ensure_longitudinal_hazard_deltas(tdir)
    assert out.attrs.get("source") == "file"
    # missing expected column added
    assert "top_contributing_domain" in out.columns
    assert out.iloc[0]["interpretation"] == "existing row"


def test_ensure_derives_and_saves_when_missing(tmp_path):
    tdir = tmp_path / "tables"
    tdir.mkdir()
    _node_deltas_with_levels().to_csv(tdir / "longitudinal_node_deltas.csv", index=False)
    target = tdir / "longitudinal_hazard_deltas.csv"
    assert not target.exists()
    out = ensure_longitudinal_hazard_deltas(tdir)
    assert out.attrs.get("source") == "derived"
    assert not out.empty
    assert list(out.columns) == LONGITUDINAL_HAZARD_DELTA_COLUMNS
    # file was saved
    assert target.exists()
    reloaded = pd.read_csv(target)
    assert "top_contributing_domain" in reloaded.columns


def test_ensure_unavailable_when_no_inputs(tmp_path):
    tdir = tmp_path / "tables"
    tdir.mkdir()
    out = ensure_longitudinal_hazard_deltas(tdir)
    assert out.empty
    assert list(out.columns) == LONGITUDINAL_HAZARD_DELTA_COLUMNS
    assert out.attrs.get("source") == "unavailable"
    assert "not found" in out.attrs.get("note", "")
