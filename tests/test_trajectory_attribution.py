"""Tests for Phase 7 trajectory attribution (synthetic data only)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neurobridge_graph.trajectory_attribution import (
    compute_node_attribution,
    compute_graph_metric_attribution,
    compute_subgraph_attribution_from_node_deltas,
    compute_hazard_context_attribution,
    compute_recovery_attribution,
    build_phase7_attribution_summary,
    load_phase6_delta_tables,
)


def _node_df() -> pd.DataFrame:
    return pd.DataFrame([
        # baseline: no change
        {"subject_id": "S1", "timepoint": "T0", "mission_phase": "baseline",
         "domain": "Cardiovascular regulation", "baseline_activation": 0.8,
         "current_activation": 0.8, "delta_activation": 0.0,
         "absolute_delta_activation": 0.0, "direction": "stable"},
        {"subject_id": "S1", "timepoint": "T0", "mission_phase": "baseline",
         "domain": "Metabolic regulation", "baseline_activation": 0.9,
         "current_activation": 0.9, "delta_activation": 0.0,
         "absolute_delta_activation": 0.0, "direction": "stable"},
        # inflight: change
        {"subject_id": "S1", "timepoint": "T2", "mission_phase": "inflight",
         "domain": "Cardiovascular regulation", "baseline_activation": 0.8,
         "current_activation": 1.1, "delta_activation": 0.3,
         "absolute_delta_activation": 0.3, "direction": "increase"},
        {"subject_id": "S1", "timepoint": "T2", "mission_phase": "inflight",
         "domain": "Metabolic regulation", "baseline_activation": 0.9,
         "current_activation": 1.0, "delta_activation": 0.1,
         "absolute_delta_activation": 0.1, "direction": "increase"},
    ])


def _graph_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"subject_id": "S1", "timepoint": "T0", "mission_phase": "baseline",
         "metric": "mean_node_activation", "baseline_value": 0.85,
         "current_value": 0.85, "delta_value": 0.0, "absolute_delta_value": 0.0},
        {"subject_id": "S1", "timepoint": "T2", "mission_phase": "inflight",
         "metric": "mean_node_activation", "baseline_value": 0.85,
         "current_value": 1.05, "delta_value": 0.2, "absolute_delta_value": 0.2},
        {"subject_id": "S1", "timepoint": "T2", "mission_phase": "inflight",
         "metric": "coactivation_edge_count", "baseline_value": 2.0,
         "current_value": 5.0, "delta_value": 3.0, "absolute_delta_value": 3.0},
    ])


# --------------------------------------------------------------------------
# Node attribution
# --------------------------------------------------------------------------

def test_node_attribution_shares_sum_to_one_when_nonzero():
    out = compute_node_attribution(_node_df())
    t2 = out[out["timepoint"] == "T2"]
    assert pytest.approx(t2["contribution_share"].sum(), abs=1e-6) == 1.0
    # cardiovascular (0.3) should dominate over metabolic (0.1)
    top = t2.sort_values("contribution_share", ascending=False).iloc[0]
    assert top["domain"] == "Cardiovascular regulation"
    assert top["attribution_rank"] == 1


def test_node_attribution_zero_change_gives_zero_shares():
    out = compute_node_attribution(_node_df())
    t0 = out[out["timepoint"] == "T0"]
    assert t0["contribution_share"].sum() == 0.0


def test_node_attribution_carries_guardrail_language():
    out = compute_node_attribution(_node_df())
    joined = " ".join(out["interpretation"])
    assert "Not diagnosis" in joined or "not diagnosis" in joined.lower()


def test_node_attribution_missing_column_raises():
    df = _node_df().drop(columns=["delta_activation"])
    with pytest.raises(ValueError):
        compute_node_attribution(df)


# --------------------------------------------------------------------------
# Graph metric attribution
# --------------------------------------------------------------------------

def test_graph_metric_attribution_handles_zero_total_delta():
    out = compute_graph_metric_attribution(_graph_df())
    t0 = out[out["timepoint"] == "T0"]
    assert t0["contribution_share"].sum() == 0.0


def test_graph_metric_attribution_shares_sum_to_one():
    out = compute_graph_metric_attribution(_graph_df())
    t2 = out[out["timepoint"] == "T2"]
    assert pytest.approx(t2["contribution_share"].sum(), abs=1e-6) == 1.0


# --------------------------------------------------------------------------
# Subgraph attribution
# --------------------------------------------------------------------------

def test_subgraph_attribution_aggregates_known_domains():
    node_attr = compute_node_attribution(_node_df())
    out = compute_subgraph_attribution_from_node_deltas(node_attr)
    cm = out[(out["timepoint"] == "T2") & (out["subgraph_name"] == "cardiometabolic")]
    assert not cm.empty
    assert cm.iloc[0]["n_available_domains"] >= 2
    assert cm.iloc[0]["total_contribution_share"] > 0


def test_subgraph_attribution_handles_missing_domains():
    node_attr = compute_node_attribution(_node_df())
    out = compute_subgraph_attribution_from_node_deltas(node_attr)
    # sleep_autonomic_recovery domains are absent from the synthetic data.
    sar = out[out["subgraph_name"] == "sleep_autonomic_recovery"]
    assert not sar.empty
    assert (sar["n_available_domains"] == 0).all()
    assert (sar["total_contribution_share"] == 0.0).all()


def test_subgraph_attribution_empty_input():
    out = compute_subgraph_attribution_from_node_deltas(pd.DataFrame())
    assert out.empty


# --------------------------------------------------------------------------
# Hazard attribution
# --------------------------------------------------------------------------

def _hazard_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"subject_id": "S1", "timepoint": "T2", "mission_phase": "inflight",
         "hazard": "gravity_fields", "baseline_hazard_relevance": 0.8,
         "current_hazard_relevance": 1.0, "delta_hazard_relevance": 0.2,
         "coverage_fraction": 0.8},
        {"subject_id": "S1", "timepoint": "T2", "mission_phase": "inflight",
         "hazard": "space_radiation", "baseline_hazard_relevance": 0.8,
         "current_hazard_relevance": 0.85, "delta_hazard_relevance": 0.05,
         "coverage_fraction": 0.6},
    ])


def test_hazard_attribution_shares_sum_to_one():
    out = compute_hazard_context_attribution(_hazard_df())
    assert pytest.approx(out["contribution_share"].sum(), abs=1e-6) == 1.0
    assert "alignment" in out.iloc[0]["interpretation"].lower()


def test_hazard_attribution_handles_missing_table_gracefully():
    out_none = compute_hazard_context_attribution(None)
    out_empty = compute_hazard_context_attribution(pd.DataFrame())
    assert out_none.empty
    assert out_empty.empty
    assert "contribution_share" in out_none.columns


# --------------------------------------------------------------------------
# Recovery attribution
# --------------------------------------------------------------------------

def _recovery_df() -> pd.DataFrame:
    return pd.DataFrame([
        # returned near baseline
        {"subject_id": "S1", "metric": "mean_node_activation", "baseline_value": 0.8,
         "peak_value": 1.2, "final_value": 0.85, "peak_delta_from_baseline": 0.4,
         "final_delta_from_baseline": 0.05, "recovery_fraction": 0.88},
        # partial recovery
        {"subject_id": "S1", "metric": "total_node_activation", "baseline_value": 5.0,
         "peak_value": 7.0, "final_value": 6.0, "peak_delta_from_baseline": 2.0,
         "final_delta_from_baseline": 1.0, "recovery_fraction": 0.5},
        # persistent shift
        {"subject_id": "S1", "metric": "graph_density", "baseline_value": 0.3,
         "peak_value": 0.6, "final_value": 0.58, "peak_delta_from_baseline": 0.3,
         "final_delta_from_baseline": 0.28, "recovery_fraction": 0.1},
        # overshoot / reversal
        {"subject_id": "S1", "metric": "max_node_activation", "baseline_value": 0.9,
         "peak_value": 1.3, "final_value": 0.7, "peak_delta_from_baseline": 0.4,
         "final_delta_from_baseline": -0.2, "recovery_fraction": 0.5},
        # insufficient data
        {"subject_id": "S1", "metric": "n_active_domains", "baseline_value": 5.0,
         "peak_value": 5.0, "final_value": 5.0, "peak_delta_from_baseline": 0.0,
         "final_delta_from_baseline": 0.0, "recovery_fraction": np.nan},
    ])


def test_recovery_attribution_categories():
    out = compute_recovery_attribution(_recovery_df())
    cat = dict(zip(out["metric"], out["recovery_category"]))
    assert cat["mean_node_activation"] == "returned_near_baseline"
    assert cat["total_node_activation"] == "partial_recovery"
    assert cat["graph_density"] == "persistent_shift"
    assert cat["max_node_activation"] == "overshoot_or_reversal"
    assert cat["n_active_domains"] == "insufficient_data"


def test_recovery_attribution_handles_missing_table():
    assert compute_recovery_attribution(None).empty
    assert compute_recovery_attribution(pd.DataFrame()).empty


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------

def test_build_summary_one_row_per_subject_timepoint():
    na = compute_node_attribution(_node_df())
    ga = compute_graph_metric_attribution(_graph_df())
    sa = compute_subgraph_attribution_from_node_deltas(na)
    ha = compute_hazard_context_attribution(_hazard_df())
    ra = compute_recovery_attribution(_recovery_df())
    summary = build_phase7_attribution_summary(na, ga, sa, ha, ra)
    n_keys = na[["subject_id", "timepoint"]].drop_duplicates().shape[0]
    assert len(summary) == n_keys
    t2 = summary[summary["timepoint"] == "T2"].iloc[0]
    assert t2["top_domain_contributor"] == "Cardiovascular regulation"
    assert t2["top_subgraph_contributor"] == "cardiometabolic"


def test_build_summary_optional_layers_missing():
    na = compute_node_attribution(_node_df())
    ga = compute_graph_metric_attribution(_graph_df())
    summary = build_phase7_attribution_summary(na, ga)
    assert (summary["top_subgraph_contributor"] == "n/a").all()
    assert (summary["top_hazard_context_contributor"] == "n/a").all()


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def test_load_phase6_missing_core_raises(tmp_path):
    (tmp_path / "tables").mkdir()
    with pytest.raises(FileNotFoundError, match="run the Phase 6 notebook"):
        load_phase6_delta_tables(tmp_path)


def test_load_phase6_loads_present_tables(tmp_path):
    tdir = tmp_path / "tables"
    tdir.mkdir()
    _node_df().to_csv(tdir / "longitudinal_node_deltas.csv", index=False)
    _graph_df().to_csv(tdir / "longitudinal_graph_deltas.csv", index=False)
    loaded = load_phase6_delta_tables(tmp_path)
    assert set(loaded.keys()) == {"node_deltas", "graph_deltas"}
