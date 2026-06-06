"""Tests for Phase 9 dashboard data helpers (mock data only, no Streamlit)."""

from __future__ import annotations

import pandas as pd

from neurobridge_graph.dashboard_data import (
    REQUIRED_TABLES,
    OPTIONAL_TABLES,
    load_dashboard_tables,
    has_required_tables,
    build_dashboard_readiness_report,
    get_available_subjects,
    get_available_timepoints,
    get_subject_timepoint_context,
    filter_subject_timepoint,
    get_domain_delta_panel_data,
    get_graph_metric_panel_data,
    get_hazard_context_panel_data,
    get_attribution_panel_data,
    get_envelope_panel_data,
    get_recovery_panel_data,
)


# --------------------------------------------------------------------------
# Mock builders
# --------------------------------------------------------------------------

def _node_deltas() -> pd.DataFrame:
    return pd.DataFrame([
        {"subject_id": "S1", "timepoint": "T0_baseline", "mission_phase": "baseline",
         "time_index": 0, "data_type": "schema_demonstration_not_scientific_evidence",
         "domain": "Cardiovascular regulation", "baseline_activation": 0.8,
         "current_activation": 0.8, "delta_activation": 0.0,
         "absolute_delta_activation": 0.0, "direction": "stable"},
        {"subject_id": "S1", "timepoint": "T2_inflight", "mission_phase": "inflight",
         "time_index": 2, "data_type": "schema_demonstration_not_scientific_evidence",
         "domain": "Cardiovascular regulation", "baseline_activation": 0.8,
         "current_activation": 1.1, "delta_activation": 0.3,
         "absolute_delta_activation": 0.3, "direction": "increase"},
        {"subject_id": "S2", "timepoint": "T0_baseline", "mission_phase": "baseline",
         "time_index": 0, "data_type": "schema_demonstration_not_scientific_evidence",
         "domain": "Metabolic regulation", "baseline_activation": 0.7,
         "current_activation": 0.7, "delta_activation": 0.0,
         "absolute_delta_activation": 0.0, "direction": "stable"},
    ])


def _graph_deltas() -> pd.DataFrame:
    return pd.DataFrame([
        {"subject_id": "S1", "timepoint": "T2_inflight", "mission_phase": "inflight",
         "time_index": 2, "metric": "mean_node_activation", "baseline_value": 0.85,
         "current_value": 1.05, "delta_value": 0.2, "absolute_delta_value": 0.2},
    ])


def _full_tables() -> dict[str, pd.DataFrame]:
    return {
        "node_deltas": _node_deltas(),
        "graph_deltas": _graph_deltas(),
        "hazard_deltas": pd.DataFrame([
            {"subject_id": "S1", "timepoint": "T2_inflight", "mission_phase": "inflight",
             "hazard": "gravity_fields", "baseline_hazard_relevance": 0.8,
             "current_hazard_relevance": 1.0, "delta_hazard_relevance": 0.2,
             "coverage_fraction": 0.8},
        ]),
        "attribution_summary": pd.DataFrame([
            {"subject_id": "S1", "timepoint": "T2_inflight", "mission_phase": "inflight",
             "top_domain_contributor": "Cardiovascular regulation",
             "top_domain_contribution_share": 0.75,
             "top_graph_metric_contributor": "mean_node_activation",
             "top_subgraph_contributor": "cardiometabolic",
             "top_hazard_context_contributor": "gravity_fields",
             "recovery_summary": "returned_near_baseline:4",
             "interpretation": "At T2_inflight, driven by Cardiovascular regulation."},
        ]),
        "node_attribution": pd.DataFrame([
            {"subject_id": "S1", "timepoint": "T2_inflight", "mission_phase": "inflight",
             "domain": "Cardiovascular regulation", "delta_activation": 0.3,
             "contribution_share": 0.75, "direction": "increase", "attribution_rank": 1},
        ]),
        "envelope_summary": pd.DataFrame([
            {"subject_id": "S1", "timepoint": "T2_inflight", "mission_phase": "inflight",
             "n_outside_node_envelope": 2, "n_outside_graph_envelope": 1,
             "n_outside_hazard_envelope": 0, "top_outside_domain": "Cardiovascular regulation",
             "top_outside_graph_metric": "mean_node_activation",
             "top_outside_hazard_context": "n/a",
             "overall_envelope_flag": "outside_expected_envelope",
             "interpretation": "Outside envelope."},
        ]),
        "node_envelope_scores": pd.DataFrame([
            {"subject_id": "S1", "timepoint": "T2_inflight", "mission_phase": "inflight",
             "domain": "Cardiovascular regulation", "delta_activation": 0.3,
             "lower_bound": -0.12, "upper_bound": 0.12, "robust_z": 4.0,
             "envelope_position": "outside_expected_envelope", "envelope_exceedance": 0.18},
        ]),
        "recovery_metrics": pd.DataFrame([
            {"subject_id": "S1", "metric": "mean_node_activation", "baseline_value": 0.83,
             "peak_value": 1.22, "final_value": 0.92, "recovery_fraction": 0.77},
        ]),
        "recovery_attribution": pd.DataFrame([
            {"subject_id": "S1", "metric": "mean_node_activation",
             "recovery_fraction": 0.77, "recovery_category": "returned_near_baseline"},
        ]),
    }


# --------------------------------------------------------------------------
# Loading / readiness
# --------------------------------------------------------------------------

def test_load_handles_missing_optional(tmp_path):
    tdir = tmp_path / "tables"
    tdir.mkdir()
    _node_deltas().to_csv(tdir / REQUIRED_TABLES["node_deltas"], index=False)
    _graph_deltas().to_csv(tdir / REQUIRED_TABLES["graph_deltas"], index=False)
    tables = load_dashboard_tables(tdir)
    assert "node_deltas" in tables and "graph_deltas" in tables
    # non-derivable optional tables simply absent, no crash
    assert "recovery_metrics" not in tables
    # hazard_deltas is derived on the fly from node deltas + hazard mapping
    assert "hazard_deltas" in tables
    assert not tables["hazard_deltas"].empty
    assert "top_contributing_domain" in tables["hazard_deltas"].columns
    assert has_required_tables(tables)


def test_load_empty_dir_no_required(tmp_path):
    tdir = tmp_path / "tables"
    tdir.mkdir()
    tables = load_dashboard_tables(tdir)
    assert tables == {}
    assert not has_required_tables(tables)


def test_readiness_includes_required_and_optional():
    tables = {"node_deltas": _node_deltas(), "graph_deltas": _graph_deltas()}
    report = build_dashboard_readiness_report(tables)
    assert set(report["required_or_optional"]) == {"required", "optional"}
    # required tables present -> loaded
    req = report[report["required_or_optional"] == "required"]
    assert (req["status"] == "loaded").all()
    # a known optional missing -> missing
    miss = report[report["table_name"] == OPTIONAL_TABLES["hazard_deltas"]]
    assert miss.iloc[0]["status"] == "missing"


# --------------------------------------------------------------------------
# Subjects / timepoints
# --------------------------------------------------------------------------

def test_available_subjects():
    assert get_available_subjects(_full_tables()) == ["S1", "S2"]


def test_available_subjects_empty():
    assert get_available_subjects({}) == []


def test_available_timepoints_sorted_by_index():
    tables = _full_tables()
    tps = get_available_timepoints(tables, "S1")
    assert tps == ["T0_baseline", "T2_inflight"]


def test_filter_subject_timepoint():
    out = filter_subject_timepoint(_node_deltas(), "S1", "T2_inflight")
    assert len(out) == 1
    assert out.iloc[0]["domain"] == "Cardiovascular regulation"


def test_filter_subject_timepoint_empty_df():
    assert filter_subject_timepoint(pd.DataFrame(), "S1", "T2").empty


# --------------------------------------------------------------------------
# Context
# --------------------------------------------------------------------------

def test_context_available():
    ctx = get_subject_timepoint_context(_full_tables(), "S1", "T2_inflight")
    assert ctx["available"] is True
    assert ctx["mission_phase"] == "inflight"
    assert "schema_demonstration" in ctx["data_type"]


def test_context_missing_selection():
    ctx = get_subject_timepoint_context(_full_tables(), "S1", "T9_none")
    assert ctx["available"] is False
    assert ctx["note"]


# --------------------------------------------------------------------------
# Trajectory panels
# --------------------------------------------------------------------------

def test_domain_panel():
    out = get_domain_delta_panel_data(_full_tables(), "S1")
    assert not out.empty
    assert set(out["timepoint"]) == {"T0_baseline", "T2_inflight"}


def test_graph_panel():
    out = get_graph_metric_panel_data(_full_tables(), "S1")
    assert not out.empty
    assert "metric" in out.columns


def test_hazard_panel_present_and_missing():
    out = get_hazard_context_panel_data(_full_tables(), "S1")
    assert not out.empty
    # missing optional table -> empty
    assert get_hazard_context_panel_data({"node_deltas": _node_deltas()}, "S1").empty


def test_recovery_panel_merges_category():
    out = get_recovery_panel_data(_full_tables(), "S1")
    assert not out.empty
    assert "recovery_category" in out.columns
    assert out.iloc[0]["recovery_category"] == "returned_near_baseline"


def test_recovery_panel_empty_when_missing():
    assert get_recovery_panel_data({"node_deltas": _node_deltas()}, "S1").empty


# --------------------------------------------------------------------------
# Attribution / envelope panels (safe dicts)
# --------------------------------------------------------------------------

def test_attribution_panel_full():
    data = get_attribution_panel_data(_full_tables(), "S1", "T2_inflight")
    assert data["available"] is True
    assert not data["top_domains"].empty
    assert data["summary"]["top_domain_contributor"] == "Cardiovascular regulation"


def test_attribution_panel_safe_dict_when_missing():
    data = get_attribution_panel_data({"node_deltas": _node_deltas(),
                                       "graph_deltas": _graph_deltas()},
                                      "S1", "T2_inflight")
    assert data["available"] is False
    assert data["note"]
    assert isinstance(data["top_domains"], pd.DataFrame)
    assert data["top_domains"].empty


def test_envelope_panel_full():
    data = get_envelope_panel_data(_full_tables(), "S1", "T2_inflight")
    assert data["available"] is True
    assert data["overall_flag"] == "outside_expected_envelope"
    assert data["n_outside_node"] == 2


def test_envelope_panel_safe_dict_when_missing():
    data = get_envelope_panel_data({"node_deltas": _node_deltas(),
                                    "graph_deltas": _graph_deltas()},
                                   "S1", "T2_inflight")
    assert data["available"] is False
    assert data["note"]
    assert data["overall_flag"] == "n/a"
