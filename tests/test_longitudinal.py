"""Tests for Phase 6 longitudinal.py — within-subject trajectory pipeline."""

import numpy as np
import pandas as pd
import pytest
import networkx as nx

from neurobridge_graph.longitudinal import (
    detect_longitudinal_columns,
    validate_longitudinal_table,
    create_example_longitudinal_table,
    identify_baseline_timepoint,
    build_timepoint_graphs,
    compute_node_activation_delta,
    compute_graph_metric_delta,
    compute_longitudinal_delta_tables,
    SCHEMA_DEMO_DATA_TYPE,
    GRAPH_DELTA_METRICS,
)


def _tiny_longitudinal() -> pd.DataFrame:
    return pd.DataFrame([
        {"subject_id": "S1", "timepoint": "T0", "mission_phase": "baseline",
         "time_index": 0, "data_type": SCHEMA_DEMO_DATA_TYPE,
         "Cardiovascular regulation": 0.8, "Metabolic regulation": 0.9},
        {"subject_id": "S1", "timepoint": "T1", "mission_phase": "inflight",
         "time_index": 1, "data_type": SCHEMA_DEMO_DATA_TYPE,
         "Cardiovascular regulation": 1.2, "Metabolic regulation": 1.1},
        {"subject_id": "S1", "timepoint": "T2", "mission_phase": "recovery",
         "time_index": 2, "data_type": SCHEMA_DEMO_DATA_TYPE,
         "Cardiovascular regulation": 0.9, "Metabolic regulation": 0.95},
    ])


def _mock_graph(activation: float, domain: str = "Cardiovascular regulation") -> nx.Graph:
    G = nx.Graph()
    G.add_node(domain, domain=domain, activation=activation,
               activation_level="low", domain_score=activation)
    G.graph["subject_id"] = "S1"
    return G


# ---------------------------------------------------------------------------
# Column detection and validation
# ---------------------------------------------------------------------------

def test_detect_longitudinal_columns():
    df = _tiny_longitudinal()
    detected = detect_longitudinal_columns(df)
    assert detected["missing_required"] == []
    assert "Cardiovascular regulation" in detected["domain_cols"]
    assert detected["has_data_type"] is True
    assert detected["n_subjects"] == 1


def test_validate_longitudinal_table():
    report = validate_longitudinal_table(_tiny_longitudinal())
    assert "check" in report.columns
    assert any(report["check"] == "baseline_phase_present")
    baseline_row = report[report["check"] == "baseline_phase_present"].iloc[0]
    assert baseline_row["status"] == "ok"


def test_create_example_longitudinal_table_marked_as_schema_demo(tmp_path):
    out = tmp_path / "example.csv"
    df = create_example_longitudinal_table(out)
    assert out.exists()
    assert (df["data_type"] == SCHEMA_DEMO_DATA_TYPE).all()
    assert df["subject_id"].nunique() == 2
    assert df["timepoint"].nunique() == 5
    assert "baseline" in df["mission_phase"].values


# ---------------------------------------------------------------------------
# Baseline identification
# ---------------------------------------------------------------------------

def test_identify_baseline_prefers_baseline_phase():
    df = _tiny_longitudinal()
    tp = identify_baseline_timepoint(df[df["subject_id"] == "S1"])
    assert tp == "T0"


def test_identify_baseline_earliest_when_no_baseline_phase():
    df = pd.DataFrame([
        {"subject_id": "S1", "timepoint": "T2", "mission_phase": "inflight", "time_index": 2},
        {"subject_id": "S1", "timepoint": "T0", "mission_phase": "pre_mission", "time_index": 0},
    ])
    assert identify_baseline_timepoint(df) == "T0"


# ---------------------------------------------------------------------------
# Graph building and deltas
# ---------------------------------------------------------------------------

def test_build_timepoint_graphs():
    graphs = build_timepoint_graphs(_tiny_longitudinal())
    assert "S1" in graphs
    assert len(graphs["S1"]) == 3
    G = graphs["S1"]["T0"]
    assert G.graph["baseline_timepoint"] == "T0"
    assert G.graph["data_type"] == SCHEMA_DEMO_DATA_TYPE
    assert G.number_of_nodes() >= 2


def test_compute_node_activation_delta():
    b = _mock_graph(0.8)
    c = _mock_graph(1.2)
    delta = compute_node_activation_delta(b, c)
    assert "delta_activation" in delta.columns
    row = delta.iloc[0]
    assert np.isclose(row["delta_activation"], 0.4)
    assert row["direction"] == "increase"


def test_compute_graph_metric_delta():
    baseline = {"mean_node_activation": 0.8, "max_node_activation": 1.0,
                "n_active_domains": 0, "graph_density": 0.2,
                "coactivation_edge_count": 0, "total_node_activation": 2.0,
                "active_domain_fraction": 0.0}
    current = {"mean_node_activation": 1.2, "max_node_activation": 1.5,
               "n_active_domains": 2, "graph_density": 0.3,
               "coactivation_edge_count": 1, "total_node_activation": 3.5,
               "active_domain_fraction": 0.33}
    deltas = compute_graph_metric_delta(baseline, current)
    for m in GRAPH_DELTA_METRICS:
        assert m in deltas
    assert deltas["mean_node_activation"]["delta_value"] == pytest.approx(0.4)


def test_compute_longitudinal_delta_tables():
    graphs = build_timepoint_graphs(_tiny_longitudinal())
    tables = compute_longitudinal_delta_tables(graphs)
    assert "node_delta_table" in tables
    assert "graph_delta_table" in tables
    assert "trajectory_summary" in tables
    node = tables["node_delta_table"]
    assert not node.empty
    # Baseline timepoint deltas should be zero.
    baseline_rows = node[node["timepoint"] == "T0"]
    assert (baseline_rows["delta_activation"].abs() < 0.01).all()


def test_missing_optional_domains_do_not_crash():
    df = pd.DataFrame([
        {"subject_id": "S1", "timepoint": "T0", "mission_phase": "baseline",
         "time_index": 0, "Cardiovascular regulation": 0.5},
        {"subject_id": "S1", "timepoint": "T1", "mission_phase": "inflight",
         "time_index": 1, "Cardiovascular regulation": 1.0},
    ])
    graphs = build_timepoint_graphs(df)
    tables = compute_longitudinal_delta_tables(graphs)
    assert not tables["node_delta_table"].empty
