"""Tests for Phase 4 graph feature extraction.

All tests use mock graphs; no real data files are required.
"""

from __future__ import annotations

import math

import networkx as nx
import pandas as pd
import pytest

from neurobridge_graph.graph_features import (
    ACTIVE_THRESHOLD,
    extract_all_edge_level_features,
    extract_all_graph_level_features,
    extract_all_node_level_features,
    extract_edge_level_features,
    extract_graph_level_features,
    extract_node_level_features,
)
from neurobridge_graph.subgraph_features import (
    SUBGRAPH_TEMPLATES,
    extract_all_subgraph_features,
    extract_subgraph_features,
    extract_subgraph_features_for_template,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_graph(subject_id: str = "test_001") -> nx.Graph:
    """Small mock graph with 3 nodes and 2 edges."""
    G = nx.Graph()
    G.graph["subject_id"] = subject_id
    G.graph["baci_score"] = 1.2
    G.graph["baci_category"] = "moderate"
    G.graph["top_domain"] = "cardiovascular regulation"
    G.graph["n_active_domains"] = 2
    G.graph["max_domain_activation"] = 1.45

    G.add_node("cardiovascular regulation", domain="cardiovascular regulation",
               activation=1.45, activation_level="moderate", domain_score=1.45,
               interpretation="Moderate activation.")
    G.add_node("metabolic regulation", domain="metabolic regulation",
               activation=1.1, activation_level="moderate", domain_score=1.1,
               interpretation="Moderate activation.")
    G.add_node("hematologic / oxygen-carrying", domain="hematologic / oxygen-carrying",
               activation=0.6, activation_level="low", domain_score=0.6,
               interpretation="Low activation.")

    G.add_edge("cardiovascular regulation", "metabolic regulation",
               edge_type="conceptual_biological_relationship",
               weight=1.0, relationship="CV and metabolic are linked.",
               interpretation="Conceptual relationship, not causal proof.")
    G.add_edge("cardiovascular regulation", "hematologic / oxygen-carrying",
               edge_type="within_subject_coactivation",
               weight=1.025, relationship="",
               interpretation="Co-activation observed.")
    return G


def _make_empty_graph(subject_id: str = "empty_001") -> nx.Graph:
    """Graph with nodes but no edges."""
    G = nx.Graph()
    G.graph["subject_id"] = subject_id
    G.add_node("cardiovascular regulation", domain="cardiovascular regulation",
               activation=0.5, activation_level="low", domain_score=0.5,
               interpretation="Low activation.")
    return G


def _make_two_graphs() -> dict[str, nx.Graph]:
    G1 = _make_graph("subj_A")
    G2 = _make_graph("subj_B")
    G2.nodes["cardiovascular regulation"]["activation"] = 0.3
    G2.nodes["cardiovascular regulation"]["activation_level"] = "low"
    return {"subj_A": G1, "subj_B": G2}


# ---------------------------------------------------------------------------
# Graph-level feature tests
# ---------------------------------------------------------------------------

class TestGraphLevelFeatures:
    def test_returns_dict(self):
        feat = extract_graph_level_features(_make_graph())
        assert isinstance(feat, dict)

    def test_expected_keys(self):
        feat = extract_graph_level_features(_make_graph())
        expected = [
            "subject_id", "n_nodes", "n_edges", "graph_density",
            "mean_node_activation", "median_node_activation",
            "max_node_activation", "total_node_activation",
            "n_active_domains", "active_domain_fraction",
            "mean_edge_weight", "max_edge_weight",
            "conceptual_edge_count", "coactivation_edge_count",
            "top_domain", "top_domain_activation",
            "baci_score", "baci_category",
        ]
        for k in expected:
            assert k in feat, f"Missing key: {k}"

    def test_n_nodes_correct(self):
        feat = extract_graph_level_features(_make_graph())
        assert feat["n_nodes"] == 3

    def test_n_edges_correct(self):
        feat = extract_graph_level_features(_make_graph())
        assert feat["n_edges"] == 2

    def test_n_active_domains(self):
        feat = extract_graph_level_features(_make_graph())
        # activations: 1.45, 1.1, 0.6 → 2 active (≥1.0)
        assert feat["n_active_domains"] == 2

    def test_active_domain_fraction(self):
        feat = extract_graph_level_features(_make_graph())
        assert abs(feat["active_domain_fraction"] - 2 / 3) < 1e-4

    def test_max_activation(self):
        feat = extract_graph_level_features(_make_graph())
        assert abs(feat["max_node_activation"] - 1.45) < 1e-4

    def test_edge_type_counts(self):
        feat = extract_graph_level_features(_make_graph())
        assert feat["conceptual_edge_count"] == 1
        assert feat["coactivation_edge_count"] == 1

    def test_baci_passthrough(self):
        feat = extract_graph_level_features(_make_graph())
        assert feat["baci_score"] == 1.2
        assert feat["baci_category"] == "moderate"

    def test_no_edges_graph(self):
        feat = extract_graph_level_features(_make_empty_graph())
        assert feat["n_edges"] == 0
        assert feat["mean_edge_weight"] == 0.0
        assert feat["graph_density"] == 0.0

    def test_extract_all_returns_dataframe(self):
        df = extract_all_graph_level_features(_make_two_graphs())
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_extract_all_subject_ids(self):
        df = extract_all_graph_level_features(_make_two_graphs())
        assert set(df["subject_id"]) == {"subj_A", "subj_B"}


# ---------------------------------------------------------------------------
# Node-level feature tests
# ---------------------------------------------------------------------------

class TestNodeLevelFeatures:
    def test_returns_dataframe(self):
        df = extract_node_level_features(_make_graph())
        assert isinstance(df, pd.DataFrame)

    def test_row_count(self):
        df = extract_node_level_features(_make_graph())
        assert len(df) == 3

    def test_expected_columns(self):
        df = extract_node_level_features(_make_graph())
        for col in ["subject_id", "domain", "activation", "activation_level",
                    "domain_score", "degree", "weighted_degree",
                    "degree_centrality", "is_active", "is_top_domain",
                    "interpretation"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_is_active_logic(self):
        df = extract_node_level_features(_make_graph())
        # Only activations ≥ 1.0 are active
        for _, row in df.iterrows():
            expected = row["activation"] >= ACTIVE_THRESHOLD
            assert row["is_active"] == expected

    def test_is_top_domain(self):
        df = extract_node_level_features(_make_graph())
        top_rows = df[df["is_top_domain"]]
        assert len(top_rows) == 1
        assert top_rows.iloc[0]["domain"] == "cardiovascular regulation"

    def test_degree_nonnegative(self):
        df = extract_node_level_features(_make_graph())
        assert (df["degree"] >= 0).all()

    def test_no_edges_graph(self):
        df = extract_node_level_features(_make_empty_graph())
        assert len(df) == 1
        assert df.iloc[0]["degree"] == 0
        assert df.iloc[0]["degree_centrality"] == 0.0

    def test_extract_all_concat(self):
        df = extract_all_node_level_features(_make_two_graphs())
        assert len(df) == 6  # 3 nodes × 2 graphs


# ---------------------------------------------------------------------------
# Edge-level feature tests
# ---------------------------------------------------------------------------

class TestEdgeLevelFeatures:
    def test_returns_dataframe(self):
        df = extract_edge_level_features(_make_graph())
        assert isinstance(df, pd.DataFrame)

    def test_row_count(self):
        df = extract_edge_level_features(_make_graph())
        assert len(df) == 2

    def test_expected_columns(self):
        df = extract_edge_level_features(_make_graph())
        for col in ["subject_id", "source", "target", "edge_type",
                    "weight", "connects_active_domains",
                    "relationship", "interpretation"]:
            assert col in df.columns

    def test_connects_active_domains_flag(self):
        df = extract_edge_level_features(_make_graph())
        # CV (1.45, active) + metabolic (1.1, active) → True
        cv_met = df[
            df["source"].isin(["cardiovascular regulation", "metabolic regulation"]) &
            df["target"].isin(["cardiovascular regulation", "metabolic regulation"])
        ]
        if not cv_met.empty:
            assert cv_met.iloc[0]["connects_active_domains"] == True  # noqa: E712

    def test_no_edges_graph_empty_df(self):
        df = extract_edge_level_features(_make_empty_graph())
        assert len(df) == 0

    def test_extract_all_concat(self):
        df = extract_all_edge_level_features(_make_two_graphs())
        assert len(df) == 4  # 2 edges × 2 graphs


# ---------------------------------------------------------------------------
# Subgraph feature tests
# ---------------------------------------------------------------------------

class TestSubgraphFeatures:
    def test_returns_dataframe(self):
        df = extract_subgraph_features(_make_graph())
        assert isinstance(df, pd.DataFrame)

    def test_one_row_per_template(self):
        df = extract_subgraph_features(_make_graph())
        assert len(df) == len(SUBGRAPH_TEMPLATES)

    def test_expected_columns(self):
        df = extract_subgraph_features(_make_graph())
        for col in ["subject_id", "subgraph_name", "available_nodes",
                    "n_available_nodes", "n_active_nodes",
                    "subgraph_activation_mean", "subgraph_activation_sum",
                    "subgraph_activation_max", "subgraph_active_fraction",
                    "dominant_node", "interpretation"]:
            assert col in df.columns

    def test_cardiometabolic_template_finds_nodes(self):
        df = extract_subgraph_features(_make_graph())
        cm = df[df["subgraph_name"] == "cardiometabolic"].iloc[0]
        # Graph has cardiovascular + metabolic → at least 2 of 3 template nodes
        assert cm["n_available_nodes"] >= 2

    def test_missing_node_template_no_crash(self):
        """Template with zero matching nodes must not crash; returns NaN stats."""
        df = extract_subgraph_features_for_template(
            _make_graph(),
            "nonexistent_template",
            ["completely nonexistent domain A", "completely nonexistent domain B"],
        )
        assert df["n_available_nodes"] == 0
        assert math.isnan(df["subgraph_activation_mean"])

    def test_active_fraction_range(self):
        df = extract_subgraph_features(_make_graph())
        valid = df[df["n_available_nodes"] > 0]
        assert (valid["subgraph_active_fraction"] >= 0.0).all()
        assert (valid["subgraph_active_fraction"] <= 1.0).all()

    def test_extract_all_shape(self):
        graphs = _make_two_graphs()
        df = extract_all_subgraph_features(graphs)
        assert len(df) == 2 * len(SUBGRAPH_TEMPLATES)

    def test_no_edges_graph_no_crash(self):
        df = extract_subgraph_features(_make_empty_graph())
        assert isinstance(df, pd.DataFrame)

    def test_active_threshold_logic(self):
        """Nodes below threshold should not count as active."""
        G = _make_empty_graph()
        # activation=0.5 < ACTIVE_THRESHOLD
        row = extract_subgraph_features_for_template(
            G, "cardiometabolic",
            ["cardiovascular regulation", "metabolic regulation"],
        )
        # Only cardiovascular is in the graph → 1 available, 0 active
        assert row["n_available_nodes"] == 1
        assert row["n_active_nodes"] == 0
