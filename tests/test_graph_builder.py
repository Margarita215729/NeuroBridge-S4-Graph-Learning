"""Tests for graph_builder.py — Phase 3."""

import pytest
import pandas as pd
import networkx as nx

from neurobridge_graph.graph_builder import (
    normalize_name,
    canonical_domain_name,
    classify_activation,
    detect_subject_id_column,
    detect_domain_columns,
    build_subject_graph,
    build_all_subject_graphs,
    export_node_table,
    export_edge_table,
    export_graph_summary,
    GUARDRAIL,
    GRAPH_TYPE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_domain_scores() -> pd.DataFrame:
    """Tiny domain-scores DataFrame for testing (index = subject IDs)."""
    return pd.DataFrame(
        {
            "Cardiovascular regulation": [0.48, 0.95],
            "Metabolic regulation":      [0.49, 0.84],
            "Hematologic / oxygen-carrying capacity": [2.11, 0.63],
        },
        index=pd.Index(["Subject_A", "Subject_B"], name="Participant"),
    )


def _mock_baci_df() -> pd.DataFrame:
    return pd.DataFrame({
        "crew":     ["Subject_A", "Subject_B"],
        "BACI":     [41.5, 22.1],
        "category": ["mild coherence", "low coherence"],
        "shifted_domains": [2, 1],
    })


# ---------------------------------------------------------------------------
# normalize_name
# ---------------------------------------------------------------------------

def test_normalize_name_strips_whitespace():
    assert normalize_name("  Hello World  ") == "hello world"


def test_normalize_name_collapses_spaces():
    assert normalize_name("a  b   c") == "a b c"


def test_normalize_name_lowercases():
    assert normalize_name("Cardiovascular Regulation") == "cardiovascular regulation"


# ---------------------------------------------------------------------------
# canonical_domain_name
# ---------------------------------------------------------------------------

def test_canonical_hematologic_variants():
    variants = [
        "Hematologic / oxygen-carrying capacity",
        "hematologic / oxygen-carrying",
        "Hematologic",
    ]
    for v in variants:
        assert canonical_domain_name(v) == "hematologic / oxygen-carrying"


def test_canonical_inflammation_variants():
    assert canonical_domain_name("Inflammation / immune-adjacent status") == \
           "inflammation / immune-adjacent"
    assert canonical_domain_name("Inflammation") == "inflammation / immune-adjacent"


def test_canonical_recovery_markers():
    assert canonical_domain_name("Recovery-related markers") == "recovery-related markers"
    assert canonical_domain_name("Recovery capacity") == "recovery capacity"


def test_canonical_unknown_passthrough():
    result = canonical_domain_name("Some Unknown Domain")
    assert isinstance(result, str)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# classify_activation
# ---------------------------------------------------------------------------

def test_classify_activation_low():
    assert classify_activation(0.0) == "low"
    assert classify_activation(0.74) == "low"


def test_classify_activation_mild():
    assert classify_activation(0.75) == "mild"
    assert classify_activation(0.99) == "mild"


def test_classify_activation_moderate():
    assert classify_activation(1.0) == "moderate"
    assert classify_activation(1.49) == "moderate"


def test_classify_activation_high():
    assert classify_activation(1.5) == "high"
    assert classify_activation(3.0) == "high"


def test_classify_activation_negative():
    # Negative domain scores should be treated by absolute value
    assert classify_activation(-2.0) == "high"


# ---------------------------------------------------------------------------
# detect_subject_id_column
# ---------------------------------------------------------------------------

def test_detect_subject_id_seqn():
    df = pd.DataFrame({"SEQN": [1, 2], "Val": [3, 4]})
    assert detect_subject_id_column(df) == "SEQN"


def test_detect_subject_id_participant():
    df = pd.DataFrame({"Participant ID": [1, 2], "Score": [3, 4]})
    assert detect_subject_id_column(df) == "Participant ID"


def test_detect_subject_id_fallback():
    df = pd.DataFrame({"col_a": [1], "col_b": [2]})
    result = detect_subject_id_column(df)
    assert result in df.columns


# ---------------------------------------------------------------------------
# detect_domain_columns
# ---------------------------------------------------------------------------

def test_detect_domain_columns_excludes_id():
    df = pd.DataFrame({
        "SEQN": [1],
        "Cardiovascular regulation": [0.5],
        "Metabolic regulation": [0.3],
    })
    subject_col = "SEQN"
    cols = detect_domain_columns(df, subject_col)
    assert "SEQN" not in cols
    assert "Cardiovascular regulation" in cols
    assert "Metabolic regulation" in cols


def test_detect_domain_columns_excludes_baci():
    df = pd.DataFrame({
        "crew": ["a"],
        "BACI": [40.0],
        "category": ["mild"],
        "Cardiovascular regulation": [0.5],
    })
    cols = detect_domain_columns(df, "crew")
    assert "BACI" not in cols
    assert "Cardiovascular regulation" in cols


# ---------------------------------------------------------------------------
# build_subject_graph
# ---------------------------------------------------------------------------

def test_build_subject_graph_returns_nx_graph():
    row = pd.Series({
        "Cardiovascular regulation": 0.48,
        "Metabolic regulation": 0.49,
        "Hematologic / oxygen-carrying capacity": 2.11,
    })
    G = build_subject_graph("test_subject", row)
    assert isinstance(G, nx.Graph)


def test_build_subject_graph_node_count():
    row = pd.Series({
        "Cardiovascular regulation": 0.48,
        "Metabolic regulation": 0.49,
        "Hematologic / oxygen-carrying capacity": 2.11,
    })
    G = build_subject_graph("test_subject", row)
    assert G.number_of_nodes() == 3


def test_build_subject_graph_node_attributes():
    row = pd.Series({"Hematologic / oxygen-carrying capacity": 2.11})
    G = build_subject_graph("sub1", row)
    node = "Hematologic / oxygen-carrying capacity"
    assert node in G.nodes
    attrs = G.nodes[node]
    assert attrs["activation_level"] == "high"
    assert attrs["node_type"] == "biological_domain"
    assert "interpretation" in attrs
    assert "data_source" in attrs
    assert attrs["subject_id"] == "sub1"


def test_build_subject_graph_graph_metadata():
    row = pd.Series({"Cardiovascular regulation": 0.48})
    G = build_subject_graph("sub_meta", row)
    assert G.graph["guardrail"] == GUARDRAIL
    assert G.graph["graph_type"] == GRAPH_TYPE
    assert G.graph["subject_id"] == "sub_meta"
    assert "n_domains" in G.graph
    assert "top_domain" in G.graph


def test_build_subject_graph_has_edges_for_known_pair():
    """Cardiovascular regulation ↔ Metabolic regulation should get a conceptual edge."""
    row = pd.Series({
        "Cardiovascular regulation": 0.5,
        "Metabolic regulation": 0.5,
    })
    G = build_subject_graph("edge_test", row)
    assert G.has_edge("Cardiovascular regulation", "Metabolic regulation") or \
           G.has_edge("Metabolic regulation", "Cardiovascular regulation")


def test_build_subject_graph_coactivation_edge():
    """Two high-activation domains should get a co-activation edge."""
    row = pd.Series({
        "Cardiovascular regulation": 2.0,
        "Metabolic regulation": 1.8,
    })
    G = build_subject_graph("coact_test", row, activation_threshold=1.0)
    edge_types = [
        attrs.get("edge_type")
        for _, _, attrs in G.edges(data=True)
    ]
    assert "within_subject_coactivation" in edge_types or any(
        attrs.get("coactivation") for _, _, attrs in G.edges(data=True)
    )


def test_build_subject_graph_no_coactivation_when_disabled():
    row = pd.Series({
        "Cardiovascular regulation": 2.0,
        "Metabolic regulation": 1.8,
    })
    G = build_subject_graph("nocoact", row, add_coactivation_edges=False)
    for _, _, attrs in G.edges(data=True):
        assert attrs.get("edge_type") != "within_subject_coactivation"


# ---------------------------------------------------------------------------
# build_all_subject_graphs
# ---------------------------------------------------------------------------

def test_build_all_subject_graphs_count():
    df = _mock_domain_scores()
    graphs = build_all_subject_graphs(df)
    assert len(graphs) == 2


def test_build_all_subject_graphs_keys():
    df = _mock_domain_scores()
    graphs = build_all_subject_graphs(df)
    assert "Subject_A" in graphs
    assert "Subject_B" in graphs


def test_build_all_subject_graphs_with_baci():
    df = _mock_domain_scores()
    baci = _mock_baci_df()
    graphs = build_all_subject_graphs(df, baci_df=baci)
    G_a = graphs["Subject_A"]
    assert G_a.graph["baci_score"] == 41.5


# ---------------------------------------------------------------------------
# export helpers
# ---------------------------------------------------------------------------

def test_export_node_table_shape():
    df = _mock_domain_scores()
    graphs = build_all_subject_graphs(df)
    node_df = export_node_table(graphs)
    assert len(node_df) == 2 * 3  # 2 subjects × 3 domains


def test_export_edge_table_nonempty():
    row = pd.Series({
        "Cardiovascular regulation": 0.5,
        "Metabolic regulation": 0.5,
    })
    graphs = {"sub": build_subject_graph("sub", row)}
    edge_df = export_edge_table(graphs)
    assert len(edge_df) >= 1


def test_export_graph_summary_columns():
    df = _mock_domain_scores()
    graphs = build_all_subject_graphs(df)
    summary = export_graph_summary(graphs)
    assert "subject_id" in summary.columns
    assert "n_nodes" in summary.columns
    assert "n_edges" in summary.columns


# ---------------------------------------------------------------------------
# save_graphml (filesystem)
# ---------------------------------------------------------------------------

def test_save_graphml_files(tmp_path):
    from neurobridge_graph.graph_builder import save_graphml_files
    row = pd.Series({"Cardiovascular regulation": 0.5})
    graphs = {"sub_99": build_subject_graph("sub_99", row)}
    paths = save_graphml_files(graphs, tmp_path / "graphs")
    assert len(paths) == 1
    assert paths[0].exists()
    assert paths[0].suffix == ".graphml"
    # Verify it can be read back
    G2 = nx.read_graphml(paths[0])
    assert G2.number_of_nodes() == 1
