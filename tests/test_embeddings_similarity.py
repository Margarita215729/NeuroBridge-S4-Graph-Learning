"""Tests for Phase 5 embeddings.py and similarity.py.

Uses small in-memory mock data; no project files required.
"""

import numpy as np
import pandas as pd
import pytest

from neurobridge_graph.embeddings import (
    load_phase4_feature_tables,
    build_hazard_aware_feature_matrix,
    select_numeric_features,
    scale_feature_matrix,
    compute_pca_embedding,
)
from neurobridge_graph.similarity import (
    compute_cosine_similarity_matrix,
    compute_euclidean_distance_matrix,
    summarize_pairwise_similarity,
    identify_most_similar_pair,
    identify_most_distinct_subject,
)


def _mock_graph_level() -> pd.DataFrame:
    return pd.DataFrame([
        {"subject_id": "S1", "mean_node_activation": 0.8, "max_node_activation": 2.0,
         "total_node_activation": 5.0, "n_active_domains": 1, "active_domain_fraction": 0.17,
         "graph_density": 0.27, "coactivation_edge_count": 0, "conceptual_edge_count": 4,
         "top_domain_activation": 2.0, "baci_score": 28.6},
        {"subject_id": "S2", "mean_node_activation": 1.05, "max_node_activation": 2.1,
         "total_node_activation": 6.3, "n_active_domains": 2, "active_domain_fraction": 0.33,
         "graph_density": 0.33, "coactivation_edge_count": 1, "conceptual_edge_count": 4,
         "top_domain_activation": 2.1, "baci_score": 41.5},
        {"subject_id": "S3", "mean_node_activation": 0.7, "max_node_activation": 1.16,
         "total_node_activation": 4.2, "n_active_domains": 2, "active_domain_fraction": 0.33,
         "graph_density": 0.27, "coactivation_edge_count": 0, "conceptual_edge_count": 4,
         "top_domain_activation": 1.16, "baci_score": 31.5},
    ])


def _mock_node_level() -> pd.DataFrame:
    rows = []
    for sid in ("S1", "S2", "S3"):
        for dom, act in [
            ("Cardiovascular regulation", 0.5),
            ("Metabolic regulation", 1.2),
            ("Hematologic / oxygen-carrying", 2.0),
        ]:
            rows.append({"subject_id": sid, "domain": dom, "activation": act})
    return pd.DataFrame(rows)


def _mock_hazard_scores() -> pd.DataFrame:
    rows = []
    for sid in ("S1", "S2", "S3"):
        for hz, val in [
            ("space_radiation", 0.9),
            ("gravity_fields", 1.1),
            ("isolation_and_confinement", float("nan")),
        ]:
            rows.append({"subject_id": sid, "hazard": hz, "hazard_relevance_score": val})
    return pd.DataFrame(rows)


def _mock_subgraph() -> pd.DataFrame:
    rows = []
    for sid in ("S1", "S2", "S3"):
        for name, mean in [("cardiometabolic", 1.0), ("immune_metabolic", 0.6)]:
            rows.append({"subject_id": sid, "subgraph_name": name,
                         "subgraph_activation_mean": mean})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# load_phase4_feature_tables
# ---------------------------------------------------------------------------

def test_load_phase4_requires_core_tables(tmp_path):
    (tmp_path / "tables").mkdir()
    with pytest.raises(FileNotFoundError):
        load_phase4_feature_tables(tmp_path)


def test_load_phase4_loads_present_tables(tmp_path):
    tdir = tmp_path / "tables"
    tdir.mkdir()
    _mock_graph_level().to_csv(tdir / "graph_level_features.csv", index=False)
    _mock_node_level().to_csv(tdir / "node_level_features.csv", index=False)
    loaded = load_phase4_feature_tables(tmp_path)
    assert "graph_level_features" in loaded
    assert "node_level_features" in loaded
    assert "subgraph_features" not in loaded  # not written


# ---------------------------------------------------------------------------
# build_hazard_aware_feature_matrix
# ---------------------------------------------------------------------------

def test_feature_matrix_includes_hazard_columns():
    fm = build_hazard_aware_feature_matrix(
        _mock_graph_level(), _mock_node_level(),
        subgraph_features=_mock_subgraph(),
        hazard_relevance_scores=_mock_hazard_scores(),
    )
    assert "subject_id" in fm.columns
    hazard_cols = [c for c in fm.columns if c.startswith("hazard_relevance__")]
    assert "hazard_relevance__space_radiation" in hazard_cols
    assert "hazard_relevance__gravity_fields" in hazard_cols
    # NaN hazard score (isolation) filled with 0.0.
    assert (fm["hazard_relevance__isolation_and_confinement"] == 0.0).all()
    # Subgraph columns present.
    assert "subgraph_activation_mean__cardiometabolic" in fm.columns
    # BACI present.
    assert "baci_score" in fm.columns
    assert len(fm) == 3


def test_feature_matrix_no_nans():
    fm = build_hazard_aware_feature_matrix(
        _mock_graph_level(), _mock_node_level(),
        subgraph_features=_mock_subgraph(),
        hazard_relevance_scores=_mock_hazard_scores(),
    )
    numeric = fm.drop(columns=["subject_id"])
    assert not numeric.isna().any().any()


def test_select_numeric_features_excludes_subject_id():
    fm = build_hazard_aware_feature_matrix(_mock_graph_level(), _mock_node_level())
    X, names = select_numeric_features(fm)
    assert "subject_id" not in names
    assert X.shape[0] == 3
    assert len(names) == X.shape[1]


# ---------------------------------------------------------------------------
# scaling + PCA
# ---------------------------------------------------------------------------

def test_scale_preserves_shape():
    fm = build_hazard_aware_feature_matrix(_mock_graph_level(), _mock_node_level())
    X, _ = select_numeric_features(fm)
    Xs, scaler = scale_feature_matrix(X, method="standard")
    assert Xs.shape == X.shape
    assert list(Xs.columns) == list(X.columns)
    assert not Xs.isna().any().any()  # constant cols -> 0, not NaN


def test_scale_unknown_method_raises():
    fm = build_hazard_aware_feature_matrix(_mock_graph_level(), _mock_node_level())
    X, _ = select_numeric_features(fm)
    with pytest.raises(ValueError):
        scale_feature_matrix(X, method="nope")


def test_pca_small_input_ok():
    fm = build_hazard_aware_feature_matrix(_mock_graph_level(), _mock_node_level())
    X, names = select_numeric_features(fm)
    Xs, _ = scale_feature_matrix(X)
    emb, pca = compute_pca_embedding(Xs, list(fm["subject_id"]), n_components=2)
    assert "PC1" in emb.columns
    assert len(emb) == 3
    assert pca is not None


def test_pca_single_subject_skips_gracefully():
    one = _mock_graph_level().iloc[:1]
    fm = build_hazard_aware_feature_matrix(one, _mock_node_level())
    X, _ = select_numeric_features(fm)
    Xs, _ = scale_feature_matrix(X)
    emb, pca = compute_pca_embedding(Xs, list(fm["subject_id"]))
    assert pca is None
    assert "note" in emb.columns


# ---------------------------------------------------------------------------
# similarity / distance
# ---------------------------------------------------------------------------

def test_cosine_diagonal_is_one():
    fm = build_hazard_aware_feature_matrix(
        _mock_graph_level(), _mock_node_level(),
        hazard_relevance_scores=_mock_hazard_scores(),
    )
    X, _ = select_numeric_features(fm)
    Xs, _ = scale_feature_matrix(X)
    ids = list(fm["subject_id"])
    sim = compute_cosine_similarity_matrix(Xs, ids)
    assert sim.shape == (3, 3)
    assert np.allclose(np.diag(sim.values), 1.0)


def test_euclidean_diagonal_is_zero():
    fm = build_hazard_aware_feature_matrix(_mock_graph_level(), _mock_node_level())
    X, _ = select_numeric_features(fm)
    Xs, _ = scale_feature_matrix(X)
    ids = list(fm["subject_id"])
    dist = compute_euclidean_distance_matrix(Xs, ids)
    assert np.allclose(np.diag(dist.values), 0.0)
    # Distance matrix is symmetric.
    assert np.allclose(dist.values, dist.values.T)


def test_summary_and_identifiers():
    fm = build_hazard_aware_feature_matrix(_mock_graph_level(), _mock_node_level())
    X, _ = select_numeric_features(fm)
    Xs, _ = scale_feature_matrix(X)
    ids = list(fm["subject_id"])
    sim = compute_cosine_similarity_matrix(Xs, ids)
    dist = compute_euclidean_distance_matrix(Xs, ids)
    summary = summarize_pairwise_similarity(sim, dist)
    # 3 subjects -> 3 unique pairs.
    assert len(summary) == 3
    assert {"subject_a", "subject_b", "cosine_similarity", "euclidean_distance"}.issubset(
        summary.columns
    )
    most_similar = identify_most_similar_pair(summary)
    assert most_similar["subject_a"] is not None
    most_distinct = identify_most_distinct_subject(dist)
    assert most_distinct["subject_id"] in ids


def test_identify_most_similar_empty():
    empty = pd.DataFrame(columns=["subject_a", "subject_b", "cosine_similarity",
                                   "euclidean_distance"])
    res = identify_most_similar_pair(empty)
    assert res["subject_a"] is None
