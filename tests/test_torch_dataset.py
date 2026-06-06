"""Tests for Phase 12 dataset construction (synthetic data only)."""

import numpy as np
import pandas as pd
import torch

from neurobridge_graph.torch_dataset import (
    build_phase12_input_readiness_report,
    build_trajectory_feature_matrix,
    encode_resilience_metadata,
    select_numeric_model_features,
    scale_model_features,
    create_masked_training_data,
    build_mask_summary,
    TrajectoryFeatureDataset,
)


def _node_deltas():
    rows = []
    for subj in ("S1", "S2"):
        for ti, (tp, ph) in enumerate([("T0", "baseline"), ("T1", "inflight"), ("T2", "recovery")]):
            for dom, val in (("metabolic regulation", 0.1 * ti), ("cardiovascular regulation", -0.2 * ti)):
                rows.append({"subject_id": subj, "timepoint": tp, "mission_phase": ph,
                             "time_index": ti, "domain": dom, "delta_activation": val})
    return pd.DataFrame(rows)


def _graph_deltas():
    rows = []
    for subj in ("S1", "S2"):
        for ti, tp in enumerate(("T0", "T1", "T2")):
            for m, v in (("density", 0.05 * ti), ("mean_node_activation", 0.02 * ti)):
                rows.append({"subject_id": subj, "timepoint": tp, "mission_phase": "x",
                             "metric": m, "delta_value": v})
    return pd.DataFrame(rows)


def _resilience():
    rows = []
    for subj in ("S1", "S2"):
        for tp, lab in (("T0", "Stable compensated trajectory"),
                        ("T1", "Systemic strain pattern"),
                        ("T2", "Localized adaptive shift")):
            rows.append({"subject_id": subj, "timepoint": tp,
                         "resilience_state_label": lab,
                         "dominant_adaptation_mode": "Multi-subgraph distributed",
                         "confidence_level": "high"})
    return pd.DataFrame(rows)


def test_readiness_report_handles_missing_optional():
    rep = build_phase12_input_readiness_report({"longitudinal_node_deltas": _node_deltas()})
    assert (rep["table_name"] == "longitudinal_node_deltas").any()
    missing = rep[rep["status"] == "missing"]
    assert not missing.empty  # optional tables reported as missing, not crashing


def test_build_feature_matrix_from_mock_tables():
    tables = {"longitudinal_node_deltas": _node_deltas(),
              "longitudinal_graph_deltas": _graph_deltas()}
    fm, cat = build_trajectory_feature_matrix(tables)
    assert len(fm) == 6  # 2 subjects x 3 timepoints
    assert "trajectory_id" in fm.columns
    assert any(c.startswith("domain_delta__") for c in fm.columns)
    assert any(c.startswith("graph_delta__") for c in fm.columns)
    assert not cat.empty


def test_build_feature_matrix_empty_when_no_node_deltas():
    fm, cat = build_trajectory_feature_matrix({})
    assert fm.empty
    assert list(cat.columns) == ["feature_name", "source_table", "feature_family", "description"]


def test_encode_resilience_metadata_when_present():
    fm, _ = build_trajectory_feature_matrix({"longitudinal_node_deltas": _node_deltas()})
    fm2, enc = encode_resilience_metadata(fm, _resilience())
    assert "resilience_state_label" in fm2.columns
    assert (fm2["resilience_state_label"] != "not_available").any()
    assert not enc.empty


def test_encode_resilience_metadata_missing_is_safe():
    fm, _ = build_trajectory_feature_matrix({"longitudinal_node_deltas": _node_deltas()})
    fm2, enc = encode_resilience_metadata(fm, None)
    assert (fm2["resilience_state_label"] == "not_available").all()
    assert enc.empty


def test_select_numeric_features_excludes_metadata():
    fm, _ = build_trajectory_feature_matrix({"longitudinal_node_deltas": _node_deltas()})
    fm, _ = encode_resilience_metadata(fm, _resilience())
    X, names = select_numeric_model_features(fm)
    for bad in ("subject_id", "timepoint", "mission_phase", "trajectory_id",
                "resilience_state_label", "dominant_adaptation_mode"):
        assert bad not in names
    assert all(pd.api.types.is_numeric_dtype(X[c]) for c in names)


def test_scaling_preserves_shape():
    fm, _ = build_trajectory_feature_matrix({"longitudinal_node_deltas": _node_deltas()})
    X, _ = select_numeric_model_features(fm)
    Xs, scaler = scale_model_features(X)
    assert Xs.shape == X.shape
    assert set(scaler["mean"]) == set(X.columns)


def test_masked_training_data_fraction_and_zeros():
    fm, _ = build_trajectory_feature_matrix({"longitudinal_node_deltas": _node_deltas()})
    X, _ = select_numeric_model_features(fm)
    Xs, _ = scale_model_features(X)
    Xm, Xt, mask = create_masked_training_data(Xs, mask_fraction=0.3, random_state=0)
    assert Xm.shape == Xt.shape == mask.shape
    # masked positions are zeroed
    assert np.all(Xm[mask.astype(bool)] == 0.0)
    # approximately the requested fraction
    assert abs(mask.mean() - 0.3) < 0.2


def test_build_mask_summary():
    mask = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    summ = build_mask_summary(mask, ["f0", "f1"])
    assert (summ["scope"] == "overall").any()
    assert (summ["scope"] == "feature").sum() == 2


def test_dataset_returns_tensors():
    X = np.random.rand(4, 3).astype(np.float32)
    ds = TrajectoryFeatureDataset(X)
    assert len(ds) == 4
    xi, xt = ds[0]
    assert isinstance(xi, torch.Tensor) and isinstance(xt, torch.Tensor)
    assert xi.shape == (3,)
