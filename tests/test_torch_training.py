"""Tests for Phase 12 training/evaluation utilities (synthetic data only)."""

import numpy as np
import pandas as pd

from neurobridge_graph.torch_autoencoder import TrajectoryAutoencoder
from neurobridge_graph.torch_dataset import TrajectoryFeatureDataset
from neurobridge_graph.torch_training import (
    set_torch_seed,
    resolve_device,
    sufficient_for_training,
    train_autoencoder,
    compute_embeddings_and_reconstructions,
    compute_reconstruction_error_table,
    compute_similarity_matrix,
    save_model_state,
)


def _mock_scaled(n_rows=6, n_feat=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_rows, n_feat)).astype(np.float32)
    cols = [f"f{i}" for i in range(n_feat)]
    Xdf = pd.DataFrame(X, columns=cols)
    meta = pd.DataFrame({
        "trajectory_id": [f"S__{i}" for i in range(n_rows)],
        "subject_id": ["S"] * n_rows,
        "timepoint": [f"T{i}" for i in range(n_rows)],
        "mission_phase": ["x"] * n_rows,
    })
    return Xdf, meta


def test_resolve_device_defaults_cpu():
    assert resolve_device(None) == "cpu"
    assert resolve_device("cpu") == "cpu"


def test_sufficient_for_training_guard():
    ok, _ = sufficient_for_training(pd.DataFrame(np.zeros((5, 3))))
    assert ok
    bad_rows, reason = sufficient_for_training(pd.DataFrame(np.zeros((2, 3))))
    assert not bad_rows and "row" in reason.lower()
    bad_feat, _ = sufficient_for_training(pd.DataFrame(np.zeros((5, 1))))
    assert not bad_feat


def test_training_loop_runs_and_reduces_loss():
    set_torch_seed(42)
    Xdf, meta = _mock_scaled()
    ds = TrajectoryFeatureDataset(Xdf.to_numpy())
    model = TrajectoryAutoencoder(input_dim=Xdf.shape[1], latent_dim=2)
    model, hist = train_autoencoder(model, ds, epochs=80, batch_size=4)
    assert len(hist) == 80
    assert {"epoch", "reconstruction_loss"}.issubset(hist.columns)
    # Learning occurred: best achieved loss is below the first-epoch loss
    # (robust to per-epoch dropout/shuffle fluctuation).
    assert hist["reconstruction_loss"].min() < hist["reconstruction_loss"].iloc[0]


def test_embeddings_and_reconstruction_shapes():
    set_torch_seed(1)
    Xdf, meta = _mock_scaled()
    model = TrajectoryAutoencoder(input_dim=Xdf.shape[1], latent_dim=3)
    emb, rowerr, featerr = compute_embeddings_and_reconstructions(model, Xdf, meta)
    assert len([c for c in emb.columns if c.startswith("latent_")]) == 3
    assert len(rowerr) == len(Xdf)
    assert "reconstruction_mse" in rowerr.columns
    assert len(featerr) == Xdf.shape[1]


def test_reconstruction_error_table_direct():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    Xr = X + 0.5
    meta = pd.DataFrame({"trajectory_id": ["a", "b"]})
    row, feat = compute_reconstruction_error_table(X, Xr, ["f0", "f1"], meta)
    assert np.allclose(row["reconstruction_mse"].values, 0.25)
    assert len(feat) == 2


def test_similarity_matrix_diagonal_is_one():
    set_torch_seed(2)
    Xdf, meta = _mock_scaled()
    model = TrajectoryAutoencoder(input_dim=Xdf.shape[1], latent_dim=3)
    emb, _, _ = compute_embeddings_and_reconstructions(model, Xdf, meta)
    sim = compute_similarity_matrix(emb)
    assert sim.shape == (len(Xdf), len(Xdf))
    assert np.allclose(np.diag(sim.to_numpy()), 1.0)


def test_save_model_state(tmp_path):
    model = TrajectoryAutoencoder(input_dim=4, latent_dim=2)
    out = save_model_state(model, tmp_path / "m.pt", {"input_dim": 4, "latent_dim": 2})
    assert out.exists()
