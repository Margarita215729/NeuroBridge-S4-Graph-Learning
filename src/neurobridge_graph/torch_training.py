"""Phase 12 — Deterministic, CPU-compatible training and evaluation utilities.

Training optimizes a self-supervised reconstruction objective only. There is no
clinical outcome label, no health risk target, and no mission-readiness target.
Reconstruction error is a representation-quality signal, not a risk score.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

MIN_TRAJECTORY_ROWS = 3
MIN_NUMERIC_FEATURES = 2


def set_torch_seed(seed: int = 42) -> None:
    """Set seeds across random, numpy, and torch for deterministic runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(device: str | None = None) -> str:
    """Resolve a compute device. Defaults to CPU; CUDA only if requested and available."""
    if device is None:
        return "cpu"
    if device == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def sufficient_for_training(X: pd.DataFrame) -> "tuple[bool, str]":
    """Return ``(ok, reason)`` describing whether there is enough data to train."""
    n_rows = 0 if X is None else int(X.shape[0])
    n_feat = 0 if X is None else int(X.shape[1])
    if n_rows < MIN_TRAJECTORY_ROWS:
        return False, (f"Only {n_rows} trajectory row(s); need >= {MIN_TRAJECTORY_ROWS}. "
                       "Training skipped.")
    if n_feat < MIN_NUMERIC_FEATURES:
        return False, (f"Only {n_feat} numeric feature(s); need >= {MIN_NUMERIC_FEATURES}. "
                       "Training skipped.")
    return True, "Sufficient data for self-supervised reconstruction training."


def train_autoencoder(
    model,
    dataset,
    epochs: int = 300,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str | None = None,
    verbose: bool = False,
) -> "tuple[object, pd.DataFrame]":
    """Train the autoencoder on the reconstruction task.

    Returns ``(trained_model, training_history)`` where the history has columns
    ``epoch`` and ``reconstruction_loss``.
    """
    dev = resolve_device(device)
    model = model.to(dev)
    bs = max(1, min(int(batch_size), len(dataset)))
    loader = DataLoader(dataset, batch_size=bs, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss()

    history: list[dict] = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for x_in, x_target in loader:
            x_in, x_target = x_in.to(dev), x_target.to(dev)
            optimizer.zero_grad()
            recon, _ = model(x_in)
            loss = loss_fn(recon, x_target)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        mean_loss = epoch_loss / max(1, n_batches)
        history.append({"epoch": epoch, "reconstruction_loss": round(mean_loss, 8)})
        if verbose and (epoch == 1 or epoch % 50 == 0 or epoch == epochs):
            print(f"epoch {epoch:4d} | reconstruction_loss {mean_loss:.6f}")

    return model, pd.DataFrame(history, columns=["epoch", "reconstruction_loss"])


def compute_reconstruction_error_table(
    X_true: np.ndarray,
    X_reconstructed: np.ndarray,
    feature_names: list[str],
    metadata_df: pd.DataFrame,
) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Return ``(row_level_errors, feature_level_errors)`` from squared errors."""
    X_true = np.asarray(X_true, dtype=float)
    X_rec = np.asarray(X_reconstructed, dtype=float)
    sq = (X_true - X_rec) ** 2

    row_err = metadata_df.reset_index(drop=True).copy()
    row_err["reconstruction_mse"] = sq.mean(axis=1).round(8)
    row_err["reconstruction_rmse"] = np.sqrt(sq.mean(axis=1)).round(8)

    feat_err = pd.DataFrame({
        "feature_name": feature_names,
        "mean_squared_error": sq.mean(axis=0).round(8),
        "root_mean_squared_error": np.sqrt(sq.mean(axis=0)).round(8),
    }).sort_values("mean_squared_error", ascending=False).reset_index(drop=True)
    return row_err, feat_err


def compute_embeddings_and_reconstructions(
    model,
    X_scaled: pd.DataFrame,
    metadata_df: pd.DataFrame,
    device: str | None = None,
) -> "tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]":
    """Return ``(embeddings_df, row_reconstruction_df, feature_reconstruction_df)``."""
    dev = resolve_device(device)
    model = model.to(dev)
    model.eval()
    feature_names = list(X_scaled.columns)
    X = np.array(X_scaled.to_numpy(dtype=np.float32), copy=True)
    with torch.no_grad():
        recon_t, z_t = model(torch.from_numpy(X).to(dev))
    recon = recon_t.cpu().numpy()
    z = z_t.cpu().numpy()

    meta = metadata_df.reset_index(drop=True).copy()
    emb = meta.copy()
    for j in range(z.shape[1]):
        emb[f"latent_{j}"] = z[:, j].round(6)

    row_err, feat_err = compute_reconstruction_error_table(X, recon, feature_names, meta)
    return emb, row_err, feat_err


def compute_similarity_matrix(
    embeddings_df: pd.DataFrame,
    id_col: str = "trajectory_id",
) -> pd.DataFrame:
    """Cosine similarity matrix over latent embeddings (diagonal == 1)."""
    latent_cols = [c for c in embeddings_df.columns if c.startswith("latent_")]
    if not latent_cols or embeddings_df.empty:
        return pd.DataFrame()
    ids = (embeddings_df[id_col].astype(str).tolist()
           if id_col in embeddings_df.columns
           else [str(i) for i in range(len(embeddings_df))])
    Z = embeddings_df[latent_cols].to_numpy(dtype=float)
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1e-12
    Zn = Z / norms
    sim = np.clip(Zn @ Zn.T, -1.0, 1.0)
    np.fill_diagonal(sim, 1.0)
    return pd.DataFrame(sim.round(6), index=ids, columns=ids)


def save_model_state(
    model,
    output_path: "str | Path",
    model_metadata: dict,
) -> Path:
    """Save the model ``state_dict`` and metadata to ``output_path`` (.pt)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": dict(model_metadata)},
               output_path)
    return output_path
