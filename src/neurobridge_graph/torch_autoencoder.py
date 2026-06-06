"""Phase 12 — Lightweight PyTorch autoencoder for graph-derived trajectory features.

This is a self-supervised representation-learning prototype. It is intentionally
**not** a clinical prediction model: it does not predict health outcomes,
diagnose conditions, classify mission readiness, measure hazard exposure, or
produce health risk scores. It learns to reconstruct baseline-relative,
within-subject graph trajectory feature vectors.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def choose_latent_dim(input_dim: int, max_latent_dim: int = 8) -> int:
    """Choose a small latent dimension based on the input dimension.

    The latent bottleneck is kept strictly smaller than the input and modest in
    absolute size to avoid overfitting a tiny dataset.
    """
    if input_dim <= 2:
        return max(1, min(input_dim, max_latent_dim))
    candidate = max(2, input_dim // 3)
    return int(min(max_latent_dim, candidate, max(1, input_dim - 1)))


def _default_hidden_dims(input_dim: int, latent_dim: int) -> list[int]:
    h1 = max(2 * latent_dim, min(64, input_dim))
    if input_dim >= 16:
        return [h1, max(latent_dim, h1 // 2)]
    return [h1]


class TrajectoryAutoencoder(nn.Module):
    """Fully connected autoencoder for graph-derived trajectory features.

    This is intentionally not a clinical prediction model. It maps a trajectory
    feature vector to a compact latent representation and back to a
    reconstructed feature vector.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 8,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.05,
    ):
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be >= 1.")
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dims = list(hidden_dims) if hidden_dims is not None else \
            _default_hidden_dims(input_dim, latent_dim)
        self.dropout = float(dropout)

        # Encoder: input -> hidden... -> latent (no activation on the bottleneck).
        enc_layers: list[nn.Module] = []
        prev = self.input_dim
        for h in self.hidden_dims:
            enc_layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(self.dropout)]
            prev = h
        enc_layers.append(nn.Linear(prev, self.latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder: latent -> reversed hidden... -> output (linear, can be negative).
        dec_layers: list[nn.Module] = []
        prev = self.latent_dim
        for h in reversed(self.hidden_dims):
            dec_layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(self.dropout)]
            prev = h
        dec_layers.append(nn.Linear(prev, self.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> "tuple[torch.Tensor, torch.Tensor]":
        """Return ``(reconstruction, latent_embedding)``."""
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
