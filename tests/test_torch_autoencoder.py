"""Tests for the Phase 12 PyTorch autoencoder (synthetic data only)."""

import torch

from neurobridge_graph.torch_autoencoder import (
    TrajectoryAutoencoder,
    choose_latent_dim,
    count_trainable_parameters,
)


def test_choose_latent_dim_is_small_and_bounded():
    assert choose_latent_dim(2) <= 2
    assert choose_latent_dim(45, max_latent_dim=8) == 8
    # latent strictly smaller than input for reasonable sizes
    assert choose_latent_dim(6) < 6
    assert choose_latent_dim(30, max_latent_dim=8) <= 8


def test_forward_returns_reconstruction_and_latent():
    model = TrajectoryAutoencoder(input_dim=12, latent_dim=4)
    x = torch.randn(5, 12)
    recon, z = model(x)
    assert recon.shape == (5, 12)
    assert z.shape == (5, 4)


def test_encode_decode_roundtrip_shapes():
    model = TrajectoryAutoencoder(input_dim=20, latent_dim=6)
    x = torch.randn(3, 20)
    z = model.encode(x)
    assert z.shape == (3, 6)
    recon = model.decode(z)
    assert recon.shape == (3, 20)


def test_two_hidden_layers_for_large_input():
    model = TrajectoryAutoencoder(input_dim=32, latent_dim=8)
    assert len(model.hidden_dims) == 2


def test_parameter_count_positive():
    model = TrajectoryAutoencoder(input_dim=10, latent_dim=3)
    n = count_trainable_parameters(model)
    assert isinstance(n, int) and n > 0


def test_small_input_dim_does_not_crash():
    model = TrajectoryAutoencoder(input_dim=2, latent_dim=choose_latent_dim(2))
    recon, z = model(torch.randn(4, 2))
    assert recon.shape == (4, 2)
