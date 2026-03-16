"""Tests for DREAM high-level API."""

import pytest
import torch
from dream_net import DREAM, DREAMStack, DREAMConfig


def test_dream_init():
    """Test DREAM model initialization."""
    model = DREAM(input_dim=80, hidden_dim=256, rank=8)

    assert model.hidden_dim == 256
    assert model.cell.config.rank == 8


def test_dream_forward():
    """Test DREAM forward pass."""
    model = DREAM(input_dim=80, hidden_dim=256, rank=8)

    x = torch.randn(4, 10, 80)  # (batch, time, input)
    output, state = model(x)

    assert output.shape == (4, 10, 256)
    assert state.h.shape == (4, 256)


def test_dream_forward_no_sequences():
    """Test DREAM with return_sequences=False."""
    model = DREAM(input_dim=80, hidden_dim=256, rank=8)

    x = torch.randn(4, 10, 80)
    output, state = model(x, return_sequences=False)

    assert output.shape == (4, 256)  # Only final timestep


def test_dreamstack_init():
    """Test DREAMStack initialization."""
    model = DREAMStack(
        input_dim=80,
        hidden_dims=[256, 128, 64],
        rank=8,
    )

    assert len(model.layers) == 3
    assert model.hidden_dims == [256, 128, 64]


def test_dreamstack_forward():
    """Test DREAMStack forward pass."""
    model = DREAMStack(
        input_dim=80,
        hidden_dims=[256, 128],
        rank=8,
    )

    x = torch.randn(4, 10, 80)
    output, states = model(x)

    assert output.shape == (4, 10, 128)
    assert len(states) == 2


def test_dreamstack_dropout():
    """Test DREAMStack with dropout."""
    model = DREAMStack(
        input_dim=80,
        hidden_dims=[256, 128],
        rank=8,
        dropout=0.1,
    )

    x = torch.randn(4, 10, 80)
    output, states = model(x)

    assert output.shape == (4, 10, 128)


def test_dream_init_state():
    """Test DREAM state initialization."""
    model = DREAM(input_dim=80, hidden_dim=256, rank=8)
    state = model.init_state(batch_size=4)

    assert state.h.shape == (4, 256)


def test_dream_with_mask():
    """Test DREAM with length masking."""
    model = DREAM(input_dim=80, hidden_dim=256, rank=8)

    x = torch.randn(4, 10, 80)
    lengths = torch.tensor([5, 8, 10, 7])  # Different lengths

    output, state = model.forward_with_mask(x, lengths)

    assert output.shape == (4, 10, 256)
    # Padded positions should be zero
    assert (output[0, 5:, :] == 0).all()
