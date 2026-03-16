"""Tests for DREAMCell."""

import pytest
import torch
from dream_net.core.config import DREAMConfig
from dream_net.core.cell import DREAMCell
from dream_net.core.state import DREAMState


def test_cell_init():
    """Test cell initialization."""
    config = DREAMConfig(input_dim=80, hidden_dim=256, rank=8)
    cell = DREAMCell(config)

    assert cell.C.shape == (80, 256)
    assert cell.W.shape == (256, 80)
    assert cell.B.shape == (256, 80)
    assert cell.V.shape == (80, 8)


def test_cell_init_state():
    """Test state initialization."""
    config = DREAMConfig(input_dim=80, hidden_dim=256, rank=8)
    cell = DREAMCell(config)
    state = cell.init_state(batch_size=4)

    assert state.h.shape == (4, 256)
    assert state.U.shape == (4, 256, 8)
    assert state.U_target.shape == (4, 256, 8)
    assert state.adaptive_tau.shape == (4,)


def test_cell_forward():
    """Test basic forward pass."""
    config = DREAMConfig(input_dim=80, hidden_dim=256, rank=8)
    cell = DREAMCell(config)
    state = cell.init_state(batch_size=4)

    x = torch.randn(4, 80)
    h_new, state_new = cell(x, state)

    assert h_new.shape == (4, 256)
    assert state_new.h.shape == (4, 256)
    assert state_new.U.shape == (4, 256, 8)


def test_cell_forward_sequence():
    """Test sequence processing."""
    config = DREAMConfig(input_dim=80, hidden_dim=256, rank=8)
    cell = DREAMCell(config)

    x_seq = torch.randn(4, 10, 80)  # (batch, time, input)
    output, state = cell.forward_sequence(x_seq, return_all=True)

    assert output.shape == (4, 10, 256)
    assert state.h.shape == (4, 256)


def test_cell_surprise_gate():
    """Test surprise gate computation."""
    config = DREAMConfig(input_dim=80, hidden_dim=256, rank=8)
    cell = DREAMCell(config)
    state = cell.init_state(batch_size=4)

    error = torch.randn(4, 80)
    error_norm = error.norm(dim=-1)

    surprise = cell.surprise_gate(error, error_norm, state)

    assert surprise.shape == (4,)
    assert (surprise >= 0).all()
    assert (surprise <= 1).all()


def test_cell_ltc_update():
    """Test LTC state update."""
    config = DREAMConfig(input_dim=80, hidden_dim=256, rank=8)
    cell = DREAMCell(config)
    state = cell.init_state(batch_size=4)

    h_prev = torch.randn(4, 256)
    input_effect = torch.randn(4, 256)
    surprise = torch.rand(4)

    h_new = cell.compute_ltc_update(h_prev, input_effect, surprise)

    assert h_new.shape == (4, 256)
    assert not torch.isnan(h_new).any()
    assert not torch.isinf(h_new).any()


def test_cell_fast_weights_update():
    """Test fast weights Hebbian update."""
    config = DREAMConfig(input_dim=80, hidden_dim=256, rank=8)
    cell = DREAMCell(config)
    state = cell.init_state(batch_size=4)

    h_prev = torch.randn(4, 256)
    error = torch.randn(4, 80)
    surprise = torch.rand(4)

    initial_U = state.U.clone()
    cell.update_fast_weights(h_prev, error, surprise, state)

    # U should change
    assert not torch.allclose(state.U, initial_U)

    # U should be bounded (normalization)
    u_norm = state.U.norm(dim=(1, 2))
    assert (u_norm <= config.target_norm * 1.5 + 1e-6).all()


def test_cell_batch_consistency():
    """Test that batch size 1 works correctly."""
    config = DREAMConfig(input_dim=80, hidden_dim=256, rank=8)
    cell = DREAMCell(config)

    # Batch size 1
    state1 = cell.init_state(batch_size=1)
    x1 = torch.randn(1, 80)
    h1, _ = cell(x1, state1)

    # Batch size 4
    state4 = cell.init_state(batch_size=4)
    x4 = torch.randn(4, 80)
    h4, _ = cell(x4, state4)

    assert h1.shape == (1, 256)
    assert h4.shape == (4, 256)


def test_cell_device_cpu():
    """Test cell works on CPU."""
    config = DREAMConfig(input_dim=80, hidden_dim=256, rank=8)
    cell = DREAMCell(config)
    state = cell.init_state(batch_size=4)

    x = torch.randn(4, 80)
    h, state = cell(x, state)

    assert h.device.type == "cpu"
    assert state.h.device.type == "cpu"
