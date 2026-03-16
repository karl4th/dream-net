"""Tests for DREAMConfig."""

import pytest
from dream_net.core.config import DREAMConfig


def test_config_defaults():
    """Test default configuration values."""
    config = DREAMConfig()

    assert config.input_dim == 39
    assert config.hidden_dim == 256
    assert config.rank == 16
    assert config.time_step == 0.1
    assert config.forgetting_rate == 0.01
    assert config.base_plasticity == 0.1


def test_config_custom():
    """Test custom configuration."""
    config = DREAMConfig(
        input_dim=80,
        hidden_dim=512,
        rank=8,
        forgetting_rate=0.05,
    )

    assert config.input_dim == 80
    assert config.hidden_dim == 512
    assert config.rank == 8
    assert config.forgetting_rate == 0.05


def test_config_adaptive_forgetting():
    """Test adaptive forgetting parameter."""
    config = DREAMConfig(adaptive_forgetting_scale=10.0)
    assert config.adaptive_forgetting_scale == 10.0
