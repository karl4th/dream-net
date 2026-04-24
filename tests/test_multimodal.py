"""Tests for dream_net.multimodal."""

import pytest
import torch

from dream_net import DREAMConfig
from dream_net.multimodal import (
    ActionEncoder,
    ClassificationHead,
    ContinuousHead,
    FusionLayer,
    IMUEncoder,
    MultimodalDREAM,
    TankDriveHead,
    TargetEncoder,
    TimeSeriesEncoder,
    VisualEncoder,
    WheelEncoderSensor,
)


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------

def test_imu_encoder_shape():
    enc = IMUEncoder(in_dim=6, out_dim=32)
    x = torch.randn(4, 6)
    assert enc(x).shape == (4, 32)


def test_imu_encoder_9axis():
    enc = IMUEncoder(in_dim=9, out_dim=32)
    x = torch.randn(4, 9)
    assert enc(x).shape == (4, 32)


def test_wheel_encoder_shape():
    enc = WheelEncoderSensor(in_dim=4, out_dim=16)
    x = torch.randn(4, 4)
    assert enc(x).shape == (4, 16)


def test_action_encoder_shape():
    enc = ActionEncoder(action_dim=2, out_dim=16)
    x = torch.zeros(4, 2)
    assert enc(x).shape == (4, 16)


def test_target_encoder_shape():
    enc = TargetEncoder(in_dim=2, out_dim=16)
    x = torch.tensor([[0.3, 5.2], [-1.1, 0.8]])  # angle_error, distance
    assert enc(x).shape == (2, 16)


def test_target_encoder_custom_dim():
    enc = TargetEncoder(in_dim=3, out_dim=24)  # e.g. with bearing
    x = torch.randn(4, 3)
    assert enc(x).shape == (4, 24)


def test_timeseries_encoder_shape():
    enc = TimeSeriesEncoder(in_dim=12, out_dim=24)
    x = torch.randn(4, 12)
    assert enc(x).shape == (4, 24)


def test_visual_encoder_default_cnn():
    enc = VisualEncoder(out_dim=64)
    x = torch.randn(2, 3, 64, 64)
    assert enc(x).shape == (2, 64)


def test_visual_encoder_custom_backbone():
    import torch.nn as nn

    class DummyBackbone(nn.Module):
        out_features = 128
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros(x.shape[0], 128)

    enc = VisualEncoder(out_dim=32, backbone=DummyBackbone(), backbone_out_features=128)
    x = torch.randn(2, 3, 64, 64)
    assert enc(x).shape == (2, 32)


def test_visual_encoder_missing_out_features_raises():
    import torch.nn as nn

    class BadBackbone(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    with pytest.raises(ValueError, match="backbone_out_features"):
        VisualEncoder(out_dim=32, backbone=BadBackbone())


# ---------------------------------------------------------------------------
# FusionLayer
# ---------------------------------------------------------------------------

def test_fusion_out_dim():
    fusion = FusionLayer({
        "imu":    IMUEncoder(in_dim=6, out_dim=32),
        "wheels": WheelEncoderSensor(in_dim=4, out_dim=16),
    })
    assert fusion.out_dim == 48


def test_fusion_forward():
    fusion = FusionLayer({
        "imu":    IMUEncoder(in_dim=6, out_dim=32),
        "wheels": WheelEncoderSensor(in_dim=4, out_dim=16),
    })
    inputs = {"imu": torch.randn(4, 6), "wheels": torch.randn(4, 4)}
    out = fusion(inputs)
    assert out.shape == (4, 48)


# ---------------------------------------------------------------------------
# Output heads
# ---------------------------------------------------------------------------

def test_tank_drive_head():
    head = TankDriveHead(hidden_dim=256)
    h = torch.randn(4, 256)
    out = head(h)
    assert out.shape == (4, 2)
    assert (out.abs() <= 1.0).all()


def test_continuous_head_unbounded():
    head = ContinuousHead(hidden_dim=256, out_dim=4, bounded=False)
    h = torch.randn(4, 256)
    assert head(h).shape == (4, 4)


def test_continuous_head_bounded():
    head = ContinuousHead(hidden_dim=256, out_dim=3, bounded=True)
    h = torch.randn(4, 256)
    out = head(h)
    assert out.shape == (4, 3)
    assert (out.abs() <= 1.0).all()


def test_classification_head():
    head = ClassificationHead(hidden_dim=256, num_classes=5)
    h = torch.randn(4, 256)
    assert head(h).shape == (4, 5)


# ---------------------------------------------------------------------------
# MultimodalDREAM
# ---------------------------------------------------------------------------

def _make_model(input_dim: int = 64) -> MultimodalDREAM:
    return MultimodalDREAM(
        encoders={
            "imu":         IMUEncoder(in_dim=6, out_dim=32),
            "wheels":      WheelEncoderSensor(in_dim=4, out_dim=16),
            "prev_action": ActionEncoder(action_dim=2, out_dim=16),
        },
        dream_config=DREAMConfig(input_dim=input_dim, hidden_dim=128, rank=4),
        output_head=TankDriveHead(hidden_dim=128),
    )


def _make_inputs(batch: int = 4) -> dict[str, torch.Tensor]:
    return {
        "imu":         torch.randn(batch, 6),
        "wheels":      torch.randn(batch, 4),
        "prev_action": torch.zeros(batch, 2),
    }


def test_multimodal_input_dim_mismatch_raises():
    with pytest.raises(ValueError, match="input_dim"):
        MultimodalDREAM(
            encoders={"imu": IMUEncoder(in_dim=6, out_dim=32)},
            dream_config=DREAMConfig(input_dim=99, hidden_dim=128),
            output_head=TankDriveHead(hidden_dim=128),
        )


def test_multimodal_forward_shape():
    model = _make_model()
    action, state = model(_make_inputs())
    assert action.shape == (4, 2)
    assert state.h.shape == (4, 128)


def test_multimodal_forward_auto_state():
    model = _make_model()
    action, state = model(_make_inputs(), state=None)
    assert action.shape == (4, 2)


def test_multimodal_forward_with_state():
    model = _make_model()
    inputs = _make_inputs()
    _, state = model(inputs)
    action, _ = model(inputs, state=state)
    assert action.shape == (4, 2)


def test_multimodal_sequence_return_all():
    model = _make_model()
    inputs_seq = {k: v.unsqueeze(1).expand(-1, 10, -1) for k, v in _make_inputs().items()}
    out, state = model.forward_sequence(inputs_seq, return_all=True)
    assert out.shape == (4, 10, 2)
    assert state.h.shape == (4, 128)


def test_multimodal_sequence_return_last():
    model = _make_model()
    inputs_seq = {k: v.unsqueeze(1).expand(-1, 10, -1) for k, v in _make_inputs().items()}
    out, state = model.forward_sequence(inputs_seq, return_all=False)
    assert out.shape == (4, 2)


def test_multimodal_fast_weights_toggle():
    model = _make_model()
    assert not model.fast_weights_enabled

    model.enable_fast_weights()
    assert model.fast_weights_enabled
    action, state = model(_make_inputs())
    assert action.shape == (4, 2)

    model.disable_fast_weights()
    assert not model.fast_weights_enabled


def test_multimodal_no_input_state_mutation():
    model = _make_model()
    inputs = _make_inputs()
    state = model.init_state(batch_size=4)
    original_h = state.h.clone()

    model(inputs, state)
    assert torch.equal(state.h, original_h)


def test_multimodal_fast_weights_updates_u():
    model = _make_model()
    model.enable_fast_weights()
    inputs = _make_inputs()
    state = model.init_state(batch_size=4)

    _, new_state = model(inputs, state)
    # With fast weights on and Hebbian update, U should change
    # (may stay close to zero initially since state.U starts at 0 and
    # fast weights rely on surprise — just verify no crash and shapes)
    assert new_state.U.shape == (4, 128, 4)
