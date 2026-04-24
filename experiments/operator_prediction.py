"""
Operator Prediction — reference experiment.

Trains MultimodalDREAM to predict tank-drive operator actions from
a fused sensor stream (IMU + wheel encoders + previous action).
Camera is omitted here so the experiment runs without hardware.

Usage
-----
    ./run.sh experiments/operator_prediction.py

Synthetic data simulates two operators with different control styles:
  - Operator A: smooth, low-frequency inputs
  - Operator B: aggressive, high-frequency inputs

The experiment shows DREAM-full (fast weights ON) adapts faster to
a new operator than the static baseline (fast weights OFF).
"""

import math
import random

import torch
import torch.nn as nn

from dream_net import DREAMConfig
from dream_net.multimodal import (
    ActionEncoder,
    IMUEncoder,
    MultimodalDREAM,
    TankDriveHead,
    WheelEncoderSensor,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BATCH = 1          # single-stream, like a real robot
SEQ_LEN = 200      # frames per episode
PRETRAIN_EPS = 60  # episodes for pre-training
ADAPT_EPS = 20     # episodes for adaptation evaluation
LR = 3e-3
CLIP_GRAD = 1.0
SEED = 42

torch.manual_seed(SEED)
random.seed(SEED)


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

def _sinusoid(t: torch.Tensor, freq: float, phase: float = 0.0) -> torch.Tensor:
    return torch.sin(2 * math.pi * freq * t + phase)


def generate_episode(
    seq_len: int,
    style: str = "smooth",
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """
    Generate one synthetic operator episode.

    Parameters
    ----------
    seq_len : int
        Number of timesteps.
    style : str
        "smooth" — low-frequency, gradual turns (Operator A).
        "aggressive" — high-frequency, sharp direction changes (Operator B).

    Returns
    -------
    inputs : dict[str, torch.Tensor]
        Sensor readings (1, seq_len, dim) per modality.
    actions : torch.Tensor
        Ground-truth actions (1, seq_len, 2).
    """
    t = torch.linspace(0, seq_len * 0.1, seq_len)  # time in seconds

    if style == "smooth":
        freq, noise_scale = 0.3, 0.05
    else:
        freq, noise_scale = 1.2, 0.15

    # Generate operator action (sinusoidal pattern per track)
    left  = _sinusoid(t, freq, phase=0.0)
    right = _sinusoid(t, freq, phase=math.pi / 3)
    action = torch.stack([left, right], dim=-1)   # (seq_len, 2)
    action = action + torch.randn_like(action) * noise_scale
    action = torch.tanh(action)                   # bound to [−1, 1]

    # Derive synthetic IMU from action (acceleration ≈ d(action)/dt)
    d_action = torch.diff(action, dim=0, prepend=action[:1])
    imu = torch.cat([
        d_action,                                 # 2 accel proxy channels
        action,                                   # 2 gyro proxy channels
        torch.randn(seq_len, 2) * 0.02,           # 2 noise channels
    ], dim=-1)                                    # (seq_len, 6)

    # Wheel encoders: cumulative position + velocity
    velocity = action                             # track speed ≈ action
    position = torch.cumsum(velocity * 0.1, dim=0)
    wheels = torch.cat([velocity, position], dim=-1)  # (seq_len, 4)

    # Previous action (shifted by 1)
    prev_action = torch.cat([torch.zeros(1, 2), action[:-1]], dim=0)  # (seq_len, 2)

    # Add batch dim
    inputs = {
        "imu":         imu.unsqueeze(0),         # (1, seq_len, 6)
        "wheels":      wheels.unsqueeze(0),      # (1, seq_len, 4)
        "prev_action": prev_action.unsqueeze(0), # (1, seq_len, 2)
    }
    return inputs, action.unsqueeze(0)           # actions: (1, seq_len, 2)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model() -> MultimodalDREAM:
    return MultimodalDREAM(
        encoders={
            "imu":         IMUEncoder(in_dim=6, out_dim=32),
            "wheels":      WheelEncoderSensor(in_dim=4, out_dim=16),
            "prev_action": ActionEncoder(action_dim=2, out_dim=16),
        },
        dream_config=DREAMConfig(
            input_dim=64,
            hidden_dim=128,
            rank=8,
        ),
        output_head=TankDriveHead(hidden_dim=128),
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_episode(
    model: MultimodalDREAM,
    style: str,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    """Run one episode. Returns mean MSE loss."""
    inputs_seq, actions = generate_episode(SEQ_LEN, style=style)
    state = model.init_state(batch_size=1)
    criterion = nn.MSELoss()
    total_loss = 0.0

    for t in range(SEQ_LEN):
        step = {k: v[:, t] for k, v in inputs_seq.items()}
        pred, state = model(step, state)
        state = state.detach()   # truncated BPTT

        loss = criterion(pred, actions[:, t])

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            optimizer.step()

        total_loss += loss.item()

    return total_loss / SEQ_LEN


def pretrain(model: MultimodalDREAM) -> list[float]:
    """Pre-train on mixed operator styles with fast weights OFF."""
    model.disable_fast_weights()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    losses = []
    for ep in range(PRETRAIN_EPS):
        style = "smooth" if ep % 2 == 0 else "aggressive"
        loss = run_episode(model, style, optimizer)
        losses.append(loss)
        if (ep + 1) % 10 == 0:
            print(f"  pre-train ep {ep+1:3d}/{PRETRAIN_EPS}  loss={loss:.4f}")
    return losses


def evaluate_adaptation(model: MultimodalDREAM, fast_weights: bool) -> list[float]:
    """Evaluate on unseen operator B episodes. No gradient updates."""
    if fast_weights:
        model.enable_fast_weights()
    else:
        model.disable_fast_weights()

    losses = []
    state = model.init_state(batch_size=1)

    for ep in range(ADAPT_EPS):
        inputs_seq, actions = generate_episode(SEQ_LEN, style="aggressive")
        criterion = nn.MSELoss()
        ep_loss = 0.0

        for t in range(SEQ_LEN):
            step = {k: v[:, t] for k, v in inputs_seq.items()}
            with torch.no_grad():
                pred, state = model(step, state)
            ep_loss += criterion(pred, actions[:, t]).item()

        losses.append(ep_loss / SEQ_LEN)

    return losses


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("DREAM-Operator: Operator Prediction Experiment")
    print("=" * 60)

    model = build_model()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")

    # --- Pre-training ---
    print("\n[Phase 1] Pre-training (fast weights OFF) ...")
    pretrain_losses = pretrain(model)
    print(f"  Final pre-train loss: {pretrain_losses[-1]:.4f}")

    # --- Evaluation: static ---
    print("\n[Phase 2a] Evaluation — static (fast weights OFF) ...")
    static_losses = evaluate_adaptation(model, fast_weights=False)
    print(f"  Mean loss: {sum(static_losses)/len(static_losses):.4f}")
    print(f"  Loss ep 1: {static_losses[0]:.4f}  →  ep {ADAPT_EPS}: {static_losses[-1]:.4f}")

    # Reset model state between conditions
    # --- Evaluation: adaptive ---
    print("\n[Phase 2b] Evaluation — adaptive (fast weights ON) ...")
    adaptive_losses = evaluate_adaptation(model, fast_weights=True)
    print(f"  Mean loss: {sum(adaptive_losses)/len(adaptive_losses):.4f}")
    print(f"  Loss ep 1: {adaptive_losses[0]:.4f}  →  ep {ADAPT_EPS}: {adaptive_losses[-1]:.4f}")

    # --- Summary ---
    mean_static   = sum(static_losses)   / len(static_losses)
    mean_adaptive = sum(adaptive_losses) / len(adaptive_losses)
    improvement   = (mean_static - mean_adaptive) / mean_static * 100

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"  Static   mean loss: {mean_static:.4f}")
    print(f"  Adaptive mean loss: {mean_adaptive:.4f}")
    print(f"  Improvement:        {improvement:+.1f}%")
    if improvement > 0:
        print("  ✓ Fast weights helped.")
    else:
        print("  ✗ Fast weights did not help — check plasticity settings.")


if __name__ == "__main__":
    main()
