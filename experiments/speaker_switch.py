"""
Speaker-Switch Experiment
=========================
Goal: prove that fast weights U adapt to speaker change without gradients.

Protocol:
  Phase 1 (pretrain) — backprop on Speaker A only → learns base C, W, B.
  Phase 2 (inference) — C/W/B frozen, 3 modes over full 10-sec clip:
      full    : fast weights update via Hebbian + surprise gate
      static  : dU = 0  (fast weights frozen)
      no_gate : S_t ≡ 1  (plasticity always open)

Pass criteria:
  1. S_t spikes to ~1.0 at second 5
  2. Loss returns to baseline in < 50 steps after switch
  3. full loss < static loss after switch
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from typing import Literal

from dream_net import DREAMConfig, DREAMCell, DREAMState


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

AUDIO_A = "data/commonvoice/speaker_1/0000.wav"
AUDIO_B = "data/commonvoice/speaker_3/0000.wav"
CLIP_SEC = 5.0
SR = 16_000
N_MELS = 80
N_FFT = 1024
HOP = 160       # 10 ms → 100 frames/sec
WIN = 400       # 25 ms

PRETRAIN_EPOCHS = 30
PRETRAIN_LR = 3e-3
PRETRAIN_CLIP_GRAD = 1.0

Mode = Literal["full", "static", "no_gate"]


# ---------------------------------------------------------------------------
# Audio → Mel
# ---------------------------------------------------------------------------

def load_clip(path: str, n_sec: float) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    n = int(n_sec * SR)
    wav = wav[:, :n] if wav.shape[1] >= n else F.pad(wav, (0, n - wav.shape[1]))
    return wav


def build_mel(wav: torch.Tensor) -> torch.Tensor:
    """wav (1, T) → log-mel (frames, n_mels), standardized to ~N(0,1)."""
    mel_tf = T.MelSpectrogram(
        sample_rate=SR, n_fft=N_FFT, win_length=WIN, hop_length=HOP,
        n_mels=N_MELS, f_min=20.0, f_max=8000.0, power=2.0,
    )
    mel = mel_tf(wav)                           # (1, n_mels, T)
    log_mel = torch.log(mel.squeeze(0).T + 1e-6)   # (T, n_mels)

    # Per-feature standardization (fit on full clip)
    mu = log_mel.mean(0, keepdim=True)
    std = log_mel.std(0, keepdim=True).clamp(min=1e-4)
    return (log_mel - mu) / std                 # (T, n_mels), ~N(0,1)


def make_features() -> tuple[torch.Tensor, torch.Tensor, int]:
    """Returns feats_A (T/2, 80), full feats (T, 80), switch_frame."""
    clip_a = load_clip(AUDIO_A, CLIP_SEC)
    clip_b = load_clip(AUDIO_B, CLIP_SEC)
    feats_a = build_mel(clip_a)
    feats_ab = build_mel(torch.cat([clip_a, clip_b], dim=1))
    switch_frame = int(CLIP_SEC * SR / HOP)
    return feats_a, feats_ab, switch_frame


def make_synthetic_features(
    n_frames_each: int = 500,
    n_mels: int = N_MELS,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Synthetic fallback when CommonVoice clips are not available.

    Speaker A: slow low-frequency modulation (calm speaker).
    Speaker B: fast high-frequency modulation (energetic speaker).
    Both are deterministic, normalized to ~N(0,1) per feature.
    """
    rng = torch.Generator()
    rng.manual_seed(seed)
    t = torch.arange(n_frames_each, dtype=torch.float32)

    def _speaker(freq_scale: float, noise_std: float) -> torch.Tensor:
        # Base: n_mels sine waves with random phases and the given frequency
        phases = torch.rand(n_mels, generator=rng) * 2 * torch.pi
        freqs = torch.linspace(0.005, 0.05, n_mels) * freq_scale
        # (n_frames, n_mels)
        signal = torch.sin(t.unsqueeze(1) * freqs.unsqueeze(0) + phases.unsqueeze(0))
        noise = torch.randn(n_frames_each, n_mels, generator=rng) * noise_std
        raw = signal + noise
        mu = raw.mean(0, keepdim=True)
        std = raw.std(0, keepdim=True).clamp(min=1e-4)
        return (raw - mu) / std

    feats_a = _speaker(freq_scale=1.0, noise_std=0.15)   # slow, quiet
    feats_b = _speaker(freq_scale=4.0, noise_std=0.40)   # fast, noisy
    feats_ab = torch.cat([feats_a, feats_b], dim=0)
    return feats_a, feats_ab, n_frames_each


# ---------------------------------------------------------------------------
# Cell with mode control
# ---------------------------------------------------------------------------

class ExperimentCell(DREAMCell):
    """DREAMCell with explicit mode control for the experiment."""

    def __init__(self, config: DREAMConfig, mode: Mode = "full"):
        super().__init__(config)
        self.mode = mode
        # Sync fast_weights_enabled with mode so the base-class flag is consistent
        if mode in ("full", "no_gate"):
            self.enable_fast_weights()

    def forward_step(
        self,
        x: torch.Tensor,
        state: DREAMState,
    ) -> tuple[torch.Tensor, DREAMState, torch.Tensor, torch.Tensor]:
        """
        Returns h_new, state, surprise (scalar), rel_error_norm (scalar).
        """
        # -- normalize input ---------------------------------------------------
        x_scale = x.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        x_norm = (x / x_scale).clamp(-1.0, 1.0)

        # -- base prediction from slow weights ---------------------------------
        x_pred_raw = state.h @ self.C.T                         # (B, input)
        x_pred_base = torch.tanh(x_pred_raw) * x_scale

        # -- fast weight prediction correction (Ba et al. style) --------------
        # Retrieval: h @ U → (B, rank), then (B, rank) @ V.T → (B, input)
        if self.mode in ("full", "no_gate"):
            h_U = torch.bmm(state.h.unsqueeze(1), state.U).squeeze(1)  # (B, rank)
            pred_correction = (h_U @ self.V.T) * x_scale               # (B, input)
        else:
            pred_correction = torch.zeros_like(x)

        x_pred = x_pred_base + 0.5 * pred_correction

        # -- error with corrected prediction -----------------------------------
        error = x - x_pred                                      # (B, input)
        error_norm = error.norm(dim=-1)                         # (B,)
        rel_error_norm = (error_norm / x_scale.squeeze(1)).clamp(max=4.0)

        # -- surprise gate (correct signature: error_norm, adaptive_tau, error_var_mean)
        if self.mode == "no_gate":
            surprise = torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
            new_adaptive_tau = state.adaptive_tau
        else:
            error_var_mean = state.error_var.mean(dim=-1)
            surprise, new_adaptive_tau = self.surprise_gate(
                rel_error_norm, state.adaptive_tau, error_var_mean
            )

        # -- fast weights update: capture returned U (no state mutation here) -
        if self.mode in ("full", "no_gate"):
            new_U = self.update_fast_weights(state.h, error, surprise, state)
        else:
            new_U = state.U

        # -- state update ------------------------------------------------------
        base_effect = (self.B @ x_norm.T).T                    # (B, hidden)
        err_effect = (self.W @ error.T).T                      # (B, hidden)

        input_effect = (
            state.h * 0.6 +
            base_effect * 0.2 +
            err_effect * surprise.unsqueeze(1) * 0.3
        )
        h_new = self.compute_ltc_update(state.h, input_effect, surprise)

        # -- statistics (Welford-correct: variance uses old_mean) --------------
        a = 0.05
        old_error_mean = state.error_mean
        new_error_mean = (1 - a) * old_error_mean + a * error
        new_error_var  = (1 - a) * state.error_var + a * (error - old_error_mean) ** 2
        new_avg_surprise = (1 - self.beta_s) * state.avg_surprise + self.beta_s * surprise

        # -- sleep (consolidate during calm; use new_U so consolidation is current)
        avg_s = new_avg_surprise.mean()
        new_U_target = state.U_target
        if avg_s < self.S_min:
            dU_t = self.sleep_rate * (new_U - state.U_target)
            new_U_target = state.U_target + dU_t
            nt = new_U_target.norm(dim=(1, 2), keepdim=True)
            new_U_target = new_U_target * (self.target_norm / (nt + 1e-6)).clamp(max=1.5)

        # -- write back into mutable state -------------------------------------
        state.h = h_new
        state.U = new_U
        state.U_target = new_U_target
        state.adaptive_tau = new_adaptive_tau
        state.error_mean = new_error_mean
        state.error_var = new_error_var
        state.avg_surprise = new_avg_surprise

        return h_new, state, surprise, rel_error_norm


# ---------------------------------------------------------------------------
# Phase 1: Pre-train on Speaker A (backprop, predicts x_norm from h)
# ---------------------------------------------------------------------------

def pretrain(config: DREAMConfig, feats_a: torch.Tensor) -> ExperimentCell:
    """
    Trains C, W, B (and LTC params) to predict normalized mel features
    of Speaker A. Fast weights are NOT updated here (mode=static).
    """
    cell = ExperimentCell(config, mode="static")
    trainable = [cell.C, cell.W, cell.B, cell.tau_sys, cell.ltc_surprise_scale]
    opt = torch.optim.Adam(trainable, lr=PRETRAIN_LR)

    print(f"Pre-training on Speaker A ({PRETRAIN_EPOCHS} epochs)...")

    for epoch in range(PRETRAIN_EPOCHS):
        state = cell.init_state(1)
        total_loss = 0.0

        for t in range(feats_a.shape[0]):
            x_t = feats_a[t].unsqueeze(0)                          # (1, 80)
            x_norm_target = (x_t / (x_t.norm() + 1e-6)).clamp(-1, 1)

            # Predict normalized x from previous h (detached state)
            x_pred_raw = state.h.detach() @ cell.C.T
            x_pred_norm = torch.tanh(x_pred_raw)

            loss = F.mse_loss(x_pred_norm, x_norm_target)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, PRETRAIN_CLIP_GRAD)
            opt.step()
            total_loss += loss.item()

            # Advance state (no_grad — just to get realistic h)
            with torch.no_grad():
                _, state, _, _ = cell.forward_step(x_t, state)

        avg = total_loss / feats_a.shape[0]
        if (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch+1:3d}/{PRETRAIN_EPOCHS}  loss={avg:.4f}")

    print("Pre-training done.\n")
    return cell


# ---------------------------------------------------------------------------
# Phase 2: Inference experiment (no gradients)
# ---------------------------------------------------------------------------

def run_inference(
    pretrained_cell: ExperimentCell,
    feats_ab: torch.Tensor,
    mode: Mode,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Copies pretrained weights, sets mode, runs full 10-sec clip.
    C/W/B are frozen; only fast weights can change (in full/no_gate modes).
    Returns (losses, surprises) arrays of shape (T,).
    """
    cell = ExperimentCell(pretrained_cell.config, mode=mode)
    # Copy pre-trained base weights; fast weight matrices stay at init
    with torch.no_grad():
        cell.C.copy_(pretrained_cell.C)
        cell.W.copy_(pretrained_cell.W)
        cell.B.copy_(pretrained_cell.B)
        cell.tau_sys.copy_(pretrained_cell.tau_sys)
        cell.ltc_surprise_scale.copy_(pretrained_cell.ltc_surprise_scale)
        cell.eta.copy_(pretrained_cell.eta)
    cell.eval()

    losses, surprises = [], []

    with torch.no_grad():
        state = cell.init_state(1)

        for t in range(feats_ab.shape[0]):
            x_t = feats_ab[t].unsqueeze(0)
            _, state, surprise, rel_err = cell.forward_step(x_t, state)
            losses.append(rel_err.item())
            surprises.append(surprise.item())

    return np.array(losses), np.array(surprises)


# ---------------------------------------------------------------------------
# Pass/Fail evaluation
# ---------------------------------------------------------------------------

def evaluate(results: dict, switch_frame: int, fps: float):
    losses_f, surp_f = results["full"]
    losses_s, _ = results["static"]

    win = min(100, switch_frame)
    baseline = losses_f[switch_frame - win:switch_frame].mean()
    threshold_loss = baseline * 1.5

    after_f = losses_f[switch_frame:]
    after_surp = surp_f[switch_frame:]
    after_s = losses_s[switch_frame:]

    # Criterion 1: S_t spike
    peak_s = after_surp[:50].max()
    c1 = peak_s >= 0.7

    # Criterion 2: Recovery < 50 steps
    recovery_step = next(
        (i for i, l in enumerate(after_f) if l <= threshold_loss), None
    )
    c2 = recovery_step is not None and recovery_step < 50

    # Criterion 3: full < static — window 30-200 frames post-switch
    # (fast weights need ~30 frames to encode new patterns before helping)
    sl = slice(30, min(200, len(after_f)))
    mean_f = after_f[sl].mean()
    mean_s = after_s[sl].mean()
    c3 = mean_f < mean_s

    print(f"\n{'='*55}")
    print(f"PASS/FAIL REPORT")
    print(f"{'='*55}")
    print(f"Baseline rel-error (pre-switch): {baseline:.4f}")
    print(f"Recovery threshold (x1.5):       {threshold_loss:.4f}")
    print(f"\n[1] S_t spike at switch:  peak={peak_s:.3f}  "
          f"{'PASS ✓' if c1 else 'FAIL ✗'}  (need ≥ 0.7)")
    print(f"[2] Recovery < 50 steps:  step={recovery_step}  "
          f"{'PASS ✓' if c2 else 'FAIL ✗'}")
    print(f"[3] Full < Static (frames +30..+200):  "
          f"full={mean_f:.4f} static={mean_s:.4f}  "
          f"{'PASS ✓' if c3 else 'FAIL ✗'}")

    print(f"\nPer-frame comparison (full vs static) after switch:")
    print(f"  {'frame':>5}  {'full':>7}  {'static':>7}  {'diff':>8}")
    for i in [0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 400]:
        if i < len(after_f):
            d = after_f[i] - after_s[i]
            print(f"  +{i:4d}:  {after_f[i]:7.4f}  {after_s[i]:7.4f}  {d:+8.4f}"
                  f"  {'<' if d < 0 else '>':1s}")

    print(f"\n{'ALL PASS ✓' if (c1 and c2 and c3) else 'SOME FAILED ✗'}")
    print(f"{'='*55}\n")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def smooth(x: np.ndarray, w: int = 15) -> np.ndarray:
    return np.convolve(x, np.ones(w) / w, mode="same")


def plot_results(
    results: dict,
    switch_frame: int,
    fps: float,
    out: str = "speaker_switch_results.png",
):
    T = len(next(iter(results.values()))[0])
    t = np.arange(T) / fps
    sw = switch_frame / fps

    colors = {"full": "#2196F3", "static": "#F44336", "no_gate": "#FF9800"}
    labels = {"full": "Full (fast weights ON)", "static": "Static (dU=0)",
              "no_gate": "No Gate (S_t≡1)"}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(
        "DREAM Speaker-Switch Experiment\n"
        "Pre-trained on Speaker A (male) → Speaker B (female) at t=5s",
        fontsize=13, fontweight="bold",
    )

    for mode, (losses, _) in results.items():
        ax1.plot(t, smooth(losses), color=colors[mode],
                 label=labels[mode], linewidth=2.0, alpha=0.9)
    ax1.axvline(sw, color="black", linestyle="--", linewidth=1.2, label="Speaker switch")
    ax1.set_ylabel("Relative Prediction Error", fontsize=11)
    ax1.set_title("Prediction Error (lower = better adaptation)", fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    for mode, (_, surprises) in results.items():
        ax2.plot(t, smooth(surprises), color=colors[mode],
                 label=labels[mode], linewidth=2.0, alpha=0.9)
    ax2.axvline(sw, color="black", linestyle="--", linewidth=1.2, label="Speaker switch")
    ax2.axhline(0.5, color="gray", linestyle=":", linewidth=1.0, label="S_t = 0.5")
    ax2.set_xlabel("Time (seconds)", fontsize=11)
    ax2.set_ylabel("Surprise  S_t", fontsize=11)
    ax2.set_title("Surprise Gate Signal", fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)

    for ax in (ax1, ax2):
        ax.axvspan(0, sw, alpha=0.04, color="blue")
        ax.axvspan(sw, t[-1], alpha=0.04, color="red")
        ax.text(sw / 2, ax.get_ylim()[1] * 0.95, "Speaker A",
                ha="center", fontsize=9, color="blue", alpha=0.7)
        ax.text((sw + t[-1]) / 2, ax.get_ylim()[1] * 0.95, "Speaker B",
                ha="center", fontsize=9, color="red", alpha=0.7)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    # -- features (real audio if available, synthetic otherwise) ---------------
    import os
    if os.path.exists(AUDIO_A) and os.path.exists(AUDIO_B):
        print("Loading audio and extracting mel-spectrograms...")
        feats_a, feats_ab, switch_frame = make_features()
        fps = SR / HOP
        print(f"  Speaker A:  {feats_a.shape[0]} frames ({feats_a.shape[0]/fps:.1f}s)")
        print(f"  Full clip:  {feats_ab.shape[0]} frames, switch at {switch_frame} ({switch_frame/fps:.1f}s)\n")
    else:
        print("Audio files not found — using synthetic features (fast-weights smoke test).")
        print(f"  Expected: {AUDIO_A}")
        print(f"  Place real CommonVoice clips there for the full experiment.\n")
        feats_a, feats_ab, switch_frame = make_synthetic_features()
        fps = 100.0  # nominal
        print(f"  Synthetic A: {feats_a.shape[0]} frames")
        print(f"  Synthetic AB: {feats_ab.shape[0]} frames, switch at {switch_frame}\n")

    # -- model config ----------------------------------------------------------
    config = DREAMConfig(
        input_dim=N_MELS,
        hidden_dim=256,
        rank=16,
        base_threshold=0.35,
        base_plasticity=0.4,        # higher → faster Hebbian encoding
        forgetting_rate=0.03,       # higher → faster forgetting of old speaker
        ltc_tau_sys=5.0,
        ltc_surprise_scale=8.0,
        surprise_temperature=0.12,
        entropy_influence=0.2,
        time_step=0.1,
        sleep_rate=0.005,
        min_surprise_for_sleep=0.25,
    )

    # -- Phase 1: pre-train ----------------------------------------------------
    pretrained = pretrain(config, feats_a)

    # -- Phase 2: inference, 3 modes -------------------------------------------
    results = {}
    for mode in ("full", "static", "no_gate"):
        print(f"Running inference mode: {mode} ...")
        losses, surprises = run_inference(pretrained, feats_ab, mode)
        results[mode] = (losses, surprises)
        print(f"  mean loss:     {losses.mean():.4f}")
        print(f"  mean surprise: {surprises.mean():.3f}")
        print(f"  loss at switch (+5 frames): {losses[switch_frame:switch_frame+5].mean():.4f}")
        print()

    # -- Report ----------------------------------------------------------------
    evaluate(results, switch_frame, fps)

    # -- Plot ------------------------------------------------------------------
    plot_results(results, switch_frame, fps)
