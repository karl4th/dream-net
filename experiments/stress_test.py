"""
Multi-Speaker Stress Test
=========================
Three sequential speaker switches: A → B → C (5 sec each, 15 sec total).

Questions under test:
  1. Does S_t spike at BOTH switch points?
  2. Does Full recover faster than Static at the SECOND switch too?
  3. Does adaptation speed improve or degrade across switches
     (i.e., does DREAM get "faster" at adapting with experience)?
  4. Catastrophic forgetting: if speaker A patterns were consolidated,
     does the Full mode perform worse on speaker C than Static?

Protocol:
  Phase 1 — Pretrain C, W, B on Speaker A via backprop (base weights only).
  Phase 2 — Inference, no gradients, 3 modes over 15-sec clip:
      full    : Hebbian fast-weight updates + adaptive surprise gate
      static  : dU = 0  (fast weights frozen at zero)
      no_gate : S_t ≡ 1  (plasticity always open, no selectivity)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import Literal

from dream_net import DREAMConfig, DREAMCell, DREAMState


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SPEAKERS = {
    "A": "data/commonvoice/speaker_1/0000.wav",   # male
    "B": "data/commonvoice/speaker_2/0000.wav",   # female
    "C": "data/commonvoice/speaker_3/0000.wav",   # second male
}
CLIP_SEC     = 5.0
SR           = 16_000
N_MELS       = 80
N_FFT        = 1024
HOP          = 160        # 10 ms → 100 frames / sec
WIN          = 400        # 25 ms window
FPS          = SR / HOP   # frames per second

PRETRAIN_EPOCHS   = 30
PRETRAIN_LR       = 3e-3
PRETRAIN_CLIP_GRAD = 1.0

Mode = Literal["full", "static", "no_gate"]

COLORS = {"full": "#2196F3", "static": "#F44336", "no_gate": "#FF9800"}
LABELS = {
    "full":    "Full  (fast weights ON)",
    "static":  "Static  (dU = 0)",
    "no_gate": "No Gate  (S_t ≡ 1)",
}


# ---------------------------------------------------------------------------
# Audio → Mel-spectrogram
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
    """wav (1, T)  →  log-mel (frames, N_MELS), globally standardised."""
    mel_tf = T.MelSpectrogram(
        sample_rate=SR, n_fft=N_FFT, win_length=WIN, hop_length=HOP,
        n_mels=N_MELS, f_min=20.0, f_max=8000.0, power=2.0,
    )
    mel   = mel_tf(wav)
    logm  = torch.log(mel.squeeze(0).T + 1e-6)          # (T, n_mels)
    mu    = logm.mean(0, keepdim=True)
    std   = logm.std(0, keepdim=True).clamp(min=1e-4)
    return (logm - mu) / std


def make_features() -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """
    Returns
    -------
    feats_a  : (T/3, N_MELS)  — Speaker A only, used for pre-training
    feats_abc: (T,   N_MELS)  — full 15-sec clip A → B → C
    switches : [frame_AB, frame_BC]
    """
    clips = {k: load_clip(v, CLIP_SEC) for k, v in SPEAKERS.items()}
    feats_a   = build_mel(clips["A"])
    feats_abc = build_mel(torch.cat([clips["A"], clips["B"], clips["C"]], dim=1))
    frames_per_speaker = int(CLIP_SEC * SR / HOP)
    switches = [frames_per_speaker, 2 * frames_per_speaker]
    return feats_a, feats_abc, switches


# ---------------------------------------------------------------------------
# Experiment cell with mode control
# ---------------------------------------------------------------------------

class ExperimentCell(DREAMCell):
    """DREAMCell with explicit mode flag for ablation experiments."""

    def __init__(self, config: DREAMConfig, mode: Mode = "full"):
        super().__init__(config)
        self.mode = mode

    def forward_step(
        self,
        x: torch.Tensor,
        state: DREAMState,
    ) -> tuple[torch.Tensor, DREAMState, torch.Tensor, torch.Tensor]:
        """Single-step forward.  Returns h_new, state, surprise, rel_error."""

        x_scale = x.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        x_norm  = (x / x_scale).clamp(-1.0, 1.0)

        # -- Base prediction (slow weights C only) ----------------------------
        x_pred = torch.tanh(state.h @ self.C.T) * x_scale      # (B, input)

        # -- Fast-weight prediction correction (associative retrieval) --------
        #    h @ U → (B, rank),  then (B, rank) @ V.T → (B, input)
        #    "given this hidden state, memory recalls: prediction was off by ~this"
        if self.mode in ("full", "no_gate"):
            h_U            = torch.bmm(state.h.unsqueeze(1), state.U).squeeze(1)
            pred_correction = (h_U @ self.V.T) * x_scale
            x_pred          = x_pred + 0.5 * pred_correction

        # -- Prediction error --------------------------------------------------
        error          = x - x_pred
        error_norm     = error.norm(dim=-1)
        rel_error_norm = (error_norm / x_scale.squeeze(1)).clamp(max=4.0)

        # -- Surprise gate -----------------------------------------------------
        if self.mode == "no_gate":
            surprise = torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
        else:
            surprise = self.surprise_gate(error, rel_error_norm, state)

        # -- Hebbian fast-weight update (store h → error associations) --------
        if self.mode in ("full", "no_gate"):
            self.update_fast_weights(state.h, error, surprise, state)

        # -- Hidden-state update (LTC) ----------------------------------------
        base_effect = (self.B @ x_norm.T).T
        err_effect  = (self.W @ error.T).T
        input_effect = (
            state.h   * 0.6 +
            base_effect * 0.2 +
            err_effect  * surprise.unsqueeze(1) * 0.3
        )
        h_new = self.compute_ltc_update(state.h, input_effect, surprise)

        # -- Statistics -------------------------------------------------------
        a = 0.05
        state.error_mean   = (1 - a) * state.error_mean   + a * error
        state.error_var    = (1 - a) * state.error_var    + a * (error - state.error_mean) ** 2
        state.avg_surprise = (1 - self.beta_s) * state.avg_surprise + self.beta_s * surprise

        # -- Sleep: consolidate fast weights during calm periods --------------
        if state.avg_surprise.mean() < self.S_min:
            dU_t = self.sleep_rate * (state.U - state.U_target)
            state.U_target = state.U_target + dU_t
            nt = state.U_target.norm(dim=(1, 2), keepdim=True)
            state.U_target = state.U_target * (self.target_norm / (nt + 1e-6)).clamp(max=1.5)

        state.h = h_new
        return h_new, state, surprise, rel_error_norm


# ---------------------------------------------------------------------------
# Phase 1 — Pre-train on Speaker A
# ---------------------------------------------------------------------------

def pretrain(config: DREAMConfig, feats_a: torch.Tensor) -> ExperimentCell:
    cell = ExperimentCell(config, mode="static")
    trainable = [cell.C, cell.W, cell.B, cell.tau_sys, cell.ltc_surprise_scale]
    opt = torch.optim.Adam(trainable, lr=PRETRAIN_LR)

    print(f"Pre-training on Speaker A  ({PRETRAIN_EPOCHS} epochs) ...")
    for epoch in range(PRETRAIN_EPOCHS):
        state      = cell.init_state(1)
        total_loss = 0.0
        for t in range(feats_a.shape[0]):
            x_t             = feats_a[t].unsqueeze(0)
            x_norm_target   = (x_t / (x_t.norm() + 1e-6)).clamp(-1, 1)
            x_pred_norm     = torch.tanh(state.h.detach() @ cell.C.T)
            loss            = F.mse_loss(x_pred_norm, x_norm_target)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(trainable, PRETRAIN_CLIP_GRAD)
            opt.step()
            total_loss += loss.item()
            with torch.no_grad():
                _, state, _, _ = cell.forward_step(x_t, state)

        if (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch+1:3d}/{PRETRAIN_EPOCHS}  loss = {total_loss/feats_a.shape[0]:.5f}")

    print("Pre-training complete.\n")
    return cell


# ---------------------------------------------------------------------------
# Phase 2 — Inference over full 15-sec clip
# ---------------------------------------------------------------------------

def run_inference(
    pretrained: ExperimentCell,
    feats_abc: torch.Tensor,
    mode: Mode,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    losses    : (T,)  relative prediction error per frame
    surprises : (T,)  surprise gate signal S_t
    u_norms   : (T,)  Frobenius norm of fast-weight matrix U
    """
    cell = ExperimentCell(pretrained.config, mode=mode)
    with torch.no_grad():
        cell.C.copy_(pretrained.C)
        cell.W.copy_(pretrained.W)
        cell.B.copy_(pretrained.B)
        cell.tau_sys.copy_(pretrained.tau_sys)
        cell.ltc_surprise_scale.copy_(pretrained.ltc_surprise_scale)
        cell.eta.copy_(pretrained.eta)
    cell.eval()

    losses, surprises, u_norms = [], [], []

    with torch.no_grad():
        state = cell.init_state(1)
        for t in range(feats_abc.shape[0]):
            x_t = feats_abc[t].unsqueeze(0)
            _, state, surprise, rel_err = cell.forward_step(x_t, state)
            losses.append(rel_err.item())
            surprises.append(surprise.item())
            u_norms.append(state.U.norm().item())

    return np.array(losses), np.array(surprises), np.array(u_norms)


# ---------------------------------------------------------------------------
# Metrics & Pass/Fail
# ---------------------------------------------------------------------------

def recovery_steps(losses: np.ndarray, switch: int, threshold: float) -> int | None:
    """Frames until loss drops below threshold after a switch."""
    after = losses[switch:]
    found = next((i for i, l in enumerate(after) if l <= threshold), None)
    return found


def evaluate(results: dict, switches: list[int], feats_abc: torch.Tensor):
    losses_f, surp_f, _ = results["full"]
    losses_s, _,    _ = results["static"]
    sw1, sw2 = switches

    # baseline: last 100 frames of Speaker A
    baseline = losses_f[sw1 - 100:sw1].mean()
    thresh   = baseline * 1.5

    print(f"\n{'='*60}")
    print(f"  STRESS TEST — PASS / FAIL REPORT")
    print(f"{'='*60}")
    print(f"  Baseline loss (Speaker A, last 100 frames): {baseline:.4f}")
    print(f"  Recovery threshold (×1.5):                  {thresh:.4f}")

    for sw_idx, (sw, name) in enumerate(zip(switches, ["A→B (switch 1)", "B→C (switch 2)"])):
        after_f    = losses_f[sw:]
        after_s    = losses_s[sw:]
        after_surp = surp_f[sw:]

        peak_s    = after_surp[:50].max()
        rec_full  = recovery_steps(losses_f, sw, thresh)
        rec_stat  = recovery_steps(losses_s, sw, thresh)

        # Adaptation advantage: mean loss [+30 .. +200] full vs static
        sl        = slice(30, min(200, len(after_f)))
        mean_f    = after_f[sl].mean()
        mean_s    = after_s[sl].mean()

        c1 = peak_s >= 0.7
        c2 = rec_full is not None and rec_full < 50
        c3 = mean_f < mean_s

        print(f"\n  ── {name} (frame {sw}) ──")
        print(f"  [1] S_t spike:     peak = {peak_s:.3f}   {'PASS ✓' if c1 else 'FAIL ✗'}")
        print(f"  [2] Recovery < 50: full = {rec_full}, static = {rec_stat}   {'PASS ✓' if c2 else 'FAIL ✗'}")
        print(f"  [3] Full < Static: full = {mean_f:.4f}, static = {mean_s:.4f}   {'PASS ✓' if c3 else 'FAIL ✗'}")

    # Adaptation speed trend
    rec1 = recovery_steps(losses_f, sw1, thresh)
    rec2 = recovery_steps(losses_f, sw2, thresh)
    trend = "faster ✓" if (rec2 is not None and rec1 is not None and rec2 <= rec1) else "same/slower"
    print(f"\n  ── Adaptation speed across switches ──")
    print(f"  Recovery frames:  switch-1 = {rec1},  switch-2 = {rec2}   ({trend})")

    print(f"\n{'='*60}\n")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def smooth(x: np.ndarray, w: int = 15) -> np.ndarray:
    return np.convolve(x, np.ones(w) / w, mode="same")


def plot_results(
    results: dict,
    switches: list[int],
    out: str = "stress_test_results.png",
):
    T   = len(next(iter(results.values()))[0])
    t   = np.arange(T) / FPS
    sw  = [s / FPS for s in switches]

    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)
    fig.suptitle(
        "DREAM Multi-Speaker Stress Test\n"
        "Speaker A (male)  →  Speaker B (female)  →  Speaker C (male)",
        fontsize=13, fontweight="bold",
    )

    # ── speaker regions ──────────────────────────────────────────────────────
    regions   = [(0, sw[0], "A", "#2196F3"), (sw[0], sw[1], "B", "#E91E63"),
                 (sw[1], t[-1], "C", "#4CAF50")]

    def shade(ax):
        for x0, x1, lbl, col in regions:
            ax.axvspan(x0, x1, alpha=0.05, color=col)
            ax.text((x0 + x1) / 2, ax.get_ylim()[1] * 0.92,
                    f"Speaker {lbl}", ha="center", fontsize=9,
                    color=col, fontweight="bold", alpha=0.85)
        for s in sw:
            ax.axvline(s, color="black", linestyle="--", linewidth=1.1)

    # ── Panel 1: Prediction Error ─────────────────────────────────────────
    ax = axes[0]
    for mode, (losses, _, _) in results.items():
        ax.plot(t, smooth(losses), color=COLORS[mode], label=LABELS[mode],
                linewidth=2.0, alpha=0.9)
    ax.set_ylabel("Relative Prediction Error", fontsize=10)
    ax.set_title("Prediction Loss  (lower = better adaptation)", fontsize=10)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.25)
    ax.set_ylim(bottom=0)
    shade(ax)

    # ── Panel 2: Surprise gate ────────────────────────────────────────────
    ax = axes[1]
    for mode, (_, surprises, _) in results.items():
        ax.plot(t, smooth(surprises), color=COLORS[mode], label=LABELS[mode],
                linewidth=2.0, alpha=0.9)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.0)
    ax.set_ylabel("Surprise  S_t", fontsize=10)
    ax.set_title("Surprise Gate Signal", fontsize=10)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.25)
    ax.set_ylim(0, 1.05)
    shade(ax)

    # ── Panel 3: U norm (fast-weight magnitude) ───────────────────────────
    ax = axes[2]
    for mode, (_, _, u_norms) in results.items():
        if mode in ("full", "no_gate"):
            ax.plot(t, smooth(u_norms), color=COLORS[mode], label=LABELS[mode],
                    linewidth=2.0, alpha=0.9)
    ax.set_xlabel("Time  (seconds)", fontsize=10)
    ax.set_ylabel("‖U‖  (Frobenius)", fontsize=10)
    ax.set_title("Fast-Weight Matrix Norm  (evidence of active adaptation)", fontsize=10)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.25)
    ax.set_ylim(bottom=0)
    shade(ax)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    # -- features --------------------------------------------------------------
    print("Loading audio and extracting mel-spectrograms...")
    feats_a, feats_abc, switches = make_features()
    print(f"  Speaker A pre-train: {feats_a.shape[0]} frames ({feats_a.shape[0]/FPS:.1f}s)")
    print(f"  Full clip A+B+C:     {feats_abc.shape[0]} frames ({feats_abc.shape[0]/FPS:.1f}s)")
    print(f"  Switch points:       frames {switches}  ({[s/FPS for s in switches]} s)\n")

    # -- model config ----------------------------------------------------------
    config = DREAMConfig(
        input_dim=N_MELS,
        hidden_dim=256,
        rank=16,
        base_threshold=0.35,
        base_plasticity=0.4,
        forgetting_rate=0.03,
        ltc_tau_sys=5.0,
        ltc_surprise_scale=8.0,
        surprise_temperature=0.12,
        entropy_influence=0.2,
        time_step=0.1,
        sleep_rate=0.005,
        min_surprise_for_sleep=0.25,
        adaptive_forgetting_scale=8.0,
    )

    # -- Phase 1 ---------------------------------------------------------------
    pretrained = pretrain(config, feats_a)

    # -- Phase 2 ---------------------------------------------------------------
    results = {}
    for mode in ("full", "static", "no_gate"):
        print(f"Running inference  [{mode}] ...")
        losses, surprises, u_norms = run_inference(pretrained, feats_abc, mode)
        results[mode] = (losses, surprises, u_norms)
        sw1, sw2 = switches
        print(f"  mean loss  A:  {losses[:sw1].mean():.4f}")
        print(f"  mean loss  B:  {losses[sw1:sw2].mean():.4f}")
        print(f"  mean loss  C:  {losses[sw2:].mean():.4f}")
        print(f"  mean surprise: {surprises.mean():.3f}\n")

    # -- Report & plot ---------------------------------------------------------
    evaluate(results, switches, feats_abc)
    plot_results(results, switches)
