"""
Spike-Triggered Reset — Fast-Weight Interference Experiment
============================================================
Tests Option A from the roadmap:

  When S_t > spike_threshold for N consecutive frames,
  multiply U by a decay factor to rapidly clear stale patterns.

Motivation
----------
In the GRU baseline experiment, DREAM-Full was *worse* than DREAM-Static
on Speaker B (2.80 vs 2.21 for B¹, 3.04 vs 2.28 for B²).  The fast weights
encode Speaker A (male) patterns during A¹ and then mis-correct predictions
for the female Speaker B.  Adaptive forgetting (λ_eff) slows the decay but
doesn't clear U fast enough at the moment of a gender switch.

The spike-triggered reset fires a hard partial reset of U at the precise
moment of high sustained surprise — exactly when a speaker switch occurs.

Grid
----
Four variants are tested against Full (no reset) and Static:

  Variant        spike_n   decay    aggression
  ─────────────────────────────────────────────
  reset_n3_d03       3      0.3     very aggressive
  reset_n5_d05       5      0.5     reference (from roadmap)
  reset_n5_d02       5      0.2     hard reset
  reset_n10_d07     10      0.7     conservative

spike_threshold = 0.7 for all variants (fixed).

Key metric: B¹ and B² mean loss vs Full (no reset).
If any variant drops B loss below Static (2.21 / 2.28), we have a clean win.

Sequence: A¹ → B¹ → C¹ → A² → B² → C² → A³  (7 × 3 s = 21 s)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from dream_net import DREAMConfig, DREAMCell, DREAMState


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CLIP_SEC = 3.0
SR       = 16_000
N_MELS   = 80
N_FFT    = 1024
HOP      = 160
WIN      = 400
FPS      = SR / HOP
HIDDEN   = 256

PRETRAIN_EPOCHS    = 30
PRETRAIN_LR        = 3e-3
PRETRAIN_CLIP_GRAD = 1.0

SPIKE_THRESHOLD = 0.7   # fixed across all grid variants

SEQUENCE = [
    ("A¹", "data/commonvoice/speaker_1/0000.wav", 1, False),
    ("B¹", "data/commonvoice/speaker_2/0000.wav", 1, False),
    ("C¹", "data/commonvoice/speaker_3/0000.wav", 1, False),
    ("A²", "data/commonvoice/speaker_1/0002.wav", 2, True),
    ("B²", "data/commonvoice/speaker_2/0001.wav", 2, True),
    ("C²", "data/commonvoice/speaker_3/0001.wav", 2, True),
    ("A³", "data/commonvoice/speaker_1/0003.wav", 3, True),
]

# (label, spike_n, decay)
GRID = [
    ("full",           None,  None),   # no reset — baseline
    ("reset_n3_d03",   3,     0.30),
    ("reset_n5_d05",   5,     0.50),
    ("reset_n5_d02",   5,     0.20),
    ("reset_n10_d07",  10,    0.70),
    ("static",         None,  None),   # no fast weights — lower bound
]

PALETTE = {
    "full":          "#1565C0",
    "reset_n3_d03":  "#E65100",
    "reset_n5_d05":  "#F57C00",
    "reset_n5_d02":  "#BF360C",
    "reset_n10_d07": "#FFB300",
    "static":        "#7986CB",
}
LEGEND = {
    "full":          "Full  (no reset)",
    "reset_n3_d03":  "Reset  n=3  decay=0.3",
    "reset_n5_d05":  "Reset  n=5  decay=0.5  (ref)",
    "reset_n5_d02":  "Reset  n=5  decay=0.2  (hard)",
    "reset_n10_d07": "Reset  n=10  decay=0.7",
    "static":        "Static  (dU = 0)",
}
SPEAKER_COLORS = {"A": "#2196F3", "B": "#E91E63", "C": "#4CAF50"}


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------

def load_clip(path, n_sec):
    wav, sr = torchaudio.load(path)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    n = int(n_sec * SR)
    return wav[:, :n] if wav.shape[1] >= n else F.pad(wav, (0, n - wav.shape[1]))


def build_mel(wav):
    mel_tf = T.MelSpectrogram(
        sample_rate=SR, n_fft=N_FFT, win_length=WIN, hop_length=HOP,
        n_mels=N_MELS, f_min=20.0, f_max=8000.0, power=2.0,
    )
    logm = torch.log(mel_tf(wav).squeeze(0).T + 1e-6)
    mu   = logm.mean(0, keepdim=True)
    std  = logm.std(0, keepdim=True).clamp(min=1e-4)
    return (logm - mu) / std


def make_features():
    clips          = [load_clip(path, CLIP_SEC) for _, path, _, _ in SEQUENCE]
    feats_pretrain = build_mel(clips[0])
    feats_full     = build_mel(torch.cat(clips, dim=1))
    fps            = int(CLIP_SEC * SR / HOP)
    switches       = [fps * i for i in range(1, len(SEQUENCE))]
    return feats_pretrain, feats_full, switches, fps


# ---------------------------------------------------------------------------
# Cell
# ---------------------------------------------------------------------------

class SpikeResetCell(DREAMCell):
    """
    CycleCell + optional spike-triggered U reset.

    mode="full"   — fast weights ON, optional spike reset
    mode="static" — fast weights OFF (dU = 0), no reset

    The reset fires when surprise > spike_threshold for spike_n *consecutive*
    frames.  On firing: U ← U * decay, counter resets to 0.
    """

    def __init__(self, config, mode="full", spike_n=None, decay=None):
        super().__init__(config)
        self.mode        = mode
        self.spike_n     = spike_n      # None → no reset
        self.decay       = decay
        self._counter    = 0            # consecutive high-surprise frames

    def _maybe_reset(self, surprise_val, state):
        """Apply spike-triggered reset if enabled and counter threshold reached."""
        if self.spike_n is None:
            return
        if surprise_val > SPIKE_THRESHOLD:
            self._counter += 1
        else:
            self._counter = 0
        if self._counter >= self.spike_n:
            state.U       = state.U * self.decay
            self._counter = 0

    def forward_step(self, x, state):
        x_scale = x.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        x_norm  = (x / x_scale).clamp(-1.0, 1.0)

        x_pred = torch.tanh(state.h @ self.C.T) * x_scale
        if self.mode == "full":
            h_U    = torch.bmm(state.h.unsqueeze(1), state.U).squeeze(1)
            x_pred = x_pred + 0.5 * (h_U @ self.V.T) * x_scale

        error   = x - x_pred
        rel_err = (error.norm(dim=-1) / x_scale.squeeze(1)).clamp(max=4.0)
        surprise = self.surprise_gate(error, rel_err, state)

        if self.mode == "full":
            # Spike-triggered reset fires BEFORE Hebbian write
            # so we clear stale patterns before writing new ones
            self._maybe_reset(surprise.item(), state)
            self.update_fast_weights(state.h, error, surprise, state)

        base_eff = (self.B @ x_norm.T).T
        err_eff  = (self.W @ error.T).T
        h_new = self.compute_ltc_update(
            state.h,
            state.h * 0.6 + base_eff * 0.2 + err_eff * surprise.unsqueeze(1) * 0.3,
            surprise,
        )

        a = 0.05
        state.error_mean   = (1 - a) * state.error_mean   + a * error
        state.error_var    = (1 - a) * state.error_var    + a * (error - state.error_mean) ** 2
        state.avg_surprise = (1 - self.beta_s) * state.avg_surprise + self.beta_s * surprise

        if state.avg_surprise.mean() < self.S_min:
            dU = self.sleep_rate * (state.U - state.U_target)
            state.U_target = state.U_target + dU
            nt = state.U_target.norm(dim=(1, 2), keepdim=True)
            state.U_target *= (self.target_norm / (nt + 1e-6)).clamp(max=1.5)

        state.h = h_new
        return h_new, state, surprise, rel_err


# ---------------------------------------------------------------------------
# Pre-train (static mode, same as all other experiments)
# ---------------------------------------------------------------------------

def pretrain(config, feats_pretrain):
    cell      = SpikeResetCell(config, mode="static")
    trainable = [cell.C, cell.W, cell.B, cell.tau_sys, cell.ltc_surprise_scale]
    opt       = torch.optim.Adam(trainable, lr=PRETRAIN_LR)

    print(f"Pre-training on A¹  ({PRETRAIN_EPOCHS} epochs) ...")
    for epoch in range(PRETRAIN_EPOCHS):
        state = cell.init_state(1)
        for t in range(feats_pretrain.shape[0]):
            x_t    = feats_pretrain[t].unsqueeze(0)
            target = (x_t / (x_t.norm() + 1e-6)).clamp(-1, 1)
            loss   = F.mse_loss(torch.tanh(state.h.detach() @ cell.C.T), target)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(trainable, PRETRAIN_CLIP_GRAD)
            opt.step()
            with torch.no_grad():
                _, state, _, _ = cell.forward_step(x_t, state)
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1}/{PRETRAIN_EPOCHS}")
    print("Done.\n")
    return cell


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(pretrained, feats_full, mode, spike_n, decay):
    cell = SpikeResetCell(pretrained.config, mode=mode, spike_n=spike_n, decay=decay)
    with torch.no_grad():
        for attr in ("C", "W", "B", "tau_sys", "ltc_surprise_scale", "eta"):
            getattr(cell, attr).copy_(getattr(pretrained, attr))
    cell.eval()

    losses, surprises, resets = [], [], []
    reset_frames = []
    with torch.no_grad():
        state = cell.init_state(1)
        for t in range(feats_full.shape[0]):
            x_t = feats_full[t].unsqueeze(0)
            _, state, surprise, rel_err = cell.forward_step(x_t, state)
            losses.append(rel_err.item())
            surprises.append(surprise.item())
            # detect when a reset just fired (counter went back to 0 after threshold)
            resets.append(cell._counter == 0 and surprise.item() > SPIKE_THRESHOLD
                          and spike_n is not None)

    return np.array(losses), np.array(surprises), resets


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def seg_means(losses, fps):
    return [losses[i * fps:(i + 1) * fps].mean() for i in range(len(SEQUENCE))]


def smooth(x, w=12):
    return np.convolve(x, np.ones(w) / w, mode="same")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(all_losses, all_surprises, switch_frames, fps,
         out="spike_reset_results.png"):
    T    = len(next(iter(all_losses.values())))
    t    = np.arange(T) / FPS
    sw_t = [s / FPS for s in switch_frames]

    fig = plt.figure(figsize=(18, 13))
    gs  = fig.add_gridspec(3, 2,
                           height_ratios=[1.5, 1.0, 1.0],
                           hspace=0.48, wspace=0.30,
                           left=0.07, right=0.97, top=0.91, bottom=0.07)

    ax_loss = fig.add_subplot(gs[0, :])
    ax_b    = fig.add_subplot(gs[1, 0])
    ax_all  = fig.add_subplot(gs[1, 1])
    ax_surp = fig.add_subplot(gs[2, :])

    fig.suptitle(
        "Spike-Triggered Reset  —  Does U flush fix first-switch interference?\n"
        "A¹ → B¹ → C¹ → A² → B² → C² → A³   (7 × 3 s = 21 s, no gradients)",
        fontsize=13, fontweight="bold",
    )

    def shade(ax, ymax):
        for i, (label, _, visit, _) in enumerate(SEQUENCE):
            x0  = i * fps / FPS
            x1  = (i + 1) * fps / FPS
            col = SPEAKER_COLORS[label[0]]
            ax.axvspan(x0, x1, alpha=0.05 if visit == 1 else 0.13, color=col)
            ax.text((x0 + x1) / 2, ymax * 0.92, label, ha="center",
                    fontsize=9, color=col,
                    fontweight="bold" if visit > 1 else "normal")
        for s in sw_t:
            ax.axvline(s, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    # ── Loss curves (all variants) ─────────────────────────────────────────
    for key, losses in all_losses.items():
        lw  = 2.2 if key == "full" else (1.8 if key == "static" else 1.5)
        ls  = "--" if key == "static" else "-"
        ax_loss.plot(t, smooth(losses), color=PALETTE[key],
                     label=LEGEND[key], linewidth=lw, linestyle=ls)
    ax_loss.set_ylabel("Relative Prediction Error", fontsize=10)
    ax_loss.set_title("Prediction Loss  —  All Variants", fontsize=10, fontweight="bold")
    ax_loss.legend(fontsize=8.5, loc="upper right", ncol=2)
    ax_loss.grid(True, alpha=0.2); ax_loss.set_ylim(0, 4.0)
    shade(ax_loss, 4.0)

    # ── Bar: B segments (interference metric) ─────────────────────────────
    b_idx    = [i for i, s in enumerate(SEQUENCE) if s[0][0] == "B"]
    b_labels = [SEQUENCE[i][0] for i in b_idx]
    x        = np.arange(len(b_idx))
    n_var    = len(GRID)
    w        = 0.8 / n_var
    bar_kw   = dict(edgecolor="white", linewidth=0.5)

    for j, (key, _, _) in enumerate(GRID):
        means = seg_means(all_losses[key], fps)
        vals  = [means[i] for i in b_idx]
        offset = (j - n_var / 2 + 0.5) * w
        bars = ax_b.bar(x + offset, vals, w, color=PALETTE[key],
                        label=LEGEND[key], **bar_kw)
        for bar, v in zip(bars, vals):
            ax_b.text(bar.get_x() + bar.get_width() / 2, v + 0.03,
                      f"{v:.2f}", ha="center", fontsize=7)
    ax_b.set_xticks(x); ax_b.set_xticklabels(b_labels, fontsize=10)
    ax_b.set_ylabel("Mean Loss", fontsize=9)
    ax_b.set_title("Speaker B  —  First-Switch Interference\n"
                   "Lower = less interference from Speaker A patterns",
                   fontsize=9, fontweight="bold")
    ax_b.legend(fontsize=7, loc="upper right")
    ax_b.grid(axis="y", alpha=0.3); ax_b.set_ylim(0, 4.5)

    # ── Bar: overall mean ─────────────────────────────────────────────────
    keys_sorted = [k for k, _, _ in GRID]
    overall     = [np.mean(seg_means(all_losses[k], fps)) for k in keys_sorted]
    colors_bar  = [PALETTE[k] for k in keys_sorted]
    bars = ax_all.bar(range(len(keys_sorted)), overall, color=colors_bar,
                      edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, overall):
        ax_all.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                    f"{v:.3f}", ha="center", fontsize=8)
    ax_all.set_xticks(range(len(keys_sorted)))
    ax_all.set_xticklabels([LEGEND[k][:18] for k in keys_sorted],
                           fontsize=7, rotation=15, ha="right")
    ax_all.set_ylabel("Mean Loss", fontsize=9)
    ax_all.set_title("Overall Mean Loss  (21 s)\nLower is better",
                     fontsize=9, fontweight="bold")
    ax_all.grid(axis="y", alpha=0.3); ax_all.set_ylim(0, 3.5)

    # ── Surprise (full vs best reset variant) ────────────────────────────
    ax_surp.plot(t, smooth(all_surprises["full"]),
                 color=PALETTE["full"], linewidth=1.8, label=LEGEND["full"])
    # pick best reset by B mean loss
    best_reset = min(
        [k for k, sn, _ in GRID if sn is not None],
        key=lambda k: np.mean([seg_means(all_losses[k], fps)[i] for i in b_idx])
    )
    ax_surp.plot(t, smooth(all_surprises[best_reset]),
                 color=PALETTE[best_reset], linewidth=1.8,
                 label=f"Best reset: {LEGEND[best_reset]}")
    ax_surp.axhline(SPIKE_THRESHOLD, color="gray", linestyle=":",
                    linewidth=1.0, label=f"Spike threshold ({SPIKE_THRESHOLD})")
    ax_surp.set_ylabel("Surprise  S_t", fontsize=10)
    ax_surp.set_xlabel("Time  (seconds)", fontsize=10)
    ax_surp.set_title("Surprise Signal  (Full vs Best Reset)",
                      fontsize=10, fontweight="bold")
    ax_surp.legend(fontsize=9, loc="upper right")
    ax_surp.grid(True, alpha=0.2); ax_surp.set_ylim(0, 1.05)
    shade(ax_surp, 1.05)

    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {out}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(all_losses, fps):
    print(f"\n{'='*80}")
    print(f"  SPIKE-TRIGGERED RESET — REPORT")
    print(f"{'='*80}")

    # Full table
    print(f"\n  {'Variant':<22}  {'A¹':>6}  {'B¹':>6}  {'C¹':>6}  "
          f"{'A²':>6}  {'B²':>6}  {'C²':>6}  {'A³':>6}  {'Mean':>7}")
    print(f"  {'-'*76}")
    for key, _, _ in GRID:
        means = seg_means(all_losses[key], fps)
        row   = "  ".join(f"{m:>6.4f}" for m in means)
        print(f"  {LEGEND[key]:<22}  {row}  {np.mean(means):>7.4f}")

    # First-switch interference delta on B
    b_idx  = [i for i, s in enumerate(SEQUENCE) if s[0][0] == "B"]
    print(f"\n  ── Speaker B interference (vs Full no-reset) ──")
    full_b = [seg_means(all_losses["full"], fps)[i] for i in b_idx]
    stat_b = [seg_means(all_losses["static"], fps)[i] for i in b_idx]
    print(f"  {'Variant':<22}  {'B¹':>8}  {'B² ':>8}  {'Δ vs Full':>10}  {'vs Static':>10}")
    print(f"  {'-'*62}")
    for key, _, _ in GRID:
        means  = seg_means(all_losses[key], fps)
        b_vals = [means[i] for i in b_idx]
        delta  = np.mean(b_vals) - np.mean(full_b)
        vs_s   = np.mean(b_vals) - np.mean(stat_b)
        print(f"  {LEGEND[key]:<22}  {b_vals[0]:>8.4f}  {b_vals[1]:>8.4f}  "
              f"  {delta:>+9.4f}  {vs_s:>+9.4f}")

    # Best reset
    reset_keys = [k for k, sn, _ in GRID if sn is not None]
    best = min(reset_keys,
               key=lambda k: np.mean([seg_means(all_losses[k], fps)[i] for i in b_idx]))
    best_b  = np.mean([seg_means(all_losses[best], fps)[i] for i in b_idx])
    full_bm = np.mean(full_b)
    stat_bm = np.mean(stat_b)
    print(f"\n  Best reset variant  : {LEGEND[best]}")
    print(f"  B mean (best reset) : {best_b:.4f}")
    print(f"  B mean (full)       : {full_bm:.4f}  (Δ {best_b - full_bm:+.4f})")
    print(f"  B mean (static)     : {stat_bm:.4f}  (Δ {best_b - stat_bm:+.4f})")

    verdict = "FIXED ✓" if best_b < stat_bm else ("IMPROVED ✓" if best_b < full_bm else "NO CHANGE ✗")
    print(f"\n  Verdict: {verdict}")
    if best_b < stat_bm:
        print(f"  → Reset beats Static on B: fast weights now HELP on first switch.")
        print(f"  → Architecture is ready for ASR integration.")
    elif best_b < full_bm:
        print(f"  → Reset reduces interference but doesn't beat Static on B.")
        print(f"  → Partial fix: fast weights still cause some interference.")
    else:
        print(f"  → Reset does not reduce B interference. Needs different approach.")

    print(f"{'='*80}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    print("Loading audio ...")
    feats_pretrain, feats_full, switch_frames, fps = make_features()
    print(f"  Total: {feats_full.shape[0]} frames ({feats_full.shape[0] / FPS:.1f}s)\n")

    config = DREAMConfig(
        input_dim=N_MELS, hidden_dim=HIDDEN, rank=8,
        base_threshold=0.35, base_plasticity=0.4,
        forgetting_rate=0.03, adaptive_forgetting_scale=8.0,
        ltc_tau_sys=5.0, ltc_surprise_scale=8.0,
        surprise_temperature=0.12, entropy_influence=0.2,
        time_step=0.1, sleep_rate=0.005, min_surprise_for_sleep=0.25,
    )

    pretrained = pretrain(config, feats_pretrain)

    all_losses    = {}
    all_surprises = {}

    for key, spike_n, decay in GRID:
        mode = "static" if key == "static" else "full"
        label = LEGEND[key]
        print(f"Running {label} ...")
        losses, surprises, _ = run_inference(pretrained, feats_full,
                                             mode=mode,
                                             spike_n=spike_n,
                                             decay=decay)
        all_losses[key]    = losses
        all_surprises[key] = surprises

    print_report(all_losses, fps)
    plot(all_losses, all_surprises, switch_frames, fps)
