"""
GRU Baseline Comparison
=======================
Compares DREAM (fast weights ON) against a standard GRU of equal hidden
dimension on the long-cycle speaker-switching task.

Sequence: A¹ → B¹ → C¹ → A² → B² → C² → A³  (7 × 3 s = 21 s)

Both models are pre-trained on Speaker A via backprop (identical protocol).
Inference is gradient-free for both. The GRU cannot adapt during inference;
DREAM adapts its fast weights U without any gradient.

Parameter count note
--------------------
GRU (hidden=256, input=80):
    weight_ih : 3 × 256 × 80  =  61 440
    weight_hh : 3 × 256 × 256 = 196 608
    bias      : 3 × 256 × 2   =   1 536
    readout C : 256 × 80       =  20 480
    TOTAL     ≈ 280 064

DREAM base (hidden=256, input=80):
    C  : 80 × 256    =  20 480
    W  : 256 × 80    =  20 480
    B  : 256 × 80    =  20 480
    TOTAL ≈ 61 440  (+fast weights U: 256×8 = 2 048, per-batch, not learned)

GRU has ~4.6× more base parameters — DREAM is at a capacity disadvantage,
making any DREAM advantage more meaningful.
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

SEQUENCE = [
    ("A¹", "data/commonvoice/speaker_1/0000.wav", 1, False),
    ("B¹", "data/commonvoice/speaker_2/0000.wav", 1, False),
    ("C¹", "data/commonvoice/speaker_3/0000.wav", 1, False),
    ("A²", "data/commonvoice/speaker_1/0002.wav", 2, True),
    ("B²", "data/commonvoice/speaker_2/0001.wav", 2, True),
    ("C²", "data/commonvoice/speaker_3/0001.wav", 2, True),
    ("A³", "data/commonvoice/speaker_1/0003.wav", 3, True),
]

SPEAKER_COLORS = {"A": "#2196F3", "B": "#E91E63", "C": "#4CAF50"}
PALETTE = {
    "dream_full":   "#1565C0",
    "dream_static": "#7986CB",
    "gru":          "#B71C1C",
}
LEGEND = {
    "dream_full":   "DREAM  (fast weights ON)",
    "dream_static": "DREAM  (static, dU = 0)",
    "gru":          "GRU  (no adaptation)",
}


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
    clips         = [load_clip(path, CLIP_SEC) for _, path, _, _ in SEQUENCE]
    feats_pretrain = build_mel(clips[0])
    feats_full    = build_mel(torch.cat(clips, dim=1))
    fps           = int(CLIP_SEC * SR / HOP)
    switches      = [fps * i for i in range(1, len(SEQUENCE))]
    return feats_pretrain, feats_full, switches, fps


# ---------------------------------------------------------------------------
# GRU model
# ---------------------------------------------------------------------------

class GRUPredictor(nn.Module):
    """
    Single-layer GRU that predicts the current mel frame from the previous
    hidden state.  Prediction: x̂_t = tanh(h_{t-1} @ C.T) · ‖x_t‖
    (same formula as DREAMCell for a fair comparison).
    """

    def __init__(self, input_dim: int = N_MELS, hidden_dim: int = HIDDEN):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRUCell(input_dim, hidden_dim)
        self.C   = nn.Parameter(torch.empty(input_dim, hidden_dim))
        nn.init.xavier_uniform_(self.C)

    def forward_step(self, x: torch.Tensor, h: torch.Tensor):
        """
        x : (1, input_dim)
        h : (1, hidden_dim)
        Returns prediction error (rel), new h.
        """
        x_scale = x.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        # predict from previous h
        x_pred = torch.tanh(h @ self.C.T) * x_scale

        # update hidden state (no gradient)
        h_new = self.gru(x, h)

        error         = x - x_pred
        rel_error     = (error.norm(dim=-1) / x_scale.squeeze(1)).clamp(max=4.0)
        return rel_error, h_new

    def init_hidden(self, device=None, dtype=None):
        return torch.randn(1, self.hidden_dim, device=device, dtype=dtype) * 0.01


# ---------------------------------------------------------------------------
# GRU pre-train
# ---------------------------------------------------------------------------

def pretrain_gru(feats_a: torch.Tensor) -> GRUPredictor:
    model = GRUPredictor()
    opt   = torch.optim.Adam(model.parameters(), lr=PRETRAIN_LR)

    print(f"Pre-training GRU  ({PRETRAIN_EPOCHS} epochs) ...")
    for epoch in range(PRETRAIN_EPOCHS):
        h          = model.init_hidden()
        total_loss = 0.0
        for t in range(feats_a.shape[0]):
            x_t    = feats_a[t].unsqueeze(0)
            target = (x_t / (x_t.norm() + 1e-6)).clamp(-1, 1)
            pred   = torch.tanh(h.detach() @ model.C.T)
            loss   = F.mse_loss(pred, target)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), PRETRAIN_CLIP_GRAD)
            opt.step()
            total_loss += loss.item()
            with torch.no_grad():
                h = model.gru(x_t, h)

        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1}/{PRETRAIN_EPOCHS}  "
                  f"loss={total_loss/feats_a.shape[0]:.5f}")
    print("Done.\n")
    return model


def run_gru(model: GRUPredictor, feats: torch.Tensor):
    model.eval()
    losses = []
    with torch.no_grad():
        h = model.init_hidden(device=feats.device, dtype=feats.dtype)
        for t in range(feats.shape[0]):
            rel_err, h = model.forward_step(feats[t].unsqueeze(0), h)
            losses.append(rel_err.item())
    return np.array(losses)


# ---------------------------------------------------------------------------
# DREAM cell (reused from long-cycle experiment)
# ---------------------------------------------------------------------------

class DREAMCell_(DREAMCell):
    def __init__(self, config, mode="full"):
        super().__init__(config)
        self.mode = mode

    def forward_step(self, x, state):
        x_scale = x.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        x_norm  = (x / x_scale).clamp(-1.0, 1.0)
        x_pred  = torch.tanh(state.h @ self.C.T) * x_scale

        if self.mode == "full":
            h_U    = torch.bmm(state.h.unsqueeze(1), state.U).squeeze(1)
            x_pred = x_pred + 0.5 * (h_U @ self.V.T) * x_scale

        error      = x - x_pred
        rel_err    = (error.norm(dim=-1) / x_scale.squeeze(1)).clamp(max=4.0)
        surprise   = self.surprise_gate(error, rel_err, state)

        if self.mode == "full":
            self.update_fast_weights(state.h, error, surprise, state)

        base_eff = (self.B @ x_norm.T).T
        err_eff  = (self.W @ error.T).T
        h_new    = self.compute_ltc_update(
            state.h,
            state.h * 0.6 + base_eff * 0.2 + err_eff * surprise.unsqueeze(1) * 0.3,
            surprise,
        )
        a = 0.05
        state.error_mean   = (1-a)*state.error_mean   + a*error
        state.error_var    = (1-a)*state.error_var    + a*(error - state.error_mean)**2
        state.avg_surprise = (1-self.beta_s)*state.avg_surprise + self.beta_s*surprise
        if state.avg_surprise.mean() < self.S_min:
            dU = self.sleep_rate * (state.U - state.U_target)
            state.U_target = state.U_target + dU
            nt = state.U_target.norm(dim=(1,2), keepdim=True)
            state.U_target *= (self.target_norm/(nt+1e-6)).clamp(max=1.5)
        state.h = h_new
        return h_new, state, surprise, rel_err


def pretrain_dream(config, feats_a):
    cell      = DREAMCell_(config, mode="static")
    trainable = [cell.C, cell.W, cell.B, cell.tau_sys, cell.ltc_surprise_scale]
    opt       = torch.optim.Adam(trainable, lr=PRETRAIN_LR)
    print(f"Pre-training DREAM  ({PRETRAIN_EPOCHS} epochs) ...")
    for epoch in range(PRETRAIN_EPOCHS):
        state = cell.init_state(1)
        for t in range(feats_a.shape[0]):
            x_t    = feats_a[t].unsqueeze(0)
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


def run_dream(pretrained, feats, mode):
    cell = DREAMCell_(pretrained.config, mode=mode)
    with torch.no_grad():
        for a in ("C","W","B","tau_sys","ltc_surprise_scale","eta"):
            getattr(cell, a).copy_(getattr(pretrained, a))
    cell.eval()
    losses = []
    with torch.no_grad():
        state = cell.init_state(1)
        for t in range(feats.shape[0]):
            _, state, _, rel = cell.forward_step(feats[t].unsqueeze(0), state)
            losses.append(rel.item())
    return np.array(losses)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def seg_means(losses, fps):
    return [losses[i*fps:(i+1)*fps].mean() for i in range(len(SEQUENCE))]


def param_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def smooth(x, w=12):
    return np.convolve(x, np.ones(w)/w, mode="same")


def plot(all_losses, switch_frames, fps, out="gru_baseline_results.png"):
    T    = len(next(iter(all_losses.values())))
    t    = np.arange(T) / FPS
    sw_t = [s / FPS for s in switch_frames]

    fig = plt.figure(figsize=(16, 11))
    gs  = fig.add_gridspec(2, 2, height_ratios=[1.6, 1.0],
                           hspace=0.42, wspace=0.30,
                           left=0.07, right=0.97, top=0.91, bottom=0.07)

    ax_loss = fig.add_subplot(gs[0, :])
    ax_seg  = fig.add_subplot(gs[1, 0])
    ax_a    = fig.add_subplot(gs[1, 1])

    fig.suptitle(
        "GRU Baseline vs DREAM  —  Long-Cycle Speaker Switch\n"
        "A¹ → B¹ → C¹ → A² → B² → C² → A³   (7 × 3 s = 21 s, no gradients during inference)",
        fontsize=13, fontweight="bold",
    )

    def shade(ax, ymax):
        for i, (label, _, visit, _) in enumerate(SEQUENCE):
            x0  = i * fps / FPS
            x1  = (i+1) * fps / FPS
            col = SPEAKER_COLORS[label[0]]
            ax.axvspan(x0, x1, alpha=0.05 if visit==1 else 0.13, color=col)
            ax.text((x0+x1)/2, ymax*0.91, label, ha="center", fontsize=9,
                    color=col, fontweight="bold" if visit>1 else "normal")
        for s in sw_t:
            ax.axvline(s, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    # ── Loss curves ───────────────────────────────────────────────────────
    for key, losses in all_losses.items():
        ax_loss.plot(t, smooth(losses), color=PALETTE[key],
                     label=LEGEND[key], linewidth=2.0,
                     linestyle="--" if key=="gru" else "-")
    ax_loss.set_ylabel("Relative Prediction Error", fontsize=10)
    ax_loss.set_xlabel("Time  (seconds)", fontsize=10)
    ax_loss.set_title("Prediction Loss  —  DREAM (full) vs DREAM (static) vs GRU",
                      fontsize=10, fontweight="bold")
    ax_loss.legend(fontsize=9, loc="upper right")
    ax_loss.grid(True, alpha=0.2); ax_loss.set_ylim(0, 4.0)
    shade(ax_loss, 4.0)

    labels  = [s[0] for s in SEQUENCE]
    x       = np.arange(len(SEQUENCE))
    w       = 0.28
    bar_kw  = dict(edgecolor="white", linewidth=0.5)

    # ── Mean loss per segment ─────────────────────────────────────────────
    for i, (key, offset) in enumerate(zip(
            ["dream_full", "dream_static", "gru"],
            [-w, 0, w])):
        means = seg_means(all_losses[key], fps)
        bars  = ax_seg.bar(x + offset, means, w, color=PALETTE[key],
                           label=LEGEND[key], **bar_kw,
                           linestyle="--" if key=="gru" else "-")
        hatches = ["///" if s[3] else "" for s in SEQUENCE]
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)

    ax_seg.set_xticks(x); ax_seg.set_xticklabels(labels, fontsize=9)
    ax_seg.set_ylabel("Mean Loss", fontsize=9)
    ax_seg.set_title("Mean Loss per Segment\n(hatched = return visit)",
                     fontsize=9, fontweight="bold")
    ax_seg.legend(fontsize=7.5, loc="upper right")
    ax_seg.grid(axis="y", alpha=0.3); ax_seg.set_ylim(0, 3.2)

    # ── Speaker A progressive improvement ────────────────────────────────
    a_idx  = [i for i, s in enumerate(SEQUENCE) if s[0][0] == "A"]
    a_lbls = [SEQUENCE[i][0] for i in a_idx]
    xa     = np.arange(len(a_idx))

    for i, (key, offset) in enumerate(zip(
            ["dream_full", "dream_static", "gru"],
            [-w, 0, w])):
        vals = [seg_means(all_losses[key], fps)[i] for i in a_idx]
        bars = ax_a.bar(xa + offset, vals, w, color=PALETTE[key],
                        label=LEGEND[key], **bar_kw)
        for bar, v in zip(bars, vals):
            ax_a.text(bar.get_x()+bar.get_width()/2, v+0.02,
                      f"{v:.2f}", ha="center", fontsize=8)

    ax_a.set_xticks(xa); ax_a.set_xticklabels(a_lbls, fontsize=10)
    ax_a.set_ylabel("Mean Loss", fontsize=9)
    ax_a.set_title("Speaker A  —  Progressive Improvement\n"
                   "Only DREAM (full) improves on return visits",
                   fontsize=9, fontweight="bold")
    ax_a.legend(fontsize=7.5); ax_a.grid(axis="y", alpha=0.3)
    ax_a.set_ylim(0, 2.2)

    p1 = mpatches.Patch(facecolor="#aaa", label="1st encounter")
    p2 = mpatches.Patch(facecolor="#aaa", hatch="///", label="Return visit")
    fig.legend(handles=[p1, p2], loc="lower center", ncol=2,
               fontsize=9, bbox_to_anchor=(0.5, 0.0))

    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {out}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(all_losses, fps, gru_params, dream_params):
    segs = {k: seg_means(v, fps) for k, v in all_losses.items()}

    print(f"\n{'='*70}")
    print(f"  GRU BASELINE COMPARISON — REPORT")
    print(f"{'='*70}")
    print(f"  Parameters:  GRU = {gru_params:,}   DREAM base = {dream_params:,}"
          f"   (GRU has {gru_params/dream_params:.1f}× more)")
    print()
    print(f"  {'Seg':>4}  {'Return':>6}  {'DREAM-Full':>11}  "
          f"{'DREAM-Stat':>11}  {'GRU':>8}  {'DREAM/GRU':>10}")
    print(f"  {'-'*60}")
    for i, (label, _, _, is_ret) in enumerate(SEQUENCE):
        df = segs["dream_full"][i]
        ds = segs["dream_static"][i]
        g  = segs["gru"][i]
        adv = (g - df) / g * 100
        ret = "yes ✓" if is_ret else "no"
        print(f"  {label:>4}  {ret:>6}  {df:>11.4f}  {ds:>11.4f}  "
              f"{g:>8.4f}  {adv:>+9.1f}%")

    print(f"\n  ── Overall mean loss (21 s) ──")
    for k in ("dream_full", "dream_static", "gru"):
        m = np.mean(segs[k])
        print(f"  {LEGEND[k]:<35} {m:.4f}")

    # Familiar-voice advantage (A visits)
    a_idx = [i for i, s in enumerate(SEQUENCE) if s[0][0] == "A"]
    print(f"\n  ── Speaker A — progression ──")
    print(f"  {'Visit':>6}  {'DREAM-Full':>11}  {'DREAM-Stat':>11}  {'GRU':>8}")
    for i in a_idx:
        lbl = SEQUENCE[i][0]
        print(f"  {lbl:>6}  {segs['dream_full'][i]:>11.4f}  "
              f"{segs['dream_static'][i]:>11.4f}  {segs['gru'][i]:>8.4f}")

    df_trend = segs["dream_full"][a_idx[-1]] < segs["dream_full"][a_idx[0]]
    gru_trend = segs["gru"][a_idx[-1]] < segs["gru"][a_idx[0]]
    print(f"\n  DREAM-Full improves on A over visits: {'YES ✓' if df_trend else 'NO ✗'}")
    print(f"  GRU        improves on A over visits: {'YES' if gru_trend else 'NO ✗  (as expected)'}")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    print("Loading audio ...")
    feats_pretrain, feats_full, switch_frames, fps = make_features()
    print(f"  Total: {feats_full.shape[0]} frames ({feats_full.shape[0]/FPS:.1f}s)\n")

    dream_config = DREAMConfig(
        input_dim=N_MELS, hidden_dim=HIDDEN, rank=8,
        base_threshold=0.35, base_plasticity=0.4,
        forgetting_rate=0.03, adaptive_forgetting_scale=8.0,
        ltc_tau_sys=5.0, ltc_surprise_scale=8.0,
        surprise_temperature=0.12, entropy_influence=0.2,
        time_step=0.1, sleep_rate=0.005, min_surprise_for_sleep=0.25,
    )

    # ── Pre-train both models on the same Speaker A data ──────────────────
    pretrained_dream = pretrain_dream(dream_config, feats_pretrain)
    pretrained_gru   = pretrain_gru(feats_pretrain)

    # ── Inference ──────────────────────────────────────────────────────────
    print("Running DREAM (full) ...")
    losses_dream_full   = run_dream(pretrained_dream, feats_full, mode="full")

    print("Running DREAM (static) ...")
    losses_dream_static = run_dream(pretrained_dream, feats_full, mode="static")

    print("Running GRU ...")
    losses_gru          = run_gru(pretrained_gru, feats_full)

    all_losses = {
        "dream_full":   losses_dream_full,
        "dream_static": losses_dream_static,
        "gru":          losses_gru,
    }

    # ── Report ─────────────────────────────────────────────────────────────
    print_report(
        all_losses, fps,
        gru_params   = param_count(pretrained_gru),
        dream_params = sum(p.numel() for p in pretrained_dream.parameters()
                          if p.requires_grad),
    )

    # ── Plot ───────────────────────────────────────────────────────────────
    plot(all_losses, switch_frames, fps)
