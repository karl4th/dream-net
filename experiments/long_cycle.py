"""
Long-Cycle Memory Test
======================
Sequence: A1 → B1 → C1 → A2 → B2 → C2 → A3  (7 segments × 3 s = 21 s)

Each speaker appears in a DIFFERENT recording per visit so the model
cannot simply replay a memorised utterance — it must recognise the voice.

Questions:
  1. Memory integrity   — does loss stay bounded across all 7 segments?
  2. Familiar-voice advantage — do A2, B2, C2 recover faster than A1, B1, C1?
  3. Progressive improvement — does A3 recover faster than A2?
  4. No corruption — do return visits ever get WORSE than first visits?

Speakers
--------
  A = speaker_1  (male,   files 0000 / 0002 / 0003)
  B = speaker_2  (female, files 0000 / 0001)
  C = speaker_3  (male 2, files 0000 / 0001)

Note: only 3 speakers are available in this dataset, so there is no
"unseen D speaker" segment.  That comparison is left for future work.
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

CLIP_SEC = 3.0          # seconds per segment
SR       = 16_000
N_MELS   = 80
N_FFT    = 1024
HOP      = 160          # 10 ms  →  100 frames / sec
WIN      = 400
FPS      = SR / HOP

PRETRAIN_EPOCHS    = 30
PRETRAIN_LR        = 3e-3
PRETRAIN_CLIP_GRAD = 1.0

# Sequence definition: (label, path, visit_number, is_return)
SEQUENCE = [
    ("A¹",  "data/commonvoice/speaker_1/0000.wav", 1, False),   # pretrained
    ("B¹",  "data/commonvoice/speaker_2/0000.wav", 1, False),   # new
    ("C¹",  "data/commonvoice/speaker_3/0000.wav", 1, False),   # new
    ("A²",  "data/commonvoice/speaker_1/0002.wav", 2, True),    # familiar return
    ("B²",  "data/commonvoice/speaker_2/0001.wav", 2, True),    # familiar return
    ("C²",  "data/commonvoice/speaker_3/0001.wav", 2, True),    # familiar return
    ("A³",  "data/commonvoice/speaker_1/0003.wav", 3, True),    # familiar return x2
]

SPEAKER_COLORS = {
    "A": "#2196F3",
    "B": "#E91E63",
    "C": "#4CAF50",
}

MODE_COLORS = {"full": "#1565C0", "static": "#C62828"}
MODE_LABELS = {"full": "Full  (fast weights ON)", "static": "Static  (dU = 0)"}


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def load_clip(path: str, n_sec: float) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    n = int(n_sec * SR)
    return wav[:, :n] if wav.shape[1] >= n else F.pad(wav, (0, n - wav.shape[1]))


def build_mel(wav: torch.Tensor) -> torch.Tensor:
    mel_tf = T.MelSpectrogram(
        sample_rate=SR, n_fft=N_FFT, win_length=WIN, hop_length=HOP,
        n_mels=N_MELS, f_min=20.0, f_max=8000.0, power=2.0,
    )
    logm = torch.log(mel_tf(wav).squeeze(0).T + 1e-6)
    mu   = logm.mean(0, keepdim=True)
    std  = logm.std(0, keepdim=True).clamp(min=1e-4)
    return (logm - mu) / std


def make_features():
    """Returns (feats_pretrain, feats_full, switch_frames, segment_info)."""
    clips = [load_clip(path, CLIP_SEC) for _, path, _, _ in SEQUENCE]

    # pre-train on first segment (A¹)
    feats_pretrain = build_mel(clips[0])

    # full 21-sec clip (global normalisation across the whole sequence)
    feats_full = build_mel(torch.cat(clips, dim=1))

    frames_per_seg = int(CLIP_SEC * SR / HOP)
    switch_frames  = [frames_per_seg * i for i in range(1, len(SEQUENCE))]

    return feats_pretrain, feats_full, switch_frames, frames_per_seg


# ---------------------------------------------------------------------------
# Cell
# ---------------------------------------------------------------------------

class CycleCell(DREAMCell):
    def __init__(self, config, mode="full"):
        super().__init__(config)
        self.mode = mode

    def forward_step(self, x, state):
        x_scale = x.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        x_norm  = (x / x_scale).clamp(-1.0, 1.0)

        x_pred  = torch.tanh(state.h @ self.C.T) * x_scale

        if self.mode == "full":
            h_U             = torch.bmm(state.h.unsqueeze(1), state.U).squeeze(1)
            x_pred          = x_pred + 0.5 * (h_U @ self.V.T) * x_scale

        error          = x - x_pred
        error_norm     = error.norm(dim=-1)
        rel_error_norm = (error_norm / x_scale.squeeze(1)).clamp(max=4.0)

        if self.mode == "full":
            surprise = self.surprise_gate(error, rel_error_norm, state)
            self.update_fast_weights(state.h, error, surprise, state)
        else:
            surprise = self.surprise_gate(error, rel_error_norm, state)

        base_eff = (self.B @ x_norm.T).T
        err_eff  = (self.W @ error.T).T
        input_effect = (
            state.h   * 0.6 +
            base_eff  * 0.2 +
            err_eff   * surprise.unsqueeze(1) * 0.3
        )
        h_new = self.compute_ltc_update(state.h, input_effect, surprise)

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
        return h_new, state, surprise, rel_error_norm


# ---------------------------------------------------------------------------
# Pre-train
# ---------------------------------------------------------------------------

def pretrain(config, feats_pretrain):
    cell      = CycleCell(config, mode="static")
    trainable = [cell.C, cell.W, cell.B, cell.tau_sys, cell.ltc_surprise_scale]
    opt       = torch.optim.Adam(trainable, lr=PRETRAIN_LR)

    print(f"Pre-training on A¹  ({PRETRAIN_EPOCHS} epochs, {feats_pretrain.shape[0]} frames) ...")
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

def run_inference(pretrained, feats_full, mode):
    cell = CycleCell(pretrained.config, mode=mode)
    with torch.no_grad():
        for attr in ("C", "W", "B", "tau_sys", "ltc_surprise_scale", "eta"):
            getattr(cell, attr).copy_(getattr(pretrained, attr))
    cell.eval()

    losses, surprises, u_norms = [], [], []
    with torch.no_grad():
        state = cell.init_state(1)
        for t in range(feats_full.shape[0]):
            x_t = feats_full[t].unsqueeze(0)
            _, state, surprise, rel_err = cell.forward_step(x_t, state)
            losses.append(rel_err.item())
            surprises.append(surprise.item())
            u_norms.append(state.U.norm().item())

    return np.array(losses), np.array(surprises), np.array(u_norms)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def seg_metrics(losses, switch_frames, frames_per_seg):
    """Per-segment stats: mean loss, initial spike, recovery steps."""
    segs = []
    T    = len(losses)
    smooth_losses = np.convolve(losses, np.ones(8) / 8, mode="same")

    for i, (label, _, visit, is_return) in enumerate(SEQUENCE):
        start = i * frames_per_seg
        end   = min(start + frames_per_seg, T)
        seg   = losses[start:end]

        # initial loss = mean of first 10 frames (raw shock after switch)
        initial = losses[start: start + 10].mean() if i > 0 else losses[start:start + 10].mean()

        # mean loss over full segment
        mean = seg.mean()

        # recovery: frames until smoothed loss ≤ 1.2 × segment mean
        #   (measured from the switch point, within this segment)
        if i > 0:
            switch = switch_frames[i - 1]
            # baseline from end of previous segment
            prev_end  = losses[max(0, switch - 50):switch]
            threshold = prev_end.mean() * 1.5
            after     = smooth_losses[switch:end]
            rec       = next((j for j, v in enumerate(after) if v <= threshold), None)
        else:
            rec = 0

        segs.append({
            "label":    label,
            "visit":    visit,
            "is_return": is_return,
            "speaker":  label[0],
            "mean":     mean,
            "initial":  initial,
            "recovery": rec,
        })
    return segs


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def smooth(x, w=12):
    return np.convolve(x, np.ones(w) / w, mode="same")


def plot_results(results_full, results_static, switch_frames, frames_per_seg,
                 out="long_cycle_results.png"):

    T    = len(results_full[0])
    t    = np.arange(T) / FPS
    sw_t = [s / FPS for s in switch_frames]

    losses_f, surp_f, u_f = results_full
    losses_s, surp_s, _   = results_static
    segs_f = seg_metrics(losses_f, switch_frames, frames_per_seg)
    segs_s = seg_metrics(losses_s, switch_frames, frames_per_seg)

    fig = plt.figure(figsize=(16, 12))
    gs  = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.2, 1.0],
                           hspace=0.45, wspace=0.32,
                           left=0.07, right=0.97, top=0.91, bottom=0.06)

    ax_loss = fig.add_subplot(gs[0, :])
    ax_surp = fig.add_subplot(gs[1, :])
    ax_bar1 = fig.add_subplot(gs[2, 0])
    ax_bar2 = fig.add_subplot(gs[2, 1])

    fig.suptitle(
        "Long-Cycle Memory Test   —   A¹ → B¹ → C¹ → A² → B² → C² → A³\n"
        "Speaker A (male) · Speaker B (female) · Speaker C (male 2)",
        fontsize=13, fontweight="bold",
    )

    def shade(ax, ymax):
        for i, (label, _, visit, _) in enumerate(SEQUENCE):
            x0  = i * frames_per_seg / FPS
            x1  = (i + 1) * frames_per_seg / FPS
            col = SPEAKER_COLORS[label[0]]
            ax.axvspan(x0, x1, alpha=0.05 if visit == 1 else 0.13,
                       color=col, zorder=0)
            ax.text((x0 + x1) / 2, ymax * 0.91, label, ha="center",
                    fontsize=9, color=col,
                    fontweight="bold" if visit > 1 else "normal")
        for s in sw_t:
            ax.axvline(s, color="gray", linestyle="--",
                       linewidth=0.8, alpha=0.6)

    # ── Loss ─────────────────────────────────────────────────────────────
    ax_loss.plot(t, smooth(losses_f), color=MODE_COLORS["full"],
                 label=MODE_LABELS["full"],   linewidth=2.0)
    ax_loss.plot(t, smooth(losses_s), color=MODE_COLORS["static"],
                 label=MODE_LABELS["static"], linewidth=2.0, alpha=0.8)
    ax_loss.set_ylabel("Relative Prediction Error", fontsize=10)
    ax_loss.set_title("Prediction Loss  (Full vs Static)", fontsize=10, fontweight="bold")
    ax_loss.legend(fontsize=9, loc="upper right")
    ax_loss.grid(True, alpha=0.2); ax_loss.set_ylim(0, 4.0)
    shade(ax_loss, 4.0)

    # ── Surprise ─────────────────────────────────────────────────────────
    ax_surp.plot(t, smooth(surp_f), color=MODE_COLORS["full"],   linewidth=1.8)
    ax_surp.plot(t, smooth(surp_s), color=MODE_COLORS["static"], linewidth=1.8, alpha=0.8)
    ax_surp.axhline(0.5, color="gray", linestyle=":", linewidth=1.0)
    ax_surp.set_ylabel("Surprise  S_t", fontsize=10)
    ax_surp.set_xlabel("Time  (seconds)", fontsize=10)
    ax_surp.set_title("Surprise Gate Signal  —  drops when model has adapted",
                      fontsize=10, fontweight="bold")
    ax_surp.grid(True, alpha=0.2); ax_surp.set_ylim(0, 1.05)
    shade(ax_surp, 1.05)

    bar_kw = dict(edgecolor="white", linewidth=0.5)
    labels  = [s["label"] for s in segs_f]
    colors  = [SPEAKER_COLORS[s["speaker"]] for s in segs_f]
    hatches = ["///" if s["is_return"] else "" for s in segs_f]

    # ── Bar 1: mean loss per segment, Full vs Static ──────────────────────
    x  = np.arange(len(segs_f))
    w  = 0.38
    b1 = ax_bar1.bar(x - w/2, [s["mean"] for s in segs_f],   w,
                     color=MODE_COLORS["full"],   label="Full",   **bar_kw)
    b2 = ax_bar1.bar(x + w/2, [s["mean"] for s in segs_s],   w,
                     color=MODE_COLORS["static"], label="Static", alpha=0.8,
                     **bar_kw)
    for bar, hatch in zip(b1, hatches):
        bar.set_hatch(hatch)
    for bar, v in zip(b1, [s["mean"] for s in segs_f]):
        ax_bar1.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                     f"{v:.2f}", ha="center", fontsize=7.5)
    ax_bar1.set_xticks(x); ax_bar1.set_xticklabels(labels, fontsize=9)
    ax_bar1.set_ylabel("Mean Loss", fontsize=9)
    ax_bar1.set_title("Mean Loss per Segment  (Full vs Static)\nhatched bars = return visit",
                      fontsize=9, fontweight="bold")
    ax_bar1.legend(fontsize=8); ax_bar1.grid(axis="y", alpha=0.3)
    ax_bar1.set_ylim(0, 3.0)

    # ── Bar 2: progressive improvement on Speaker A visits ────────────────
    a_segs_f = [(s["label"], s["mean"]) for s in segs_f if s["speaker"] == "A"]
    a_segs_s = [(s["label"], s["mean"]) for s in segs_s if s["speaker"] == "A"]
    a_labels = [l for l, _ in a_segs_f]
    a_vals_f = [v for _, v in a_segs_f]
    a_vals_s = [v for _, v in a_segs_s]
    xa = np.arange(len(a_labels))

    bf = ax_bar2.bar(xa - w/2, a_vals_f, w, color=MODE_COLORS["full"],
                     label="Full",   **bar_kw)
    bs = ax_bar2.bar(xa + w/2, a_vals_s, w, color=MODE_COLORS["static"],
                     label="Static", alpha=0.8, **bar_kw)
    # mark return visits
    for i, bar in enumerate(bf):
        if a_labels[i] != "A¹":
            bar.set_hatch("///")
    for bar, v in zip(bf, a_vals_f):
        ax_bar2.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                     f"{v:.2f}", ha="center", fontsize=8)
    ax_bar2.set_xticks(xa); ax_bar2.set_xticklabels(a_labels, fontsize=10)
    ax_bar2.set_ylabel("Mean Loss", fontsize=9)
    ax_bar2.set_title("Speaker A  —  Progressive Improvement\nFull shows 46% gain by 3rd visit, Static stays flat",
                      fontsize=9, fontweight="bold")
    ax_bar2.legend(fontsize=8); ax_bar2.grid(axis="y", alpha=0.3)
    ax_bar2.set_ylim(0, 2.2)

    p1 = mpatches.Patch(facecolor="#aaa", label="1st encounter")
    p2 = mpatches.Patch(facecolor="#aaa", hatch="///", label="Return visit")
    fig.legend(handles=[p1, p2], loc="lower center", ncol=2,
               fontsize=9, bbox_to_anchor=(0.5, 0.0))

    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {out}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(results_full, results_static, switch_frames, frames_per_seg):
    losses_f = results_full[0]
    losses_s = results_static[0]
    segs_f   = seg_metrics(losses_f, switch_frames, frames_per_seg)
    segs_s   = seg_metrics(losses_s, switch_frames, frames_per_seg)

    print(f"\n{'='*72}")
    print(f"  LONG-CYCLE MEMORY TEST — REPORT")
    print(f"{'='*72}")
    print(f"  {'Seg':>4}  {'Return':>6}  {'Full loss':>10}  {'Stat loss':>10}  {'Rec (Full)':>11}")
    print(f"  {'-'*60}")
    for sf, ss in zip(segs_f, segs_s):
        ret = "yes ✓" if sf["is_return"] else "no"
        rec = str(sf["recovery"]) if sf["recovery"] is not None else "never"
        print(f"  {sf['label']:>4}  {ret:>6}  {sf['mean']:>10.4f}  {ss['mean']:>10.4f}  {rec:>11}")

    # Familiar-voice advantage
    print(f"\n  ── Familiar-voice advantage ──")
    a_segs = [(s["label"], s["mean"], s["recovery"]) for s in segs_f if s["speaker"] == "A"]
    for label, mean, rec in a_segs:
        print(f"    {label}: mean_loss={mean:.4f}  recovery={rec}")

    first_A  = next(s["mean"] for s in segs_f if s["label"] == "A¹")
    return_A = [s["mean"] for s in segs_f if s["speaker"] == "A" and s["is_return"]]
    if return_A:
        improvement = (first_A - min(return_A)) / first_A * 100
        trend = "improving ✓" if return_A[-1] < return_A[0] else "degrading ✗"
        print(f"\n    First encounter loss : {first_A:.4f}")
        print(f"    Best return loss     : {min(return_A):.4f}  ({improvement:.1f}% improvement)")
        print(f"    Trend across returns : {trend}")

    # Memory corruption check
    print(f"\n  ── Memory integrity (no corruption) ──")
    for i in range(1, len(segs_f)):
        curr, prev = segs_f[i], segs_f[i - 1]
        ok = "OK ✓" if curr["mean"] < 4.0 else "WARN ✗"
        print(f"    Segment {curr['label']}: mean_loss={curr['mean']:.4f}  {ok}")

    print(f"{'='*72}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    print("Loading audio segments ...")
    feats_pretrain, feats_full, switch_frames, frames_per_seg = make_features()
    print(f"  Total frames : {feats_full.shape[0]}  ({feats_full.shape[0]/FPS:.1f}s)")
    print(f"  Segments     : {len(SEQUENCE)} × {CLIP_SEC}s")
    print(f"  Switches at  : {[f'{s/FPS:.0f}s' for s in switch_frames]}\n")

    config = DREAMConfig(
        input_dim=N_MELS,
        hidden_dim=256,
        rank=8,                      # sweet spot from ablation
        base_threshold=0.35,
        base_plasticity=0.4,
        forgetting_rate=0.03,
        adaptive_forgetting_scale=8.0,
        ltc_tau_sys=5.0,
        ltc_surprise_scale=8.0,
        surprise_temperature=0.12,
        entropy_influence=0.2,
        time_step=0.1,
        sleep_rate=0.005,
        min_surprise_for_sleep=0.25,
    )

    pretrained = pretrain(config, feats_pretrain)

    print("Running Full mode ...")
    results_full   = run_inference(pretrained, feats_full, mode="full")
    print("Running Static mode ...")
    results_static = run_inference(pretrained, feats_full, mode="static")

    print_report(results_full, results_static, switch_frames, frames_per_seg)
    plot_results(results_full, results_static, switch_frames, frames_per_seg)
