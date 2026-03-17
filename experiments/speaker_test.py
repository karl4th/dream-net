"""
Multi-Speaker Adaptation Test
==============================
Three CommonVoice speakers concatenated into one 44-second stream.
All models run on the full sequence — no state reset at speaker boundaries.

DREAM (full mode): adapts online via fast weights + habituation + surprise gate.
LSTM / Transformer: frozen weights, no adaptation.

Expected:
  - DREAM: MSE spikes at speaker switch, then drops as it adapts
  - LSTM / TF: MSE varies by speaker but no downward trend within segments

Outputs:
  speaker_test_mse.png          — per-frame MSE curves + speaker boundaries
  results/speaker_original.wav
  results/speaker_dream.wav
  results/speaker_lstm.wav
  results/speaker_tf.wav
"""

import os, sys, warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore", message="enable_nested_tensor")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dream_net import DREAMConfig, DREAMCell, DREAMState

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SR           = 16_000
N_MELS       = 80
N_FFT        = 1024
HOP          = 160
WIN          = 400
N_STFT       = N_FFT // 2 + 1
FPS          = SR / HOP

DREAM_HIDDEN = 833
LSTM_HIDDEN  = 178
TF_D         = 100; TF_FF = 256; TF_NHEAD = 4; TF_NLAYERS = 2

RESULTS_DIR  = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

SPEAKER_FILES = [
    ("Speaker 1", "data/commonvoice/speaker_1/0000.wav"),
    ("Speaker 2", "data/commonvoice/speaker_2/0002.wav"),
    ("Speaker 3", "data/commonvoice/speaker_3/0002.wav"),
]

SMOOTH_WIN = 50   # frames = 0.5 seconds for MSE smoothing
PALETTE    = {"dream": "#1565C0", "lstm": "#C62828", "tf": "#2E7D32"}

# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def load_wav(path):
    wav, sr = torchaudio.load(path)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    return wav

_MEL_TF = None
def mel_transform():
    global _MEL_TF
    if _MEL_TF is None:
        _MEL_TF = T.MelSpectrogram(sample_rate=SR, n_fft=N_FFT, win_length=WIN,
                                    hop_length=HOP, n_mels=N_MELS,
                                    f_min=20.0, f_max=8000.0, power=2.0)
    return _MEL_TF

def wav_to_logmel(wav):
    return torch.log(mel_transform()(wav).squeeze(0).T + 1e-6)  # (T, 80)

def mel_to_wav(norm_mel, mu, std, n_iter=128):
    logm      = norm_mel.detach().cpu() * std + mu
    mel_power = (torch.exp(logm) - 1e-6).clamp(min=1e-10)
    inv_mel   = T.InverseMelScale(n_stft=N_STFT, n_mels=N_MELS, sample_rate=SR,
                                   f_min=20.0, f_max=8000.0)
    linear    = inv_mel(mel_power.T)
    gl        = T.GriffinLim(n_fft=N_FFT, win_length=WIN, hop_length=HOP,
                              power=2.0, n_iter=n_iter)
    return gl(linear).unsqueeze(0)


def build_stream():
    """
    Load all speakers, concatenate, normalise globally.
    Returns (feats [T,80], mu [1,80], std [1,80], boundaries [list of int]).
    boundaries[i] = first frame of speaker i+1.
    """
    logmels, boundaries = [], []
    offset = 0
    for name, path in SPEAKER_FILES:
        wav  = load_wav(path)
        logm = wav_to_logmel(wav)
        logmels.append(logm)
        offset += logm.shape[0]
        boundaries.append(offset)
        print(f"  {name}: {logm.shape[0]} frames  ({logm.shape[0]/FPS:.1f}s)")

    all_logmel = torch.cat(logmels, dim=0)          # (T_total, 80)
    mu  = all_logmel.mean(0, keepdim=True)
    std = all_logmel.std(0, keepdim=True).clamp(min=1e-4)
    feats = (all_logmel - mu) / std

    boundaries = boundaries[:-1]  # drop last (= end of stream)
    return feats, mu, std, boundaries


# ---------------------------------------------------------------------------
# Models  (identical to listen.py)
# ---------------------------------------------------------------------------

class DREAMPredictor(DREAMCell):
    def __init__(self, config, mode="full"):
        super().__init__(config)
        self.mode = mode

    def forward_step(self, x, state):
        x_scale  = x.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        x_norm   = (x / x_scale).clamp(-1.0, 1.0)
        x_pred   = torch.tanh(state.h @ self.C.T) * x_scale
        if self.mode == "full":
            h_U    = torch.bmm(state.h.unsqueeze(1), state.U).squeeze(1)
            x_pred = x_pred + 0.5 * (h_U @ self.V.T) * x_scale
        error    = x - x_pred
        rel_err  = (error.norm(dim=-1) / x_scale.squeeze(1)).clamp(max=4.0)
        surprise = self.surprise_gate(error, rel_err, state)
        if self.mode == "full":
            self.update_fast_weights(state.h, error, surprise, state)
        base_eff = (self.B @ x_norm.T).T
        err_eff  = (self.W @ error.T).T
        h_new    = self.compute_ltc_update(
            state.h,
            state.h * 0.6 + base_eff * 0.2 + err_eff * surprise.unsqueeze(1) * 0.3,
            surprise,
        )
        state.error_mean   = (1-0.05)*state.error_mean   + 0.05*error.detach()
        state.error_var    = (1-0.05)*state.error_var    + 0.05*(error.detach()-state.error_mean)**2
        state.avg_surprise = (1-self.beta_s)*state.avg_surprise + self.beta_s*surprise.detach()
        state.h = h_new
        return x_pred, state

    def predict_sequence(self, feats):
        preds  = []
        state  = self.init_state(1)
        with torch.no_grad():
            for t in range(feats.shape[0]):
                x_pred, state = self.forward_step(feats[t].unsqueeze(0), state)
                preds.append(x_pred)
        return torch.cat(preds, 0)


class LSTMPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTMCell(N_MELS, LSTM_HIDDEN)
        self.C    = nn.Parameter(torch.empty(N_MELS, LSTM_HIDDEN))
        nn.init.xavier_uniform_(self.C)

    def predict_sequence(self, feats):
        preds = []
        h = c = torch.zeros(1, LSTM_HIDDEN)
        with torch.no_grad():
            for t in range(feats.shape[0]):
                x_t     = feats[t].unsqueeze(0)
                x_scale = x_t.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                preds.append(torch.tanh(h @ self.C.T) * x_scale)
                h, c = self.lstm(x_t, (h, c))
        return torch.cat(preds, 0)


class TransformerPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed   = nn.Linear(N_MELS, TF_D)
        layer = nn.TransformerEncoderLayer(d_model=TF_D, nhead=TF_NHEAD,
                                            dim_feedforward=TF_FF, dropout=0.0,
                                            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=TF_NLAYERS)
        self.readout = nn.Linear(TF_D, N_MELS)

    def predict_sequence(self, feats):
        n    = feats.shape[0]
        x_in = torch.cat([torch.zeros(1, N_MELS), feats[:-1]], dim=0)
        h    = self.embed(x_in.unsqueeze(0))
        mask = nn.Transformer.generate_square_subsequent_mask(n)
        with torch.no_grad():
            out = self.encoder(h, mask=mask, is_causal=True)
        return self.readout(out.squeeze(0)).detach()


# ---------------------------------------------------------------------------
# Per-frame MSE + smoothing
# ---------------------------------------------------------------------------

def per_frame_mse(preds, feats):
    """Returns (T,) tensor of squared errors per frame."""
    return ((preds - feats) ** 2).mean(dim=-1).cpu().numpy()

def smooth(x, w):
    return np.convolve(x, np.ones(w) / w, mode="same")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_mse(mse_dict, boundaries, total_frames, out="speaker_test_mse.png"):
    fig, ax = plt.subplots(figsize=(16, 5))
    t = np.arange(total_frames) / FPS

    for key, mse in mse_dict.items():
        label = {"dream": "DREAM (full — all features)", "lstm": "LSTM", "tf": "Transformer"}[key]
        lw    = 2.5 if key == "dream" else 1.5
        ax.plot(t, smooth(mse, SMOOTH_WIN), color=PALETTE[key], label=label,
                linewidth=lw, alpha=0.9)

    # speaker boundary lines
    for i, b in enumerate(boundaries):
        ax.axvline(b / FPS, color="black", linestyle="--", linewidth=1.2, alpha=0.6)
        ax.text(b / FPS + 0.1, ax.get_ylim()[1] * 0.92,
                f"→ Speaker {i+2}", fontsize=9, color="black", alpha=0.7)

    # speaker labels
    starts = [0] + [b / FPS for b in boundaries]
    ends   = [b / FPS for b in boundaries] + [total_frames / FPS]
    for i, (s, e) in enumerate(zip(starts, ends)):
        ax.text((s + e) / 2, ax.get_ylim()[1] * 0.05,
                f"Speaker {i+1}", ha="center", fontsize=10,
                color="gray", fontweight="bold")

    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel(f"MSE  (smoothed, window={SMOOTH_WIN} frames)", fontsize=11)
    ax.set_title("Multi-Speaker Adaptation Test — MSE over time\n"
                 "DREAM adapts online; LSTM/TF weights frozen",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0, total_frames / FPS)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {out}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(mse_dict, boundaries, total_frames):
    segs = list(zip([0] + boundaries, boundaries + [total_frames]))
    print(f"\n{'='*68}")
    print(f"  MULTI-SPEAKER ADAPTATION TEST — MSE per speaker")
    print(f"{'='*68}")
    print(f"\n  {'Segment':<14}", end="")
    for key in ("dream", "lstm", "tf"):
        label = {"dream": "DREAM (full)", "lstm": "LSTM", "tf": "TF"}[key]
        print(f"  {label:>12}", end="")
    print()
    print(f"  {'-'*52}")

    for i, (s, e) in enumerate(segs):
        print(f"  Speaker {i+1:<6}", end="")
        row = {}
        for key, mse in mse_dict.items():
            row[key] = mse[s:e].mean()
            print(f"  {row[key]:>12.4f}", end="")
        print()

    print(f"  {'-'*52}")
    print(f"  {'Overall':<14}", end="")
    for key, mse in mse_dict.items():
        print(f"  {mse.mean():>12.4f}", end="")
    print()

    print(f"\n  Adaptation effect (DREAM full vs static within each speaker):")
    dream_mse = mse_dict["dream"]
    for i, (s, e) in enumerate(segs):
        mid   = (s + e) // 2
        first = dream_mse[s: s + (e-s)//4].mean()   # first quarter
        last  = dream_mse[e - (e-s)//4: e].mean()   # last quarter
        trend = "↓ adapting" if last < first else "→ flat"
        print(f"    Speaker {i+1}: first quarter={first:.4f}  last quarter={last:.4f}  {trend}")

    print(f"\n{'='*68}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = DREAMConfig(
        input_dim=N_MELS, hidden_dim=DREAM_HIDDEN, rank=8,
        base_threshold=0.35, base_plasticity=0.4,
        forgetting_rate=0.03, adaptive_forgetting_scale=8.0,
        ltc_tau_sys=5.0, ltc_surprise_scale=8.0,
        surprise_temperature=0.12, entropy_influence=0.2,
        time_step=0.1, sleep_rate=0.005, min_surprise_for_sleep=0.25,
    )

    # ── Load weights ──────────────────────────────────────────────────────
    w = {
        "dream": "results/weights_dream_full.pt",
        "lstm":  "results/weights_lstm.pt",
        "tf":    "results/weights_tf.pt",
    }
    for k, p in w.items():
        assert os.path.exists(p), f"Missing: {p} — run reconstruction.py + dream_full_train.py first"

    dream = DREAMPredictor(config, mode="full")
    dream.load_state_dict(torch.load(w["dream"], weights_only=True))
    dream.eval()

    lstm = LSTMPredictor()
    lstm.load_state_dict(torch.load(w["lstm"], weights_only=True))
    lstm.eval()

    tf = TransformerPredictor()
    tf.load_state_dict(torch.load(w["tf"], weights_only=True))
    tf.eval()

    print("Weights loaded.\n")

    # ── Build stream ──────────────────────────────────────────────────────
    print("Building audio stream ...")
    feats, mu, std, boundaries = build_stream()
    n_frames = feats.shape[0]
    print(f"  Total: {n_frames} frames  ({n_frames/FPS:.1f}s)")
    print(f"  Speaker boundaries at frames: {boundaries}  "
          f"({[f'{b/FPS:.1f}s' for b in boundaries]})\n")

    # ── Inference ─────────────────────────────────────────────────────────
    print("Running inference ...")
    preds = {}
    for key, model in [("dream", dream), ("lstm", lstm), ("tf", tf)]:
        print(f"  {key} ...", end=" ", flush=True)
        preds[key] = model.predict_sequence(feats)
        mse = F.mse_loss(preds[key], feats).item()
        print(f"overall MSE={mse:.4f}")

    # ── Per-frame MSE ─────────────────────────────────────────────────────
    mse_dict = {k: per_frame_mse(preds[k], feats) for k in preds}

    print_report(mse_dict, boundaries, n_frames)
    plot_mse(mse_dict, boundaries, n_frames)

    # ── Audio ─────────────────────────────────────────────────────────────
    print("Reconstructing audio (n_iter=128) ...")
    for key in ["dream", "lstm", "tf"]:
        wav  = mel_to_wav(preds[key], mu, std)
        path = os.path.join(RESULTS_DIR, f"speaker_{key}.wav")
        torchaudio.save(path, wav, SR)
        print(f"  → {path}")

    wav = mel_to_wav(feats, mu, std)
    torchaudio.save(os.path.join(RESULTS_DIR, "speaker_original.wav"), wav, SR)
    print(f"  → results/speaker_original.wav")

    print("\nDone.")
