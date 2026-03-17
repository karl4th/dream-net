"""
Mel Spectrogram Reconstruction — DREAM vs LSTM vs Transformer
==============================================================
Dataset : LJSpeech (single female speaker, English narration)
Task    : Next-frame mel prediction (teacher-forced).
          Reconstruct audio via InverseMelScale + Griffin-Lim.

All models are matched at ~200 K trainable parameters:
  DREAM       hidden=833              → 199,923
  LSTM        hidden=178              → 199,360
  Transformer d=100 ff=256 nhead=4    → 200,892

Training protocols (each uses its natural optimisation):
  DREAM       detach-h pretrain (C, W, B, tau, scale) — proven_v1
  LSTM        truncated BPTT, chunk=50
  Transformer full-sequence causal, one pass per file

Inference:
  DREAM       fast weights ON (adapts online during the test clip)
  LSTM        stateful, no adaptation
  Transformer full causal attention on the test clip

Test clips:
  Seen    — LJ001-0010 (in training set)
  Unseen  — LJ001-0055 (never shown during training)

Outputs:
  reconstruction_metrics.png
  reconstruction_spectrograms_seen.png
  reconstruction_spectrograms_unseen.png
  results/reconstruction_<model>_<condition>.wav
  results/reconstruction_original_<condition>.wav  (oracle Griffin-Lim)
"""

import os, warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore", message="enable_nested_tensor")

from dream_net import DREAMConfig, DREAMCell, DREAMState


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SR          = 16_000
N_MELS      = 80
N_FFT       = 1024
HOP         = 160
WIN         = 400
N_STFT      = N_FFT // 2 + 1
FPS         = SR / HOP          # 100 fps

# ── Architecture sizes (matched at ~200K params) ──────────────────────────
DREAM_HIDDEN = 833
LSTM_HIDDEN  = 178
TF_D         = 100
TF_FF        = 256
TF_NHEAD     = 4
TF_NLAYERS   = 2

PARAM_TARGET  = 200_000
PARAM_TOL_PCT = 1.5             # abort if any model deviates more than this %

# ── Training ──────────────────────────────────────────────────────────────
CLIP_FRAMES = 400
EPOCHS      = 50
LR          = 3e-3
CLIP_GRAD   = 1.0
TBPTT_CHUNK = 50                # LSTM truncated BPTT window

# ── Data ──────────────────────────────────────────────────────────────────
DATA_DIR    = "data/ljspeech"
ALL_FILES   = sorted(os.listdir(DATA_DIR))
TRAIN_FILES = ALL_FILES[:50]
FILE_SEEN   = "LJ001-0010.wav"
FILE_UNSEEN = "LJ001-0055.wav"

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Colours ───────────────────────────────────────────────────────────────
PALETTE = {"dream": "#1565C0", "lstm": "#C62828", "tf": "#2E7D32"}
LABELS  = {
    "dream": "DREAM  (fast weights ON)",
    "lstm":  "LSTM  (no adaptation)",
    "tf":    "Transformer  (no adaptation)",
}


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def load_wav(path: str) -> torch.Tensor:
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
        _MEL_TF = T.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, win_length=WIN, hop_length=HOP,
            n_mels=N_MELS, f_min=20.0, f_max=8000.0, power=2.0,
        )
    return _MEL_TF


def build_mel(wav: torch.Tensor) -> tuple:
    """Returns (norm_logmel [T,80], mu [1,80], std [1,80])."""
    logm = torch.log(mel_transform()(wav).squeeze(0).T + 1e-6)  # (T,80)
    mu   = logm.mean(0, keepdim=True)
    std  = logm.std(0, keepdim=True).clamp(min=1e-4)
    return (logm - mu) / std, mu, std


def load_features(fname: str, max_frames: int | None = None):
    wav = load_wav(os.path.join(DATA_DIR, fname))
    feat, mu, std = build_mel(wav)
    if max_frames:
        feat = feat[:max_frames]
    return feat, mu, std


def mel_to_wav(norm_mel: torch.Tensor, mu: torch.Tensor, std: torch.Tensor,
               n_iter: int = 128) -> torch.Tensor:
    """Reconstruct waveform from z-scored log-mel spectrogram."""
    logm      = norm_mel.detach().cpu() * std + mu          # (T, 80)
    mel_power = (torch.exp(logm) - 1e-6).clamp(min=1e-10)  # (T, 80)
    inv_mel   = T.InverseMelScale(
        n_stft=N_STFT, n_mels=N_MELS, sample_rate=SR,
        f_min=20.0, f_max=8000.0,
    )
    linear_power = inv_mel(mel_power.T)                     # (N_STFT, T)
    gl = T.GriffinLim(n_fft=N_FFT, win_length=WIN, hop_length=HOP,
                      power=2.0, n_iter=n_iter)
    return gl(linear_power).unsqueeze(0)                    # (1, samples)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

# ── DREAM ──────────────────────────────────────────────────────────────────

class DREAMPredictor(DREAMCell):
    """DREAMCell with CycleCell-style forward_step and mode switch."""

    def __init__(self, config: DREAMConfig, mode: str = "static"):
        super().__init__(config)
        self.mode = mode

    def forward_step(self, x: torch.Tensor, state: DREAMState):
        """x: (1, N_MELS).  Returns (x_pred, h_new, state, surprise, rel_err)."""
        x_scale = x.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        x_norm  = (x / x_scale).clamp(-1.0, 1.0)

        # prediction in mel space  ←  collect this for reconstruction
        x_pred = torch.tanh(state.h @ self.C.T) * x_scale
        if self.mode == "full":
            h_U    = torch.bmm(state.h.unsqueeze(1), state.U).squeeze(1)
            x_pred = x_pred + 0.5 * (h_U @ self.V.T) * x_scale

        error   = x - x_pred
        rel_err = (error.norm(dim=-1) / x_scale.squeeze(1)).clamp(max=4.0)
        surprise = self.surprise_gate(error, rel_err, state)

        if self.mode == "full":
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
        return x_pred, h_new, state, surprise, rel_err

    def predict_sequence(self, feats: torch.Tensor):
        """Returns (preds [T,80] in mel space, rel_errs [T])."""
        preds, errs = [], []
        state = self.init_state(1)
        with torch.no_grad():
            for t in range(feats.shape[0]):
                x_t = feats[t].unsqueeze(0)
                x_pred, _, state, _, rel = self.forward_step(x_t, state)
                preds.append(x_pred)
                errs.append(rel.item())
        return torch.cat(preds, 0), np.array(errs)


# ── LSTM ───────────────────────────────────────────────────────────────────

class LSTMPredictor(nn.Module):
    def __init__(self, input_dim: int = N_MELS, hidden_dim: int = LSTM_HIDDEN):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTMCell(input_dim, hidden_dim)
        self.C    = nn.Parameter(torch.empty(input_dim, hidden_dim))
        nn.init.xavier_uniform_(self.C)

    def init_state(self):
        z = torch.zeros(1, self.hidden_dim)
        return z, z.clone()

    def predict_sequence(self, feats: torch.Tensor):
        """Teacher-forced; returns (preds [T,80], rel_errs [T])."""
        preds, errs = [], []
        h, c = self.init_state()
        with torch.no_grad():
            for t in range(feats.shape[0]):
                x_t     = feats[t].unsqueeze(0)
                x_scale = x_t.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                x_pred  = torch.tanh(h @ self.C.T) * x_scale
                preds.append(x_pred)
                error   = x_t - x_pred
                rel     = (error.norm(dim=-1) / x_scale.squeeze(1)).clamp(max=4.0)
                errs.append(rel.item())
                h, c = self.lstm(x_t, (h, c))
        return torch.cat(preds, 0), np.array(errs)


# ── Transformer ────────────────────────────────────────────────────────────

class TransformerPredictor(nn.Module):
    """Causal TransformerEncoder; readout directly in mel space."""

    def __init__(self, input_dim: int = N_MELS,
                 d_model: int = TF_D, nhead: int = TF_NHEAD,
                 nlayers: int = TF_NLAYERS, ff: int = TF_FF):
        super().__init__()
        self.embed   = nn.Linear(input_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.readout = nn.Linear(d_model, input_dim)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """x_seq: (T, input_dim).  Returns predictions (T, input_dim) in mel space."""
        T     = x_seq.shape[0]
        # shift: pred[t] = f(x[0..t-1])
        x_in  = torch.cat([torch.zeros(1, x_seq.shape[1], device=x_seq.device),
                            x_seq[:-1]], dim=0)
        h     = self.embed(x_in.unsqueeze(0))
        mask  = nn.Transformer.generate_square_subsequent_mask(T, device=x_seq.device)
        out   = self.encoder(h, mask=mask, is_causal=True)
        return self.readout(out.squeeze(0))

    def predict_sequence(self, feats: torch.Tensor):
        with torch.no_grad():
            preds   = self(feats)
        x_scale = feats.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        errs    = ((feats - preds).norm(dim=-1) / x_scale.squeeze(1)).clamp(max=4.0)
        return preds.detach(), errs.cpu().numpy()


# ---------------------------------------------------------------------------
# Parameter count check  —  must run before any training
# ---------------------------------------------------------------------------

def check_params(dream, lstm, tf):
    n = {
        "dream": sum(p.numel() for p in dream.parameters() if p.requires_grad),
        "lstm":  sum(p.numel() for p in lstm.parameters()  if p.requires_grad),
        "tf":    sum(p.numel() for p in tf.parameters()    if p.requires_grad),
    }
    print(f"\n{'─'*52}")
    print(f"  Parameter count check  (target ≈ {PARAM_TARGET:,})")
    print(f"{'─'*52}")
    ok = True
    for key, count in n.items():
        pct = abs(count - PARAM_TARGET) / PARAM_TARGET * 100
        flag = "✓" if pct <= PARAM_TOL_PCT else "✗  EXCEEDS TOLERANCE"
        print(f"  {LABELS[key]:<38} {count:>8,}  ({pct:.2f}%)  {flag}")
        if pct > PARAM_TOL_PCT:
            ok = False
    print(f"{'─'*52}\n")
    if not ok:
        raise RuntimeError("Parameter mismatch exceeds tolerance — fix hidden dims first.")
    return n


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_dream(config: DREAMConfig) -> tuple:
    cell      = DREAMPredictor(config, mode="static")
    trainable = [cell.C, cell.W, cell.B, cell.tau_sys, cell.ltc_surprise_scale]
    opt       = torch.optim.Adam(trainable, lr=LR)
    hist      = []

    print(f"Training DREAM  ({EPOCHS} epochs, detach-h pretrain) ...")
    for epoch in range(EPOCHS):
        total, count = 0.0, 0
        for fname in TRAIN_FILES:
            feats, _, _ = load_features(fname, CLIP_FRAMES)
            state = cell.init_state(1)
            for t in range(feats.shape[0]):
                x_t    = feats[t].unsqueeze(0)
                target = (x_t / (x_t.norm() + 1e-6)).clamp(-1, 1)
                pred   = torch.tanh(state.h.detach() @ cell.C.T)
                loss   = F.mse_loss(pred, target)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(trainable, CLIP_GRAD)
                opt.step()
                total += loss.item(); count += 1
                with torch.no_grad():
                    _, _, state, _, _ = cell.forward_step(x_t, state)
        hist.append(total / count)
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1:2d}/{EPOCHS}  loss={hist[-1]:.5f}")
    print("Done.\n")
    return cell, hist


def train_lstm() -> tuple:
    model = LSTMPredictor()
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    hist  = []

    print(f"Training LSTM  ({EPOCHS} epochs, truncated BPTT chunk={TBPTT_CHUNK}) ...")
    for epoch in range(EPOCHS):
        total, count = 0.0, 0
        for fname in TRAIN_FILES:
            feats, _, _ = load_features(fname, CLIP_FRAMES)
            T    = feats.shape[0]
            h, c = model.init_state()

            # Truncated BPTT: gradient flows within each chunk, state carries across
            for start in range(0, T, TBPTT_CHUNK):
                chunk  = feats[start: start + TBPTT_CHUNK]
                h_c    = h.detach()
                c_c    = c.detach()
                preds  = []
                for t in range(chunk.shape[0]):
                    x_t     = chunk[t].unsqueeze(0)
                    x_scale = x_t.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                    x_pred  = torch.tanh(h_c @ model.C.T) * x_scale
                    preds.append(x_pred)
                    h_c, c_c = model.lstm(x_t, (h_c, c_c))

                loss = F.mse_loss(torch.cat(preds, 0), chunk)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
                opt.step()
                total += loss.item(); count += 1
                h, c = h_c.detach(), c_c.detach()

        hist.append(total / count)
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1:2d}/{EPOCHS}  loss={hist[-1]:.5f}")
    print("Done.\n")
    return model, hist


def train_transformer() -> tuple:
    model = TransformerPredictor()
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    hist  = []

    print(f"Training Transformer  ({EPOCHS} epochs, full-sequence causal) ...")
    for epoch in range(EPOCHS):
        total, count = 0.0, 0
        for fname in TRAIN_FILES:
            feats, _, _ = load_features(fname, CLIP_FRAMES)
            preds = model(feats)
            loss  = F.mse_loss(preds, feats)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            opt.step()
            total += loss.item(); count += 1
        hist.append(total / count)
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1:2d}/{EPOCHS}  loss={hist[-1]:.5f}")
    print("Done.\n")
    return model, hist


# ---------------------------------------------------------------------------
# Inference (copy weights to full-mode cell for DREAM)
# ---------------------------------------------------------------------------

def infer_dream(pretrained: DREAMPredictor, feats: torch.Tensor):
    cell = DREAMPredictor(pretrained.config, mode="full")
    with torch.no_grad():
        for attr in ("C", "W", "B", "tau_sys", "ltc_surprise_scale", "eta"):
            getattr(cell, attr).copy_(getattr(pretrained, attr))
    cell.eval()
    return cell.predict_sequence(feats)


def infer_lstm(model: LSTMPredictor, feats: torch.Tensor):
    model.eval()
    return model.predict_sequence(feats)


def infer_tf(model: TransformerPredictor, feats: torch.Tensor):
    model.eval()
    return model.predict_sequence(feats)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def smooth(x, w=5):
    return np.convolve(x, np.ones(w) / w, mode="same")


def plot_metrics(histories: dict, mse_all: dict, params: dict,
                 out="reconstruction_metrics.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"LJSpeech Reconstruction  —  All models ≈ {PARAM_TARGET//1000}K params\n"
        f"Training: {EPOCHS} epochs, 50 files, CLIP={CLIP_FRAMES} frames",
        fontsize=12, fontweight="bold",
    )

    # ── Training loss curves ─────────────────────────────────────────────
    ax = axes[0]
    for key, hist in histories.items():
        ax.plot(range(1, len(hist)+1), smooth(hist, 3),
                color=PALETTE[key], label=LABELS[key], linewidth=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE (mel space)")
    ax.set_title("Training Loss", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25)

    # ── Reconstruction MSE bars ─────────────────────────────────────────
    ax = axes[1]
    x = np.arange(2)
    w = 0.25
    for i, key in enumerate(("dream", "lstm", "tf")):
        vals = [mse_all[key]["seen"], mse_all[key]["unseen"]]
        bars = ax.bar(x + (i - 1) * w, vals, w,
                      color=PALETTE[key], edgecolor="white", linewidth=0.5,
                      label=f"{LABELS[key]}  ({params[key]:,}p)")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                    f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["Seen\n(LJ001-0010)", "Unseen\n(LJ001-0055)"], fontsize=10)
    ax.set_ylabel("MSE (normalised mel space)")
    ax.set_title("Reconstruction MSE  —  Seen vs Unseen", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7.5, loc="upper left"); ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {out}")


def plot_spectrograms(feats_orig, preds_dict, condition, out):
    T      = feats_orig.shape[0]
    panels = [("Original", feats_orig.cpu().numpy())] + \
             [(LABELS[k], p.cpu().numpy()) for k, p in preds_dict.items()]

    vmin = feats_orig.min().item()
    vmax = feats_orig.max().item()

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4),
                             gridspec_kw={"wspace": 0.22})
    fig.suptitle(
        f"Log-Mel Spectrogram  —  {condition}\n"
        "(normalised · 80 mel bands · brighter = higher energy)",
        fontsize=11, fontweight="bold",
    )
    for ax, (title, data) in zip(axes, panels):
        im = ax.imshow(data.T, origin="lower", aspect="auto",
                       extent=[0, T / FPS, 0, N_MELS],
                       vmin=vmin, vmax=vmax, cmap="magma")
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("Mel band", fontsize=8)
        ax.tick_params(labelsize=7)
    plt.colorbar(im, ax=axes[-1], fraction=0.035, pad=0.03,
                 label="Normalised log energy")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {out}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(mse_all: dict, params: dict):
    print(f"\n{'='*74}")
    print(f"  RECONSTRUCTION EXPERIMENT — REPORT")
    print(f"{'='*74}")
    print(f"\n  {'Model':<38} {'Params':>8}  {'Seen MSE':>10}  {'Unseen MSE':>11}")
    print(f"  {'-'*70}")
    for key in ("dream", "lstm", "tf"):
        print(f"  {LABELS[key]:<38} {params[key]:>8,}  "
              f"{mse_all[key]['seen']:>10.4f}  {mse_all[key]['unseen']:>11.4f}")

    print(f"\n  ── Seen→Unseen degradation ──")
    for key in ("dream", "lstm", "tf"):
        s, u = mse_all[key]["seen"], mse_all[key]["unseen"]
        print(f"  {key.upper():<10}: seen={s:.4f}  unseen={u:.4f}  Δ={u-s:+.4f} ({(u-s)/s*100:+.1f}%)")

    print(f"\n  ── DREAM advantage on unseen vs other models ──")
    dream_u = mse_all["dream"]["unseen"]
    for key in ("lstm", "tf"):
        other_u = mse_all[key]["unseen"]
        sign = "lower" if dream_u < other_u else "higher"
        pct  = abs(dream_u - other_u) / other_u * 100
        print(f"  DREAM vs {key.upper():<3}: {pct:.1f}% {sign} MSE on unseen")

    print(f"\n  DREAM advantage: fast weights adapt online during the test clip.")
    print(f"  LSTM/TF: weights frozen after training, no in-context adaptation.")
    print(f"{'='*74}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    # ── Instantiate once to check params ─────────────────────────────────
    config = DREAMConfig(
        input_dim=N_MELS, hidden_dim=DREAM_HIDDEN, rank=8,
        base_threshold=0.35, base_plasticity=0.4,
        forgetting_rate=0.03, adaptive_forgetting_scale=8.0,
        ltc_tau_sys=5.0, ltc_surprise_scale=8.0,
        surprise_temperature=0.12, entropy_influence=0.2,
        time_step=0.1, sleep_rate=0.005, min_surprise_for_sleep=0.25,
    )
    _dream_chk = DREAMPredictor(config)
    _lstm_chk  = LSTMPredictor()
    _tf_chk    = TransformerPredictor()

    params = check_params(_dream_chk, _lstm_chk, _tf_chk)
    del _dream_chk, _lstm_chk, _tf_chk

    print(f"Dataset : {DATA_DIR}")
    print(f"Train   : {len(TRAIN_FILES)} files  (LJ001-0001 … LJ001-0050)")
    print(f"Seen    : {FILE_SEEN}   Unseen: {FILE_UNSEEN}\n")

    # ── Train (or load if weights already saved) ──────────────────────────
    dream_path = os.path.join(RESULTS_DIR, "weights_dream.pt")
    lstm_path  = os.path.join(RESULTS_DIR, "weights_lstm.pt")
    tf_path    = os.path.join(RESULTS_DIR, "weights_tf.pt")

    if os.path.exists(dream_path):
        dream_model = DREAMPredictor(config, mode="static")
        dream_model.load_state_dict(torch.load(dream_path, weights_only=True))
        dream_hist = []
        print(f"Loaded DREAM weights from {dream_path}\n")
    else:
        dream_model, dream_hist = train_dream(config)
        torch.save(dream_model.state_dict(), dream_path)
        print(f"Saved DREAM weights → {dream_path}\n")

    if os.path.exists(lstm_path):
        lstm_model = LSTMPredictor()
        lstm_model.load_state_dict(torch.load(lstm_path, weights_only=True))
        lstm_hist = []
        print(f"Loaded LSTM weights from {lstm_path}\n")
    else:
        lstm_model, lstm_hist = train_lstm()
        torch.save(lstm_model.state_dict(), lstm_path)
        print(f"Saved LSTM weights → {lstm_path}\n")

    if os.path.exists(tf_path):
        tf_model = TransformerPredictor()
        tf_model.load_state_dict(torch.load(tf_path, weights_only=True))
        tf_hist = []
        print(f"Loaded Transformer weights from {tf_path}\n")
    else:
        tf_model, tf_hist = train_transformer()
        torch.save(tf_model.state_dict(), tf_path)
        print(f"Saved Transformer weights → {tf_path}\n")

    # ── Load test clips ───────────────────────────────────────────────────
    print("Loading test clips ...")
    feats_seen,   mu_s, std_s = load_features(FILE_SEEN)
    feats_unseen, mu_u, std_u = load_features(FILE_UNSEEN)
    print(f"  Seen   : {feats_seen.shape[0]} frames")
    print(f"  Unseen : {feats_unseen.shape[0]} frames\n")

    # ── Inference ─────────────────────────────────────────────────────────
    print("Running inference ...")
    results = {}
    for key, fn, model in [
        ("dream", infer_dream, dream_model),
        ("lstm",  infer_lstm,  lstm_model),
        ("tf",    infer_tf,    tf_model),
    ]:
        results[key] = {
            "seen":   fn(model, feats_seen),
            "unseen": fn(model, feats_unseen),
        }

    # ── MSE ───────────────────────────────────────────────────────────────
    mse_all = {
        key: {
            "seen":   F.mse_loss(results[key]["seen"][0],   feats_seen).item(),
            "unseen": F.mse_loss(results[key]["unseen"][0], feats_unseen).item(),
        }
        for key in ("dream", "lstm", "tf")
    }

    print_report(mse_all, params)

    # ── Plots ─────────────────────────────────────────────────────────────
    plot_metrics(
        {"dream": dream_hist, "lstm": lstm_hist, "tf": tf_hist},
        mse_all, params,
    )
    for condition, feats in [("seen", feats_seen), ("unseen", feats_unseen)]:
        plot_spectrograms(
            feats,
            {k: results[k][condition][0] for k in ("dream", "lstm", "tf")},
            condition=f"{'Seen' if condition == 'seen' else 'Unseen'} "
                      f"({'LJ001-0010' if condition == 'seen' else 'LJ001-0055'})",
            out=f"reconstruction_spectrograms_{condition}.png",
        )

    # ── Audio (Griffin-Lim, n_iter=128) ───────────────────────────────────
    print("Reconstructing audio  (n_iter=128, this takes ~1 min) ...")
    for condition, feats, mu, std in [
        ("seen",   feats_seen,   mu_s, std_s),
        ("unseen", feats_unseen, mu_u, std_u),
    ]:
        # oracle: actual mel → audio  (Griffin-Lim quality ceiling)
        wav = mel_to_wav(feats, mu, std)
        path = os.path.join(RESULTS_DIR, f"reconstruction_original_{condition}.wav")
        torchaudio.save(path, wav, SR); print(f"  {path}")

        for key in ("dream", "lstm", "tf"):
            pred_mel = results[key][condition][0]
            try:
                wav  = mel_to_wav(pred_mel, mu, std)
                path = os.path.join(RESULTS_DIR, f"reconstruction_{key}_{condition}.wav")
                torchaudio.save(path, wav, SR); print(f"  {path}")
            except Exception as e:
                print(f"  WARN {key}/{condition}: {e}")

    print("\nDone.")
