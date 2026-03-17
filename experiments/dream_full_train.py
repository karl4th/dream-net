"""
DREAM Full-Mode Training with BPTT
====================================
Hypothesis: detach-h pretrain (proven_v1) produces a useless C and
uninformative h because gradients never flow through the h trajectory.

Fix: train with fast weights ON + truncated BPTT (chunk=50).
  - Gradients flow: loss → x_pred → h_t → h_{t-1} → ... → C, W, B
  - Loss: mel-space MSE (same objective as LSTM / Transformer)
  - All parameters trained: C, W, B, tau_sys, ltc_surprise_scale, eta

LSTM and Transformer weights are loaded from results/ — not retrained.

After training, runs the same repr_tests diagnostics so results are
directly comparable to the detach-h baseline.
"""

import os, sys, warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import numpy as np

warnings.filterwarnings("ignore", message="enable_nested_tensor")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dream_net import DREAMConfig, DREAMCell, DREAMState


# ---------------------------------------------------------------------------
# Constants  (must match reconstruction.py)
# ---------------------------------------------------------------------------

SR           = 16_000
N_MELS       = 80
N_FFT        = 1024
HOP          = 160
WIN          = 400
N_STFT       = N_FFT // 2 + 1

DREAM_HIDDEN = 833
LSTM_HIDDEN  = 178
TF_D         = 100
TF_FF        = 256
TF_NHEAD     = 4
TF_NLAYERS   = 2

CLIP_FRAMES  = 400
EPOCHS       = 50
LR           = 1e-3          # lower than proven_v1 — BPTT needs stability
CLIP_GRAD    = 0.5           # tighter clip — gradients can explode through h
TBPTT_CHUNK  = 20            # shorter chunks for BPTT stability

DATA_DIR     = "data/ljspeech"
ALL_FILES    = sorted(os.listdir(DATA_DIR))
TRAIN_FILES  = ALL_FILES[:50]
FILE_SEEN    = "LJ001-0010.wav"
FILE_UNSEEN  = "LJ001-0055.wav"

RESULTS_DIR  = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

DREAM_FULL_WEIGHTS = os.path.join(RESULTS_DIR, "weights_dream_full.pt")


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
        _MEL_TF = T.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, win_length=WIN, hop_length=HOP,
            n_mels=N_MELS, f_min=20.0, f_max=8000.0, power=2.0,
        )
    return _MEL_TF

def build_mel(wav):
    logm = torch.log(mel_transform()(wav).squeeze(0).T + 1e-6)
    mu   = logm.mean(0, keepdim=True)
    std  = logm.std(0, keepdim=True).clamp(min=1e-4)
    return (logm - mu) / std, mu, std

def load_features(fname, max_frames=None):
    wav  = load_wav(os.path.join(DATA_DIR, fname))
    feat, mu, std = build_mel(wav)
    if max_frames:
        feat = feat[:max_frames]
    return feat, mu, std


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

class DREAMPredictor(DREAMCell):
    def __init__(self, config, mode="static"):
        super().__init__(config)
        self.mode = mode

    def forward_step(self, x, state):
        x_scale = x.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        x_norm  = (x / x_scale).clamp(-1.0, 1.0)
        x_pred  = torch.tanh(state.h @ self.C.T) * x_scale
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
        state.error_mean   = (1 - a) * state.error_mean   + a * error.detach()
        state.error_var    = (1 - a) * state.error_var    + a * (error.detach() - state.error_mean) ** 2
        state.avg_surprise = (1 - self.beta_s) * state.avg_surprise + self.beta_s * surprise.detach()
        state.h = h_new
        return x_pred, h_new, state, surprise, rel_err

    def predict_sequence(self, feats):
        preds, errs = [], []
        state = self.init_state(1)
        with torch.no_grad():
            for t in range(feats.shape[0]):
                x_t = feats[t].unsqueeze(0)
                x_pred, _, state, _, rel = self.forward_step(x_t, state)
                preds.append(x_pred)
                errs.append(rel.item())
        return torch.cat(preds, 0), np.array(errs)


class LSTMPredictor(nn.Module):
    def __init__(self, input_dim=N_MELS, hidden_dim=LSTM_HIDDEN):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTMCell(input_dim, hidden_dim)
        self.C    = nn.Parameter(torch.empty(input_dim, hidden_dim))
        nn.init.xavier_uniform_(self.C)

    def init_state(self):
        z = torch.zeros(1, self.hidden_dim)
        return z, z.clone()

    def predict_sequence(self, feats):
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


class TransformerPredictor(nn.Module):
    def __init__(self, input_dim=N_MELS, d_model=TF_D, nhead=TF_NHEAD,
                 nlayers=TF_NLAYERS, ff=TF_FF):
        super().__init__()
        self.embed   = nn.Linear(input_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.readout = nn.Linear(d_model, input_dim)

    def forward(self, x_seq):
        T    = x_seq.shape[0]
        x_in = torch.cat([torch.zeros(1, x_seq.shape[1], device=x_seq.device),
                           x_seq[:-1]], dim=0)
        h    = self.embed(x_in.unsqueeze(0))
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x_seq.device)
        out  = self.encoder(h, mask=mask, is_causal=True)
        return self.readout(out.squeeze(0))

    def predict_sequence(self, feats):
        with torch.no_grad():
            preds = self(feats)
        x_scale = feats.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        errs    = ((feats - preds).norm(dim=-1) / x_scale.squeeze(1)).clamp(max=4.0)
        return preds.detach(), errs.cpu().numpy()


# ---------------------------------------------------------------------------
# DREAM Full-Mode BPTT Training
# ---------------------------------------------------------------------------

def bptt_step(cell, h, x_t, use_fast_weights=True):
    """
    Single step with gradient flowing through h.
    U (fast weights) updates happen on detached h to avoid in-place autograd issues.
    Returns (x_pred, h_new) — both differentiable w.r.t. h and parameters.
    """
    x_scale = x_t.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    x_norm  = (x_t / x_scale).clamp(-1.0, 1.0)

    # ── prediction  (gradient flows through h) ────────────────────────────
    x_pred = torch.tanh(h @ cell.C.T) * x_scale

    # fast weight contribution: use detached h for U lookup to avoid
    # in-place modification breaking autograd
    if use_fast_weights:
        with torch.no_grad():
            h_U    = torch.bmm(h.detach().unsqueeze(1), cell._bptt_U).squeeze(1)
        x_pred = x_pred + 0.5 * (h_U @ cell.V.T) * x_scale

    # ── state update — surprise + LTC  (gradient flows through h) ─────────
    error   = (x_t - x_pred).detach()   # detach error for state update (stability)
    rel_err = (error.norm(dim=-1) / x_scale.squeeze(1).detach()).clamp(max=4.0)

    # simplified surprise (use stored stats from last forward pass)
    surprise = torch.sigmoid((rel_err - 0.35) / 0.12)

    base_eff = (cell.B @ x_norm.detach().T).T
    err_eff  = (cell.W @ error.T).T

    # Detach the recurrent self-connection to prevent gradient explosion over BPTT.
    # Gradient still flows through the (1 - dt/tau)*h term inside compute_ltc_update.
    # Jacobian per step drops from ~1.29 to ~0.99 → stable over 50 steps.
    input_h = h.detach() * 0.6 + base_eff * 0.2 + err_eff * surprise.unsqueeze(1) * 0.3

    h_new = cell.compute_ltc_update(h, input_h, surprise)

    # ── fast weight update (side effect, not part of grad graph) ──────────
    if use_fast_weights:
        with torch.no_grad():
            eta     = cell.eta
            S_t     = surprise.unsqueeze(1).unsqueeze(2)
            lam_eff = cell.lambda_ * (1.0 + cell.config.adaptive_forgetting_scale * S_t)
            outer   = torch.bmm(h.detach().unsqueeze(2), error.unsqueeze(1))
            hebbian = outer @ cell.V.unsqueeze(0)
            dU      = -lam_eff * cell._bptt_U + eta * S_t * hebbian
            cell._bptt_U = cell._bptt_U + dU * cell.config.time_step
            # normalise U to prevent divergence (same as update_fast_weights)
            U_norm = cell._bptt_U.norm(dim=(1, 2), keepdim=True)
            cell._bptt_U = cell._bptt_U * (cell.target_norm / (U_norm + 1e-6)).clamp(max=1.5)

    return x_pred, h_new


def train_dream_full(config):
    cell = DREAMPredictor(config, mode="full")
    opt  = torch.optim.Adam(cell.parameters(), lr=LR)
    hist = []

    print(f"Training DREAM full-mode BPTT  ({EPOCHS} epochs, chunk={TBPTT_CHUNK}) ...")
    print(f"  LR={LR}  clip_grad={CLIP_GRAD}  loss=mel-space MSE\n")

    for epoch in range(EPOCHS):
        total, count = 0.0, 0

        for fname in TRAIN_FILES:
            feats, _, _ = load_features(fname, CLIP_FRAMES)
            T = feats.shape[0]

            # initialise h and U for this file
            h = torch.zeros(1, DREAM_HIDDEN)
            cell._bptt_U = torch.zeros(1, DREAM_HIDDEN, config.rank)

            for start in range(0, T, TBPTT_CHUNK):
                chunk = feats[start: start + TBPTT_CHUNK]
                h     = h.detach()           # detach at chunk boundary
                preds = []

                for t in range(chunk.shape[0]):
                    x_t    = chunk[t].unsqueeze(0)
                    x_pred, h = bptt_step(cell, h, x_t, use_fast_weights=True)
                    preds.append(x_pred)

                loss = F.mse_loss(torch.cat(preds, 0), chunk)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(cell.parameters(), CLIP_GRAD)
                opt.step()

                total += loss.item()
                count += 1

        hist.append(total / count)
        if (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch+1:2d}/{EPOCHS}  loss={hist[-1]:.5f}")

    print("\nDone.\n")
    return cell, hist


# ---------------------------------------------------------------------------
# Linear Probe diagnostic  (copy from repr_tests.py)
# ---------------------------------------------------------------------------

class LinearProbe(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
    def forward(self, h):
        return self.linear(h)


def collect_h_states(dream, train_files):
    dream.eval()
    h_list, mel_list = [], []
    print(f"  Collecting h states ({len(train_files)} files) ...")
    with torch.no_grad():
        dream._bptt_U = torch.zeros(1, DREAM_HIDDEN, dream.config.rank)
        for fname in train_files:
            feats, _, _ = load_features(fname, CLIP_FRAMES)
            h = torch.zeros(1, DREAM_HIDDEN)
            dream._bptt_U = torch.zeros(1, DREAM_HIDDEN, dream.config.rank)
            for t in range(feats.shape[0]):
                x_t = feats[t].unsqueeze(0)
                h_list.append(h.squeeze(0).clone())
                mel_list.append(feats[t].clone())
                _, h = bptt_step(dream, h, x_t, use_fast_weights=True)
    H   = torch.stack(h_list)
    MEL = torch.stack(mel_list)
    print(f"  {H.shape[0]:,} pairs  h:{H.shape[1]}")
    return H, MEL


def train_probe(H, MEL, epochs=100, lr=1e-3):
    probe = LinearProbe(H.shape[1], MEL.shape[1])
    opt   = torch.optim.Adam(probe.parameters(), lr=lr)
    print(f"  Training probe ({epochs} epochs) ...")
    for ep in range(epochs):
        preds = probe(H.detach())
        loss  = F.mse_loss(preds, MEL.detach())
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 25 == 0:
            print(f"    epoch {ep+1:3d}/{epochs}  loss={loss.item():.5f}")
    print("  Done.\n")
    return probe


def eval_probe_on_clip(dream, probe, feats):
    dream.eval()
    h_list = []
    h = torch.zeros(1, DREAM_HIDDEN)
    dream._bptt_U = torch.zeros(1, DREAM_HIDDEN, dream.config.rank)
    with torch.no_grad():
        for t in range(feats.shape[0]):
            x_t = feats[t].unsqueeze(0)
            h_list.append(h.squeeze(0).clone())
            _, h = bptt_step(dream, h, x_t, use_fast_weights=True)
        H     = torch.stack(h_list)
        preds = probe(H)
    return preds.detach()


def static_inference(dream, feats):
    """DREAM with fast weights OFF — base C only."""
    dream.eval()
    preds = []
    h = torch.zeros(1, DREAM_HIDDEN)
    with torch.no_grad():
        for t in range(feats.shape[0]):
            x_t     = feats[t].unsqueeze(0)
            x_scale = x_t.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            x_pred  = torch.tanh(h @ dream.C.T) * x_scale
            preds.append(x_pred)
            error   = (x_t - x_pred)
            x_norm  = (x_t / x_scale).clamp(-1, 1)
            rel_err = (error.norm(dim=-1) / x_scale.squeeze(1)).clamp(max=4.0)
            surprise = torch.sigmoid((rel_err - 0.35) / 0.12)
            base_eff = (dream.B @ x_norm.T).T
            err_eff  = (dream.W @ error.T).T
            h = dream.compute_ltc_update(
                h,
                h * 0.6 + base_eff * 0.2 + err_eff * surprise.unsqueeze(1) * 0.3,
                surprise,
            )
    return torch.cat(preds, 0)


def full_inference(dream, feats):
    """DREAM with fast weights ON."""
    dream.eval()
    preds = []
    h = torch.zeros(1, DREAM_HIDDEN)
    dream._bptt_U = torch.zeros(1, DREAM_HIDDEN, dream.config.rank)
    with torch.no_grad():
        for t in range(feats.shape[0]):
            x_t = feats[t].unsqueeze(0)
            x_pred, h = bptt_step(dream, h, x_t, use_fast_weights=True)
            preds.append(x_pred)
    return torch.cat(preds, 0)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def verdict_probe(probe_mse, lstm_mse):
    r = probe_mse / lstm_mse
    if r < 1.5:   return f"GOOD  ({r:.1f}x LSTM — h linearly encodes mel)"
    elif r < 5.0: return f"WEAK  ({r:.1f}x LSTM — partial mel info in h)"
    else:         return f"FAIL  ({r:.1f}x LSTM — h does not encode mel)"

def verdict_static(static_mse, lstm_mse, full_mse):
    r = static_mse / lstm_mse
    g = static_mse / full_mse if full_mse > 0 else float("inf")
    if r < 2.0:   return f"GOOD  ({r:.1f}x LSTM — base C competitive)"
    elif r < 10:  return f"WEAK  ({r:.1f}x LSTM — adaptation helps {g:.1f}x but base weak)"
    else:         return f"FAIL  ({r:.1f}x LSTM — C is bad, adaptation starts from garbage)"


def print_report(t1, t2, label="FULL-MODE BPTT"):
    W = 74
    print(f"\n{'='*W}")
    print(f"  REPRESENTATION QUALITY — DREAM {label}")
    print(f"{'='*W}")

    print(f"\n  TEST 1 — Linear Probe on h")
    print(f"  {'─'*68}")
    print(f"\n  {'Condition':<12} {'Model':<28} {'MSE':>8}")
    print(f"  {'─'*50}")
    for cond in ("seen", "unseen"):
        print(f"  {cond:<12} {'DREAM probe on h':<28} {t1['probe'][cond]:>8.4f}")
        print(f"  {cond:<12} {'LSTM (baseline)':<28} {t1['lstm'][cond]:>8.4f}")
        print(f"  {cond:<12} {'Transformer (baseline)':<28} {t1['tf'][cond]:>8.4f}")
        print()
    print(f"  Verdict (seen):   {verdict_probe(t1['probe']['seen'],   t1['lstm']['seen'])}")
    print(f"  Verdict (unseen): {verdict_probe(t1['probe']['unseen'], t1['lstm']['unseen'])}")

    print(f"\n  TEST 2 — Static vs Full mode")
    print(f"  {'─'*68}")
    print(f"\n  {'Condition':<12} {'Mode':<28} {'MSE':>10}")
    print(f"  {'─'*52}")
    for cond in ("seen", "unseen"):
        print(f"  {cond:<12} {'DREAM static':<28} {t2['static'][cond]:>10.4f}")
        print(f"  {cond:<12} {'DREAM full':<28} {t2['full'][cond]:>10.4f}")
        print(f"  {cond:<12} {'LSTM':<28} {t2['lstm'][cond]:>10.4f}")
        print(f"  {cond:<12} {'Transformer':<28} {t2['tf'][cond]:>10.4f}")
        print()
    print(f"  Verdict (seen):   {verdict_static(t2['static']['seen'],   t2['lstm']['seen'],   t2['full']['seen'])}")
    print(f"  Verdict (unseen): {verdict_static(t2['static']['unseen'], t2['lstm']['unseen'], t2['full']['unseen'])}")
    print(f"\n{'='*W}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    config = DREAMConfig(
        input_dim=N_MELS, hidden_dim=DREAM_HIDDEN, rank=8,
        base_threshold=0.35, base_plasticity=0.4,
        forgetting_rate=0.03, adaptive_forgetting_scale=8.0,
        ltc_tau_sys=5.0, ltc_surprise_scale=8.0,
        surprise_temperature=0.12, entropy_influence=0.2,
        time_step=0.1, sleep_rate=0.005, min_surprise_for_sleep=0.25,
    )

    # ── Load LSTM / TF (no retraining) ────────────────────────────────────
    lstm_path = os.path.join(RESULTS_DIR, "weights_lstm.pt")
    tf_path   = os.path.join(RESULTS_DIR, "weights_tf.pt")

    assert os.path.exists(lstm_path), f"Not found: {lstm_path} — run reconstruction.py first"
    assert os.path.exists(tf_path),   f"Not found: {tf_path}  — run reconstruction.py first"

    lstm = LSTMPredictor()
    lstm.load_state_dict(torch.load(lstm_path, weights_only=True))
    lstm.eval()
    print(f"Loaded LSTM  → {lstm_path}")

    tf = TransformerPredictor()
    tf.load_state_dict(torch.load(tf_path, weights_only=True))
    tf.eval()
    print(f"Loaded TF    → {tf_path}\n")

    # ── Train DREAM full-mode BPTT (or load if exists) ────────────────────
    if os.path.exists(DREAM_FULL_WEIGHTS):
        dream = DREAMPredictor(config, mode="full")
        dream.load_state_dict(torch.load(DREAM_FULL_WEIGHTS, weights_only=True))
        dream._bptt_U = torch.zeros(1, DREAM_HIDDEN, config.rank)
        print(f"Loaded DREAM full → {DREAM_FULL_WEIGHTS}\n")
    else:
        dream, hist = train_dream_full(config)
        torch.save(dream.state_dict(), DREAM_FULL_WEIGHTS)
        print(f"Saved → {DREAM_FULL_WEIGHTS}\n")

    # ── Test clips ────────────────────────────────────────────────────────
    print("Loading test clips ...")
    feats_seen,   _, _ = load_features(FILE_SEEN)
    feats_unseen, _, _ = load_features(FILE_UNSEEN)
    print(f"  Seen   : {feats_seen.shape[0]} frames  ({FILE_SEEN})")
    print(f"  Unseen : {feats_unseen.shape[0]} frames  ({FILE_UNSEEN})\n")

    # ── Baselines (LSTM / TF) ─────────────────────────────────────────────
    lstm_seen,  _ = lstm.predict_sequence(feats_seen)
    lstm_unseen,_ = lstm.predict_sequence(feats_unseen)
    tf_seen,  _   = tf.predict_sequence(feats_seen)
    tf_unseen,_   = tf.predict_sequence(feats_unseen)

    lstm_mse = {
        "seen":   F.mse_loss(lstm_seen,   feats_seen).item(),
        "unseen": F.mse_loss(lstm_unseen, feats_unseen).item(),
    }
    tf_mse = {
        "seen":   F.mse_loss(tf_seen,   feats_seen).item(),
        "unseen": F.mse_loss(tf_unseen, feats_unseen).item(),
    }

    # ── TEST 1: Linear Probe ──────────────────────────────────────────────
    print("=" * 60)
    print("  TEST 1: Linear Probe on h")
    print("=" * 60)
    H_train, MEL_train = collect_h_states(dream, TRAIN_FILES)
    probe = train_probe(H_train, MEL_train, epochs=100, lr=1e-3)

    probe_seen   = eval_probe_on_clip(dream, probe, feats_seen)
    probe_unseen = eval_probe_on_clip(dream, probe, feats_unseen)

    t1 = {
        "probe": {
            "seen":   F.mse_loss(probe_seen,   feats_seen).item(),
            "unseen": F.mse_loss(probe_unseen, feats_unseen).item(),
        },
        "lstm": lstm_mse,
        "tf":   tf_mse,
    }

    # ── TEST 2: Static vs Full ────────────────────────────────────────────
    print("=" * 60)
    print("  TEST 2: DREAM static vs full mode")
    print("=" * 60)
    print("  Running static ...")
    ds = static_inference(dream, feats_seen)
    du = static_inference(dream, feats_unseen)
    print("  Running full ...")
    df_s = full_inference(dream, feats_seen)
    df_u = full_inference(dream, feats_unseen)

    t2 = {
        "static": {
            "seen":   F.mse_loss(ds,   feats_seen).item(),
            "unseen": F.mse_loss(du,   feats_unseen).item(),
        },
        "full": {
            "seen":   F.mse_loss(df_s, feats_seen).item(),
            "unseen": F.mse_loss(df_u, feats_unseen).item(),
        },
        "lstm": lstm_mse,
        "tf":   tf_mse,
    }

    print_report(t1, t2)
