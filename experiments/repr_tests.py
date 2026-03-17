"""
Representation Quality Tests  —  DREAM pre-ASR diagnostics
===========================================================
Run AFTER experiments/reconstruction.py (loads saved weights from results/).

TEST 1 — Linear Probe on h
  Collect (h_t, mel_t) pairs from DREAM (full mode) on training files.
  Train a single Linear(DREAM_HIDDEN → N_MELS) layer on these pairs.
  Measure probe MSE on seen / unseen clips.

  Interpretation:
    probe MSE ≈ LSTM/TF  →  h linearly encodes mel  →  representations useful
    probe MSE >> LSTM/TF  →  h is not linearly informative  →  bad sign for ASR

TEST 2 — DREAM static vs full mode
  Run DREAM inference with fast weights OFF (mode="static").
  Compare to DREAM-full, LSTM, Transformer.

  Interpretation:
    static MSE ≈ LSTM/TF  →  base C is competitive, adaptation is a bonus
    static MSE >> LSTM/TF  →  C is bad, adaptation starts from garbage
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
# Constants  (must match reconstruction.py exactly)
# ---------------------------------------------------------------------------

SR          = 16_000
N_MELS      = 80
N_FFT       = 1024
HOP         = 160
WIN         = 400
N_STFT      = N_FFT // 2 + 1

DREAM_HIDDEN = 833
LSTM_HIDDEN  = 178
TF_D         = 100
TF_FF        = 256
TF_NHEAD     = 4
TF_NLAYERS   = 2

CLIP_FRAMES  = 400
EPOCHS       = 50
LR           = 3e-3
CLIP_GRAD    = 1.0
TBPTT_CHUNK  = 50

DATA_DIR     = "data/ljspeech"
ALL_FILES    = sorted(os.listdir(DATA_DIR))
TRAIN_FILES  = ALL_FILES[:50]
FILE_SEEN    = "LJ001-0010.wav"
FILE_UNSEEN  = "LJ001-0055.wav"

RESULTS_DIR  = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Audio helpers  (identical to reconstruction.py)
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
# Model definitions  (identical to reconstruction.py)
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
# Training fallback  (used only if results/weights_*.pt not found)
# ---------------------------------------------------------------------------

def train_dream(config):
    cell      = DREAMPredictor(config, mode="static")
    trainable = [cell.C, cell.W, cell.B, cell.tau_sys, cell.ltc_surprise_scale]
    opt       = torch.optim.Adam(trainable, lr=LR)
    print(f"Training DREAM ({EPOCHS} epochs) ...")
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
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1:2d}/{EPOCHS}  loss={total/count:.5f}")
    print("Done.\n")
    return cell

def train_lstm():
    model = LSTMPredictor()
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    print(f"Training LSTM ({EPOCHS} epochs) ...")
    for epoch in range(EPOCHS):
        total, count = 0.0, 0
        for fname in TRAIN_FILES:
            feats, _, _ = load_features(fname, CLIP_FRAMES)
            T    = feats.shape[0]
            h, c = model.init_state()
            for start in range(0, T, TBPTT_CHUNK):
                chunk = feats[start: start + TBPTT_CHUNK]
                h_c, c_c = h.detach(), c.detach()
                preds = []
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
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1:2d}/{EPOCHS}  loss={total/count:.5f}")
    print("Done.\n")
    return model

def train_transformer():
    model = TransformerPredictor()
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    print(f"Training Transformer ({EPOCHS} epochs) ...")
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
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1:2d}/{EPOCHS}  loss={total/count:.5f}")
    print("Done.\n")
    return model


def load_or_train(config):
    """Load saved weights; re-train only if not found."""
    dream_path = os.path.join(RESULTS_DIR, "weights_dream.pt")
    lstm_path  = os.path.join(RESULTS_DIR, "weights_lstm.pt")
    tf_path    = os.path.join(RESULTS_DIR, "weights_tf.pt")

    if os.path.exists(dream_path):
        dream = DREAMPredictor(config, mode="static")
        dream.load_state_dict(torch.load(dream_path, weights_only=True))
        print(f"Loaded DREAM  → {dream_path}")
    else:
        print(f"WARN: {dream_path} not found — training from scratch")
        dream = train_dream(config)
        torch.save(dream.state_dict(), dream_path)

    if os.path.exists(lstm_path):
        lstm = LSTMPredictor()
        lstm.load_state_dict(torch.load(lstm_path, weights_only=True))
        print(f"Loaded LSTM   → {lstm_path}")
    else:
        print(f"WARN: {lstm_path} not found — training from scratch")
        lstm = train_lstm()
        torch.save(lstm.state_dict(), lstm_path)

    if os.path.exists(tf_path):
        tf = TransformerPredictor()
        tf.load_state_dict(torch.load(tf_path, weights_only=True))
        print(f"Loaded TF     → {tf_path}")
    else:
        print(f"WARN: {tf_path} not found — training from scratch")
        tf = train_transformer()
        torch.save(tf.state_dict(), tf_path)

    return dream, lstm, tf


# ---------------------------------------------------------------------------
# TEST 1 — Linear Probe on h
# ---------------------------------------------------------------------------

class LinearProbe(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, h):
        return self.linear(h)


def collect_h_states(dream, train_files):
    """Run DREAM (full mode) on training files. Collect (h_t, mel_t) pairs."""
    cell = DREAMPredictor(dream.config, mode="full")
    with torch.no_grad():
        for attr in ("C", "W", "B", "tau_sys", "ltc_surprise_scale", "eta"):
            getattr(cell, attr).copy_(getattr(dream, attr))
    cell.eval()

    h_list, mel_list = [], []
    print(f"  Collecting h states from {len(train_files)} training files ...")
    with torch.no_grad():
        for fname in train_files:
            feats, _, _ = load_features(fname, CLIP_FRAMES)
            state = cell.init_state(1)
            for t in range(feats.shape[0]):
                x_t = feats[t].unsqueeze(0)
                h_list.append(state.h.squeeze(0).clone())
                mel_list.append(feats[t].clone())
                _, _, state, _, _ = cell.forward_step(x_t, state)

    H   = torch.stack(h_list)   # (N, hidden_dim)
    MEL = torch.stack(mel_list) # (N, N_MELS)
    print(f"  Collected {H.shape[0]:,} pairs  h:{H.shape[1]}  mel:{MEL.shape[1]}")
    return H, MEL


def train_probe(H, MEL, epochs=100, lr=1e-3):
    probe = LinearProbe(H.shape[1], MEL.shape[1])
    opt   = torch.optim.Adam(probe.parameters(), lr=lr)
    print(f"  Training linear probe ({epochs} epochs) ...")
    for ep in range(epochs):
        preds = probe(H.detach())
        loss  = F.mse_loss(preds, MEL.detach())
        opt.zero_grad(); loss.backward()
        opt.step()
        if (ep + 1) % 25 == 0:
            print(f"    epoch {ep+1:3d}/{epochs}  loss={loss.item():.5f}")
    print("  Done.\n")
    return probe


def eval_probe(dream, probe, feats):
    """Run DREAM full mode on feats, collect h, apply probe → mel predictions."""
    cell = DREAMPredictor(dream.config, mode="full")
    with torch.no_grad():
        for attr in ("C", "W", "B", "tau_sys", "ltc_surprise_scale", "eta"):
            getattr(cell, attr).copy_(getattr(dream, attr))
    cell.eval()

    h_list = []
    state  = cell.init_state(1)
    with torch.no_grad():
        for t in range(feats.shape[0]):
            x_t = feats[t].unsqueeze(0)
            h_list.append(state.h.squeeze(0).clone())
            _, _, state, _, _ = cell.forward_step(x_t, state)
        H     = torch.stack(h_list)
        preds = probe(H)
    return preds.detach()


# ---------------------------------------------------------------------------
# TEST 2 — DREAM static vs full mode
# ---------------------------------------------------------------------------

def dream_inference(dream, feats, mode):
    cell = DREAMPredictor(dream.config, mode=mode)
    with torch.no_grad():
        for attr in ("C", "W", "B", "tau_sys", "ltc_surprise_scale", "eta"):
            getattr(cell, attr).copy_(getattr(dream, attr))
    cell.eval()
    preds, _ = cell.predict_sequence(feats)
    return preds


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def verdict(probe_mse, lstm_mse):
    ratio = probe_mse / lstm_mse
    if ratio < 1.5:
        return f"GOOD  (probe {ratio:.1f}x LSTM — h linearly encodes mel)"
    elif ratio < 5.0:
        return f"WEAK  (probe {ratio:.1f}x LSTM — partial mel info in h)"
    else:
        return f"FAIL  (probe {ratio:.1f}x LSTM — h does not encode mel)"

def verdict_static(static_mse, lstm_mse, full_mse):
    ratio = static_mse / lstm_mse
    gain  = static_mse / full_mse if full_mse > 0 else float("inf")
    if ratio < 2.0:
        return f"GOOD  (static {ratio:.1f}x LSTM — base C competitive)"
    elif ratio < 10.0:
        return f"WEAK  (static {ratio:.1f}x LSTM — adaptation helps {gain:.1f}x but base is weak)"
    else:
        return f"FAIL  (static {ratio:.1f}x LSTM — C is bad, adaptation starts from garbage)"


def print_report(t1, t2):
    W = 76
    print(f"\n{'='*W}")
    print(f"  REPRESENTATION QUALITY TESTS  —  DREAM pre-ASR diagnostics")
    print(f"{'='*W}")

    print(f"\n  TEST 1 — Linear Probe on h")
    print(f"  {'─'*70}")
    print(f"  Does h contain mel information? (single linear layer on frozen h)")
    print(f"\n  {'Condition':<12} {'Model':<28} {'MSE':>8}")
    print(f"  {'─'*52}")
    for cond in ("seen", "unseen"):
        print(f"  {cond:<12} {'DREAM (probe on h)':<28} {t1['probe'][cond]:>8.4f}")
        print(f"  {cond:<12} {'LSTM (baseline)':<28} {t1['lstm'][cond]:>8.4f}")
        print(f"  {cond:<12} {'Transformer (baseline)':<28} {t1['tf'][cond]:>8.4f}")
        print()
    print(f"  Verdict (seen):   {verdict(t1['probe']['seen'],   t1['lstm']['seen'])}")
    print(f"  Verdict (unseen): {verdict(t1['probe']['unseen'], t1['lstm']['unseen'])}")

    print(f"\n  TEST 2 — DREAM static vs full mode (no fast weights vs fast weights)")
    print(f"  {'─'*70}")
    print(f"  Does the base C matrix produce meaningful predictions?")
    print(f"\n  {'Condition':<12} {'Mode':<28} {'MSE':>10}")
    print(f"  {'─'*54}")
    for cond in ("seen", "unseen"):
        print(f"  {cond:<12} {'DREAM  static (no adapt)':<28} {t2['static'][cond]:>10.4f}")
        print(f"  {cond:<12} {'DREAM  full   (adapt ON)':<28} {t2['full'][cond]:>10.4f}")
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

    print("Loading models ...")
    dream, lstm, tf = load_or_train(config)
    dream.eval(); lstm.eval(); tf.eval()

    print("\nLoading test clips ...")
    feats_seen,   _, _ = load_features(FILE_SEEN)
    feats_unseen, _, _ = load_features(FILE_UNSEEN)
    print(f"  Seen   : {feats_seen.shape[0]} frames  ({FILE_SEEN})")
    print(f"  Unseen : {feats_unseen.shape[0]} frames  ({FILE_UNSEEN})\n")

    # ── TEST 1: Linear Probe ────────────────────────────────────────────────
    print("=" * 60)
    print("  TEST 1: Linear Probe on h")
    print("=" * 60)
    H_train, MEL_train = collect_h_states(dream, TRAIN_FILES)
    probe = train_probe(H_train, MEL_train, epochs=100, lr=1e-3)

    probe_preds_seen   = eval_probe(dream, probe, feats_seen)
    probe_preds_unseen = eval_probe(dream, probe, feats_unseen)

    lstm_preds_seen,  _ = lstm.predict_sequence(feats_seen)
    lstm_preds_unseen,_ = lstm.predict_sequence(feats_unseen)

    tf_preds_seen,  _ = tf.predict_sequence(feats_seen)
    tf_preds_unseen,_ = tf.predict_sequence(feats_unseen)

    t1 = {
        "probe": {
            "seen":   F.mse_loss(probe_preds_seen,   feats_seen).item(),
            "unseen": F.mse_loss(probe_preds_unseen, feats_unseen).item(),
        },
        "lstm": {
            "seen":   F.mse_loss(lstm_preds_seen,   feats_seen).item(),
            "unseen": F.mse_loss(lstm_preds_unseen, feats_unseen).item(),
        },
        "tf": {
            "seen":   F.mse_loss(tf_preds_seen,   feats_seen).item(),
            "unseen": F.mse_loss(tf_preds_unseen, feats_unseen).item(),
        },
    }

    # ── TEST 2: Static vs Full mode ─────────────────────────────────────────
    print("=" * 60)
    print("  TEST 2: DREAM static vs full mode")
    print("=" * 60)
    print("  Running DREAM static ...")
    dream_static_seen   = dream_inference(dream, feats_seen,   mode="static")
    dream_static_unseen = dream_inference(dream, feats_unseen, mode="static")
    print("  Running DREAM full ...")
    dream_full_seen   = dream_inference(dream, feats_seen,   mode="full")
    dream_full_unseen = dream_inference(dream, feats_unseen, mode="full")

    t2 = {
        "static": {
            "seen":   F.mse_loss(dream_static_seen,   feats_seen).item(),
            "unseen": F.mse_loss(dream_static_unseen, feats_unseen).item(),
        },
        "full": {
            "seen":   F.mse_loss(dream_full_seen,   feats_seen).item(),
            "unseen": F.mse_loss(dream_full_unseen, feats_unseen).item(),
        },
        "lstm": t1["lstm"],
        "tf":   t1["tf"],
    }

    print_report(t1, t2)
