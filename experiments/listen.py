"""
Audio reconstruction — loads saved weights, outputs WAV files.
No training. Run reconstruction.py + dream_full_train.py first.

Outputs in results/:
  listen_original_seen.wav
  listen_original_unseen.wav
  listen_dream_detach_seen.wav      (proven_v1, broken)
  listen_dream_bptt_seen.wav        (full BPTT, new)
  listen_lstm_seen.wav
  listen_tf_seen.wav
  ... same for unseen
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

SR           = 16_000
N_MELS       = 80
N_FFT        = 1024
HOP          = 160
WIN          = 400
N_STFT       = N_FFT // 2 + 1
DREAM_HIDDEN = 833
LSTM_HIDDEN  = 178
TF_D         = 100; TF_FF = 256; TF_NHEAD = 4; TF_NLAYERS = 2
DATA_DIR     = "data/ljspeech"
RESULTS_DIR  = "results"
FILE_SEEN    = "LJ001-0010.wav"
FILE_UNSEEN  = "LJ001-0055.wav"


# ---------------------------------------------------------------------------
# Audio
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

def build_mel(wav):
    logm = torch.log(mel_transform()(wav).squeeze(0).T + 1e-6)
    mu   = logm.mean(0, keepdim=True)
    std  = logm.std(0, keepdim=True).clamp(min=1e-4)
    return (logm - mu) / std, mu, std

def load_features(fname):
    wav = load_wav(os.path.join(DATA_DIR, fname))
    return build_mel(wav)

def mel_to_wav(norm_mel, mu, std, n_iter=128):
    logm      = norm_mel.detach().cpu() * std + mu
    mel_power = (torch.exp(logm) - 1e-6).clamp(min=1e-10)
    inv_mel   = T.InverseMelScale(n_stft=N_STFT, n_mels=N_MELS, sample_rate=SR,
                                   f_min=20.0, f_max=8000.0)
    linear    = inv_mel(mel_power.T)
    gl        = T.GriffinLim(n_fft=N_FFT, win_length=WIN, hop_length=HOP,
                              power=2.0, n_iter=n_iter)
    return gl(linear).unsqueeze(0)

def save_wav(path, wav):
    torchaudio.save(path, wav, SR)
    print(f"  → {path}")


# ---------------------------------------------------------------------------
# Models
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
        state.error_mean   = (1 - 0.05) * state.error_mean   + 0.05 * error.detach()
        state.error_var    = (1 - 0.05) * state.error_var    + 0.05 * (error.detach() - state.error_mean) ** 2
        state.avg_surprise = (1 - self.beta_s) * state.avg_surprise + self.beta_s * surprise.detach()
        state.h = h_new
        return x_pred, h_new, state

    def predict_sequence(self, feats):
        preds = []
        state = self.init_state(1)
        with torch.no_grad():
            for t in range(feats.shape[0]):
                x_pred, _, state = self.forward_step(feats[t].unsqueeze(0), state)
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
        T    = feats.shape[0]
        x_in = torch.cat([torch.zeros(1, N_MELS), feats[:-1]], dim=0)
        h    = self.embed(x_in.unsqueeze(0))
        mask = nn.Transformer.generate_square_subsequent_mask(T)
        with torch.no_grad():
            out = self.encoder(h, mask=mask, is_causal=True)
        return self.readout(out.squeeze(0)).detach()


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
        "dream_detach": os.path.join(RESULTS_DIR, "weights_dream.pt"),
        "dream_bptt":   os.path.join(RESULTS_DIR, "weights_dream_full.pt"),
        "lstm":         os.path.join(RESULTS_DIR, "weights_lstm.pt"),
        "tf":           os.path.join(RESULTS_DIR, "weights_tf.pt"),
    }
    for k, p in w.items():
        assert os.path.exists(p), f"Missing: {p}"

    dream_detach = DREAMPredictor(config, mode="full")
    dream_detach.load_state_dict(torch.load(w["dream_detach"], weights_only=True))
    dream_detach.eval()

    dream_bptt = DREAMPredictor(config, mode="full")
    dream_bptt.load_state_dict(torch.load(w["dream_bptt"], weights_only=True))
    dream_bptt.eval()

    lstm = LSTMPredictor()
    lstm.load_state_dict(torch.load(w["lstm"], weights_only=True))
    lstm.eval()

    tf = TransformerPredictor()
    tf.load_state_dict(torch.load(w["tf"], weights_only=True))
    tf.eval()

    print("All weights loaded.\n")

    # ── Reconstruct ───────────────────────────────────────────────────────
    print("Reconstructing audio (n_iter=128) ...")
    for fname, tag in [(FILE_SEEN, "seen"), (FILE_UNSEEN, "unseen")]:
        feats, mu, std = load_features(fname)
        print(f"\n  [{tag}]  {fname}  ({feats.shape[0]} frames)")

        # original
        save_wav(os.path.join(RESULTS_DIR, f"listen_original_{tag}.wav"),
                 mel_to_wav(feats, mu, std))

        models = [
            ("dream_detach",  dream_detach),
            ("dream_bptt",    dream_bptt),
            ("lstm",          lstm),
            ("tf",            tf),
        ]
        for name, model in models:
            preds = model.predict_sequence(feats)
            mse   = F.mse_loss(preds, feats).item()
            wav   = mel_to_wav(preds, mu, std)
            save_wav(os.path.join(RESULTS_DIR, f"listen_{name}_{tag}.wav"), wav)
            print(f"     MSE={mse:.4f}")

    print("\nDone. Open results/listen_*.wav and compare.")
