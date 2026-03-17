"""
Acoustic Model — DREAM, phoneme CE with TextGrid alignment
===========================================================
Goal: DREAM acoustic model outputting P(phoneme | frame).
Next step: DREAM language model on top.

Dataset: data/dataset/ — 15 files with exact TextGrid phoneme boundaries.
Phonemes: vocab.txt (Arpabet with stress + <sil>, 72 classes total).
Training: CrossEntropy on frame-level labels from TextGrid.
          Gradient accumulation — 1 optimizer step per epoch.
Metric: PER (silence/blank/unk excluded from evaluation).
"""

import os, sys, re, warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as Tr
import numpy as np

warnings.filterwarnings("ignore", message="enable_nested_tensor")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dream_net import DREAMConfig, DREAMCell


# ---------------------------------------------------------------------------
# Phoneme vocabulary — loaded from vocab.txt
# ---------------------------------------------------------------------------

DATASET_DIR = "/content/dream-net/data/dataset"
VOCAB_PATH  = os.path.join(DATASET_DIR, "vocab.txt")
METADATA    = os.path.join(DATASET_DIR, "metadata.csv")
AUDIO_DIR   = os.path.join(DATASET_DIR, "audio")
TG_DIR      = os.path.join(DATASET_DIR, "textgrid")
RESULTS     = "results"
os.makedirs(RESULTS, exist_ok=True)

with open(VOCAB_PATH) as f:
    VOCAB = [line.strip() for line in f if line.strip()]

N_CLS  = len(VOCAB)                          # 72
PH2IDX = {p: i for i, p in enumerate(VOCAB)}
IDX2PH = {i: p for p, i in PH2IDX.items()}

SIL_IDX   = PH2IDX.get('<sil>',   2)
BLANK_IDX = PH2IDX.get('<blank>', 0)
UNK_IDX   = PH2IDX.get('<unk>',   1)

# tokens to skip in PER evaluation
_SKIP = {BLANK_IDX, UNK_IDX, SIL_IDX}


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SR      = 16_000
N_MELS  = 80
N_FFT   = 1024
HOP     = 160
WIN     = 400

TRAIN_N         = 100
EPOCHS          = 100
STEPS_PER_EPOCH = 10   # 10 optimizer steps per epoch → 1000 total gradient steps
LR              = 3e-3
CLIP            = 1.0

DREAM_H  = 833   # 1-layer config
DREAM_H2 = 256   # 2-layer config (each layer)


# ---------------------------------------------------------------------------
# TextGrid parser
# ---------------------------------------------------------------------------

def parse_textgrid_phones(path: str) -> list:
    """
    Parse Praat TextGrid → list of (frame_start, frame_end, label) for phones tier.
    Empty intervals → '<sil>'.
    """
    with open(path, encoding='utf-8') as f:
        text = f.read()

    # isolate the phones IntervalTier section
    m = re.search(r'name = "phones"\s+(.*?)(?=\s+item \[|\Z)', text, re.DOTALL)
    if not m:
        return []

    section = m.group(1)
    intervals = re.findall(
        r'xmin\s*=\s*([\d.]+)\s+xmax\s*=\s*([\d.]+)\s+text\s*=\s*"(.*?)"',
        section
    )

    result = []
    for xmin_s, xmax_s, label in intervals:
        xmin = float(xmin_s)
        xmax = float(xmax_s)
        f_start = int(round(xmin * SR / HOP))
        f_end   = int(round(xmax * SR / HOP))
        ph = label.strip() if label.strip() else '<sil>'
        result.append((f_start, f_end, ph))
    return result


def build_frame_labels(T: int, intervals: list) -> torch.Tensor:
    """Convert (frame_start, frame_end, phoneme) list to per-frame label tensor."""
    labels = torch.full((T,), SIL_IDX, dtype=torch.long)
    for f_start, f_end, ph in intervals:
        f_end = min(f_end, T)
        idx   = PH2IDX.get(ph, SIL_IDX)
        if f_start < f_end:
            labels[f_start:f_end] = idx
    return labels


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def load_wav(path):
    wav, sr = torchaudio.load(path)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    return wav.mean(0, keepdim=True) if wav.shape[0] > 1 else wav

_MEL = None
def mel_tf():
    global _MEL
    if _MEL is None:
        _MEL = Tr.MelSpectrogram(sample_rate=SR, n_fft=N_FFT, win_length=WIN,
                                  hop_length=HOP, n_mels=N_MELS,
                                  f_min=20.0, f_max=8000.0, power=2.0)
    return _MEL

def build_mel(wav):
    logm = torch.log(mel_tf()(wav).squeeze(0).T + 1e-6)
    mu   = logm.mean(0, keepdim=True)
    std  = logm.std(0, keepdim=True).clamp(min=1e-4)
    return (logm - mu) / std


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_dataset(n=TRAIN_N):
    meta = {}
    with open(METADATA, encoding='utf-8') as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split('|')
            if len(parts) >= 4:
                meta[parts[0]] = parts[3]   # textgrid_path (relative)

    ds = []
    for fname in sorted(os.listdir(AUDIO_DIR)):
        if len(ds) >= n:
            break
        if not fname.endswith('.wav'):
            continue
        fid    = fname.replace('.wav', '')
        tg_rel = meta.get(fid, '')
        if not tg_rel:
            continue
        tg_path = os.path.join(DATASET_DIR, tg_rel)
        if not os.path.exists(tg_path):
            continue

        feats     = build_mel(load_wav(os.path.join(AUDIO_DIR, fname)))
        T         = feats.shape[0]
        intervals = parse_textgrid_phones(tg_path)
        if not intervals:
            continue
        labels    = build_frame_labels(T, intervals)

        # reference phoneme sequence (no sil/blank/unk) for PER
        ref = [ph for _, _, ph in intervals
               if ph not in ('<sil>', '<blank>', '<unk>', '')]
        if not ref:
            continue

        ds.append((feats, labels, ref))

    return ds


# ---------------------------------------------------------------------------
# PER helpers
# ---------------------------------------------------------------------------

def edit_dist(a, b):
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            tmp = dp[j]
            dp[j] = prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = tmp
    return dp[n]

def per(pred: list, ref: list) -> float:
    return edit_dist(pred, ref) / max(len(ref), 1)

SMOOTH_WIN = 11   # frames to average before argmax (~110ms at 10ms/frame)

def decode_phones(logits: torch.Tensor) -> list:
    """
    Smooth logits over time → argmax → collapse → skip sil/blank/unk.
    Smoothing prevents T-N-T-N flicker within a single phoneme segment.
    """
    if logits.shape[0] > SMOOTH_WIN:
        pad = SMOOTH_WIN // 2
        # avg_pool1d expects (N, C, L) — treat classes as channels
        smooth = F.avg_pool1d(
            logits.T.unsqueeze(0), SMOOTH_WIN, stride=1, padding=pad
        ).squeeze(0).T                               # back to (T, N_CLS)
    else:
        smooth = logits
    idx = smooth.argmax(-1).tolist()
    phones, prev = [], -1
    for i in idx:
        if i != prev and i not in _SKIP:
            phones.append(IDX2PH[i])
        prev = i
    return phones


# ---------------------------------------------------------------------------
# DREAM acoustic model
# ---------------------------------------------------------------------------

class DREAMAcoustic(nn.Module):
    """
    DREAMCell + Linear → P(phoneme | frame).
    Head reads cat(h, base_eff):
      h        = slow context (LTC, speaker/prosody)
      base_eff = B @ x_norm  (fast per-frame features, direct gradient to B)
    """
    def __init__(self, config: DREAMConfig):
        super().__init__()
        self.cell = DREAMCell(config)
        # head reads only base_eff (direct per-frame mel→phoneme projection).
        # h (LTC context) is maintained for future LM use but excluded here:
        # h decays ~2%/step → carries 100s of frames history → hurts frame-level task.
        self.head = nn.Linear(config.hidden_dim * 2, N_CLS)
        self.hdim = config.hidden_dim

    def _step(self, h, x_t):
        cell     = self.cell
        x_scale  = x_t.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        x_norm   = (x_t / x_scale).clamp(-1.0, 1.0)
        error    = x_t - torch.tanh(h.detach() @ cell.C.T) * x_scale
        rel_err  = (error.norm(dim=-1) / x_scale.squeeze(1).detach()).clamp(max=4.0)
        surprise = torch.sigmoid((rel_err - cell.tau_0) / cell.gamma)
        base_eff = (cell.B @ x_norm.T).T
        err_eff  = (cell.W @ error.T).T
        input_h  = h.detach() * 0.2 + base_eff * 0.6 + err_eff * surprise.unsqueeze(1) * 0.2
        h_new    = cell.compute_ltc_update(h, input_h, surprise)
        return h_new, base_eff

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """Single sequence, used for evaluation. feats: (T, N_MELS) → (T, N_CLS)."""
        h      = torch.zeros(1, self.hdim)
        h_list = []
        for t in range(feats.shape[0]):
            h, base_eff = self._step(h, feats[t].unsqueeze(0))
            h_list.append(torch.cat([h, base_eff], dim=-1).squeeze(0))
        return self.head(torch.stack(h_list))

    def forward_files(self, feats_pad: torch.Tensor) -> torch.Tensor:
        """
        Process all files in parallel, full BPTT (no h detach).
        feats_pad : (B, T_max, N_MELS)
        Returns   : (B, T_max, N_CLS)

        LTC Jacobian ≈0.98/step is contractive — no gradient explosion
        even at T_max=750 steps. Same as the sequential approach but B files at once.
        """
        B, T_max, _ = feats_pad.shape
        h = torch.zeros(B, self.hdim)
        h_list = []
        for t in range(T_max):
            h, base_eff = self._step(h, feats_pad[:, t, :])
            h_list.append(torch.cat([h, base_eff], dim=-1))      # (B, 2H)
        return self.head(torch.stack(h_list, dim=1))             # (B, T_max, N_CLS)

    def decode(self, feats: torch.Tensor) -> list:
        with torch.no_grad():
            return decode_phones(self.forward(feats))


# ---------------------------------------------------------------------------
# DREAM acoustic — N stacked layers (general)
# ---------------------------------------------------------------------------

class DREAMAcousticNL(nn.Module):
    """
    N stacked DREAMCells.
      Layer 0 : mel (80)   → h0, base_eff0  (hidden_dim of configs[0])
      Layer i : base_eff_{i-1} → h_i, base_eff_i
      Head    : cat(h_{N-1}, base_eff_{N-1}) → N_CLS
    """
    def __init__(self, configs: list):
        super().__init__()
        self.cells = nn.ModuleList([DREAMCell(cfg) for cfg in configs])
        self.head  = nn.Linear(configs[-1].hidden_dim * 2, N_CLS)
        self.hdims = [cfg.hidden_dim for cfg in configs]

    @staticmethod
    def _step(cell, h, x_t):
        """One DREAMCell step: (h, x_t) → (h_new, base_eff)."""
        x_scale  = x_t.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        x_norm   = (x_t / x_scale).clamp(-1.0, 1.0)
        error    = x_t - torch.tanh(h.detach() @ cell.C.T) * x_scale
        rel_err  = (error.norm(dim=-1) / x_scale.squeeze(1).detach()).clamp(max=4.0)
        surprise = torch.sigmoid((rel_err - cell.tau_0) / cell.gamma)
        base_eff = (cell.B @ x_norm.T).T
        err_eff  = (cell.W @ error.T).T
        input_h  = h.detach() * 0.2 + base_eff * 0.6 + err_eff * surprise.unsqueeze(1) * 0.2
        h_new    = cell.compute_ltc_update(h, input_h, surprise)
        return h_new, base_eff

    def _run(self, get_x_t, T, B):
        """Shared loop for forward / forward_files."""
        hs  = [torch.zeros(B, d) for d in self.hdims]
        out = []
        for t in range(T):
            x = get_x_t(t)
            be = x  # will be overwritten each layer
            for i, (cell, h) in enumerate(zip(self.cells, hs)):
                h_new, be = self._step(cell, h, x)
                hs[i] = h_new
                x = be                         # layer output feeds next layer
            out.append(torch.cat([hs[-1], be], dim=-1))
        return torch.stack(out, dim=1 if B > 1 else 0)   # (B,T,2H) or (T,2H)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """Single file eval. feats: (T, N_MELS) → (T, N_CLS)."""
        T = feats.shape[0]
        stacked = self._run(lambda t: feats[t].unsqueeze(0), T, B=1)
        return self.head(stacked.squeeze(1) if stacked.dim() == 3 else stacked)

    def forward_files(self, feats_pad: torch.Tensor) -> torch.Tensor:
        """Parallel batch. (B, T_max, N_MELS) → (B, T_max, N_CLS)."""
        B, T_max, _ = feats_pad.shape
        stacked = self._run(lambda t: feats_pad[:, t, :], T_max, B=B)
        return self.head(stacked)              # (B, T_max, N_CLS)

    def decode(self, feats: torch.Tensor) -> list:
        with torch.no_grad():
            return decode_phones(self.forward(feats))


# Keep alias for backwards compat with saved weights
DREAMAcoustic2L = DREAMAcousticNL


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(model, dataset, epochs, save_path=None):
    """
    All files processed in parallel each epoch.
    h carries full context across each file; BPTT detached every CHUNK_SIZE steps.
    Updates per epoch = ceil(T_max / CHUNK_SIZE) — much more than 1.
    """
    # build padded batch once
    feats_list  = [d[0] for d in dataset]
    labels_list = [d[1] for d in dataset]
    T_max       = max(f.shape[0] for f in feats_list)
    B           = len(dataset)

    feats_pad  = torch.zeros(B, T_max, N_MELS)
    labels_pad = torch.full((B, T_max), SIL_IDX, dtype=torch.long)
    mask       = torch.zeros(B, T_max, dtype=torch.bool)   # True = real frame

    for i, (f, l) in enumerate(zip(feats_list, labels_list)):
        T = f.shape[0]
        feats_pad[i, :T]  = f
        labels_pad[i, :T] = l
        mask[i, :T]       = True

    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=1e-5)
    n_p   = sum(p.numel() for p in model.parameters())
    best_per = 1.0

    print(f"\nTraining DREAM-acoustic  ({n_p:,} params, {epochs} epochs × {STEPS_PER_EPOCH} steps)")
    print(f"  files={B}  T_max={T_max}  |  {N_CLS} classes\n")

    for ep in range(epochs):
        model.train()
        last_loss = 0.0

        for _ in range(STEPS_PER_EPOCH):
            logits = model.forward_files(feats_pad)   # (B, T_max, N_CLS)

            flat_logits = logits[mask]                # (N_real, N_CLS)
            flat_labels = labels_pad[mask]            # (N_real,)
            loss = F.cross_entropy(flat_logits, flat_labels)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
            last_loss = loss.item()

        sched.step()  # once per epoch

        if (ep + 1) % 5 == 0:
            mean_per = evaluate(model, dataset, quiet=True)
            marker = " ←best" if mean_per < best_per else ""
            print(f"  epoch {ep+1:3d}/{epochs}  "
                  f"ce={last_loss:.4f}  PER={mean_per*100:.1f}%{marker}")
            if mean_per < best_per:
                best_per = mean_per
                if save_path:
                    torch.save(model.state_dict(), save_path)

    if save_path and os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, weights_only=True))
    print(f"\nDone.  best PER={best_per*100:.1f}%\n")
    return best_per


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, dataset, label=None, verbose=False, quiet=False):
    model.eval()
    scores = []
    for feats, _, ref_phones in dataset:
        pred = model.decode(feats)
        scores.append(per(pred, ref_phones))
        if verbose:
            print(f"  REF : {' '.join(ref_phones[:15])} ...")
            print(f"  PRED: {' '.join(pred[:15])} ...")
            print(f"  PER : {scores[-1]:.3f}\n")
    mean = float(np.mean(scores))
    if not quiet:
        tag = label or "DREAM-acoustic"
        print(f"  {tag:<30}  PER = {mean:.4f}  ({mean*100:.1f}%)")
    return mean


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def make_config(input_dim, hidden_dim):
    return DREAMConfig(
        input_dim=input_dim, hidden_dim=hidden_dim, rank=8,
        base_threshold=0.35, base_plasticity=0.4,
        forgetting_rate=0.03, adaptive_forgetting_scale=8.0,
        ltc_tau_sys=5.0, ltc_surprise_scale=8.0,
        surprise_temperature=0.12, entropy_influence=0.2,
        time_step=0.1, sleep_rate=0.005, min_surprise_for_sleep=0.25,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["1L", "2L", "3L", "4L", "both"], default="both",
                        help="1L=single 833, 2L/3L/4L=N×256 layers, both=1L+2L")
    args = parser.parse_args()

    torch.manual_seed(42)

    print("\nLoading dataset ...")
    dataset = load_dataset(TRAIN_N)
    print(f"  {len(dataset)} files loaded")
    if dataset:
        ex = dataset[0]
        print(f"  Example: T={ex[0].shape[0]} frames  "
              f"ref_len={len(ex[2])}  [{' '.join(ex[2][:8])} ...]")

    results = {}

    # ── 1-layer DREAM (hidden=833) ────────────────────────────────────────
    if args.model in ("1L", "both"):
        dream1 = DREAMAcoustic(make_config(N_MELS, DREAM_H))
        path1  = os.path.join(RESULTS, "weights_acoustic_dream_1L.pt")
        print(f"\n{'─'*55}")
        print(f"  1L  DREAM acoustic  params : {sum(p.numel() for p in dream1.parameters()):,}")
        print(f"{'─'*55}")
        if os.path.exists(path1):
            dream1.load_state_dict(torch.load(path1, weights_only=True))
            print(f"  Loaded weights from {path1}")
            per1 = evaluate(dream1, dataset, quiet=True)
        else:
            per1 = train_model(dream1, dataset, EPOCHS, save_path=path1)
        results["1L (h=833)"] = per1

    # ── N-layer DREAM helpers ─────────────────────────────────────────────
    def run_nl(n_layers, tag):
        dims = [N_MELS] + [DREAM_H2] * n_layers
        cfgs = [make_config(dims[i], dims[i+1]) for i in range(n_layers)]
        model = DREAMAcousticNL(cfgs)
        path  = os.path.join(RESULTS, f"weights_acoustic_dream_{tag}.pt")
        n_p   = sum(p.numel() for p in model.parameters())
        layer_ps = [sum(p.numel() for p in c.parameters()) for c in model.cells]
        head_p   = sum(p.numel() for p in model.head.parameters())
        print(f"\n{'─'*55}")
        print(f"  {tag}  DREAM acoustic  params : {n_p:,}")
        print(f"      layers=[{', '.join(str(p) for p in layer_ps)}]  head={head_p:,}")
        print(f"{'─'*55}")
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, weights_only=True))
            print(f"  Loaded weights from {path}")
            p = evaluate(model, dataset, quiet=True)
        else:
            p = train_model(model, dataset, EPOCHS, save_path=path)
        results[tag] = p
        return model

    # ── 2-layer DREAM (two×256) ───────────────────────────────────────────
    if args.model in ("2L", "both"):
        run_nl(2, "2L")

    # ── 3-layer DREAM (three×256) ─────────────────────────────────────────
    if args.model == "3L":
        run_nl(3, "3L")

    # ── 4-layer DREAM (four×256) ──────────────────────────────────────────
    if args.model == "4L":
        run_nl(4, "4L")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  FINAL COMPARISON — {len(dataset)} training files")
    print(f"{'='*55}")
    for name, p in results.items():
        print(f"  {name:<20}  PER = {p*100:.1f}%")

    print("Done.")
