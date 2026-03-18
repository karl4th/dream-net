"""
DREAM Acoustic Model — GPU Training Script
===========================================
Usage:
    uv run experiments/asr_train.py
    uv run experiments/asr_train.py --layers 2 --n_files 100 --epochs 200
    uv run experiments/asr_train.py --layers 3 --n_files 500 --epochs 300 --steps 20

Arguments:
    --layers   N   Number of stacked DREAMCells (default: 2)
    --n_files  N   Number of training files (default: 100)
    --epochs   N   Training epochs (default: 100)
    --steps    N   Gradient steps per epoch (default: 10)
    --lr       F   Learning rate (default: 3e-3)
    --device   S   Device: auto / cpu / cuda / cuda:0 (default: auto)
    --save     S   Weight save path (default: auto-named in results/)
    --load     S   Resume from checkpoint (optional)
"""

import os, sys, re, argparse, warnings
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
# Paths — auto-detect, override with DREAM_DATASET_DIR env var
# ---------------------------------------------------------------------------

_ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_DS = os.path.join(_ROOT, "data", "dataset")
DATASET_DIR = os.environ.get("DREAM_DATASET_DIR", _DEFAULT_DS)
VOCAB_PATH  = os.path.join(DATASET_DIR, "vocab.txt")
METADATA    = os.path.join(DATASET_DIR, "metadata.csv")
AUDIO_DIR   = os.path.join(DATASET_DIR, "audio")
TG_DIR      = os.path.join(DATASET_DIR, "textgrid")
RESULTS     = os.path.join(_ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

with open(VOCAB_PATH) as f:
    VOCAB = [line.strip() for line in f if line.strip()]

N_CLS     = len(VOCAB)
PH2IDX    = {p: i for i, p in enumerate(VOCAB)}
IDX2PH    = {i: p for p, i in PH2IDX.items()}
SIL_IDX   = PH2IDX.get('<sil>',   2)
BLANK_IDX = PH2IDX.get('<blank>', 0)
UNK_IDX   = PH2IDX.get('<unk>',   1)
_SKIP     = {BLANK_IDX, UNK_IDX, SIL_IDX}


# ---------------------------------------------------------------------------
# Audio / feature constants
# ---------------------------------------------------------------------------

SR     = 16_000
N_MELS = 80
N_FFT  = 1_024
HOP    = 160
WIN    = 400

DREAM_H2 = 256   # hidden dim per layer in N-layer config


# ---------------------------------------------------------------------------
# TextGrid parser
# ---------------------------------------------------------------------------

def parse_textgrid_phones(path: str) -> list:
    with open(path, encoding='utf-8') as f:
        text = f.read()
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
        f_start = int(round(float(xmin_s) * SR / HOP))
        f_end   = int(round(float(xmax_s) * SR / HOP))
        ph = label.strip() if label.strip() else '<sil>'
        result.append((f_start, f_end, ph))
    return result


def build_frame_labels(T: int, intervals: list) -> torch.Tensor:
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

def load_wav(path: str) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    return wav.mean(0, keepdim=True) if wav.shape[0] > 1 else wav

_MEL = None
def _mel_tf():
    global _MEL
    if _MEL is None:
        _MEL = Tr.MelSpectrogram(sample_rate=SR, n_fft=N_FFT, win_length=WIN,
                                  hop_length=HOP, n_mels=N_MELS,
                                  f_min=20.0, f_max=8000.0, power=2.0)
    return _MEL

def build_mel(wav: torch.Tensor) -> torch.Tensor:
    logm = torch.log(_mel_tf()(wav).squeeze(0).T + 1e-6)
    mu   = logm.mean(0, keepdim=True)
    std  = logm.std(0,  keepdim=True).clamp(min=1e-4)
    return (logm - mu) / std


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def _find_textgrid(fid: str, tg_rel: str) -> str:
    for p in [
        os.path.join(DATASET_DIR, tg_rel) if tg_rel else '',
        tg_rel,
        os.path.join(TG_DIR, os.path.basename(tg_rel)) if tg_rel else '',
        os.path.join(TG_DIR, fid + '.TextGrid'),
    ]:
        if p and os.path.exists(p):
            return p
    return ''


def load_dataset(n: int) -> list:
    meta = {}
    try:
        with open(METADATA, encoding='utf-8') as f:
            next(f)
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 4:
                    meta[parts[0]] = parts[3]
    except FileNotFoundError:
        pass

    ds = []
    no_tg = no_ph = 0
    for fname in sorted(os.listdir(AUDIO_DIR)):
        if len(ds) >= n:
            break
        if not fname.endswith('.wav'):
            continue
        fid     = fname.replace('.wav', '')
        tg_path = _find_textgrid(fid, meta.get(fid, ''))
        if not tg_path:
            no_tg += 1
            continue

        feats     = build_mel(load_wav(os.path.join(AUDIO_DIR, fname)))
        T         = feats.shape[0]
        intervals = parse_textgrid_phones(tg_path)
        if not intervals:
            no_ph += 1
            continue
        labels = build_frame_labels(T, intervals)
        ref    = [ph for _, _, ph in intervals
                  if ph not in ('<sil>', '<blank>', '<unk>', '')]
        if not ref:
            no_ph += 1
            continue
        ds.append((feats, labels, ref))

    if no_tg or no_ph:
        print(f"  [skipped] no_textgrid={no_tg}  no_phonemes={no_ph}")
    return ds


# ---------------------------------------------------------------------------
# PER / decode
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

def per(pred, ref):
    return edit_dist(pred, ref) / max(len(ref), 1)

SMOOTH_WIN = 11

def decode_phones(logits: torch.Tensor) -> list:
    """logits: (T, N_CLS) on any device → list of phoneme strings."""
    logits = logits.cpu()
    if logits.shape[0] > SMOOTH_WIN:
        pad    = SMOOTH_WIN // 2
        smooth = F.avg_pool1d(
            logits.T.unsqueeze(0), SMOOTH_WIN, stride=1, padding=pad
        ).squeeze(0).T
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
# Model — N-layer DREAM acoustic
# ---------------------------------------------------------------------------

class DREAMAcousticNL(nn.Module):
    """N stacked DREAMCells, device-aware."""

    def __init__(self, configs: list):
        super().__init__()
        self.cells = nn.ModuleList([DREAMCell(cfg) for cfg in configs])
        self.head  = nn.Linear(configs[-1].hidden_dim * 2, N_CLS)
        self.hdims = [cfg.hidden_dim for cfg in configs]

    @staticmethod
    def _step(cell, h, x_t):
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

    def _init_h(self, B: int) -> list:
        """Create zero hidden states on the model's device."""
        dev = next(self.parameters()).device
        return [torch.zeros(B, d, device=dev) for d in self.hdims]

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """Single-file eval. feats: (T, N_MELS) → (T, N_CLS)."""
        hs  = self._init_h(1)
        out = []
        for t in range(feats.shape[0]):
            x = feats[t].unsqueeze(0)
            for i, (cell, h) in enumerate(zip(self.cells, hs)):
                h, be = self._step(cell, h, x)
                hs[i] = h
                x = be
            out.append(torch.cat([hs[-1], be], dim=-1).squeeze(0))
        return self.head(torch.stack(out))

    def forward_files(self, feats_pad: torch.Tensor) -> torch.Tensor:
        """Batched training. (B, T_max, N_MELS) → (B, T_max, N_CLS)."""
        B, T_max, _ = feats_pad.shape
        hs  = self._init_h(B)
        out = []
        for t in range(T_max):
            x = feats_pad[:, t, :]
            for i, (cell, h) in enumerate(zip(self.cells, hs)):
                h, be = self._step(cell, h, x)
                hs[i] = h
                x = be
            out.append(torch.cat([hs[-1], be], dim=-1))
        return self.head(torch.stack(out, dim=1))

    def decode(self, feats: torch.Tensor) -> list:
        with torch.no_grad():
            return decode_phones(self.forward(feats))


# ---------------------------------------------------------------------------
# Config factory
# ---------------------------------------------------------------------------

def make_config(input_dim: int, hidden_dim: int) -> DREAMConfig:
    return DREAMConfig(
        input_dim=input_dim, hidden_dim=hidden_dim, rank=8,
        base_threshold=0.35, base_plasticity=0.4,
        forgetting_rate=0.03, adaptive_forgetting_scale=8.0,
        ltc_tau_sys=5.0, ltc_surprise_scale=8.0,
        surprise_temperature=0.12, entropy_influence=0.2,
        time_step=0.1, sleep_rate=0.005, min_surprise_for_sleep=0.25,
    )


def build_model(n_layers: int, H2: int = DREAM_H2) -> DREAMAcousticNL:
    dims = [N_MELS] + [H2] * n_layers
    cfgs = [make_config(dims[i], dims[i + 1]) for i in range(n_layers)]
    return DREAMAcousticNL(cfgs)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(model: DREAMAcousticNL, dataset: list, epochs: int,
          steps_per_epoch: int, lr: float, device: torch.device,
          save_path: str, max_frames: int = 0) -> float:

    if not dataset:
        print("  [error] dataset is empty — check paths")
        return 1.0

    # build padded batch on GPU (full sequences, used for random crop each step)
    feats_list  = [d[0] for d in dataset]
    labels_list = [d[1] for d in dataset]
    T_full = max(f.shape[0] for f in feats_list)
    B      = len(dataset)

    feats_full  = torch.zeros(B, T_full, N_MELS)
    labels_full = torch.full((B, T_full), SIL_IDX, dtype=torch.long)
    lengths     = torch.zeros(B, dtype=torch.long)
    for i, (f, l) in enumerate(zip(feats_list, labels_list)):
        T = f.shape[0]
        feats_full[i, :T]  = f
        labels_full[i, :T] = l
        lengths[i]         = T

    feats_full  = feats_full.to(device)
    labels_full = labels_full.to(device)

    # crop window per step: limits Python loop length → higher GPU-Util
    crop = max_frames if max_frames > 0 else T_full
    crop = min(crop, T_full)

    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=1e-5)
    n_p   = sum(p.numel() for p in model.parameters())
    best_per = 1.0

    crop_info = f"crop={crop}" if max_frames > 0 else f"T_max={T_full}"
    print(f"\nTraining {n_p:,} params | {B} files | T_full={T_full} | {crop_info} | "
          f"{epochs} epochs × {steps_per_epoch} steps | device={device}")
    print(f"{'─'*60}\n")

    for ep in range(epochs):
        model.train()
        last_loss = 0.0

        for _ in range(steps_per_epoch):
            # random crop: each step sees a different window → better coverage
            if max_frames > 0 and T_full > crop:
                t0 = torch.randint(0, T_full - crop + 1, (1,)).item()
            else:
                t0 = 0
            t1 = t0 + crop

            feats_crop  = feats_full[:, t0:t1, :]    # (B, crop, N_MELS)
            labels_crop = labels_full[:, t0:t1]       # (B, crop)

            # mask: frames that actually belong to each file in this window
            mask = torch.zeros(B, crop, dtype=torch.bool, device=device)
            for i in range(B):
                valid_end = min(int(lengths[i].item()) - t0, crop)
                if valid_end > 0:
                    mask[i, :valid_end] = True

            logits      = model.forward_files(feats_crop)    # (B, crop, N_CLS)
            flat_logits = logits[mask]
            flat_labels = labels_crop[mask]
            loss        = F.cross_entropy(flat_logits, flat_labels)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            last_loss = loss.item()

        sched.step()

        if (ep + 1) % 5 == 0:
            mean_per = evaluate(model, dataset, device, quiet=True)
            marker   = " ←best" if mean_per < best_per else ""
            print(f"  epoch {ep+1:3d}/{epochs}  "
                  f"ce={last_loss:.4f}  PER={mean_per*100:.1f}%{marker}")
            if mean_per < best_per:
                best_per = mean_per
                torch.save(model.state_dict(), save_path)

    # reload best
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, weights_only=True, map_location=device))
    print(f"\nDone.  best PER={best_per*100:.1f}%\n")
    return best_per


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model: DREAMAcousticNL, dataset: list, device: torch.device,
             verbose: bool = False, quiet: bool = False) -> float:
    model.eval()
    scores = []
    for feats, _, ref in dataset:
        feats = feats.to(device)
        pred  = model.decode(feats)
        scores.append(per(pred, ref))
        if verbose:
            print(f"  REF : {' '.join(ref[:15])} ...")
            print(f"  PRED: {' '.join(pred[:15])} ...")
            print(f"  PER : {scores[-1]:.3f}\n")
    mean = float(np.mean(scores))
    if not quiet:
        print(f"  PER = {mean:.4f}  ({mean*100:.1f}%)")
    return mean


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DREAM Acoustic Model — GPU training")
    parser.add_argument("--layers",   type=int,   default=2,      help="Number of DREAM layers (default: 2)")
    parser.add_argument("--n_files",  type=int,   default=100,    help="Training files to load (default: 100)")
    parser.add_argument("--epochs",   type=int,   default=100,    help="Training epochs (default: 100)")
    parser.add_argument("--steps",    type=int,   default=10,     help="Gradient steps per epoch (default: 10)")
    parser.add_argument("--lr",       type=float, default=3e-3,   help="Learning rate (default: 3e-3)")
    parser.add_argument("--device",   type=str,   default="auto", help="Device: auto/cpu/cuda/cuda:0 (default: auto)")
    parser.add_argument("--save",     type=str,   default="",     help="Save path for weights (auto if empty)")
    parser.add_argument("--load",       type=str,   default="",  help="Resume from checkpoint")
    parser.add_argument("--max_frames", type=int,   default=400, help="Max frames per crop window during training (0=full, default: 400)")
    args = parser.parse_args()

    # ── Device ────────────────────────────────────────────────────────────
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # ── Save path ─────────────────────────────────────────────────────────
    save_path = args.save or os.path.join(
        RESULTS, f"weights_dream_{args.layers}L_{args.n_files}f.pt"
    )

    # ── Header ────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  DREAM Acoustic  |  {args.layers}L × {DREAM_H2}  |  device={device}")
    print(f"  dataset: {DATASET_DIR}")
    print(f"  save:    {save_path}")
    print(f"{'═'*60}")

    torch.manual_seed(42)

    # ── Dataset ───────────────────────────────────────────────────────────
    print(f"\nLoading dataset (n={args.n_files}) ...")
    dataset = load_dataset(args.n_files)
    if not dataset:
        print("  [fatal] 0 files loaded — check AUDIO_DIR and TG_DIR paths")
        print(f"  AUDIO_DIR : {AUDIO_DIR}")
        print(f"  TG_DIR    : {TG_DIR}")
        sys.exit(1)
    ex = dataset[0]
    print(f"  {len(dataset)} files loaded")
    print(f"  Example: T={ex[0].shape[0]} frames  ref_len={len(ex[2])}  "
          f"[{' '.join(ex[2][:6])} ...]")

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(args.layers).to(device)
    n_p   = sum(p.numel() for p in model.parameters())
    layer_ps = [sum(p.numel() for p in c.parameters()) for c in model.cells]
    print(f"\n  params={n_p:,}  layers=[{', '.join(str(p) for p in layer_ps)}]  "
          f"head={sum(p.numel() for p in model.head.parameters()):,}")

    if args.load and os.path.exists(args.load):
        model.load_state_dict(torch.load(args.load, weights_only=True, map_location=device))
        print(f"  Resumed from: {args.load}")

    # ── Train ─────────────────────────────────────────────────────────────
    train(model, dataset, args.epochs, args.steps, args.lr, device, save_path,
          max_frames=args.max_frames)

    # ── Final eval ────────────────────────────────────────────────────────
    print(f"{'═'*60}")
    print(f"  FINAL  —  {len(dataset)} files")
    print(f"{'═'*60}")
    evaluate(model, dataset, device)

    print(f"\n{'═'*60}")
    print(f"  SAMPLE PREDICTIONS — first 3 files")
    print(f"{'═'*60}")
    evaluate(model, dataset[:3], device, verbose=True, quiet=True)

    print("Done.")


if __name__ == "__main__":
    main()
