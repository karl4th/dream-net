# DREAM-Net: Technical Specification
**Dynamic Recall and Elastic Adaptive Memory — Neural Architecture for Non-Stationary Sequence Processing**

> *Foundation document for scientific publication. Status: draft v0.1 — sections marked [TBD] to be filled as experiments progress.*

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Architecture Overview](#2-architecture-overview)
3. [Mathematical Formulation](#3-mathematical-formulation)
4. [Acoustic Model (ASR Foundation)](#4-acoustic-model-asr-foundation)
5. [Experimental Results](#5-experimental-results)
6. [Comparison with Baselines](#6-comparison-with-baselines)
7. [Implementation Details](#7-implementation-details)
8. [Known Limitations and Open Problems](#8-known-limitations-and-open-problems)
9. [Roadmap](#9-roadmap)
10. [Glossary](#10-glossary)

---

## 1. Problem Statement

### 1.1 Non-Stationarity in Real-World Sequences

Standard recurrent neural networks (LSTM, GRU, Transformer) are trained offline and deployed with fixed weights. This creates a fundamental mismatch: real-world speech streams are **non-stationary** — speakers change, acoustic environments shift, dialects and speech rates vary within a single session. A fixed-weight model has no mechanism to adapt to a new speaker mid-stream without expensive re-training.

**The core question:** Can a neural network carry a small, fast-adapting memory that updates in real-time at inference — without gradient descent — and improves performance on familiar patterns while not degrading on unfamiliar ones?

### 1.2 Design Goals

1. **Online adaptation** — fast weights update per-timestep using a Hebbian rule, no backpropagation at inference time
2. **Selective plasticity** — only update when the input is genuinely surprising (informative), not on every frame
3. **Stability** — no catastrophic forgetting; long-term patterns are not overwritten by short-term noise
4. **Efficiency** — the entire adaptation memory fits in 8 KB per stream (rank=8, hidden=256, float32)
5. **Composability** — the core cell is an LSTM-compatible drop-in; can be stacked, used with CTC, connected to language models

### 1.3 Target Applications

- Streaming automatic speech recognition (ASR) with speaker adaptation
- Online speaker diarization
- Personalized voice assistants with no fine-tuning
- Any sequence model on non-stationary distributions (financial time series, sensor streams, etc.)

---

## 2. Architecture Overview

### 2.1 High-Level Structure

DREAM-Net processes sequences timestep-by-timestep, maintaining two kinds of memory:

| Memory Type | Container | Update Rule | Speed |
|---|---|---|---|
| **Slow context** | Hidden state `h` | LTC (continuous-time ODE) | Slow (τ ≈ 5–50 steps) |
| **Fast weights** | Low-rank matrix `U` | Hebbian (no backprop) | Immediate (every step) |

At each timestep `t`, the cell:
1. **Predicts** the current input from the slow state (predictive coding)
2. **Computes surprise** — how wrong was the prediction?
3. **Updates fast weights** — if surprising, encode the error pattern via Hebbian rule
4. **Updates slow state** — via Liquid Time-Constant ODE, modulated by surprise
5. **Sleep consolidation** — during calm periods, transfer fast → long-term memory

### 2.2 Four-Block Decomposition

```
Input x_t ∈ ℝ^d
      │
      ▼
┌─────────────────────────────────────────────────┐
│  BLOCK 1 — Predictive Coding                    │
│  x̂_t = tanh(h_{t-1} C^T) · ‖x_t‖              │
│  e_t  = x_t − x̂_t                              │
│                                                 │
│  Params: C ∈ ℝ^{d×H}, W ∈ ℝ^{H×d}, B ∈ ℝ^{H×d} │
└───────────────────────┬─────────────────────────┘
                        │ e_t
                        ▼
┌─────────────────────────────────────────────────┐
│  BLOCK 2 — Surprise Gate                        │
│  S_t = σ((‖e_t‖/‖x_t‖ − τ_eff) / γ) ∈ [0,1]   │
│  τ_eff = 0.3·τ_classical + 0.7·τ_adaptive       │
│                                                 │
│  Gate opens  (S→1): error is informative        │
│  Gate closes (S→0): input is predictable        │
└───────────────────────┬─────────────────────────┘
                S_t     │
       ┌────────────────┤
       │                │
       ▼                ▼
┌─────────────┐  ┌──────────────────────────────────┐
│  BLOCK 4    │  │  BLOCK 3 — Fast Weights (Hebbian)  │
│  LTC Update │  │  λ_eff = λ·(1 + k·S_t)             │
│             │  │  dU = −λ_eff·(U−U_tgt)             │
│  τ_dyn(t) = │  │       + η·S_t·outer(h,e)@V         │
│  τ/( 1+S·k)│  │  U ← U + dU·dt                     │
│             │  │  Norm U to target ‖U‖=W_target     │
│  h_t = LTC  │  └────────────────────────────────────┘
└─────────────┘             │
       │                    │ (calm: S < S_min)
       │                    ▼
       │          ┌──────────────────────────┐
       │          │  Sleep Consolidation      │
       │          │  U_tgt ← U_tgt + ζ·(U−U_tgt) │
       │          │  Normalize ‖U_tgt‖        │
       │          └──────────────────────────┘
       │
       ▼
   (h_t, U_t)  →  next step
```

### 2.3 Stacked Architecture (DREAMAcousticNL)

For the acoustic model, cells are stacked N-deep. Each layer receives the `base_eff` output of the previous layer as its input:

```
mel frame x_t (80-dim)
       │
       ▼
   DREAMCell₁ (input=80, hidden=256)
       │  base_eff₁ = B₁ @ x_norm
       ▼
   DREAMCell₂ (input=256, hidden=256)
       │  base_eff₂ = B₂ @ x_norm₂
       ▼
   ...
       │
       ▼
   DREAMCellₙ (input=256, hidden=256)
       │
       ▼
   Head: Linear(cat(hₙ, base_effₙ), N_phonemes)
```

The `base_eff = B @ x_norm` at each layer is the fast phoneme feature channel — it carries the cross-entropy gradient directly to B at every frame, bypassing the slow LTC dynamics.

---

## 3. Mathematical Formulation

### 3.1 Notation

| Symbol | Meaning | Shape |
|---|---|---|
| `x_t` | Input at time t | `(B, d)` |
| `h_t` | Hidden (slow) state | `(B, H)` |
| `U_t` | Fast weights | `(B, H, r)` |
| `U_tgt` | Long-term (consolidated) fast weights | `(B, H, r)` |
| `V` | Shared right factor (fixed, SVD-init) | `(d, r)` |
| `C` | Predictive coding matrix | `(d, H)` |
| `W` | Error projection | `(H, d)` |
| `B` | Base input projection | `(H, d)` |
| `e_t` | Prediction error | `(B, d)` |
| `S_t` | Surprise | `(B,)` |
| `τ` | Time constant | scalar |
| `dt` | Integration time step | scalar |

### 3.2 Predictive Coding (Block 1)

Input is normalized by its L2 norm, then predicted from the current hidden state:

```
x_norm_t = x_t / (‖x_t‖ + ε),  clamp ∈ [−1, 1]

x̂_t = tanh(h_{t-1} C^T) · ‖x_t‖         [prediction in original scale]

e_t = x_t − x̂_t                           [prediction error]

rel_error_t = ‖e_t‖ / (‖x_t‖ + ε)        [scale-invariant error, ∈ [0, ~2]]
```

The scaling by `‖x_t‖` ensures the surprise gate fires independently of input amplitude and dimension.

### 3.3 Surprise Gate with Habituation (Block 2)

The effective threshold combines entropy-based classical signal and an adaptive habituation component:

```
entropy_t = 0.5 · log(2πe · mean(error_var_t))     [Shannon entropy of error]

τ_classical = τ₀ · (1 + α · entropy_t)             [entropy-inflated threshold]

τ_adaptive ← (1 − ρ) · τ_adaptive + ρ · rel_error_t    [ρ = 0.001, slow]
τ_adaptive = min(τ_adaptive, 0.8)                    [prevent "deafness"]

τ_eff = 0.3 · τ_classical + 0.7 · τ_adaptive

S_t = σ((rel_error_t − τ_eff) / γ)
```

**Intuition:** A model that has seen many similar inputs will have high `τ_adaptive` (habituated), so the same raw error produces lower surprise. Novel inputs break habituation and produce high S_t.

### 3.4 Fast Weights Update — Hebbian with Adaptive Forgetting (Block 3)

Fast weights encode recent (h → e) associations in low-rank form:

```
λ_eff = λ · (1 + k · S_t)              [adaptive forgetting: high surprise clears stale patterns]

hebbian = outer(h_{t-1}, e_t) @ V      [(B, H, r): project to rank-r subspace]

dU = −λ_eff · (U_t − U_tgt) + η · S_t · hebbian

U_{t+1} = U_t + dU · dt

‖U_{t+1}‖ normalized to W_target       [homeostasis: prevents norm explosion]
```

**Key property:** `λ_eff` increases with surprise, meaning high-surprise events both **write** new patterns and **erase** old conflicting patterns in one step. This prevents interference on speaker switches.

### 3.5 Liquid Time-Constant Update (Block 4)

The slow state integrates input effects at a rate modulated by surprise:

```
base_eff  = B @ x_norm_t^T                               [direct per-frame projection]
err_eff   = W @ e_t^T                                    [error contribution]

input_effect = 0.7·h_{t-1} + 0.2·base_eff + 0.3·err_eff·S_t    [weighted combination]

τ_dyn(t) = τ_sys / (1 + S_t · exp(p_scale))    [p_scale: learnable in log-space]
τ_dyn    = clamp(τ_dyn, τ_min=0.01, τ_max=50)

dt_τ = clamp(dt / (τ_dyn + dt),  0.01, 0.5)    [Euler step fraction]

h_t = (1 − dt_τ) · h_{t-1} + dt_τ · tanh(input_effect)
```

**Continuous-time interpretation:** This is the Euler discretization of `dh/dt = (−h + tanh(input_effect)) / τ_dyn(t)`, a first-order ODE with time-varying time constant.

**Jacobian:** `dh_t/dh_{t-1} ≈ (1 − dt_τ) ≈ 0.9–0.98/step`. This ensures BPTT through hundreds of steps without gradient explosion.

### 3.6 Sleep Consolidation

During calm periods (low average surprise), fast weights are transferred to long-term memory:

```
if avg_surprise < S_min:
    dU_tgt = ζ_sleep · (U_t − U_tgt)
    U_tgt ← U_tgt + dU_tgt
    U_tgt ← U_tgt · clamp(W_target / (‖U_tgt‖ + ε), max=1.5)
```

This mirrors biological sleep consolidation: episodic memory (U) → semantic memory (U_tgt).

### 3.7 Acoustic Model Head

For the N-layer acoustic stack, the final phoneme probabilities are:

```
logits_t = W_head · cat(h_t^{(N)}, base_eff_t^{(N)}) + b_head    ∈ ℝ^{N_ph}

P(phoneme | frame_t) = softmax(logits_t)
```

The head reads both `h` (slow context) and `base_eff` (fast phoneme features). During training, CE loss gradient flows directly to `B` via `base_eff` at every frame, while `h` receives gradient through the LTC chain.

---

## 4. Acoustic Model (ASR Foundation)

### 4.1 Task Definition

**Goal:** Learn P(phoneme | frame) from mel-spectrogram features using forced alignment labels.

**Input:** Log-mel spectrogram, normalized per file: `x ∈ ℝ^T×80`
**Output:** Frame-level phoneme probabilities `∈ ℝ^T×72`
**Labels:** Arpabet phonemes with stress markers (72 classes) from Praat TextGrid files
**Metric:** Phoneme Error Rate (PER) — edit distance / reference length, excluding silence/blank/unk

### 4.2 Dataset

| Property | Value |
|---|---|
| Source | Custom dataset with forced alignment |
| Total files | 992 audio + TextGrid pairs |
| Audio format | WAV, 16 kHz, mono |
| TextGrid tiers | `words`, `phones` |
| Vocabulary | 72 classes: `<blank>`, `<unk>`, `<sil>`, 69 Arpabet phonemes |
| Training split used | 15 files (overfit / proof-of-concept) |
| Frame rate | 10 ms (HOP=160 samples at SR=16kHz) |

### 4.3 Feature Extraction

```
SR      = 16,000 Hz
N_FFT   = 1,024  samples
WIN     = 400    samples  (~25 ms)
HOP     = 160    samples  (~10 ms)
N_MELS  = 80
f_min   = 20 Hz,  f_max = 8,000 Hz
power   = 2.0  (energy spectrogram)

log_mel = log(MelSpectrogram(wav) + 1e-6)

per-file normalization:
    μ = mean(log_mel, dim=time)
    σ = std(log_mel,  dim=time).clamp(min=1e-4)
    x = (log_mel − μ) / σ
```

### 4.4 Model Configurations

#### Single-layer (1L) — baseline
```
DREAMCell(input=80, hidden=833)
Head: Linear(833×2, 72)           [cat(h, base_eff)]
Total: 319,947 parameters
```

#### Two-layer (2L) — recommended
```
DREAMCell₁(input=80,  hidden=256)
DREAMCell₂(input=256, hidden=256)
Head: Linear(256×2, 72)
Total: 294,990 parameters
```

#### Three-layer (3L) — for larger datasets
```
DREAMCell₁(input=80,  hidden=256)
DREAMCell₂(input=256, hidden=256)
DREAMCell₃(input=256, hidden=256)
Head: Linear(256×2, 72)
Total: 491,601 parameters
```

#### Four-layer (4L) — for large-scale training
```
DREAMCell₁(input=80,  hidden=256)
DREAMCell₂₋₄(input=256, hidden=256) × 3
Head: Linear(256×2, 72)
Total: 688,212 parameters
```

### 4.5 Training Protocol

```
Optimizer     : Adam(lr=3e-3)
Scheduler     : CosineAnnealingLR(T_max=epochs, eta_min=1e-5)
Gradient clip : 1.0
Loss          : CrossEntropy (frame-level, real frames only via boolean mask)
Steps/epoch   : 10  (gradient steps, not passes over data)
Total steps   : 100 epochs × 10 steps = 1,000 gradient steps
Batch         : all 15 files padded to T_max and processed in parallel
BPTT          : full (no hidden state detach across sequence)
```

**Decode (inference):**
```
smooth_logits = avg_pool1d(logits.T, kernel=11, stride=1, padding=5).T
phones        = collapse(argmax(smooth_logits)), skip {sil, blank, unk}
```

### 4.6 Results Summary

| Model | Params | Best PER | Notes |
|---|---|---|---|
| 1L (h=833) | 319,947 | **10.7%** | Baseline |
| **2L (h=256×2)** | **294,990** | **5.9%** | Best overall |
| 3L (h=256×3) | 491,601 | 8.9% | Candidate for 500+ files |
| 4L (h=256×4) | 688,212 | 8.2% | Candidate for 1000+ files |

All models trained on 15 files, 100 epochs (1,000 gradient steps). Weights saved in `results/`.

**Observation:** Depth outperforms width at fixed-or-lower parameter count. 2L (295K) beats 1L (320K). 3L and 4L converge slower than 2L on 15 files due to gradient attenuation through stacked LTC layers — but are expected to outperform on larger data where more epochs are available.

---

## 5. Experimental Results

### 5.1 Speaker Switching (Single Switch)

**Setup:** Pre-train on Speaker A (30 epochs). Inference on 10s: 5s Speaker A → 5s Speaker B (no gradients, Hebbian only).

| Condition | Loss post-switch (frames 0–200) |
|---|---|
| Full (Hebbian ON) | — recovering |
| Static (dU=0) | — no recovery |
| No gate (S_t ≡ 1) | — noisy, slow |

**Pass criteria (all met):**
1. S_t spike ≥ 0.7 at speaker switch ✓
2. Loss recovery within 50 steps ✓
3. Full loss < Static loss in frames +30..+200 ✓

### 5.2 Multi-Speaker Stress Test (A → B → C)

**Setup:** 15 seconds: Speaker A (5s male) → B (5s female) → C (5s male). Pre-trained on A only.

| Metric | Without adaptive forgetting | With adaptive forgetting |
|---|---|---|
| Loss on B | 3.04 | **2.34** (−23%) |
| Loss on C | 1.91 | **0.66** (−65%) |

**Cross-speaker transfer observed:** C (male) benefits from A (male) patterns even after B (female) in between.

### 5.3 Rank Ablation

| Rank | Memory (bytes) | Mean Loss | Speed (ms/frame) |
|---|---|---|---|
| 2 | 2 KB | 0.641 | 0.27 |
| 4 | 4 KB | 0.601 | 0.28 |
| **8** | **8 KB** | **0.574** | **0.29** |
| 16 | 16 KB | 0.541 | 0.30 |
| 32 | 32 KB | 0.512 | 0.31 |
| 64 | 64 KB | 0.501 | 0.33 |

**Conclusion:** rank=8 is optimal. Only 14.6% quality loss vs rank=64 at 12.5% of the memory.

### 5.4 Long-Cycle Memory (A¹ → B¹ → C¹ → A² → B² → C² → A³)

**Setup:** 21 seconds, 7 segments of 3s each. Each revisit uses a different recording of the same speaker.

| Speaker A visit | Loss |
|---|---|
| 1st encounter | 1.483 |
| 2nd visit | 0.836 (−43.6%) |
| 3rd visit | 0.792 (−46.6%) |

**Key finding:** Progressive monotonic improvement across visits, no catastrophic forgetting.
**Static mode:** 1.051 → 1.675 → 1.671 (no improvement, slight degradation).

### 5.5 Reconstruction Quality

| Model | MSE (unseen) |
|---|---|
| DREAM v1 (broken, h-detach) | 89.5 |
| LSTM | **0.124** |
| Transformer | 0.159 |
| **DREAM v2 (full BPTT, mel loss)** | **0.150** (fast weights ON) |
| DREAM v2 static | 0.162 |

**Key fix:** Removing `h.detach()` during training. With full BPTT, DREAM v2 beats Transformer and approaches LSTM on reconstruction quality.

---

## 6. Comparison with Baselines

### 6.1 DREAM vs GRU (21-second speaker sequence)

| Model | Params | Mean Loss (7 segments) |
|---|---|---|
| GRU (hidden=256) | 280,064 | 3.24 |
| DREAM static | 61,443 | 1.82 |
| **DREAM full** | **61,443** | **1.51** |

DREAM full = **53% lower loss** with **4.6× fewer parameters**.
DREAM beats GRU on **all 7 individual segments** (margins +15.7% to +78.1%).

### 6.2 CTC Comparison (Character-level ASR)

| Model | Params | CER (after 300 epochs) | Notes |
|---|---|---|---|
| LSTM-CTC (h=210) | ~250K | ~20–30% | Stable |
| DREAM-CTC (h=833) | ~320K | ~18–30% | Unstable oscillations |

**Conclusion:** CTC on small datasets (15 files) with 72 phoneme classes is unreliable due to collapse to frequent tokens. Frame-level CE+TextGrid labels are the correct approach for the acoustic model.

### 6.3 Acoustic Model: DREAM vs LSTM-CTC (historical)

| Approach | Epochs | PER |
|---|---|---|
| DREAM-CTC (char-level) | 300 | ~18–30% (unstable) |
| DREAM-CE 1 step/epoch | 100 | 81.8% |
| DREAM-CE 10 steps/epoch | 100 | **5.9% (2L)** |

The key variable was not the architecture — it was the number of gradient steps.

---

## 7. Implementation Details

### 7.1 Project Structure

```
dream_base/
├── src/dream_net/
│   ├── __init__.py              # Public API: DREAMConfig, DREAMCell, DREAM, DREAMStack
│   ├── core/
│   │   ├── config.py            # DREAMConfig dataclass (all hyperparameters)
│   │   ├── cell.py              # DREAMCell — 4-block RNN cell
│   │   └── state.py             # DREAMState — per-batch state container
│   ├── layers/
│   │   └── layer.py             # DREAM (high-level), DREAMStack (multi-layer)
│   └── utils/
│       └── statistics.py        # RunningStatistics (EMA utility)
├── experiments/
│   ├── asr_acoustic.py          # Acoustic model — current main experiment
│   ├── asr_ctc.py               # CTC experiments (char-level)
│   ├── speaker_switch.py        # Single speaker switch
│   ├── stress_test.py           # Multi-speaker A→B→C
│   ├── rank_ablation.py         # Fast weights rank sweep
│   ├── long_cycle.py            # 21-second memory test
│   ├── gru_baseline.py          # GRU comparison
│   ├── spike_reset.py           # Spike-triggered reset (rejected)
│   └── reconstruction.py        # Mel-space reconstruction quality
├── data/dataset/
│   ├── audio/                   # WAV files (16kHz mono)
│   ├── textgrid/                # Praat TextGrid forced alignment
│   ├── metadata.csv             # filename|text|phonemes|textgrid_path
│   └── vocab.txt                # 72 Arpabet phonemes + special tokens
├── results/                     # Saved model weights
├── configs/
│   └── proven_v2.yaml           # Validated configuration
├── ASR_TECH_REPORT.md           # Acoustic model tech report
└── TECHNICAL_REPORT.md          # Full architecture tech report
```

### 7.2 DREAMConfig — Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `input_dim` | 39 | Input feature dimension |
| `hidden_dim` | 256 | Hidden state size H |
| `rank` | 16 | Fast weights rank r |
| `time_step` | 0.1 | ODE integration dt |
| `ltc_tau_sys` | 10.0 | Base time constant τ_sys |
| `ltc_surprise_scale` | 10.0 | Surprise modulation of τ |
| `base_threshold` | 0.5 | Surprise gate threshold τ₀ |
| `surprise_temperature` | 0.1 | Gate sharpness γ |
| `entropy_influence` | 0.2 | Entropy weight α |
| `forgetting_rate` | 0.01 | Hebbian decay λ |
| `base_plasticity` | 0.1 | Hebbian learning rate η |
| `adaptive_forgetting_scale` | 8.0 | Adaptive forgetting k |
| `sleep_rate` | 0.005 | Consolidation rate ζ |
| `min_surprise_for_sleep` | 0.2 | Consolidation threshold S_min |
| `target_norm` | 2.0 | U homeostasis target W_target |

### 7.3 Acoustic Model Hyperparameters (Validated)

```python
DREAMConfig(
    input_dim=80, hidden_dim=256, rank=8,
    base_threshold=0.35, base_plasticity=0.4,
    forgetting_rate=0.03, adaptive_forgetting_scale=8.0,
    ltc_tau_sys=5.0, ltc_surprise_scale=8.0,
    surprise_temperature=0.12, entropy_influence=0.2,
    time_step=0.1, sleep_rate=0.005, min_surprise_for_sleep=0.25,
)

STEPS_PER_EPOCH = 10
LR = 3e-3  (CosineAnnealingLR → 1e-5)
GRAD_CLIP = 1.0
SMOOTH_WIN = 11  (decode only, not training)
```

### 7.4 Critical Engineering Lessons

These discoveries were made experimentally and are non-obvious:

1. **STEPS_PER_EPOCH = 10 is the most important hyperparameter for small datasets.** 1 step/epoch → too few gradient steps regardless of architecture. 10 steps/epoch took PER from 81.8% → 10.7% with no architecture change.

2. **No h.detach() in forward_files.** Detaching hidden state every N steps kills context gradient. LTC Jacobian ≈ 0.9/step is already contractive — no explosion risk with full BPTT.

3. **Head = cat(h, base_eff), not base_eff alone.** base_eff-only head plateaus at CE=2.27. h provides speaker/prosody context that disambiguates phonemes.

4. **No smoothness loss in training.** `(probs[t+1]-probs[t])^2` penalizes correct sharp transitions at phoneme boundaries. Smooth only at decode time via avg_pool1d.

5. **CosineAnnealingLR > ReduceLROnPlateau** for monotonically improving loss. ReduceLROnPlateau never fires when loss improves every epoch.

6. **CTC requires thousands of files, not tens.** With 72 phoneme classes, CTC collapses to frequent tokens on small datasets. Use forced alignment (TextGrid) for small-scale experiments.

7. **proven_v1 → proven_v2 critical fix.** Any h.detach() during base weights training prevents C from learning real structure. Full BPTT required.

---

## 8. Known Limitations and Open Problems

### 8.1 Current Limitations

| # | Limitation | Impact | Proposed Fix |
|---|---|---|---|
| 1 | Acoustic model tested on 15 files only | Unknown OOD generalization | Scale to full dataset |
| 2 | LTC τ too large for CTC (τ_sys=5) | Phoneme boundaries too slow for CTC | Reduce τ to 0.5–1.0 for ASR, or use auxiliary frame loss |
| 3 | BPTT through full sequence at T=749 | Memory O(B·T·H) | Truncated BPTT with overlap, or reversible layers |
| 4 | No language model component | Higher WER than acoustic+LM systems | DREAM-LM on phoneme sequences [TBD] |
| 5 | Deeper models (3L, 4L) slower to converge | Needs more gradient steps | Expected to improve on larger datasets |
| 6 | First-switch interference | Brief quality drop on speaker switch | Already partially addressed by adaptive forgetting |
| 7 | Single-stream inference only | Cannot handle multi-speaker overlap | Multi-stream batching [TBD] |
| 8 | No evaluation on standard benchmarks | Cannot compare with LibriSpeech SOTA | LibriSpeech evaluation [TBD] |

### 8.2 Open Research Questions

1. **Does fast-weight adaptation transfer to unseen speakers?** Current validation is on pre-trained speakers only. The key claim — adaptation without gradient descent — needs evaluation on a held-out speaker set.

2. **What is the optimal depth-width tradeoff as a function of dataset size?** 2L wins at 15 files. At 1000 files, 3L or 4L may be better. A scaling law analysis is needed.

3. **Can DREAM replace the encoder in end-to-end ASR?** Current work is acoustic model only. A full S2T pipeline with DREAM encoder + CTC/attention decoder is the target.

4. **Hebbian learning vs. gradient-based fast adaptation (MAML, MetaNet).** DREAM updates fast weights without gradients. How does this compare to gradient-based meta-learning methods in terms of adaptation speed and memory efficiency?

5. **Sleep consolidation in continuous operation.** In a real-time streaming scenario, "calm periods" may be rare. What happens to U_tgt if S_t > S_min for a long time?

---

## 9. Roadmap

### Phase 1 — Acoustic Model Scaling [Current]
- [x] Prove architecture can learn phonemes (15 files, PER < 10%)
- [x] Identify optimal depth (2L = best at small scale)
- [x] Identify 3L and 4L as candidates for larger data
- [ ] Train on 100, 500, full dataset — compare 1L/2L/3L/4L
- [ ] Evaluate on held-out speakers (generalization test)
- [ ] Benchmark against standard hybrid ASR pipeline

### Phase 2 — Language Model Integration [TBD]
- [ ] Build DREAM-LM: autoregressive model on phoneme sequences
- [ ] Train on phoneme transcripts from forced alignment labels
- [ ] Connect acoustic → LM via beam search or greedy decode
- [ ] Evaluate joint WER on test set

### Phase 3 — End-to-End ASR [TBD]
- [ ] Joint training of acoustic + language model
- [ ] Replace CTC training instability with joint CE+LM objective
- [ ] Speaker adaptation: fine-tune fast weights U from reference audio only

### Phase 4 — Publication [TBD]
- [ ] Full evaluation on LibriSpeech test-clean / test-other
- [ ] Ablation study: each DREAM component (surprise gate, fast weights, LTC, sleep)
- [ ] Comparison with LSTM, GRU, Transformer, wav2vec 2.0 on adaptation speed
- [ ] Streaming latency benchmarks (edge deployment)

---

## 10. Glossary

| Term | Definition |
|---|---|
| **DREAM** | Dynamic Recall and Elastic Adaptive Memory |
| **LTC** | Liquid Time-Constant — continuous-time ODE for hidden state update |
| **Surprise gate** | Sigmoid function that opens plasticity when prediction error is high |
| **Fast weights** | Low-rank matrix U updated by Hebbian rule at each inference step |
| **Slow weights** | Standard parameters C, W, B, τ — updated by backpropagation during training |
| **Hebbian rule** | `ΔU ∝ outer(h, e)` — "neurons that fire together wire together" |
| **Adaptive forgetting** | `λ_eff = λ · (1 + k·S_t)` — high surprise accelerates erasure of stale patterns |
| **Sleep consolidation** | Transfer of episodic fast weights U → long-term U_tgt during low-surprise periods |
| **base_eff** | `B @ x_norm` — direct per-frame mel → hidden projection; receives CE gradient at every step |
| **PER** | Phoneme Error Rate = edit_distance(predicted, reference) / len(reference) |
| **CER** | Character Error Rate — same as PER but for character-level ASR |
| **TextGrid** | Praat forced-alignment file containing phoneme time boundaries |
| **Arpabet** | American English phoneme notation system (AA0, AH1, etc.) with stress markers 0/1/2 |
| **BPTT** | Backpropagation Through Time — gradient computation through the recurrent chain |
| **rank** | Decomposition rank r of fast weights: U ∈ ℝ^{H×r}, V ∈ ℝ^{d×r} |

---

*Document maintained alongside the codebase. Update when new experimental results are obtained.*
*For implementation details see: `experiments/asr_acoustic.py`, `src/dream_net/core/cell.py`*
*For experiment results see: `ASR_TECH_REPORT.md`, `TECHNICAL_REPORT.md`*
