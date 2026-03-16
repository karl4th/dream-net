# DREAM-Net

**Dynamic Recall and Elastic Adaptive Memory**

A PyTorch implementation of continuous-time recurrent neural networks with surprise-driven plasticity, liquid time-constants (LTC), and fast weights with Hebbian learning.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![GitHub](https://img.shields.io/github/stars/karl4th/dream-net?style=social)](https://github.com/karl4th/dream-net)

> **Research by:** [Manifestro](https://manifestro.io) — bagzhankarl@manifestro.io

---

## 📋 Overview

DREAM-Net is a novel RNN architecture designed for **non-stationary signal processing** — scenarios where the input distribution changes over time (e.g., speaker changes in audio, domain shift in streaming data).

Unlike standard architectures (Transformer, LSTM, GRU) whose parameters are frozen after training, DREAM maintains a set of **fast weights** that adapt in real-time to the current input distribution **without backpropagation**.

### Key Features

| Feature | Description |
|---------|-------------|
| **Fast Weights** | Low-rank associative memory (Ba et al. style) that updates via Hebbian learning |
| **Surprise Gate** | Adaptive plasticity — opens only when prediction error is informative |
| **Liquid Time-Constants** | Continuous-time dynamics with adaptive integration speeds |
| **Sleep Consolidation** | Transfers fast weights to long-term memory during calm periods |
| **Adaptive Forgetting** | Clears stale patterns faster when surprise is high |

---

## 📦 Installation

### From Source (Development)

```bash
# Clone the repository
git clone https://github.com/karl4th/dream-net.git
cd dream-net

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e ".[dev]"
```

### From PyPI 

```bash
pip install dream-audio
```

---

## 🚀 Quick Start

### Basic Usage

```python
import torch
from dream_net import DREAM, DREAMConfig

# Create model
model = DREAM(
    input_dim=80,      # e.g., mel spectrogram bands
    hidden_dim=256,
    rank=8,            # fast weight rank (sweet spot)
)

# Process sequence
x = torch.randn(4, 100, 80)  # (batch, time, features)
output, state = model(x)

print(output.shape)  # (4, 100, 256)
```

### Using DREAMCell Directly

```python
from dream_net import DREAMCell, DREAMConfig

config = DREAMConfig(
    input_dim=80,
    hidden_dim=256,
    rank=8,
    base_plasticity=0.4,
    forgetting_rate=0.03,
    adaptive_forgetting_scale=8.0,
)

cell = DREAMCell(config)
state = cell.init_state(batch_size=4)

# Process step-by-step
for t in range(100):
    x_t = x[:, t, :]  # (4, 80)
    h, state = cell(x_t, state)
```

### Multi-Layer Stack

```python
from dream_net import DREAMStack

model = DREAMStack(
    input_dim=80,
    hidden_dims=[256, 256, 128],  # 3 layers
    rank=8,
    dropout=0.1,
)

output, states = model(x)
print(output.shape)  # (4, 100, 128)
```

---

## 🧪 Experiments

The `experiments/` directory contains reproduction scripts for all experiments from the technical report.

### Running Experiments

```bash
# Using the run script (recommended)
./run.sh experiments/speaker_switch.py

# Or with PYTHONPATH
export PYTHONPATH=src:$PYTHONPATH
python experiments/speaker_switch.py
```

### Experiment 1: Single Speaker Switch

```bash
./run.sh experiments/speaker_switch.py
```

**Goal:** Prove fast weights adapt to speaker change without gradients.

**Result:** Full mode recovers 2× faster than static after speaker switch.

### Experiment 2: Multi-Speaker Stress Test

```bash
./run.sh experiments/stress_test.py
```

**Goal:** Test adaptation across multiple sequential speaker changes (A→B→C).

**Result:** Cross-speaker transfer emerges naturally (male→male benefits).

### Experiment 3: Rank Ablation

```bash
./run.sh experiments/rank_ablation.py
```

**Goal:** Find optimal fast-weight rank for efficiency.

**Result:** `rank=8` is sweet spot (8 KB memory, 11% worse than rank=64).

### Experiment 4: Long-Cycle Memory

```bash
./run.sh experiments/long_cycle.py
```

**Goal:** Test memory integrity across 21 seconds of continuous operation.

**Result:** 47% improvement on 3rd encounter with same speaker.

---

## 📊 Results Summary

| Finding | Evidence |
|---------|----------|
| Fast weights adapt without gradients | Exp 1: Full < Static after switch |
| Adaptive forgetting reduces interference | Exp 2: loss B improved 23% |
| Surprise gate provides selective plasticity | No Gate always worse than Full |
| **rank=8** is practical sweet spot | Exp 3: 8 KB, near-optimal quality |
| Familiar voices recovered faster | Exp 4: 47% improvement by 3rd visit |
| Memory stable over 21s | Exp 4: no corruption |

See [`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md) for full analysis.

---

## 🏗️ Architecture

```
Input x_t
    │
    ▼
┌─────────────────────────────────┐
│  Block 1: Predictive Coding     │  x̂_t = tanh(h_{t-1} @ C.T) · ‖x_t‖
│  Prediction error e_t = x_t − x̂_t           │
└────────────────┬────────────────┘
                 │ e_t
                 ▼
┌─────────────────────────────────┐
│  Block 2: Surprise Gate         │  S_t = σ((‖e_t‖/‖x_t‖ − τ_eff) / γ)
│  τ_eff = classical + adaptive   │  τ adapts via habituation
└────────────────┬────────────────┘
                 │ S_t
          ┌──────┴──────┐
          │             │
          ▼             ▼
┌──────────────┐  ┌───────────────────────────────────────┐
│  LTC Update  │  │  Block 3: Fast Weights (Hebbian/STDP) │
│  τ_dyn =     │  │  dU = −λ_eff(U−U_target)              │
│  τ/(1+S·k)   │  │      + η · S_t · outer(h,e) @ V       │
│  h_t = LTC   │  │  λ_eff = λ·(1 + k_adap · S_t)         │
└──────────────┘  └───────────────────────────────────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │  Block 4: Sleep      │
                  │  Consolidation       │
                  │  (when S_t < S_min)  │
                  │  U_target ← slow U   │
                  └─────────────────────┘
```

---

## 📁 Project Structure

```
dream-net/
├── src/dream_net/          # Main package
│   ├── __init__.py
│   ├── core/               # Core components
│   │   ├── cell.py         # DREAMCell
│   │   ├── config.py       # DREAMConfig
│   │   └── state.py        # DREAMState
│   ├── layers/             # High-level layers
│   │   └── layer.py        # DREAM, DREAMStack
│   └── utils/              # Utilities
│       └── statistics.py   # RunningStatistics
├── experiments/            # Experiment scripts
│   ├── speaker_switch.py
│   ├── stress_test.py
│   ├── rank_ablation.py
│   └── long_cycle.py
├── data/                   # Audio data (gitignored)
├── results/                # Generated plots
├── notebooks/              # Jupyter analysis (future)
├── configs/                # YAML configs (future)
├── tests/                  # Unit tests
├── docs/                   # Documentation (future)
├── LICENSE
├── CITATION.cff
├── pyproject.toml
└── TECHNICAL_REPORT.md
```

---

## 🔬 Configuration

Default hyperparameters (optimized for ASR mel-spectrograms):

```python
DREAMConfig(
    input_dim=80,              # mel bands
    hidden_dim=256,
    rank=8,                    # sweet spot
    base_plasticity=0.4,
    forgetting_rate=0.03,
    adaptive_forgetting_scale=8.0,
    ltc_tau_sys=5.0,
    ltc_surprise_scale=8.0,    # stored in log-space
    surprise_temperature=0.12,
    base_threshold=0.35,
    entropy_influence=0.2,
    time_step=0.1,
    sleep_rate=0.005,
    min_surprise_for_sleep=0.25,
)
```

---

## 📝 License

MIT License — see [`LICENSE`](LICENSE) for details.

---

## 🔖 Citation

If you use DREAM-Net in your research, please cite:

```bibtex
@software{dream_net_2026,
  title = {DREAM-Net: Dynamic Recall and Elastic Adaptive Memory},
  author = {Karl, Bagzhan},
  year = {2026},
  url = {https://github.com/karl4th/dream-net},
  version = {0.2.0},
}
```

Or use the [`CITATION.cff`](CITATION.cff) file for citation metadata.

---

## 📧 Contact

- **GitHub:** [github.com/karl4th/dream-net](https://github.com/karl4th/dream-net)
- **Issues:** [GitHub Issues](https://github.com/karl4th/dream-net/issues)
- **Email:** bagzhankarl@manifestro.io
- **Research:** [manifestro.io](https://manifestro.io)

---

## 🗺️ Roadmap

### Immediate
- [ ] GRU baseline comparison experiment
- [ ] Unit tests for core components
- [ ] PyPI publication

### Short-Term
- [ ] Fix first-switch interference
- [ ] Learnable V matrix
- [ ] Multi-stream evaluation

### Long-Term
- [ ] Hierarchical fast weights
- [ ] Streaming ASR integration
- [ ] Sphinx documentation

See [`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md) §8 for full roadmap.
