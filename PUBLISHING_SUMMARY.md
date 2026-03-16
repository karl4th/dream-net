# DREAM-Net — Project Summary

**Author:** Bagzhan Karl  
**Organization:** Manifestro  
**Email:** bagzhankarl@manifestro.io  
**GitHub:** https://github.com/karl4th/dream-net  
**License:** MIT  

---

## 📦 What's Ready for Publication

### ✅ Core Package (`src/dream_net/`)

| File | Description |
|------|-------------|
| `__init__.py` | Public API exports |
| `core/__init__.py` | Core module exports |
| `core/config.py` | `DREAMConfig` dataclass |
| `core/state.py` | `DREAMState` dataclass |
| `core/cell.py` | `DREAMCell` — main forward logic |
| `layers/__init__.py` | Layer exports |
| `layers/layer.py` | `DREAM`, `DREAMStack` (high-level API) |
| `utils/__init__.py` | Utils exports |
| `utils/statistics.py` | `RunningStatistics` utility |

### ✅ Experiments (`experiments/`)

| File | Description |
|------|-------------|
| `__init__.py` | Package marker |
| `speaker_switch.py` | Experiment 1: Single speaker switch |
| `stress_test.py` | Experiment 2: Multi-speaker stress test |
| `rank_ablation.py` | Experiment 3: Rank ablation study |
| `long_cycle.py` | Experiment 4: Long-cycle memory test |

### ✅ Tests (`tests/`)

| File | Description |
|------|-------------|
| `__init__.py` | Package marker |
| `test_config.py` | Config unit tests |
| `test_cell.py` | DREAMCell unit tests (10+ tests) |
| `test_layer.py` | DREAM/DREAMStack unit tests |

### ✅ Documentation

| File | Description |
|------|-------------|
| `README.md` | Main documentation with examples |
| `CONTRIBUTING.md` | Contribution guidelines |
| `QUICKSTART.md` | Quick start guide |
| `CITATION.cff` | Academic citation metadata |
| `TECHNICAL_REPORT.md` | Full technical report |

### ✅ Configuration

| File | Description |
|------|-------------|
| `pyproject.toml` | Package config, dependencies, tool settings |
| `LICENSE` | MIT License (Copyright: Bagzhan Karl, Manifestro) |
| `.gitignore` | Proper ignores for research project |
| `.gitattributes` | Git LFS for audio files |
| `.python-version` | Python 3.13 |
| `run.sh` | Experiment runner script |

---

## 🏗️ Project Structure

```
dream-net/
├── src/dream_net/              # Main package
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── state.py
│   │   └── cell.py
│   ├── layers/
│   │   ├── __init__.py
│   │   └── layer.py
│   └── utils/
│       ├── __init__.py
│       └── statistics.py
│
├── experiments/                # Experiment scripts
│   ├── __init__.py
│   ├── speaker_switch.py
│   ├── stress_test.py
│   ├── rank_ablation.py
│   └── long_cycle.py
│
├── tests/                      # Unit tests
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_cell.py
│   └── test_layer.py
│
├── data/                       # Audio data (gitignored)
│   ├── commonvoice/
│   └── ljspeech/
│
├── results/                    # Generated plots (gitignored)
├── notebooks/                  # Jupyter (future)
├── configs/                    # YAML configs (future)
├── scripts/                    # Helper scripts (future)
├── docs/                       # Sphinx docs (future)
│
├── README.md                   # Main documentation ✨
├── CONTRIBUTING.md             # Contribution guide ✨
├── QUICKSTART.md               # Quick start ✨
├── CITATION.cff                # Citation metadata ✨
├── LICENSE                     # MIT License ✨
├── TECHNICAL_REPORT.md         # Technical report
├── pyproject.toml              # Package config ✨
├── run.sh                      # Experiment runner ✨
├── .gitignore                  # Git ignores ✨
└── .gitattributes              # Git LFS ✨
```

---

## 🚀 How to Use

### Run Experiments

```bash
# Clone
git clone https://github.com/karl4th/dream-net.git
cd dream-net

# Install
uv sync

# Run experiment
./run.sh experiments/speaker_switch.py
```

### Use as Library

```python
from dream_net import DREAM, DREAMConfig

model = DREAM(input_dim=80, hidden_dim=256, rank=8)
x = torch.randn(4, 100, 80)
output, state = model(x)
```

---

## 📊 Key Results

| Finding | Metric |
|---------|--------|
| Fast weights adapt without gradients | Full < Static after switch |
| Adaptive forgetting reduces interference | 23% improvement |
| rank=8 is sweet spot | 8 KB, near-optimal |
| Familiar voices recovered faster | 47% improvement by 3rd visit |
| Memory stable over 21s | No corruption |

---

## 📝 Next Steps Before Publishing

1. **Add your ORCID** to `CITATION.cff` (currently placeholder)
2. **Create Zenodo deposit** for DOI (optional, for citability)
3. **Run final tests** to ensure everything works
4. **Push to GitHub**
5. **Add GitHub Topics**: `deep-learning`, `rnn`, `continual-learning`, `pytorch`

---

## 📧 Contact

- **GitHub:** https://github.com/karl4th/dream-net
- **Email:** bagzhankarl@manifestro.io
- **Research:** https://manifestro.io

---

*Last updated: March 16, 2026*
