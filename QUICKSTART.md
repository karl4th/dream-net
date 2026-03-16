# DREAM-Net — Quick Start Guide

## 1. Installation

```bash
# Clone repository
git clone https://github.com/karl4th/dream-net.git
cd dream-net

# Install dependencies (requires uv: https://docs.astral.sh/uv/)
uv sync

# Alternative: use existing .venv
python -m venv .venv
source .venv/bin/activate
pip install torch torchaudio matplotlib numpy
```

## 2. Run Experiments

```bash
# Using the run script (recommended)
./run.sh experiments/speaker_switch.py

# Or manually
export PYTHONPATH=src:$PYTHONPATH
python experiments/speaker_switch.py
```

## 3. Use as Library

```python
from dream_net import DREAM, DREAMConfig

model = DREAM(input_dim=80, hidden_dim=256, rank=8)
x = torch.randn(4, 100, 80)
output, state = model(x)
```

## 4. Run Tests

```bash
./run.sh -m pytest tests/ -v
```

## Troubleshooting

**ModuleNotFoundError: No module named 'dream_net'**
→ Make sure PYTHONPATH includes `src/`: `export PYTHONPATH=src:$PYTHONPATH`

**uv not found**
→ Install: `curl -LsSf https://astral.sh/uv/install.sh | sh`

---

For full documentation, see [README.md](README.md).
