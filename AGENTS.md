# AGENTS.md

This file provides guidance to Qoder (qoder.com) when working with code in this repository.

## Commands

```bash
# Install (uv recommended)
uv sync

# Run tests
pytest                                          # all tests
pytest tests/test_cell.py                       # single test file
pytest tests/test_cell.py::test_cell_forward    # single test
pytest --cov=src/dream_net                      # with coverage

# Lint
ruff check src/ tests/
ruff check --fix src/ tests/                    # auto-fix

# Type check
mypy src/

# Run experiments
./run.sh experiments/speaker_switch.py
./run.sh experiments/stress_test.py
./run.sh experiments/rank_ablation.py
./run.sh experiments/long_cycle.py
```

## Architecture

DREAM-Net is a continuous-time RNN with surprise-driven plasticity and liquid time-constants (LTC), designed for non-stationary signal processing (e.g., speaker changes in audio streams).

### Core Data Flow (DREAMCell.forward)

The cell processes one timestep at a time through four blocks:

1. **Predictive Coding** — `x_pred = tanh(h @ C.T) * ||x||`, error `e = x - x_pred`
2. **Surprise Gate** — `S = sigmoid((rel_error_norm - effective_tau) / gamma)`. Effective threshold blends classical entropy-based threshold with adaptive habituation. Surprise ∈ [0,1].
3. **Fast Weights** — Low-rank Hebbian update `dU = -λ_eff(U - U_target) + η * S * outer(h,e) @ V`. **Currently DISABLED** in the forward pass (see below).
4. **LTC Update** — `tau_dynamic = tau_sys / (1 + S * exp(ltc_surprise_scale))`, then `h_new = (1 - dt/tau) * h_prev + (dt/tau) * tanh(input_effect)`. High surprise → small tau → fast updates.

Sleep consolidation runs when `avg_surprise < S_min`: slowly transfers `U → U_target` with homeostatic normalization.

### Important: Fast Weights Are Disabled

In `cell.py:forward()`, the fast weights update block is replaced with `pass`. The model currently relies on LTC dynamics + surprise-gated error injection for adaptation. The `update_fast_weights()` method still exists and is tested, but is not called during forward. The `U` state tensor is kept for compatibility.

### Module Hierarchy

- `DREAMConfig` (dataclass, `core/config.py`) — all hyperparameters with defaults tuned for 80-band log-mel spectrograms
- `DREAMState` (dataclass, `core/state.py`) — mutable state container: `h`, `U`, `U_target`, `adaptive_tau`, `error_mean`, `error_var`, `avg_surprise`. All tensors are batched.
- `DREAMCell` (`core/cell.py`) — single-step RNN cell with `forward(x, state)` and `forward_sequence(x_seq, state, return_all)`
- `DREAM` (`layers/layer.py`) — `nn.LSTM`-like wrapper that iterates `DREAMCell` over a sequence. Supports `forward_with_mask()` for padded sequences.
- `DREAMStack` (`layers/layer.py`) — multi-layer stack of `DREAM` modules with optional inter-layer dropout
- `RunningStatistics` (`utils/statistics.py`) — standalone EMA tracker (not used in current cell; cell tracks stats inline in state)

### Key Design Details

- `ltc_surprise_scale` is stored in log-space as an `nn.Parameter` (`exp(param)` gives the actual scale), initialized to `log(config.ltc_surprise_scale)`
- `V` (fast weight right factor) is an `nn.Buffer` (not a Parameter), initialized via SVD for stability
- `C[0, :]` is forced positive after Xavier init as a stability constraint
- Error norm is normalized by input norm (`rel_error_norm = error_norm / x_scale`) to keep surprise scale-invariant across input dimensions
- `adaptive_tau` in state implements habituation: slowly adapts toward recent error norms, capped at 0.8 to prevent "deafness"
- `tau_effective` is clamped to [0.01, 50.0]; `dt_over_tau` clamped to [0.01, 0.5]
- Fast weights `U` are per-batch (shape `(batch, hidden, rank)`) for independent adaptation across sequences

## Locked Configuration

`configs/proven_v1.yaml` contains validated hyperparameters locked after 6 experiments. Do not change these values without re-running the full experiment suite. The key deviations from `DREAMConfig` defaults: `input_dim=80`, `rank=8`, `base_plasticity=0.4`, `forgetting_rate=0.03`, `base_threshold=0.35`, `surprise_temperature=0.12`, `ltc_tau_sys=5.0`, `ltc_surprise_scale=8.0`.

## Code Style

- Python 3.13+, line length 100
- Ruff with E/W/F/I/B/C4/UP/ARG/TCH rules (E501, B028, B904 ignored)
- Mypy strict mode with `torch.*`, `torchaudio.*`, `matplotlib.*`, `numpy.*` import ignores
- NumPy-style docstrings required for all public functions/classes
- Hatchling build system (`src/dream_net` layout)
