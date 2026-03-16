"""
DREAM-Net: Dynamic Recall and Elastic Adaptive Memory

A PyTorch implementation of continuous-time RNN cells with:
- Surprise-driven plasticity
- Liquid Time-Constants (LTC)
- Fast weights with Hebbian learning
- Sleep consolidation

Example
-------
>>> from dream_net import DREAM, DREAMConfig, DREAMCell
>>> model = DREAM(input_dim=64, hidden_dim=128, rank=8)
>>> x = torch.randn(4, 50, 64)  # (batch, time, features)
>>> output, state = model(x)
"""

from dream_net.core.config import DREAMConfig
from dream_net.core.state import DREAMState
from dream_net.core.cell import DREAMCell
from dream_net.layers.layer import DREAM, DREAMStack
from dream_net.utils.statistics import RunningStatistics

__version__ = "0.2.0"
__all__ = [
    # Config & State
    "DREAMConfig",
    "DREAMState",
    
    # Core
    "DREAMCell",
    "RunningStatistics",
    
    # High-level API
    "DREAM",
    "DREAMStack",
]
