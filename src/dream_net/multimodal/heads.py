"""Output heads for MultimodalDREAM."""

import torch
import torch.nn as nn


class OutputHead(nn.Module):
    """
    Base class for output heads.

    Takes h_t from DREAMCell and maps it to the desired output.
    Subclass this to add a new task type.

    Examples
    --------
    >>> class MyHead(OutputHead):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.linear = nn.Linear(256, 3)
    ...     def forward(self, h):
    ...         return self.linear(h)
    """

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TankDriveHead(OutputHead):
    """
    Action head for differential drive (tank) robots.

    Outputs [track_left, track_right] ∈ [−1, 1]² via tanh.

    Parameters
    ----------
    hidden_dim : int
        DREAMCell hidden_dim (input to this head).

    Examples
    --------
    >>> head = TankDriveHead(hidden_dim=256)
    >>> h = torch.randn(4, 256)
    >>> head(h).shape
    torch.Size([4, 2])
    >>> (head(h).abs() <= 1).all()
    tensor(True)
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, 2)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.linear(h))


class ContinuousHead(OutputHead):
    """
    Generic regression head for continuous action spaces.

    Parameters
    ----------
    hidden_dim : int
        DREAMCell hidden_dim (input to this head).
    out_dim : int
        Number of output dimensions.
    bounded : bool
        If True, applies tanh to bound outputs to [−1, 1].

    Examples
    --------
    >>> head = ContinuousHead(hidden_dim=256, out_dim=4)
    >>> h = torch.randn(4, 256)
    >>> head(h).shape
    torch.Size([4, 4])
    """

    def __init__(self, hidden_dim: int = 256, out_dim: int = 1, bounded: bool = False):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, out_dim)
        self.bounded = bounded

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        out = self.linear(h)
        return torch.tanh(out) if self.bounded else out


class ClassificationHead(OutputHead):
    """
    Discrete action head (softmax over N classes).

    Parameters
    ----------
    hidden_dim : int
        DREAMCell hidden_dim (input to this head).
    num_classes : int
        Number of discrete action classes.

    Examples
    --------
    >>> head = ClassificationHead(hidden_dim=256, num_classes=5)
    >>> h = torch.randn(4, 256)
    >>> head(h).shape
    torch.Size([4, 5])
    """

    def __init__(self, hidden_dim: int = 256, num_classes: int = 5):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.linear(h)  # raw logits — apply F.cross_entropy in training
