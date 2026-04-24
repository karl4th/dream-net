"""High-level DREAM sequence model (nn.LSTM-like interface)."""


from typing import Any

import torch
import torch.nn as nn

from dream_net.core.cell import DREAMCell
from dream_net.core.config import DREAMConfig
from dream_net.core.state import DREAMState


class DREAM(nn.Module):
    """
    High-level DREAM sequence model.

    Provides a simple interface similar to nn.LSTM or nn.RNN for processing
    sequences with the DREAM cell.

    Parameters
    ----------
    input_dim : int
        Dimension of input features
    hidden_dim : int
        Dimension of hidden state
    rank : int
        Fast weights rank
    **kwargs
        Additional arguments passed to DREAMConfig

    Examples
    --------
    >>> from dream import DREAM
    >>> model = DREAM(input_dim=64, hidden_dim=128, rank=8)

    >>> # Process sequence
    >>> x = torch.randn(32, 50, 64)  # (batch, time, features)
    >>> output, state = model(x)
    >>> print(output.shape)  # (batch, time, hidden_dim) if return_sequences=True

    >>> # Process with initial state
    >>> state = model.init_state(32)
    >>> output, state = model(x, state)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        rank: int = 8,
        **kwargs: Any,
    ):
        super().__init__()
        self.config = DREAMConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            rank=rank,
            **kwargs
        )
        self.cell = DREAMCell(self.config)
        self.hidden_dim = hidden_dim

    def init_state(
        self,
        batch_size: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> DREAMState:
        """
        Initialize model state.

        Parameters
        ----------
        batch_size : int
            Batch size
        device : torch.device, optional
            Device for tensors
        dtype : torch.dtype, optional
            Data type for tensors

        Returns
        -------
        DREAMState
            Initialized state
        """
        return self.cell.init_state(batch_size, device, dtype)

    def forward(
        self,
        x: torch.Tensor,
        state: DREAMState | None = None,
        return_sequences: bool = True
    ) -> tuple[torch.Tensor, DREAMState]:
        """
        Process input sequence.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, time, input_dim)
        state : DREAMState, optional
            Initial state. If None, initialized from config.
        return_sequences : bool, default=True
            If True, return all hidden states.
            If False, return only final hidden state.

        Returns
        -------
        output : torch.Tensor
            If return_sequences=True: (batch, time, hidden_dim)
            If return_sequences=False: (batch, hidden_dim)
        state : DREAMState
            Final state after processing sequence
        """
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input (batch, time, input_dim), got {x.shape}")

        batch_size, time_steps, _ = x.shape

        if state is None:
            state = self.init_state(batch_size, device=x.device, dtype=x.dtype)

        outputs: list[torch.Tensor] = []
        h = state.h

        for t in range(time_steps):
            x_t = x[:, t, :]  # (batch, input_dim)
            h, state = self.cell(x_t, state)

            if return_sequences:
                outputs.append(h.unsqueeze(1))

        if return_sequences:
            output = torch.cat(outputs, dim=1)  # (batch, time, hidden_dim)
        else:
            output = h  # (batch, hidden_dim) — state.h if time_steps == 0

        return output, state

    def forward_with_mask(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        state: DREAMState | None = None
    ) -> tuple[torch.Tensor, DREAMState]:
        """
        Process padded sequence with length masking.

        Parameters
        ----------
        x : torch.Tensor
            Padded input tensor (batch, time, input_dim)
        lengths : torch.Tensor
            Actual sequence lengths (batch,)
        state : DREAMState, optional
            Initial state

        Returns
        -------
        output : torch.Tensor
            Output tensor with zeros for padded positions
        state : DREAMState
            Final state (state at last valid timestep for each sequence)
        """
        batch_size, time_steps, _ = x.shape

        if state is None:
            state = self.init_state(batch_size, device=x.device, dtype=x.dtype)

        # Initialize output
        outputs = torch.zeros(batch_size, time_steps, self.hidden_dim,
                             device=x.device, dtype=x.dtype)

        # Process each timestep
        for t in range(time_steps):
            # Create mask for active sequences at this timestep
            mask = (lengths > t).float().unsqueeze(1)  # (batch, 1)

            x_t = x[:, t, :]  # (batch, input_dim)
            h, state = self.cell(x_t, state)

            # Only update output for active sequences
            outputs[:, t:t+1, :] = h.unsqueeze(1) * mask.unsqueeze(2)

        return outputs, state


class DREAMStack(nn.Module):
    """
    Stack of multiple DREAM layers.

    Parameters
    ----------
    input_dim : int
        Input dimension for first layer
    hidden_dims : list of int
        Hidden dimensions for each layer
    rank : int
        Fast weights rank for all layers
    dropout : float
        Dropout between layers (applied to hidden states)
    **kwargs
        Additional arguments for DREAMConfig

    Examples
    --------
    >>> from dream import DREAMStack
    >>> model = DREAMStack(
    ...     input_dim=64,
    ...     hidden_dims=[128, 128, 64],  # 3 layers
    ...     rank=8,
    ...     dropout=0.1
    ... )
    >>> x = torch.randn(32, 50, 64)
    >>> output, states = model(x)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        rank: int = 8,
        dropout: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__()

        self.layers = nn.ModuleList()

        # First layer
        self.layers.append(DREAM(input_dim, hidden_dims[0], rank, **kwargs))

        # Subsequent layers: each takes the previous layer's output as input
        for i, hidden_dim in enumerate(hidden_dims[1:]):
            self.layers.append(DREAM(hidden_dims[i], hidden_dim, rank, **kwargs))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.hidden_dims = hidden_dims

    def init_state(
        self,
        batch_size: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> list[DREAMState]:
        """Initialize state for all layers."""
        return [
            layer.init_state(batch_size, device, dtype)  # type: ignore[operator]
            for layer in self.layers
        ]

    def forward(
        self,
        x: torch.Tensor,
        states: list[DREAMState] | None = None,
        return_sequences: bool = True
    ) -> tuple[torch.Tensor, list[DREAMState]]:
        """
        Process sequence through all layers.

        Parameters
        ----------
        x : torch.Tensor
            Input (batch, time, input_dim)
        states : list, optional
            Initial states for each layer
        return_sequences : bool
            Whether to return all timesteps

        Returns
        -------
        output : torch.Tensor
            Output from final layer
        states : list
            Final states for all layers
        """
        if states is None:
            states = self.init_state(x.shape[0], device=x.device, dtype=x.dtype)

        output = x
        last = len(self.layers) - 1

        for i, layer in enumerate(self.layers):
            # Intermediate layers must always return sequences so the next
            # layer receives 3D input; only the final layer respects the flag.
            layer_return_seq = True if i < last else return_sequences
            output, states[i] = layer(output, states[i], layer_return_seq)

            # Apply dropout between layers (not on last layer)
            if self.dropout is not None and i < last:
                output = self.dropout(output)

        return output, states
