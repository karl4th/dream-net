"""MultimodalDREAM: pluggable multimodal adaptive sequence model."""

import torch
import torch.nn as nn

from dream_net.core.cell import DREAMCell
from dream_net.core.config import DREAMConfig
from dream_net.core.state import DREAMState
from dream_net.multimodal.encoders import SensorEncoder
from dream_net.multimodal.fusion import FusionLayer
from dream_net.multimodal.heads import OutputHead


class MultimodalDREAM(nn.Module):
    """
    Multimodal adaptive sequence model built on DREAMCell.

    Combines a set of pluggable SensorEncoders, a FusionLayer, a DREAMCell,
    and an OutputHead into a single nn.Module. Any developer can:

    1. Write a SensorEncoder subclass for their modality (~10 lines).
    2. Plug it into the encoders dict.
    3. Choose or subclass an OutputHead for their task.
    4. Get DREAM's surprise-driven fast-weight adaptation for free.

    Parameters
    ----------
    encoders : dict[str, SensorEncoder]
        Named sensor encoders. Their out_dims must sum to
        dream_config.input_dim.
    dream_config : DREAMConfig
        Configuration for the DREAMCell. Set input_dim to match
        the sum of encoder out_dims.
    output_head : OutputHead
        Maps DREAMCell hidden state to task output.

    Examples
    --------
    >>> from dream_net.multimodal import (
    ...     MultimodalDREAM, IMUEncoder, WheelEncoderSensor,
    ...     ActionEncoder, TankDriveHead,
    ... )
    >>> from dream_net import DREAMConfig
    >>>
    >>> model = MultimodalDREAM(
    ...     encoders={
    ...         "imu":         IMUEncoder(in_dim=6, out_dim=32),
    ...         "wheels":      WheelEncoderSensor(in_dim=4, out_dim=16),
    ...         "prev_action": ActionEncoder(action_dim=2, out_dim=16),
    ...     },
    ...     dream_config=DREAMConfig(input_dim=64, hidden_dim=256, rank=8),
    ...     output_head=TankDriveHead(hidden_dim=256),
    ... )
    >>>
    >>> inputs = {
    ...     "imu":         torch.randn(4, 6),
    ...     "wheels":      torch.randn(4, 4),
    ...     "prev_action": torch.zeros(4, 2),
    ... }
    >>> action, state = model(inputs)
    >>> action.shape
    torch.Size([4, 2])

    Fast-weight adaptation
    ----------------------
    >>> model.enable_fast_weights()   # after pre-training
    >>> model.disable_fast_weights()  # back to pre-training mode
    """

    def __init__(
        self,
        encoders: dict[str, SensorEncoder],
        dream_config: DREAMConfig,
        output_head: OutputHead,
    ):
        super().__init__()
        self.fusion = FusionLayer(encoders)

        if self.fusion.out_dim != dream_config.input_dim:
            raise ValueError(
                f"Encoder output dims sum to {self.fusion.out_dim} "
                f"but dream_config.input_dim={dream_config.input_dim}. "
                f"Adjust encoder out_dims or input_dim."
            )

        self.cell = DREAMCell(dream_config)
        self.head = output_head

    # ------------------------------------------------------------------
    # Fast-weight toggle (proxies to DREAMCell)
    # ------------------------------------------------------------------

    def enable_fast_weights(self) -> None:
        """Enable in-context fast-weight adaptation (call after pre-training)."""
        self.cell.enable_fast_weights()

    def disable_fast_weights(self) -> None:
        """Disable fast-weight adaptation (use during pre-training)."""
        self.cell.disable_fast_weights()

    @property
    def fast_weights_enabled(self) -> bool:
        return self.cell.fast_weights_enabled

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def init_state(
        self,
        batch_size: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> DREAMState:
        """Initialise DREAMCell state."""
        return self.cell.init_state(batch_size, device, dtype)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        state: DREAMState | None = None,
    ) -> tuple[torch.Tensor, DREAMState]:
        """
        Process one timestep.

        Parameters
        ----------
        inputs : dict[str, torch.Tensor]
            Sensor readings for this timestep, keyed by encoder name.
            Each value is (batch, raw_dim) or (batch, C, H, W).
        state : DREAMState | None
            Previous state. Initialised from config if None.

        Returns
        -------
        output : torch.Tensor
            Task output from the head (batch, output_dim).
        state : DREAMState
            Updated state.
        """
        # Pick any tensor to get batch size and device
        sample = next(iter(inputs.values()))
        batch_size = sample.shape[0]

        if state is None:
            state = self.init_state(batch_size, device=sample.device, dtype=sample.dtype)

        x_t = self.fusion(inputs)
        h_t, new_state = self.cell(x_t, state)
        return self.head(h_t), new_state

    def forward_sequence(
        self,
        inputs_seq: dict[str, torch.Tensor],
        state: DREAMState | None = None,
        return_all: bool = True,
    ) -> tuple[torch.Tensor, DREAMState]:
        """
        Process a full sequence.

        Parameters
        ----------
        inputs_seq : dict[str, torch.Tensor]
            Sensor sequences keyed by encoder name.
            Each value is (batch, time, raw_dim) or (batch, time, C, H, W).
        state : DREAMState | None
            Initial state.
        return_all : bool
            If True, return outputs for every timestep (batch, time, out_dim).
            If False, return only the final timestep output (batch, out_dim).

        Returns
        -------
        output : torch.Tensor
            (batch, time, out_dim) if return_all else (batch, out_dim).
        state : DREAMState
            Final state.
        """
        # Determine (batch, time) from any sensor
        sample = next(iter(inputs_seq.values()))
        batch_size, time_steps = sample.shape[0], sample.shape[1]

        if state is None:
            state = self.init_state(batch_size, device=sample.device, dtype=sample.dtype)

        outputs: list[torch.Tensor] = []
        # Get initial output shape for fallback
        out_placeholder, _ = self.forward(
            {k: v[:, 0] for k, v in inputs_seq.items()}, state
        )
        output = out_placeholder  # fallback for time_steps == 0

        if time_steps == 0:
            if return_all:
                return output.unsqueeze(1)[:, :0, :], state
            return output, state

        for t in range(time_steps):
            step_inputs = {k: v[:, t] for k, v in inputs_seq.items()}
            output, state = self.forward(step_inputs, state)
            if return_all:
                outputs.append(output.unsqueeze(1))

        if return_all:
            return torch.cat(outputs, dim=1), state
        return output, state
