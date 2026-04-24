"""Multimodal sensor fusion for MultimodalDREAM."""

import torch
import torch.nn as nn

from dream_net.multimodal.encoders import SensorEncoder


class FusionLayer(nn.Module):
    """
    Concatenates encoded sensor streams into a single x_t vector for DREAMCell.

    Each sensor is encoded independently by its SensorEncoder, then the
    resulting embeddings are concatenated along the feature dimension.
    The output dimensionality equals the sum of all encoder out_dims.

    Parameters
    ----------
    encoders : dict[str, SensorEncoder]
        Named encoders. Order determines concatenation order.
        Keys are used to match inputs in forward().

    Examples
    --------
    >>> from dream_net.multimodal.encoders import IMUEncoder, WheelEncoderSensor
    >>> fusion = FusionLayer({
    ...     "imu":      IMUEncoder(in_dim=6, out_dim=32),
    ...     "encoders": WheelEncoderSensor(in_dim=4, out_dim=16),
    ... })
    >>> fusion.out_dim
    48
    >>> inputs = {"imu": torch.randn(4, 6), "encoders": torch.randn(4, 4)}
    >>> fusion(inputs).shape
    torch.Size([4, 48])
    """

    def __init__(self, encoders: dict[str, SensorEncoder]):
        super().__init__()
        self.encoders = nn.ModuleDict(encoders)  # type: ignore[arg-type]
        self._encoder_keys = list(encoders.keys())
        self.out_dim: int = sum(e.out_dim for e in encoders.values())

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode and fuse all sensor streams.

        Parameters
        ----------
        inputs : dict[str, torch.Tensor]
            Sensor readings keyed by encoder name. Each tensor is
            (batch, raw_dim) or (batch, C, H, W) for visual inputs.

        Returns
        -------
        torch.Tensor
            Fused embedding (batch, out_dim).
        """
        parts = [
            self.encoders[k](inputs[k])  # type: ignore[index]
            for k in self._encoder_keys
        ]
        return torch.cat(parts, dim=-1)
