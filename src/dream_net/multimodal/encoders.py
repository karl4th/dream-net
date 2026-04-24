"""Pluggable sensor encoders for MultimodalDREAM."""

import torch
import torch.nn as nn


class SensorEncoder(nn.Module):
    """
    Base class for all sensor encoders.

    Subclass this and implement forward() to add a new modality.
    Must set self._out_dim in __init__.

    Examples
    --------
    >>> class MyEncoder(SensorEncoder):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self._out_dim = 32
    ...         self.net = nn.Linear(10, 32)
    ...     def forward(self, x):
    ...         return self.net(x)
    """

    @property
    def out_dim(self) -> int:
        if not hasattr(self, '_out_dim'):
            raise NotImplementedError("Subclasses must set self._out_dim in __init__")
        return self._out_dim  # type: ignore[return-value]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class IMUEncoder(SensorEncoder):
    """
    Encodes IMU readings into a fixed-size vector.

    Designed for MPU-6050 and compatible sensors.

    Parameters
    ----------
    in_dim : int
        Raw IMU channels. 6 for MPU-6050 (ax,ay,az,gx,gy,gz),
        9 for 9-axis sensors (adds magnetometer mx,my,mz).
    out_dim : int
        Output embedding size.

    Examples
    --------
    >>> enc = IMUEncoder(in_dim=6, out_dim=32)
    >>> x = torch.randn(4, 6)   # batch of MPU-6050 readings
    >>> enc(x).shape
    torch.Size([4, 32])
    """

    def __init__(self, in_dim: int = 6, out_dim: int = 32):
        super().__init__()
        self._out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class WheelEncoderSensor(SensorEncoder):
    """
    Encodes wheel encoder readings (velocity + position delta).

    Parameters
    ----------
    in_dim : int
        Number of encoder channels.
        4 for differential drive: [v_left, v_right, pos_left, pos_right].
    out_dim : int
        Output embedding size.

    Examples
    --------
    >>> enc = WheelEncoderSensor(in_dim=4, out_dim=16)
    >>> x = torch.randn(4, 4)
    >>> enc(x).shape
    torch.Size([4, 16])
    """

    def __init__(self, in_dim: int = 4, out_dim: int = 16):
        super().__init__()
        self._out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActionEncoder(SensorEncoder):
    """
    Encodes the previous action as temporal context.

    Giving the model its own last output as input closes the
    prediction loop and helps DREAM anticipate operator intent.

    Parameters
    ----------
    action_dim : int
        Dimensionality of the action. 2 for tank drive [left, right].
    out_dim : int
        Output embedding size.

    Examples
    --------
    >>> enc = ActionEncoder(action_dim=2, out_dim=16)
    >>> prev_action = torch.zeros(4, 2)
    >>> enc(prev_action).shape
    torch.Size([4, 16])
    """

    def __init__(self, action_dim: int = 2, out_dim: int = 16):
        super().__init__()
        self._out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(action_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TimeSeriesEncoder(SensorEncoder):
    """
    Generic encoder for any 1D time-series modality.

    Use this when none of the built-in encoders fit your sensor.

    Parameters
    ----------
    in_dim : int
        Raw sensor channels.
    out_dim : int
        Output embedding size.
    hidden_dim : int
        Hidden layer size.

    Examples
    --------
    >>> enc = TimeSeriesEncoder(in_dim=12, out_dim=32)
    >>> x = torch.randn(4, 12)
    >>> enc(x).shape
    torch.Size([4, 32])
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 64):
        super().__init__()
        self._out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TargetEncoder(SensorEncoder):
    """
    Encodes navigation target signal relative to the robot's current pose.

    Parameters
    ----------
    in_dim : int
        2 by default: [target_angle_error, target_distance].
        Extend to 3 if you also pass bearing or elevation.
    out_dim : int
        Output embedding size.

    Input semantics
    ---------------
    target_angle_error : float
        Signed heading error in radians (from gyroscope + odometry).
        Positive = target is to the right, negative = to the left.
    target_distance : float
        Euclidean distance to the goal in metres.

    Examples
    --------
    >>> enc = TargetEncoder(in_dim=2, out_dim=16)
    >>> x = torch.tensor([[0.3, 5.2]])   # 0.3 rad error, 5.2 m away
    >>> enc(x).shape
    torch.Size([1, 16])
    """

    def __init__(self, in_dim: int = 2, out_dim: int = 16):
        super().__init__()
        self._out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _DefaultCNN(nn.Module):
    """Minimal 4-layer CNN — no torchvision dependency required."""

    out_features: int = 256

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VisualEncoder(SensorEncoder):
    """
    Encodes camera frames via a CNN backbone + linear projection.

    The default backbone is a minimal 4-layer CNN that requires no
    external dependencies. Pass your own pretrained backbone via
    ``backbone`` for better visual representations.

    Parameters
    ----------
    out_dim : int
        Output embedding size after projection.
    backbone : nn.Module | None
        Custom CNN backbone. Must have an ``out_features: int``
        class or instance attribute indicating its output size.
        If None, a simple 4-layer CNN is used.
    backbone_out_features : int | None
        Required when providing a custom backbone. Tells the encoder
        how many features the backbone outputs.

    Examples
    --------
    >>> enc = VisualEncoder(out_dim=64)
    >>> x = torch.randn(2, 3, 224, 224)
    >>> enc(x).shape
    torch.Size([2, 64])

    Using a torchvision backbone:
    >>> import torchvision.models as models
    >>> resnet = models.resnet18(pretrained=False)
    >>> resnet.fc = nn.Identity()           # remove classification head
    >>> enc = VisualEncoder(out_dim=64, backbone=resnet, backbone_out_features=512)
    """

    def __init__(
        self,
        out_dim: int = 64,
        backbone: nn.Module | None = None,
        backbone_out_features: int | None = None,
    ):
        super().__init__()
        self._out_dim = out_dim

        if backbone is None:
            self.backbone: nn.Module = _DefaultCNN()
            features = _DefaultCNN.out_features
        else:
            if backbone_out_features is None:
                raise ValueError(
                    "backbone_out_features is required when providing a custom backbone"
                )
            self.backbone = backbone
            features = backbone_out_features

        self.proj = nn.Sequential(
            nn.Linear(features, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.backbone(x))
