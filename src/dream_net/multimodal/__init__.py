"""
dream_net.multimodal — pluggable multimodal adaptive modeling.

Typical usage
-------------
>>> from dream_net.multimodal import (
...     MultimodalDREAM,
...     IMUEncoder, WheelEncoderSensor, ActionEncoder, VisualEncoder,
...     TimeSeriesEncoder,
...     TankDriveHead, ContinuousHead, ClassificationHead,
... )
"""

from dream_net.multimodal.encoders import (
    SensorEncoder,
    IMUEncoder,
    WheelEncoderSensor,
    ActionEncoder,
    TargetEncoder,
    TimeSeriesEncoder,
    VisualEncoder,
)
from dream_net.multimodal.fusion import FusionLayer
from dream_net.multimodal.heads import (
    OutputHead,
    TankDriveHead,
    ContinuousHead,
    ClassificationHead,
)
from dream_net.multimodal.model import MultimodalDREAM

__all__ = [
    # Encoders
    "SensorEncoder",
    "IMUEncoder",
    "WheelEncoderSensor",
    "ActionEncoder",
    "TargetEncoder",
    "TimeSeriesEncoder",
    "VisualEncoder",
    # Fusion
    "FusionLayer",
    # Heads
    "OutputHead",
    "TankDriveHead",
    "ContinuousHead",
    "ClassificationHead",
    # Model
    "MultimodalDREAM",
]
