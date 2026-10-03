"""Small immutable data models shared by pipeline stages."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatImage = NDArray[np.float32]
BoolImage = NDArray[np.bool_]


@dataclass(frozen=True)
class FrameObservation:
    image: FloatImage
    true_shift_lr: tuple[float, float]
    is_outlier: bool
    outlier_mask_lr: BoolImage | None = None


@dataclass(frozen=True)
class RegistrationEstimate:
    shift_lr: tuple[float, float]
    response: float
    residual: float
    sharpness: float
    reliability: float
