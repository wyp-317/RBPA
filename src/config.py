"""Typed configuration for the reproducible baseline."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SynthesisConfig:
    """Controls the synthetic multi-frame degradation model."""

    scale: int = 2
    frame_count: int = 10
    max_shift_lr: float = 0.9
    blur_sigma_min: float = 0.55
    blur_sigma_max: float = 1.05
    noise_std: float = 0.008
    outlier_fraction: float = 0.3
    outlier_mode: Literal["blur", "occlusion", "noise", "mixed"] = "mixed"
    seed: int = 7

    def __post_init__(self) -> None:
        if self.scale < 2:
            raise ValueError("scale must be at least 2")
        if self.frame_count < 2:
            raise ValueError("frame_count must be at least 2")
        if not 0.0 <= self.outlier_fraction < 1.0:
            raise ValueError("outlier_fraction must be in [0, 1)")
        if self.blur_sigma_min <= 0 or self.blur_sigma_max < self.blur_sigma_min:
            raise ValueError("invalid blur sigma range")


@dataclass(frozen=True)
class SelectionConfig:
    """Controls reliability-complementarity frame selection."""

    selected_count: int = 6
    complementarity_weight: float = 0.35

    def __post_init__(self) -> None:
        if self.selected_count < 1:
            raise ValueError("selected_count must be positive")
        if self.complementarity_weight < 0:
            raise ValueError("complementarity_weight must be non-negative")


@dataclass(frozen=True)
class ReconstructionConfig:
    """Controls robust fusion and iterative back-projection."""

    scale: int = 2
    irls_iterations: int = 4
    back_projection_iterations: int = 6
    huber_delta: float = 0.035
    step_size: float = 0.22

    def __post_init__(self) -> None:
        if self.scale < 2:
            raise ValueError("scale must be at least 2")
        if self.irls_iterations < 1 or self.back_projection_iterations < 0:
            raise ValueError("iteration counts must be non-negative")
        if self.huber_delta <= 0:
            raise ValueError("huber_delta must be positive")
        if not 0 < self.step_size <= 1:
            raise ValueError("step_size must be in (0, 1]")
