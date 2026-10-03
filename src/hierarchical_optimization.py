"""Hierarchical half-quadratic optimization of frame and pixel reliability."""

from dataclasses import dataclass

import numpy as np
from skimage.restoration import denoise_tv_chambolle

from .models import FloatImage, FrameObservation, RegistrationEstimate
from .spatial_optimization import aligned_candidates


@dataclass(frozen=True)
class HierarchicalOptimizationConfig:
    scale: int = 2
    iterations: int = 8
    pixel_barrier_strength: float = 1e-3
    frame_barrier_strength: float = 1e-3
    frame_prior_strength: float = 5e-3
    image_tv_strength: float = 0.002
    minimum_weight: float = 1e-3
    update_damping: float = 0.8
    line_search_steps: int = 8


@dataclass(frozen=True)
class HierarchicalOptimizationResult:
    image: FloatImage
    frame_weights: np.ndarray
    pixel_weights: np.ndarray
    objective_history: list[float]

    @property
    def combined_weights(self) -> np.ndarray:
        return self.frame_weights[:, None, None] * self.pixel_weights


def _tv_mean(image: np.ndarray) -> float:
    return float(
        0.5
        * (
            np.mean(np.abs(np.diff(image, axis=-1)))
            + np.mean(np.abs(np.diff(image, axis=-2)))
        )
    )


def hierarchical_objective(
    image: FloatImage,
    frame_weights: np.ndarray,
    pixel_weights: np.ndarray,
    candidates: np.ndarray,
    frame_priors: np.ndarray,
    config: HierarchicalOptimizationConfig,
) -> float:
    residual_squared = (candidates - image[None, ...]) ** 2
    safe_frames = np.clip(
        frame_weights,
        config.minimum_weight,
        1.0,
    )
    safe_pixels = np.clip(
        pixel_weights,
        config.minimum_weight,
        1.0,
    )
    combined = safe_frames[:, None, None] * safe_pixels
    data_term = float(np.mean(combined * residual_squared))
    pixel_barrier = config.pixel_barrier_strength * float(
        np.mean(safe_pixels - np.log(safe_pixels) - 1.0)
    )
    frame_barrier = config.frame_barrier_strength * float(
        np.mean(safe_frames - np.log(safe_frames) - 1.0)
    )
    frame_prior = config.frame_prior_strength * float(
        np.mean((safe_frames - frame_priors) ** 2)
    )
    image_tv = config.image_tv_strength * _tv_mean(image)
    return data_term + pixel_barrier + frame_barrier + frame_prior + image_tv


def _frame_weight_update(
    residual_squared: np.ndarray,
    frame_priors: np.ndarray,
    config: HierarchicalOptimizationConfig,
) -> np.ndarray:
    aggregate_residual = np.mean(residual_squared, axis=(1, 2))
    barrier = config.frame_barrier_strength
    prior = config.frame_prior_strength
    if prior == 0:
        return barrier / (barrier + aggregate_residual)
    coefficient = (
        aggregate_residual + barrier - 2.0 * prior * frame_priors
    )
    discriminant = coefficient**2 + 8.0 * prior * barrier
    return (
        -coefficient + np.sqrt(np.maximum(discriminant, 0.0))
    ) / (4.0 * prior)


def _propose_update(
    image: FloatImage,
    candidates: np.ndarray,
    frame_priors: np.ndarray,
    config: HierarchicalOptimizationConfig,
) -> tuple[FloatImage, np.ndarray, np.ndarray]:
    residual_squared = (candidates - image[None, ...]) ** 2
    frame_weights = _frame_weight_update(
        residual_squared,
        frame_priors,
        config,
    )
    pixel_weights = config.pixel_barrier_strength / (
        config.pixel_barrier_strength
        + frame_weights[:, None, None] * residual_squared
    )
    frame_weights = np.clip(
        frame_weights,
        config.minimum_weight,
        1.0,
    ).astype(np.float32)
    pixel_weights = np.clip(
        pixel_weights,
        config.minimum_weight,
        1.0,
    ).astype(np.float32)
    combined = frame_weights[:, None, None] * pixel_weights
    fused = np.sum(combined * candidates, axis=0) / np.maximum(
        np.sum(combined, axis=0),
        config.minimum_weight,
    )
    if config.image_tv_strength > 0:
        fused = denoise_tv_chambolle(
            fused,
            weight=config.image_tv_strength,
            channel_axis=None,
        )
    return (
        np.clip(fused, 0.0, 1.0).astype(np.float32),
        frame_weights,
        pixel_weights,
    )


def optimize_hierarchical_reliability(
    frames: list[FrameObservation],
    estimates: list[RegistrationEstimate],
    config: HierarchicalOptimizationConfig = HierarchicalOptimizationConfig(),
) -> HierarchicalOptimizationResult:
    if not frames:
        raise ValueError("at least one frame is required")
    candidates = aligned_candidates(frames, estimates, config.scale)
    priors = np.clip(
        np.asarray(
            [estimate.reliability for estimate in estimates],
            dtype=np.float32,
        ),
        config.minimum_weight,
        1.0,
    )
    frame_weights = priors.copy()
    pixel_weights = np.ones_like(candidates, dtype=np.float32)
    combined = frame_weights[:, None, None] * pixel_weights
    image = (
        np.sum(combined * candidates, axis=0)
        / np.maximum(np.sum(combined, axis=0), config.minimum_weight)
    ).astype(np.float32)
    history = [
        hierarchical_objective(
            image,
            frame_weights,
            pixel_weights,
            candidates,
            priors,
            config,
        )
    ]

    for _ in range(config.iterations):
        proposed_image, proposed_frames, proposed_pixels = _propose_update(
            image,
            candidates,
            priors,
            config,
        )
        accepted = False
        for line_search_index in range(config.line_search_steps):
            damping = config.update_damping * (0.5**line_search_index)
            candidate_image = (
                (1.0 - damping) * image + damping * proposed_image
            ).astype(np.float32)
            candidate_frames = (
                (1.0 - damping) * frame_weights
                + damping * proposed_frames
            ).astype(np.float32)
            candidate_pixels = (
                (1.0 - damping) * pixel_weights
                + damping * proposed_pixels
            ).astype(np.float32)
            objective = hierarchical_objective(
                candidate_image,
                candidate_frames,
                candidate_pixels,
                candidates,
                priors,
                config,
            )
            if objective <= history[-1] + 1e-10:
                image = candidate_image
                frame_weights = candidate_frames
                pixel_weights = candidate_pixels
                history.append(objective)
                accepted = True
                break
        if not accepted:
            history.append(history[-1])
            break

    return HierarchicalOptimizationResult(
        image=np.clip(image, 0.0, 1.0).astype(np.float32),
        frame_weights=np.clip(
            frame_weights,
            config.minimum_weight,
            1.0,
        ).astype(np.float32),
        pixel_weights=np.clip(
            pixel_weights,
            config.minimum_weight,
            1.0,
        ).astype(np.float32),
        objective_history=history,
    )
