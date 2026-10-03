"""Mixed-variable differential evolution for robust MFSR observation design."""

from dataclasses import dataclass

import cv2
import numpy as np

from .config import ReconstructionConfig
from .models import FrameObservation, RegistrationEstimate
from .registration import align_to_reference
from .reconstruction import reconstruct
from .uncertainty_reconstruction import (
    UncertaintyReconstructionConfig,
    reconstruct_uncertainty_aware,
)


@dataclass(frozen=True)
class MixedVariableDEConfig:
    """Controls the budget-matched outer optimizer."""

    selected_count: int = 6
    population_size: int = 20
    generations: int = 25
    mutation_factor: float = 0.7
    crossover_rate: float = 0.8
    minimum_weight: float = 0.05
    scale: int = 2
    inner_irls_iterations: int = 2
    optimize_registration: bool = False
    max_shift_correction_lr: float = 0.35
    uncertainty_adaptive_registration: bool = False
    minimum_shift_correction_lr: float = 0.02
    uncertainty_search_margin_lr: float = 0.03
    uncertainty_search_gain: float = 1.25
    registration_penalty: float = 0.002
    uncertainty_aware_inner: bool = False
    inner_spatial_iterations: int = 2
    seed: int = 19

    def __post_init__(self) -> None:
        if self.selected_count < 2:
            raise ValueError("selected_count must be at least two")
        if self.population_size < 4:
            raise ValueError("population_size must be at least four")
        if self.generations < 1:
            raise ValueError("generations must be positive")
        if not 0 < self.mutation_factor <= 2:
            raise ValueError("mutation_factor must be in (0, 2]")
        if not 0 <= self.crossover_rate <= 1:
            raise ValueError("crossover_rate must be in [0, 1]")
        if not 0 < self.minimum_weight < 1:
            raise ValueError("minimum_weight must be in (0, 1)")
        if self.scale < 2 or self.inner_irls_iterations < 1:
            raise ValueError("invalid inner reconstruction settings")
        if self.max_shift_correction_lr <= 0 or self.registration_penalty < 0:
            raise ValueError("invalid registration optimization settings")
        if not 0 <= self.minimum_shift_correction_lr <= self.max_shift_correction_lr:
            raise ValueError("invalid minimum registration search radius")
        if self.uncertainty_search_margin_lr < 0:
            raise ValueError("uncertainty_search_margin_lr must be non-negative")
        if self.uncertainty_search_gain <= 0:
            raise ValueError("uncertainty_search_gain must be positive")
        if self.inner_spatial_iterations < 1:
            raise ValueError("inner_spatial_iterations must be positive")


@dataclass(frozen=True)
class MixedVariableDEResult:
    selected_indices: list[int]
    frame_weights: list[float]
    objective: float
    objective_history: list[float]
    evaluations: int
    shift_corrections_lr: list[tuple[float, float]]
    registration_uncertainty: list[float]
    registration_search_bounds_lr: list[float]
    residual_registration_priors_lr: list[tuple[float, float]]


class MixedVariableDifferentialEvolution:
    """Jointly optimize binary frame selection and continuous frame weights."""

    def __init__(
        self,
        frames: list[FrameObservation],
        estimates: list[RegistrationEstimate],
        config: MixedVariableDEConfig = MixedVariableDEConfig(),
    ) -> None:
        if len(frames) != len(estimates):
            raise ValueError("frames and estimates must have the same length")
        if config.selected_count > len(frames):
            raise ValueError("selected_count exceeds the burst size")
        if len(frames) < 2:
            raise ValueError("at least two frames are required")
        self.frames = frames
        self.estimates = estimates
        self.config = config
        self.frame_count = len(frames)
        self.aligned = np.stack(
            [
                align_to_reference(frame.image, estimate.shift_lr)
                for frame, estimate in zip(frames, estimates, strict=True)
            ]
        ).astype(np.float32)
        self.reliability = np.asarray(
            [estimate.reliability for estimate in estimates],
            dtype=np.float64,
        )
        self.reliability[0] = 1.0
        (
            self.residual_registration_priors,
            self.registration_uncertainty,
            self.registration_search_bounds,
        ) = self._registration_search_model()
        reference_hr = cv2.resize(
            self.frames[0].image,
            (
                self.frames[0].image.shape[1] * self.config.scale,
                self.frames[0].image.shape[0] * self.config.scale,
            ),
            interpolation=cv2.INTER_CUBIC,
        )
        reference_laplacian = cv2.Laplacian(
            reference_hr,
            cv2.CV_32F,
        )
        self.reference_sharpness = max(
            float(np.var(reference_laplacian)),
            1e-8,
        )

    def _registration_search_model(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        priors = np.zeros((self.frame_count, 2), dtype=np.float64)
        uncertainty = np.zeros(self.frame_count, dtype=np.float64)
        if not self.config.optimize_registration:
            return priors, uncertainty, uncertainty.copy()
        if not self.config.uncertainty_adaptive_registration:
            bounds = np.full(
                self.frame_count,
                self.config.max_shift_correction_lr,
                dtype=np.float64,
            )
            bounds[0] = 0.0
            return priors, uncertainty, bounds

        reference = self.frames[0].image
        window = cv2.createHanningWindow(
            (reference.shape[1], reference.shape[0]),
            cv2.CV_32F,
        )
        for index in range(1, self.frame_count):
            residual_shift, response = cv2.phaseCorrelate(
                reference.copy(),
                self.aligned[index].copy(),
                window,
            )
            prior = np.clip(
                np.asarray(residual_shift, dtype=np.float64),
                -self.config.max_shift_correction_lr,
                self.config.max_shift_correction_lr,
            )
            priors[index] = prior
            residual_motion = float(np.max(np.abs(prior)))
            confidence = float(np.clip(response, 0.0, 1.0))
            uncertainty[index] = np.clip(
                residual_motion
                / self.config.max_shift_correction_lr
                * (0.5 + 0.5 * confidence),
                0.0,
                1.0,
            )

        bounds = np.clip(
            self.config.minimum_shift_correction_lr
            + self.config.uncertainty_search_margin_lr
            + self.config.uncertainty_search_gain
            * np.max(np.abs(priors), axis=1),
            self.config.minimum_shift_correction_lr,
            self.config.max_shift_correction_lr,
        )
        bounds[0] = 0.0
        return priors, uncertainty, bounds

    def _decode(
        self,
        vector: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        selection_scores = vector[: self.frame_count].copy()
        selection_scores[0] = np.inf
        selected = np.argpartition(
            selection_scores,
            -self.config.selected_count,
        )[-self.config.selected_count :]
        selected = np.sort(selected)
        raw_weights = vector[self.frame_count :][selected]
        weights = self.config.minimum_weight + (
            1.0 - self.config.minimum_weight
        ) * raw_weights
        reference_position = np.flatnonzero(selected == 0)
        weights[reference_position] = np.maximum(
            weights[reference_position],
            0.5,
        )
        corrections = np.zeros((self.frame_count, 2), dtype=np.float64)
        if self.config.optimize_registration:
            start = 2 * self.frame_count
            corrections[:, 0] = (
                2.0 * vector[start : start + self.frame_count] - 1.0
            ) * self.registration_search_bounds
            corrections[:, 1] = (
                2.0 * vector[start + self.frame_count :] - 1.0
            ) * self.registration_search_bounds
            corrections[0] = 0.0
        return selected, weights, corrections

    def _phase_penalty(self, selected: np.ndarray) -> float:
        phases = np.asarray(
            [
                (
                    self.estimates[index].shift_lr[0] % 1.0,
                    self.estimates[index].shift_lr[1] % 1.0,
                )
                for index in selected
            ],
            dtype=np.float64,
        )
        phase_bins = np.floor(phases * self.config.scale).astype(np.int64)
        phase_bins = np.clip(
            phase_bins,
            0,
            self.config.scale - 1,
        )
        occupied = len({(int(x), int(y)) for x, y in phase_bins})
        target_occupancy = min(
            self.config.selected_count,
            self.config.scale**2,
        )
        return (target_occupancy - occupied) / target_occupancy

    def _corrected_estimates(
        self,
        corrections: np.ndarray,
    ) -> list[RegistrationEstimate]:
        return [
            RegistrationEstimate(
                shift_lr=(
                    estimate.shift_lr[0] + corrections[index, 0],
                    estimate.shift_lr[1] + corrections[index, 1],
                ),
                response=estimate.response,
                residual=estimate.residual,
                sharpness=estimate.sharpness,
                reliability=estimate.reliability,
            )
            for index, estimate in enumerate(self.estimates)
        ]

    def _forward_residual(
        self,
        reconstruction: np.ndarray,
        estimates: list[RegistrationEstimate],
    ) -> np.ndarray:
        lr_height, lr_width = self.frames[0].image.shape
        hr_height, hr_width = reconstruction.shape
        residuals = []
        for frame, estimate in zip(
            self.frames,
            estimates,
            strict=True,
        ):
            dx, dy = estimate.shift_lr
            matrix = np.asarray(
                [
                    [1.0, 0.0, dx * self.config.scale],
                    [0.0, 1.0, dy * self.config.scale],
                ],
                dtype=np.float32,
            )
            predicted_hr = cv2.warpAffine(
                reconstruction,
                matrix,
                (hr_width, hr_height),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REFLECT101,
            )
            predicted_lr = cv2.resize(
                predicted_hr,
                (lr_width, lr_height),
                interpolation=cv2.INTER_AREA,
            )
            residuals.append(
                float(
                    np.sqrt(
                        (frame.image - predicted_lr) ** 2 + 1e-6
                    ).mean()
                )
            )
        return np.asarray(residuals, dtype=np.float64)

    def objective(self, vector: np.ndarray) -> float:
        selected, weights, corrections = self._decode(vector)
        corrected_estimates = self._corrected_estimates(corrections)
        normalized_weights = weights / max(float(weights.sum()), 1e-8)
        inner_config = ReconstructionConfig(
            scale=self.config.scale,
            irls_iterations=self.config.inner_irls_iterations,
            back_projection_iterations=0,
        )
        if self.config.uncertainty_aware_inner:
            candidate = reconstruct_uncertainty_aware(
                self.frames,
                corrected_estimates,
                selected.tolist(),
                weights,
                self.registration_uncertainty,
                UncertaintyReconstructionConfig(
                    scale=self.config.scale,
                    iterations=self.config.inner_spatial_iterations,
                ),
                inner_config,
            ).image
        else:
            candidate = reconstruct(
                self.frames,
                corrected_estimates,
                selected.tolist(),
                inner_config,
                frame_weights=weights,
                robust=True,
            )
        residual = self._forward_residual(candidate, corrected_estimates)
        selected_loss = float(
            np.sum(normalized_weights * residual[selected])
        )
        validation_loss = float(
            np.sum(self.reliability * residual)
            / max(float(self.reliability.sum()), 1e-8)
        )
        reliability_risk = float(
            np.sum(normalized_weights * (1.0 - self.reliability[selected]))
        )
        effective_count = float(
            weights.sum() ** 2 / max(float(np.sum(weights**2)), 1e-8)
        )
        count_penalty = max(
            0.0,
            0.65 * self.config.selected_count - effective_count,
        ) / self.config.selected_count
        consensus_sharpness = float(
            np.var(cv2.Laplacian(candidate, cv2.CV_32F))
        )
        detail_penalty = max(
            0.0,
            1.0 - consensus_sharpness / self.reference_sharpness,
        )
        normalized_correction = (
            corrections[selected] / self.config.max_shift_correction_lr
        )
        registration_penalty = float(
            np.mean(np.sum(normalized_correction**2, axis=1))
        )
        return (
            0.50 * validation_loss
            + 0.30 * selected_loss
            + 0.018 * reliability_risk
            + 0.012 * count_penalty
            + 0.004 * detail_penalty
            + 0.003 * self._phase_penalty(selected)
            + self.config.registration_penalty * registration_penalty
        )

    def run(self) -> MixedVariableDEResult:
        rng = np.random.default_rng(self.config.seed)
        dimension = (
            4 * self.frame_count
            if self.config.optimize_registration
            else 2 * self.frame_count
        )
        population = rng.uniform(
            0.0,
            1.0,
            size=(self.config.population_size, dimension),
        )
        # Seed with reliability Top-K so the optimizer must improve on it.
        population[0, : self.frame_count] = self.reliability
        population[0, self.frame_count : 2 * self.frame_count] = np.clip(
            (self.reliability - self.config.minimum_weight)
            / (1.0 - self.config.minimum_weight),
            0.0,
            1.0,
        )
        if self.config.optimize_registration:
            population[0, 2 * self.frame_count :] = 0.5
            if (
                self.config.uncertainty_adaptive_registration
                and self.config.population_size > 1
            ):
                population[1] = population[0]
                safe_bounds = np.maximum(
                    self.registration_search_bounds,
                    1e-8,
                )
                population[1, 2 * self.frame_count : 3 * self.frame_count] = (
                    0.5
                    + 0.5
                    * self.residual_registration_priors[:, 0]
                    / safe_bounds
                )
                population[1, 3 * self.frame_count :] = (
                    0.5
                    + 0.5
                    * self.residual_registration_priors[:, 1]
                    / safe_bounds
                )
                population[1, 2 * self.frame_count :] = np.clip(
                    population[1, 2 * self.frame_count :],
                    0.0,
                    1.0,
                )
                population[1, 2 * self.frame_count] = 0.5
                population[1, 3 * self.frame_count] = 0.5
        objectives = np.asarray(
            [self.objective(individual) for individual in population],
            dtype=np.float64,
        )
        evaluations = self.config.population_size
        history = [float(objectives.min())]

        for _ in range(self.config.generations):
            for target_index in range(self.config.population_size):
                candidates = np.delete(
                    np.arange(self.config.population_size),
                    target_index,
                )
                first, second, third = rng.choice(
                    candidates,
                    size=3,
                    replace=False,
                )
                mutant = population[first] + self.config.mutation_factor * (
                    population[second] - population[third]
                )
                mutant = np.clip(mutant, 0.0, 1.0)
                crossover = (
                    rng.random(dimension) < self.config.crossover_rate
                )
                crossover[rng.integers(0, dimension)] = True
                trial = np.where(crossover, mutant, population[target_index])
                trial_objective = self.objective(trial)
                evaluations += 1
                if trial_objective <= objectives[target_index]:
                    population[target_index] = trial
                    objectives[target_index] = trial_objective
            history.append(float(objectives.min()))

        best = int(np.argmin(objectives))
        selected, weights, corrections = self._decode(population[best])
        return MixedVariableDEResult(
            selected_indices=selected.tolist(),
            frame_weights=weights.astype(float).tolist(),
            objective=float(objectives[best]),
            objective_history=history,
            evaluations=evaluations,
            shift_corrections_lr=[
                (float(dx), float(dy)) for dx, dy in corrections
            ],
            registration_uncertainty=self.registration_uncertainty.astype(
                float
            ).tolist(),
            registration_search_bounds_lr=self.registration_search_bounds.astype(
                float
            ).tolist(),
            residual_registration_priors_lr=[
                (float(dx), float(dy))
                for dx, dy in self.residual_registration_priors
            ],
        )


def optimize_selection_and_weights(
    frames: list[FrameObservation],
    estimates: list[RegistrationEstimate],
    config: MixedVariableDEConfig = MixedVariableDEConfig(),
) -> MixedVariableDEResult:
    return MixedVariableDifferentialEvolution(
        frames,
        estimates,
        config,
    ).run()
