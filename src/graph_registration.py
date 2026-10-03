"""Globally consistent translation estimates and self-consistency diagnostics."""

from dataclasses import dataclass, replace

import cv2
import numpy as np
from numpy.typing import NDArray

from .models import FrameObservation, RegistrationEstimate
from .registration import estimate_registrations


@dataclass(frozen=True)
class GraphRegistrationResult:
    estimates: list[RegistrationEstimate]
    direct_estimates: list[RegistrationEstimate]
    self_consistency_risk: NDArray[np.float32]
    marginal_uncertainty: NDArray[np.float32]
    pairwise_residuals: NDArray[np.float32]


def _pairwise_measurements(
    frames: list[FrameObservation],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    count = len(frames)
    shape = frames[0].image.shape
    window = cv2.createHanningWindow((shape[1], shape[0]), cv2.CV_32F)
    pairs = []
    shifts = []
    responses = []
    for left in range(count):
        for right in range(left + 1, count):
            shift, response = cv2.phaseCorrelate(
                frames[left].image.copy(),
                frames[right].image.copy(),
                window,
            )
            pairs.append((left, right))
            shifts.append((float(shift[0]), float(shift[1])))
            responses.append(float(np.clip(response, 0.0, 1.0)))
    return (
        np.asarray(pairs, dtype=np.int64),
        np.asarray(shifts, dtype=np.float64),
        np.asarray(responses, dtype=np.float64),
    )


def _design_matrix(
    pairs: NDArray[np.int64],
    frame_count: int,
) -> NDArray[np.float64]:
    matrix = np.zeros((len(pairs), frame_count - 1), dtype=np.float64)
    for row, (left, right) in enumerate(pairs):
        if left:
            matrix[row, left - 1] = -1.0
        if right:
            matrix[row, right - 1] = 1.0
    return matrix


def _solve_weighted(
    matrix: NDArray[np.float64],
    observations: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    normal_matrix = matrix.T @ (weights[:, None] * matrix)
    normal_vector = matrix.T @ (weights * observations)
    # The translation graph is tiny and anchored at frame zero.  A small ridge
    # keeps weak-response graphs numerically stable and avoids a Windows
    # NumPy/MKL lstsq abort observed when PyTorch is loaded in the same process.
    ridge = 1e-8 * max(float(np.trace(normal_matrix)), 1.0)
    return np.linalg.solve(
        normal_matrix + ridge * np.eye(normal_matrix.shape[0]),
        normal_vector,
    )


def estimate_graph_registrations(
    frames: list[FrameObservation],
    *,
    iterations: int = 6,
    huber_delta: float = 0.12,
) -> GraphRegistrationResult:
    """Synchronize all pairwise translations with robust graph least squares.

    The returned risk is an internal diagnostic: it uses only disagreement
    among pairwise measurements and never accesses synthetic ground truth.
    """

    if len(frames) < 2:
        raise ValueError("at least two frames are required")
    shape = frames[0].image.shape
    if any(frame.image.shape != shape for frame in frames):
        raise ValueError("all frames must share a shape")
    if iterations < 1 or huber_delta <= 0:
        raise ValueError("invalid robust synchronization settings")

    direct = estimate_registrations(frames)
    pairs, pairwise_shifts, responses = _pairwise_measurements(frames)
    matrix = _design_matrix(pairs, len(frames))
    base_weights = np.maximum(responses, 0.03) ** 2
    weights = base_weights.copy()
    solution = np.zeros((len(frames) - 1, 2), dtype=np.float64)

    for _ in range(iterations):
        solution[:, 0] = _solve_weighted(
            matrix,
            pairwise_shifts[:, 0],
            weights,
        )
        solution[:, 1] = _solve_weighted(
            matrix,
            pairwise_shifts[:, 1],
            weights,
        )
        residual_vectors = matrix @ solution - pairwise_shifts
        residuals = np.linalg.norm(residual_vectors, axis=1)
        robust_weights = np.minimum(
            1.0,
            huber_delta / np.maximum(residuals, 1e-8),
        )
        weights = base_weights * robust_weights

    translations = np.vstack([np.zeros((1, 2)), solution])
    residual_vectors = matrix @ solution - pairwise_shifts
    residuals = np.linalg.norm(residual_vectors, axis=1)
    information = matrix.T @ (weights[:, None] * matrix)
    information_ridge = 1e-8 * max(float(np.trace(information)), 1.0)
    covariance_shape = np.linalg.solve(
        information + information_ridge * np.eye(information.shape[0]),
        np.eye(information.shape[0]),
    )
    robust_scale = max(
        1.4826 * float(np.median(np.abs(residuals - np.median(residuals)))),
        1e-3,
    )
    marginal_uncertainty = np.zeros(len(frames), dtype=np.float64)
    marginal_uncertainty[1:] = np.sqrt(
        2.0 * robust_scale**2 * np.maximum(np.diag(covariance_shape), 0.0)
    )

    incident_weight = np.zeros(len(frames), dtype=np.float64)
    incident_residual = np.zeros(len(frames), dtype=np.float64)
    for edge, (left, right) in enumerate(pairs):
        edge_weight = max(responses[edge], 0.03)
        incident_weight[left] += edge_weight
        incident_weight[right] += edge_weight
        incident_residual[left] += edge_weight * residuals[edge]
        incident_residual[right] += edge_weight * residuals[edge]
    consistency = incident_residual / np.maximum(incident_weight, 1e-8)
    # A frame can look consistent merely because all of its incident edges are
    # weak.  The marginal term penalizes that poorly constrained case.
    self_consistency_risk = consistency + marginal_uncertainty
    self_consistency_risk[0] = 0.0

    graph_estimates = [
        replace(
            estimate,
            shift_lr=(
                float(translations[index, 0]),
                float(translations[index, 1]),
            ),
        )
        for index, estimate in enumerate(direct)
    ]
    return GraphRegistrationResult(
        estimates=graph_estimates,
        direct_estimates=direct,
        self_consistency_risk=self_consistency_risk.astype(np.float32),
        marginal_uncertainty=marginal_uncertainty.astype(np.float32),
        pairwise_residuals=residuals.astype(np.float32),
    )
