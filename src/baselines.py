"""Budget-matched Gate 1 baselines with a shared reconstruction backend."""

from dataclasses import dataclass

import numpy as np

from .config import ReconstructionConfig, SelectionConfig
from .models import FloatImage, FrameObservation, RegistrationEstimate
from .reconstruction import bicubic_reference, reconstruct
from .selection import (
    select_complementary_frames,
    select_frames,
    select_top_reliability,
    select_top_sharpness,
)


@dataclass(frozen=True)
class BaselineOutput:
    image: FloatImage
    selected_indices: list[int]


def run_gate1_baselines(
    frames: list[FrameObservation],
    estimates: list[RegistrationEstimate],
    selected_count: int,
    reconstruction: ReconstructionConfig,
    complementarity_weight: float = 0.35,
) -> dict[str, BaselineOutput]:
    """Run selection ablations while holding registration and recovery fixed."""

    all_indices = list(range(len(frames)))
    equal_all = np.ones(len(all_indices), dtype=np.float32)
    reliability_all = [estimate.reliability for estimate in estimates]

    sharpness_indices = select_top_sharpness(estimates, selected_count)
    reliability_indices = select_top_reliability(estimates, selected_count)
    complementary_indices = select_complementary_frames(estimates, selected_count)
    joint_indices = select_frames(
        estimates,
        SelectionConfig(
            selected_count=selected_count,
            complementarity_weight=complementarity_weight,
        ),
    )
    equal_selected = np.ones(len(joint_indices), dtype=np.float32)
    joint_weights = [estimates[index].reliability for index in joint_indices]
    reliability_weights = [
        estimates[index].reliability for index in reliability_indices
    ]

    def selected_reconstruction(indices: list[int]) -> FloatImage:
        return reconstruct(
            frames,
            estimates,
            indices,
            reconstruction,
            frame_weights=np.ones(len(indices), dtype=np.float32),
            robust=True,
        )

    return {
        "bicubic": BaselineOutput(
            image=bicubic_reference(frames[0].image, reconstruction.scale),
            selected_indices=[0],
        ),
        "all_equal": BaselineOutput(
            image=reconstruct(
                frames,
                estimates,
                all_indices,
                reconstruction,
                frame_weights=equal_all,
                robust=False,
            ),
            selected_indices=all_indices,
        ),
        "all_robust": BaselineOutput(
            image=reconstruct(
                frames,
                estimates,
                all_indices,
                reconstruction,
                frame_weights=equal_all,
                robust=True,
            ),
            selected_indices=all_indices,
        ),
        "reliability_soft_all": BaselineOutput(
            image=reconstruct(
                frames,
                estimates,
                all_indices,
                reconstruction,
                frame_weights=reliability_all,
                robust=True,
            ),
            selected_indices=all_indices,
        ),
        "sharpness_topk": BaselineOutput(
            image=selected_reconstruction(sharpness_indices),
            selected_indices=sharpness_indices,
        ),
        "reliability_topk": BaselineOutput(
            image=selected_reconstruction(reliability_indices),
            selected_indices=reliability_indices,
        ),
        "reliability_weighted": BaselineOutput(
            image=reconstruct(
                frames,
                estimates,
                reliability_indices,
                reconstruction,
                frame_weights=reliability_weights,
                robust=True,
            ),
            selected_indices=reliability_indices,
        ),
        "complementarity_only": BaselineOutput(
            image=selected_reconstruction(complementary_indices),
            selected_indices=complementary_indices,
        ),
        "joint_equal": BaselineOutput(
            image=reconstruct(
                frames,
                estimates,
                joint_indices,
                reconstruction,
                frame_weights=equal_selected,
                robust=True,
            ),
            selected_indices=joint_indices,
        ),
        "joint_weighted": BaselineOutput(
            image=reconstruct(
                frames,
                estimates,
                joint_indices,
                reconstruction,
                frame_weights=joint_weights,
                robust=True,
            ),
            selected_indices=joint_indices,
        ),
    }
