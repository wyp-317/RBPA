"""End-to-end orchestration for the baseline method and controls."""

from dataclasses import dataclass

from .config import ReconstructionConfig, SelectionConfig
from .models import FloatImage, FrameObservation, RegistrationEstimate
from .reconstruction import bicubic_reference, reconstruct
from .registration import estimate_registrations
from .selection import select_frames


@dataclass(frozen=True)
class PipelineResult:
    registrations: list[RegistrationEstimate]
    selected_indices: list[int]
    bicubic: FloatImage
    all_frames: FloatImage
    selected_frames: FloatImage


def run_pipeline(
    frames: list[FrameObservation],
    selection: SelectionConfig = SelectionConfig(),
    reconstruction: ReconstructionConfig = ReconstructionConfig(),
) -> PipelineResult:
    registrations = estimate_registrations(frames)
    selected_indices = select_frames(registrations, selection)
    return PipelineResult(
        registrations=registrations,
        selected_indices=selected_indices,
        bicubic=bicubic_reference(frames[0].image, reconstruction.scale),
        all_frames=reconstruct(
            frames,
            registrations,
            list(range(len(frames))),
            reconstruction,
        ),
        selected_frames=reconstruct(
            frames,
            registrations,
            selected_indices,
            reconstruction,
        ),
    )

