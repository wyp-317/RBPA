"""Reliability-aware robust multi-frame super-resolution."""

from .config import ReconstructionConfig, SelectionConfig, SynthesisConfig
from .degradation import generate_synthetic_burst
from .graph_registration import (
    GraphRegistrationResult,
    estimate_graph_registrations,
)
from .metrics import image_metrics
from .mixed_variable_de import (
    MixedVariableDEConfig,
    MixedVariableDEResult,
    optimize_selection_and_weights,
)
from .pipeline import PipelineResult, run_pipeline
from .real_data import (
    RealBurst,
    center_crop_burst,
    load_realbsr_burst,
    make_observations,
    real_image_metrics,
    reconstruct_rgb,
    rgb_to_luminance,
)
from .hierarchical_optimization import (
    HierarchicalOptimizationConfig,
    HierarchicalOptimizationResult,
    optimize_hierarchical_reliability,
)
from .spatial_optimization import (
    SpatialOptimizationConfig,
    SpatialOptimizationResult,
    optimize_spatial_reliability,
)
from .uncertainty_reconstruction import (
    UncertaintyReconstructionConfig,
    UncertaintyReconstructionResult,
    reconstruct_uncertainty_aware,
    spatial_reliability_maps,
)

__all__ = [
    "PipelineResult",
    "GraphRegistrationResult",
    "HierarchicalOptimizationConfig",
    "HierarchicalOptimizationResult",
    "MixedVariableDEConfig",
    "MixedVariableDEResult",
    "ReconstructionConfig",
    "RealBurst",
    "SelectionConfig",
    "SynthesisConfig",
    "SpatialOptimizationConfig",
    "SpatialOptimizationResult",
    "UncertaintyReconstructionConfig",
    "UncertaintyReconstructionResult",
    "generate_synthetic_burst",
    "center_crop_burst",
    "estimate_graph_registrations",
    "image_metrics",
    "load_realbsr_burst",
    "make_observations",
    "optimize_hierarchical_reliability",
    "optimize_selection_and_weights",
    "optimize_spatial_reliability",
    "reconstruct_uncertainty_aware",
    "real_image_metrics",
    "reconstruct_rgb",
    "rgb_to_luminance",
    "run_pipeline",
    "spatial_reliability_maps",
]
