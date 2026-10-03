"""Run the early synthetic MFSR pipeline using the first packaged GT image.

This script is intentionally separate from the frozen RBPA formal evaluation.
It uses a RealBSR ground-truth image only as a clean HR source for the early
synthetic degradation pipeline, then evaluates traditional MFSR outputs
against that known HR source.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from robust_mfsr import (  # noqa: E402
    ReconstructionConfig,
    SelectionConfig,
    SynthesisConfig,
    generate_synthetic_burst,
    image_metrics,
    run_pipeline,
)
from robust_mfsr.baselines import run_gate1_baselines  # noqa: E402


GT_ROOT = PROJECT_ROOT / "data" / "original" / "testpatch"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "first_gt_synthetic_mfsr"


def _read_rgb(path: Path) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _save_gray(path: Path, image: np.ndarray) -> None:
    array = np.clip(image, 0.0, 1.0)
    encoded = (array * 255.0).round().astype(np.uint8)
    cv2.imencode(".png", encoded)[1].tofile(path)


def main() -> None:
    gt_paths = sorted(GT_ROOT.glob("*/*_x4warp.png"))
    if not gt_paths:
        raise RuntimeError(f"no x4warp GT images found under {GT_ROOT}")
    gt_path = gt_paths[0]
    high_resolution = _read_rgb(gt_path)

    synthesis = SynthesisConfig(
        scale=2,
        frame_count=10,
        max_shift_lr=0.9,
        outlier_fraction=0.3,
        outlier_mode="mixed",
        seed=7,
    )
    selection = SelectionConfig(selected_count=6, complementarity_weight=0.35)
    reconstruction = ReconstructionConfig(
        scale=synthesis.scale,
        irls_iterations=4,
        back_projection_iterations=6,
        huber_delta=0.035,
        step_size=0.22,
    )

    hr, frames = generate_synthetic_burst(high_resolution, synthesis)
    result = run_pipeline(frames, selection, reconstruction)
    baselines = run_gate1_baselines(
        frames,
        result.registrations,
        selection.selected_count,
        reconstruction,
        selection.complementarity_weight,
    )

    outputs = {
        "00_input_gt_gray": hr,
        "01_reference_lr_bicubic": result.bicubic,
        "02_all_frames": result.all_frames,
        "03_selected_frames": result.selected_frames,
        "04_baseline_joint_weighted": baselines["joint_weighted"].image,
        "05_baseline_reliability_topk": baselines["reliability_topk"].image,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for stem, image in outputs.items():
        _save_gray(OUTPUT_DIR / f"{stem}.png", image)

    metric_rows = {
        name: image_metrics(hr, image)
        for name, image in outputs.items()
        if name != "00_input_gt_gray"
    }
    summary = {
        "status": "early synthetic MFSR run from first packaged GT",
        "gt_path": str(gt_path.relative_to(PROJECT_ROOT)),
        "synthesis": synthesis.__dict__,
        "selection": selection.__dict__,
        "reconstruction": reconstruction.__dict__,
        "selected_indices": result.selected_indices,
        "registrations": [
            {
                "shift_lr": estimate.shift_lr,
                "response": estimate.response,
                "residual": estimate.residual,
                "sharpness": estimate.sharpness,
                "reliability": estimate.reliability,
                "is_outlier": frames[index].is_outlier,
                "true_shift_lr": frames[index].true_shift_lr,
            }
            for index, estimate in enumerate(result.registrations)
        ],
        "metrics": metric_rows,
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"GT input: {summary['gt_path']}")
    print(f"Output:   {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")
    print(f"Selected frame indices: {result.selected_indices}")
    for name, values in metric_rows.items():
        print(
            f"{name}: PSNR={values['psnr']:.4f} dB, "
            f"SSIM={values['ssim']:.6f}"
        )


if __name__ == "__main__":
    main()
