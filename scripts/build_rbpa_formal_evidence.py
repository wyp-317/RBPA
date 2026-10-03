"""Build quantitative and qualitative evidence for the RBPA formal test."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
SCRIPT_ROOT = Path(__file__).resolve().parent
for root in (SOURCE_ROOT, SCRIPT_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from robust_mfsr import real_image_metrics  # noqa: E402
from robust_mfsr.learned import (  # noqa: E402
    calibrated_priors,
    hierarchical_spatial_priors,
    load_reliable_sbfburst,
    risk_budgeted_precision_priors,
    spatial_anomaly_maps,
)
from run_hierarchical_spatial_screen import _load_frame_parameters  # noqa: E402
from run_sbfburst_calibrated_de import (  # noqa: E402
    _output_numpy,
    _prepare_case,
)


CONDITIONS = (
    ("mixed", 0.2, "Mix 20"),
    ("mixed", 0.4, "Mix 40"),
    ("mixed", 0.6, "Mix 60"),
    ("occlusion", 0.2, "Occ 20"),
    ("occlusion", 0.4, "Occ 40"),
    ("occlusion", 0.6, "Occ 60"),
)
COLORS = {
    "scalar": "#C55A11",
    "cubic": "#4472C4",
    "official": "#7F8C8D",
    "topk": "#6B8E23",
    "rbpa": "#8E5AA7",
}


def _save(figure: plt.Figure, output: Path, stem: str) -> None:
    for suffix in ("png", "pdf"):
        figure.savefig(
            output / f"{stem}.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)


def _condition_figure(summary: dict, output: Path) -> None:
    scalar = {
        (row["protocol"], row["fraction"]): row
        for row in summary["condition_vs_scalar"]
    }
    cubic = {
        (row["protocol"], row["fraction"]): row
        for row in summary["condition_vs_cubic"]
    }
    x = np.arange(len(CONDITIONS))
    figure, axis = plt.subplots(figsize=(9.2, 4.2))
    for offset, lookup, name in (
        (-0.10, scalar, "vs scalar"),
        (0.10, cubic, "vs cubic"),
    ):
        rows = [lookup[(protocol, fraction)] for protocol, fraction, _ in CONDITIONS]
        values = np.asarray([row["mean_delta"] for row in rows])
        errors = np.asarray(
            [
                [
                    value - row["ci95"][0] for value, row in zip(values, rows)
                ],
                [
                    row["ci95"][1] - value for value, row in zip(values, rows)
                ],
            ]
        )
        axis.errorbar(
            x + offset,
            values,
            yerr=errors,
            fmt="o",
            markersize=6,
            capsize=3,
            linewidth=1.6,
            color=COLORS["scalar" if name == "vs scalar" else "cubic"],
            label=name,
        )
    axis.axhline(0.0, color="#333333", linewidth=0.8)
    axis.set_xticks(x, [label for _, _, label in CONDITIONS])
    axis.set_ylabel("RBPA PSNR gain (dB)")
    axis.set_title("Frozen 24-scene formal holdout")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    _save(figure, output, "rbpa_formal_condition_gains")


def _overall_figure(summary: dict, output: Path) -> None:
    lookup = {
        (row["baseline"], row["metric"]): row
        for row in summary["overall_comparisons"]
    }
    baselines = ("official", "topk", "scalar", "cubic")
    labels = ("Official", "Top-K", "Scalar", "Cubic")
    panels = (
        ("psnr_rgb", "PSNR gain (dB)", 1.0),
        ("ssim_rgb", "SSIM gain", 1.0),
        ("lpips", "LPIPS reduction", -1.0),
    )
    figure, axes = plt.subplots(1, 3, figsize=(11.2, 3.8))
    for axis, (metric, ylabel, direction) in zip(
        axes,
        panels,
        strict=True,
    ):
        values = [
            direction * lookup[(baseline, metric)]["mean_delta"]
            for baseline in baselines
        ]
        bars = axis.bar(
            np.arange(len(baselines)),
            values,
            color=[COLORS[baseline] for baseline in baselines],
        )
        axis.axhline(0.0, color="#333333", linewidth=0.8)
        axis.set_xticks(np.arange(len(labels)), labels, rotation=20)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.2)
        for bar, value in zip(bars, values, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:+.4g}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=7,
            )
    axes[0].set_title("(a) Reconstruction")
    axes[1].set_title("(b) Structure")
    axes[2].set_title("(c) Perceptual quality")
    figure.tight_layout()
    _save(figure, output, "rbpa_formal_overall_metrics")


def _select_visual(records: list[dict]) -> dict:
    keyed = {
        (record["case"], record["method"]): record for record in records
    }
    candidates = []
    for case in sorted({record["case"] for record in records}):
        rbpa = keyed[(case, "rbpa")]
        if rbpa["protocol"] != "occlusion" or rbpa["fraction"] != 0.6:
            continue
        delta = rbpa["psnr_rgb"] - keyed[(case, "scalar")]["psnr_rgb"]
        candidates.append((case, delta, rbpa))
    median = float(np.median([delta for _, delta, _ in candidates]))
    case, delta, record = min(
        candidates,
        key=lambda item: (abs(item[1] - median), item[0]),
    )
    return {
        "rule": (
            "Occlusion 60%; case with RBPA-vs-scalar PSNR gain closest "
            "to the 48-case median; lexical tie break."
        ),
        "condition_median_delta_db": median,
        "selected_case": case,
        "selected_delta_db": delta,
        "record": record,
    }


def _crop_location(
    target: np.ndarray,
    scalar: np.ndarray,
    rbpa: np.ndarray,
    crop_size: int,
    border: int,
) -> tuple[int, int]:
    improvement = cv2.GaussianBlur(
        np.abs(scalar - target).mean(axis=2)
        - np.abs(rbpa - target).mean(axis=2),
        (0, 0),
        4.0,
    )
    valid = improvement.copy()
    valid[:border] = -np.inf
    valid[-border:] = -np.inf
    valid[:, :border] = -np.inf
    valid[:, -border:] = -np.inf
    center_y, center_x = np.unravel_index(
        int(np.argmax(valid)),
        valid.shape,
    )
    top = int(
        np.clip(
            center_y - crop_size // 2,
            0,
            target.shape[0] - crop_size,
        )
    )
    left = int(
        np.clip(
            center_x - crop_size // 2,
            0,
            target.shape[1] - crop_size,
        )
    )
    return top, left


def _qualitative_figure(
    selection: dict,
    args,
) -> dict:
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    cubic_summary = json.loads(
        args.cubic_summary.read_text(encoding="utf-8")
    )
    frame_parameters = _load_frame_parameters(args.frame_summary)
    cubic_parameters = {
        name: float(value)
        for name, value in cubic_summary["spatial_calibration"][
            "parameters"
        ].items()
    }
    rbpa_parameters = {
        name: float(value)
        for name, value in freeze["parameters"].items()
    }
    record = selection["record"]
    model = load_reliable_sbfburst(args.sbfburst_root)
    prepared = _prepare_case(
        model,
        args.aligned_root,
        args.original_root,
        record["burst_name"],
        record["protocol"],
        record["fraction"],
        record["seed"],
        args.proxy_size,
    )
    scalar = calibrated_priors(
        prepared["reliabilities"],
        frame_parameters,
        selected_count=args.select,
    )[0]
    anomaly_maps = torch.from_numpy(
        spatial_anomaly_maps(
            prepared["aligned_frames_rgb"],
            smoothing_sigma=args.smoothing_sigma,
        )
    ).cuda()
    cubic_priors = hierarchical_spatial_priors(
        scalar,
        anomaly_maps,
        cubic_parameters,
    )
    rbpa_priors = risk_budgeted_precision_priors(
        scalar,
        anomaly_maps,
        rbpa_parameters,
    )
    height, width = anomaly_maps.shape[-2:]
    scalar_priors = scalar[
        None, :, None, None
    ].expand(1, -1, height, width)
    with torch.inference_mode():
        predictions = model.decode_cached(
            prepared["cache"],
            torch.cat([scalar_priors, cubic_priors, rbpa_priors], dim=0),
        ).clamp(0.0, 1.0)
    images = {
        "target": prepared["target_numpy"],
        "scalar": _output_numpy(predictions[0:1]),
        "cubic": _output_numpy(predictions[1:2]),
        "rbpa": _output_numpy(predictions[2:3]),
    }
    top, left = _crop_location(
        images["target"],
        images["scalar"],
        images["rbpa"],
        args.crop_size,
        args.border,
    )
    methods = (
        ("target", "Ground truth"),
        ("scalar", "Scalar"),
        ("cubic", "Cubic"),
        ("rbpa", "RBPA"),
    )
    figure, axes = plt.subplots(2, 4, figsize=(11.6, 6.0))
    metric_rows = {
        method: real_image_metrics(
            images["target"],
            images[method],
            border=args.border,
        )
        for method in ("scalar", "cubic", "rbpa")
    }
    for column, (method, label) in enumerate(methods):
        metric = (
            ""
            if method == "target"
            else f"\n{metric_rows[method]['psnr_rgb']:.2f} dB"
        )
        axes[0, column].imshow(images[method])
        axes[0, column].set_title(label + metric)
        axes[0, column].axis("off")
        axes[1, column].imshow(
            images[method][
                top : top + args.crop_size,
                left : left + args.crop_size,
            ],
            interpolation="nearest",
        )
        axes[1, column].axis("off")
    figure.suptitle(
        f"Median-gain formal case: {selection['selected_case']}"
    )
    figure.tight_layout()
    _save(figure, args.evidence, "rbpa_formal_qualitative")
    return {
        **{key: value for key, value in selection.items() if key != "record"},
        "crop": {
            "top": top,
            "left": left,
            "size": args.crop_size,
        },
        "metrics": metric_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--statistics",
        type=Path,
        default=Path("outputs/rbpa_formal_holdout/statistical_summary.json"),
    )
    parser.add_argument(
        "--records",
        type=Path,
        default=Path("outputs/rbpa_formal_holdout/records.json"),
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=Path("paper/evidence"),
    )
    parser.add_argument("--aligned-root", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--sbfburst-root", type=Path, required=True)
    parser.add_argument(
        "--freeze",
        type=Path,
        default=Path("paper/RBPA_METHOD_FREEZE.json"),
    )
    parser.add_argument(
        "--frame-summary",
        type=Path,
        default=Path("outputs/calibburst_optimizer_comparison/summary.json"),
    )
    parser.add_argument(
        "--cubic-summary",
        type=Path,
        default=Path("outputs/calibburst_hierarchical_screen/summary.json"),
    )
    parser.add_argument("--proxy-size", type=int, default=64)
    parser.add_argument("--select", type=int, default=8)
    parser.add_argument("--smoothing-sigma", type=float, default=2.0)
    parser.add_argument("--border", type=int, default=40)
    parser.add_argument("--crop-size", type=int, default=128)
    args = parser.parse_args()
    summary = json.loads(args.statistics.read_text(encoding="utf-8"))
    records = json.loads(args.records.read_text(encoding="utf-8"))
    args.evidence.mkdir(parents=True, exist_ok=True)
    _condition_figure(summary, args.evidence)
    _overall_figure(summary, args.evidence)
    selection = _select_visual(records)
    visual_result = _qualitative_figure(selection, args)
    (args.evidence / "rbpa_formal_qualitative_selection.json").write_text(
        json.dumps(visual_result, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
