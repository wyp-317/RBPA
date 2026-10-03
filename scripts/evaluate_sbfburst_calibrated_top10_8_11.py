"""Evaluate newly calibrated 8-11 scalar priors on the top-10 tuning bursts."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
SCRIPT_ROOT = Path(__file__).resolve().parent
for root in (SOURCE_ROOT, SCRIPT_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from robust_mfsr import (  # noqa: E402
    center_crop_burst,
    load_realbsr_burst,
    make_observations,
    real_image_metrics,
    rgb_to_luminance,
)
from robust_mfsr.learned import calibrated_priors, load_reliable_sbfburst  # noqa: E402
from robust_mfsr.registration import estimate_registrations  # noqa: E402
from run_sbfburst_calibrated_de_8_11 import (  # noqa: E402
    _append_jsonl,
    _condition_list,
    _corrupt_burst,
    _load_names,
    _output_numpy,
    _parse_values,
    _safe_relative,
)


def _load_parameters(path: Path) -> dict[str, dict[str, float]]:
    values = json.loads(path.read_text(encoding="utf-8"))
    return {
        name: {key: float(value) for key, value in item["parameters"].items()}
        for name, item in values.items()
    }


def _summarize(records: list[dict]) -> dict:
    methods = sorted({record["method"] for record in records})
    conditions = sorted({(record["protocol"], record["fraction"]) for record in records})
    aggregates = []
    for protocol, fraction in conditions:
        for method in methods:
            subset = [
                record
                for record in records
                if record["protocol"] == protocol
                and record["fraction"] == fraction
                and record["method"] == method
            ]
            if subset:
                aggregates.append(
                    {
                        "protocol": protocol,
                        "fraction": fraction,
                        "method": method,
                        "psnr_rgb": float(np.mean([row["psnr_rgb"] for row in subset])),
                        "ssim_rgb": float(np.mean([row["ssim_rgb"] for row in subset])),
                        "psnr_y": float(np.mean([row["psnr_y"] for row in subset])),
                        "ssim_y": float(np.mean([row["ssim_y"] for row in subset])),
                        "samples": len(subset),
                    }
                )

    comparisons = []
    for target in ("a2_mean_de_8-11", "a3_robust_de_8-11"):
        for protocol, fraction in conditions:
            cases = sorted(
                {
                    record["case"]
                    for record in records
                    if record["protocol"] == protocol
                    and record["fraction"] == fraction
                }
            )
            deltas = []
            for case in cases:
                official = next(
                    record
                    for record in records
                    if record["case"] == case and record["method"] == "sbf_official"
                )
                candidate = next(
                    record
                    for record in records
                    if record["case"] == case and record["method"] == target
                )
                deltas.append(candidate["psnr_rgb"] - official["psnr_rgb"])
            comparisons.append(
                {
                    "protocol": protocol,
                    "fraction": fraction,
                    "comparison": f"{target}_vs_sbf_official",
                    "psnr_rgb_delta": float(np.mean(deltas)),
                    "win_rate": float(np.mean([delta > 0.0 for delta in deltas])),
                    "samples": len(deltas),
                }
            )
    return {"aggregates": aggregates, "comparisons": comparisons}


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tuning-root", type=Path, default=Path("\u8c03\u53c2-data"))
    parser.add_argument("--sbfburst-root", type=Path, default=Path("external/SBFBurst"))
    parser.add_argument(
        "--parameters",
        type=Path,
        default=Path("outputs/sbfburst_calibrated_de_top10_8-11/parameters_8-11.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/sbfburst_calibrated_eval_top10_8-11"),
    )
    parser.add_argument("--protocols", default="mixed,occlusion")
    parser.add_argument("--fractions", default="0.40")
    parser.add_argument("--seeds", default="17")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--proxy-size", type=int, default=64)
    parser.add_argument("--burst-size", type=int, default=14)
    parser.add_argument("--select", type=int, default=8)
    parser.add_argument("--border", type=int, default=40)
    parser.add_argument("--split-seed", type=int, default=20260811)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SBFBurst evaluation")

    project_root = Path(__file__).resolve().parents[1]
    tuning_root = (project_root / args.tuning_root).resolve()
    aligned_root = tuning_root / "aligned"
    original_root = tuning_root / "original"
    sbfburst_root = (project_root / args.sbfburst_root).resolve()
    parameters_path = (project_root / args.parameters).resolve()
    output = (project_root / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "progress_8-11.jsonl"
    if progress_path.exists():
        progress_path.unlink()

    protocols = _parse_values(args.protocols, str)
    fractions = _parse_values(args.fractions, float)
    seeds = _parse_values(args.seeds, int)
    conditions = _condition_list(protocols, fractions, seeds)
    burst_names = _load_names(aligned_root, original_root, args.limit)
    parameters = _load_parameters(parameters_path)
    device = torch.device(args.device)
    started = time.perf_counter()
    _append_jsonl(
        progress_path,
        {
            "event": "run_start",
            "burst_names": burst_names,
            "conditions": conditions,
            "total_cases": len(burst_names) * len(conditions),
            "parameters": _safe_relative(parameters_path, project_root),
        },
    )

    model = load_reliable_sbfburst(sbfburst_root, device=device)
    records = []
    total = len(burst_names) * len(conditions)
    completed = 0
    for sample_index, burst_name in enumerate(burst_names):
        for protocol, fraction, condition_seed in conditions:
            seed = args.split_seed + sample_index * 101 + condition_seed
            aligned_sample = load_realbsr_burst(
                aligned_root,
                original_root,
                burst_name,
                burst_size=args.burst_size,
            )
            original_sample = load_realbsr_burst(
                original_root,
                original_root,
                burst_name,
                burst_size=args.burst_size,
            )
            aligned, abnormal = _corrupt_burst(
                aligned_sample.frames_rgb,
                fraction,
                protocol,
                seed,
            )
            original, _ = _corrupt_burst(
                original_sample.frames_rgb,
                fraction,
                protocol,
                seed,
            )
            proxy = center_crop_burst(aligned, args.proxy_size)
            observations = make_observations(
                rgb_to_luminance(proxy),
                abnormal_frames=abnormal,
            )
            estimates = estimate_registrations(observations)
            reliabilities = torch.tensor(
                [estimate.reliability for estimate in estimates],
                dtype=torch.float32,
                device=device,
            )
            burst_tensor = (
                torch.from_numpy(original)
                .permute(0, 3, 1, 2)
                .unsqueeze(0)
                .to(device)
            )
            official_prediction, cache = model.prepare_cache(burst_tensor)
            outputs = {"sbf_official": _output_numpy(official_prediction)}
            for method, method_parameters in parameters.items():
                priors = calibrated_priors(
                    reliabilities,
                    method_parameters,
                    selected_count=args.select,
                )
                prediction = model.decode_cached(cache, priors).clamp(0.0, 1.0)
                outputs[method] = _output_numpy(prediction)

            case = f"{burst_name}_{protocol}_{fraction}_{condition_seed}"
            for method, image in outputs.items():
                metrics = real_image_metrics(
                    original_sample.target_rgb,
                    image,
                    border=args.border,
                )
                records.append(
                    {
                        "case": case,
                        "burst_name": burst_name,
                        "protocol": protocol,
                        "fraction": fraction,
                        "seed": seed,
                        "method": method,
                        **metrics,
                    }
                )
            completed += 1
            _append_jsonl(
                progress_path,
                {
                    "event": "case_evaluated",
                    "completed": completed,
                    "total": total,
                    "burst_name": burst_name,
                    "protocol": protocol,
                    "fraction": fraction,
                    "elapsed_seconds": time.perf_counter() - started,
                },
            )
            print(f"evaluated {completed}/{total}", flush=True)

    summary = {
        "status": "complete",
        "configuration": {
            "tuning_root": _safe_relative(tuning_root, project_root),
            "burst_names": burst_names,
            "conditions": conditions,
            "cases": total,
            "methods": ["sbf_official", *parameters.keys()],
            "parameters": _safe_relative(parameters_path, project_root),
            "gpu": torch.cuda.get_device_name() if device.type == "cuda" else str(device),
            "elapsed_seconds": time.perf_counter() - started,
        },
        **_summarize(records),
    }
    (output / "records_8-11.json").write_text(
        json.dumps(records, indent=2),
        encoding="utf-8",
    )
    (output / "summary_8-11.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    _append_jsonl(
        progress_path,
        {
            "event": "run_done",
            "elapsed_seconds": time.perf_counter() - started,
            "summary_file": "summary_8-11.json",
        },
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
