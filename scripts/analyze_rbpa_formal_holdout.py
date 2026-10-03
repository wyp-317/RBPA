"""Pre-registered paired statistics for the frozen RBPA formal holdout."""

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np


METHODS = ("official", "topk", "scalar", "cubic", "rbpa")
CONDITIONS = (
    ("mixed", 0.2),
    ("mixed", 0.4),
    ("mixed", 0.6),
    ("occlusion", 0.2),
    ("occlusion", 0.4),
    ("occlusion", 0.6),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bootstrap_ci(
    values: list[float] | np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        len(array),
        size=(resamples, len(array)),
    )
    bootstrap_means = array[indices].mean(axis=1)
    return [
        float(np.quantile(bootstrap_means, 0.025)),
        float(np.quantile(bootstrap_means, 0.975)),
    ]


def _sign_flip_pvalue(
    values: list[float] | np.ndarray,
    *,
    permutations: int,
    seed: int,
    alternative: str,
) -> float:
    array = np.asarray(values, dtype=np.float64)
    observed = float(array.mean())
    rng = np.random.default_rng(seed)
    exceedances = 0
    remaining = permutations
    while remaining:
        count = min(20_000, remaining)
        signs = rng.choice(
            np.asarray([-1.0, 1.0]),
            size=(count, len(array)),
        )
        null_means = (signs * array).mean(axis=1)
        if alternative == "greater":
            exceedances += int(np.sum(null_means >= observed))
        elif alternative == "two-sided":
            exceedances += int(
                np.sum(np.abs(null_means) >= abs(observed))
            )
        else:
            raise ValueError(f"unknown alternative: {alternative}")
        remaining -= count
    return (exceedances + 1.0) / (permutations + 1.0)


def _holm_adjust(rows: list[dict], p_key: str = "p_value") -> None:
    order = sorted(range(len(rows)), key=lambda index: rows[index][p_key])
    running = 0.0
    count = len(rows)
    for rank, index in enumerate(order):
        adjusted = min(1.0, (count - rank) * rows[index][p_key])
        running = max(running, adjusted)
        rows[index]["p_value_holm"] = running
        rows[index]["holm_reject_0_05"] = running < 0.05


def _scene_differences(
    records: list[dict],
    method: str,
    baseline: str,
    metric: str,
    *,
    condition: tuple[str, float] | None = None,
    clean: bool = False,
) -> dict[str, float]:
    grouped = defaultdict(list)
    keyed = {
        (record["case"], record["method"]): record for record in records
    }
    cases = sorted({record["case"] for record in records})
    for case in cases:
        candidate = keyed[(case, method)]
        if clean and candidate["protocol"] != "clean":
            continue
        if not clean and candidate["protocol"] == "clean":
            continue
        if condition is not None and (
            candidate["protocol"],
            float(candidate["fraction"]),
        ) != condition:
            continue
        delta = (
            float(candidate[metric])
            - float(keyed[(case, baseline)][metric])
        )
        grouped[candidate["scene"]].append(delta)
    return {
        scene: statistics.mean(values)
        for scene, values in sorted(grouped.items())
    }


def _comparison(
    records: list[dict],
    method: str,
    baseline: str,
    metric: str,
    *,
    condition: tuple[str, float] | None,
    clean: bool,
    resamples: int,
    permutations: int,
    seed: int,
    alternative: str,
) -> dict:
    scene_values = _scene_differences(
        records,
        method,
        baseline,
        metric,
        condition=condition,
        clean=clean,
    )
    values = list(scene_values.values())
    return {
        "method": method,
        "baseline": baseline,
        "metric": metric,
        "protocol": "clean" if clean else (
            condition[0] if condition is not None else "all_degraded"
        ),
        "fraction": (
            0.0 if clean else (
                condition[1] if condition is not None else None
            )
        ),
        "mean_delta": statistics.mean(values),
        "ci95": _bootstrap_ci(
            values,
            resamples=resamples,
            seed=seed,
        ),
        "p_value": _sign_flip_pvalue(
            values,
            permutations=permutations,
            seed=seed + 10_000,
            alternative=alternative,
        ),
        "alternative": alternative,
        "scene_win_rate": statistics.mean(value > 0 for value in values),
        "scenes": len(values),
        "scene_deltas": scene_values,
    }


def _method_aggregates(records: list[dict]) -> list[dict]:
    rows = []
    for scope, predicate in (
        ("clean", lambda record: record["protocol"] == "clean"),
        ("all_degraded", lambda record: record["protocol"] != "clean"),
    ):
        for method in METHODS:
            items = [
                record
                for record in records
                if record["method"] == method and predicate(record)
            ]
            rows.append(
                {
                    "scope": scope,
                    "method": method,
                    "psnr_rgb": statistics.mean(
                        item["psnr_rgb"] for item in items
                    ),
                    "ssim_rgb": statistics.mean(
                        item["ssim_rgb"] for item in items
                    ),
                    "lpips": statistics.mean(
                        item["lpips"] for item in items
                    ),
                    "samples": len(items),
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "protocol",
        "fraction",
        "baseline",
        "mean_delta",
        "ci95_low",
        "ci95_high",
        "p_value",
        "p_value_holm",
        "scene_win_rate",
        "scenes",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "protocol": row["protocol"],
                    "fraction": row["fraction"],
                    "baseline": row["baseline"],
                    "mean_delta": row["mean_delta"],
                    "ci95_low": row["ci95"][0],
                    "ci95_high": row["ci95"][1],
                    "p_value": row["p_value"],
                    "p_value_holm": row.get("p_value_holm"),
                    "scene_win_rate": row["scene_win_rate"],
                    "scenes": row["scenes"],
                }
            )


def _write_report(summary: dict, evidence: Path) -> None:
    aggregates = {
        (row["scope"], row["method"]): row
        for row in summary["method_aggregates"]
    }
    confirmatory = summary["confirmatory"]
    clean = summary["clean_safety"]
    lines = [
        "# RBPA Formal Holdout",
        "",
        "This is the one-time frozen evaluation on 24 previously unused "
        "RealBSR-RGB scenes (312 cases).",
        "",
        "## Confirmatory result",
        "",
        f"- RBPA versus scalar CalibBurst, all degraded cases: "
        f"**{confirmatory['mean_delta']:+.4f} dB** "
        f"(95% CI [{confirmatory['ci95'][0]:+.4f}, "
        f"{confirmatory['ci95'][1]:+.4f}]), "
        f"one-sided sign-flip p={confirmatory['p_value']:.6g}.",
        f"- Clean change versus scalar: **{clean['mean_delta']:+.4f} dB** "
        f"(safety threshold -0.0200 dB).",
        f"- Confirmatory decision: "
        f"**{'PASS' if summary['confirmatory_pass'] else 'FAIL'}**.",
        "",
        "## Mean performance",
        "",
        "| Method | Clean PSNR | Degraded PSNR | Degraded SSIM | Degraded LPIPS |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        clean_row = aggregates[("clean", method)]
        degraded = aggregates[("all_degraded", method)]
        lines.append(
            f"| {method} | {clean_row['psnr_rgb']:.4f} | "
            f"{degraded['psnr_rgb']:.4f} | "
            f"{degraded['ssim_rgb']:.5f} | "
            f"{degraded['lpips']:.5f} |"
        )
    lines.extend(
        [
            "",
            "## RBPA PSNR gains by condition",
            "",
            "| Condition | vs scalar | 95% CI | Holm p | vs cubic | 95% CI |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    scalar_lookup = {
        (row["protocol"], row["fraction"]): row
        for row in summary["condition_vs_scalar"]
    }
    cubic_lookup = {
        (row["protocol"], row["fraction"]): row
        for row in summary["condition_vs_cubic"]
    }
    for condition in CONDITIONS:
        scalar = scalar_lookup[condition]
        cubic = cubic_lookup[condition]
        lines.append(
            f"| {condition[0]} {int(100 * condition[1])}% | "
            f"{scalar['mean_delta']:+.4f} | "
            f"[{scalar['ci95'][0]:+.4f}, {scalar['ci95'][1]:+.4f}] | "
            f"{scalar['p_value_holm']:.4g} | "
            f"{cubic['mean_delta']:+.4f} | "
            f"[{cubic['ci95'][0]:+.4f}, {cubic['ci95'][1]:+.4f}] |"
        )
    lines.extend(
        [
            "",
            "Positive PSNR/SSIM deltas and negative LPIPS deltas favor RBPA. "
            "All confidence intervals resample scenes, not individual cases.",
            "",
            "Machine-readable statistics: "
            "`../../outputs/rbpa_formal_holdout/statistical_summary.json`.",
        ]
    )
    (evidence / "RBPA_FORMAL_HOLDOUT.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--records",
        type=Path,
        default=Path("outputs/rbpa_formal_holdout/records.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/rbpa_formal_holdout"),
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=Path("paper/evidence"),
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--permutations", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=2093)
    args = parser.parse_args()
    records = json.loads(args.records.read_text(encoding="utf-8"))
    keyed = {
        (record["case"], record["method"]) for record in records
    }
    cases = sorted({record["case"] for record in records})
    scenes = sorted({record["scene"] for record in records})
    if len(records) != 1560 or len(cases) != 312 or len(scenes) != 24:
        raise ValueError("formal record dimensions do not match the protocol")
    if any((case, method) not in keyed for case in cases for method in METHODS):
        raise ValueError("at least one formal case is missing a method")

    confirmatory = _comparison(
        records,
        "rbpa",
        "scalar",
        "psnr_rgb",
        condition=None,
        clean=False,
        resamples=args.bootstrap_resamples,
        permutations=args.permutations,
        seed=args.seed,
        alternative="greater",
    )
    clean_safety = _comparison(
        records,
        "rbpa",
        "scalar",
        "psnr_rgb",
        condition=None,
        clean=True,
        resamples=args.bootstrap_resamples,
        permutations=args.permutations,
        seed=args.seed + 1,
        alternative="two-sided",
    )
    condition_vs_scalar = [
        _comparison(
            records,
            "rbpa",
            "scalar",
            "psnr_rgb",
            condition=condition,
            clean=False,
            resamples=args.bootstrap_resamples,
            permutations=args.permutations,
            seed=args.seed + 10 + index,
            alternative="two-sided",
        )
        for index, condition in enumerate(CONDITIONS)
    ]
    _holm_adjust(condition_vs_scalar)
    condition_vs_cubic = [
        _comparison(
            records,
            "rbpa",
            "cubic",
            "psnr_rgb",
            condition=condition,
            clean=False,
            resamples=args.bootstrap_resamples,
            permutations=args.permutations,
            seed=args.seed + 30 + index,
            alternative="two-sided",
        )
        for index, condition in enumerate(CONDITIONS)
    ]
    _holm_adjust(condition_vs_cubic)

    overall_comparisons = []
    for baseline_index, baseline in enumerate(
        ("official", "topk", "scalar", "cubic")
    ):
        for metric_index, metric in enumerate(
            ("psnr_rgb", "ssim_rgb", "lpips")
        ):
            overall_comparisons.append(
                _comparison(
                    records,
                    "rbpa",
                    baseline,
                    metric,
                    condition=None,
                    clean=False,
                    resamples=args.bootstrap_resamples,
                    permutations=args.permutations,
                    seed=(
                        args.seed
                        + 100
                        + baseline_index * 10
                        + metric_index
                    ),
                    alternative="two-sided",
                )
            )
    confirmatory_pass = (
        confirmatory["ci95"][0] > 0.0
        and clean_safety["mean_delta"] >= -0.02
    )
    summary = {
        "status": "formal statistical analysis complete",
        "protocol": {
            "scenes": 24,
            "cases": 312,
            "records": 1560,
            "bootstrap_resamples": args.bootstrap_resamples,
            "permutation_samples": args.permutations,
            "scene_level_pairing": True,
            "primary_metric": "RGB PSNR",
        },
        "confirmatory": confirmatory,
        "clean_safety": clean_safety,
        "confirmatory_pass": confirmatory_pass,
        "condition_vs_scalar": condition_vs_scalar,
        "condition_vs_cubic": condition_vs_cubic,
        "overall_comparisons": overall_comparisons,
        "method_aggregates": _method_aggregates(records),
        "integrity": {
            "records_sha256": _sha256(args.records),
            "unique_scenes": len(scenes),
            "unique_cases": len(cases),
            "records": len(records),
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    args.evidence.mkdir(parents=True, exist_ok=True)
    (args.output / "statistical_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    _write_csv(
        args.evidence / "rbpa_formal_condition_comparisons.csv",
        condition_vs_scalar,
    )
    _write_report(summary, args.evidence)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
