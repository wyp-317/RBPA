"""Create or verify the immutable RBPA formal-result manifest."""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path


RESULT_FILES = (
    "paper/RBPA_METHOD_FREEZE.json",
    "paper/RBPA_NEW_HOLDOUT_MANIFEST.json",
    "paper/RBPA_HOLDOUT_OPENING.json",
    "paper/RBPA_EVALUATION_PROTOCOL.md",
    "src/robust_mfsr/learned/risk_budget_allocation.py",
    "scripts/run_rbpa_formal_holdout.py",
    "scripts/analyze_rbpa_formal_holdout.py",
    "scripts/build_rbpa_formal_evidence.py",
    "outputs/rbpa_formal_holdout/test_plan.json",
    "outputs/rbpa_formal_holdout/records.json",
    "outputs/rbpa_formal_holdout/raw_summary.json",
    "outputs/rbpa_formal_holdout/run_state.json",
    "outputs/rbpa_formal_holdout/statistical_summary.json",
    "paper/evidence/rbpa_formal_condition_comparisons.csv",
    "paper/evidence/RBPA_FORMAL_HOLDOUT.md",
    "paper/evidence/rbpa_formal_condition_gains.png",
    "paper/evidence/rbpa_formal_condition_gains.pdf",
    "paper/evidence/rbpa_formal_overall_metrics.png",
    "paper/evidence/rbpa_formal_overall_metrics.pdf",
    "paper/evidence/rbpa_formal_qualitative.png",
    "paper/evidence/rbpa_formal_qualitative.pdf",
    "paper/evidence/rbpa_formal_qualitative_selection.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _create(project_root: Path, manifest_path: Path) -> None:
    statistics_path = (
        project_root
        / "outputs/rbpa_formal_holdout/statistical_summary.json"
    )
    statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
    missing = [
        relative
        for relative in RESULT_FILES
        if not (project_root / relative).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing result files: {missing}")
    manifest = {
        "method": "Risk-Budgeted Precision Allocation",
        "version": "RBPA-1.0-formal-result",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "formal holdout complete and immutable",
        "dimensions": statistics["protocol"],
        "confirmatory_pass": statistics["confirmatory_pass"],
        "primary_result": {
            key: statistics["confirmatory"][key]
            for key in (
                "method",
                "baseline",
                "metric",
                "protocol",
                "mean_delta",
                "ci95",
                "p_value",
                "alternative",
                "scene_win_rate",
                "scenes",
            )
        },
        "sha256": {
            relative: _sha256(project_root / relative)
            for relative in RESULT_FILES
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "created": str(manifest_path),
                "files_frozen": len(RESULT_FILES),
                "confirmatory_pass": manifest["confirmatory_pass"],
            },
            indent=2,
        )
    )


def _verify(project_root: Path, manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mismatches = []
    for relative, expected in manifest["sha256"].items():
        path = project_root / relative
        actual = _sha256(path) if path.is_file() else None
        if actual != expected:
            mismatches.append(
                {
                    "path": relative,
                    "expected": expected,
                    "actual": actual,
                }
            )
    result = {
        "version": manifest["version"],
        "files_checked": len(manifest["sha256"]),
        "confirmatory_pass": manifest["confirmatory_pass"],
        "verified": not mismatches,
        "mismatches": mismatches,
    }
    print(json.dumps(result, indent=2))
    if mismatches:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("paper/RBPA_FORMAL_RESULT_FREEZE.json"),
    )
    parser.add_argument("--create", action="store_true")
    args = parser.parse_args()
    project_root = args.manifest.resolve().parents[1]
    if args.create:
        _create(project_root, args.manifest)
    else:
        _verify(project_root, args.manifest)


if __name__ == "__main__":
    main()
