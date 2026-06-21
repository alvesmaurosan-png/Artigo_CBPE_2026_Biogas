from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from cbpe.data import DatasetReport, sha256_file


PACKAGES = ("numpy", "pandas", "matplotlib", "ortools", "PyYAML")


def git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_manifest(
    root: Path,
    profile: str,
    dataset: DatasetReport,
    outputs: list[Path],
    runs: list[dict[str, object]],
    duration_seconds: float,
) -> dict[str, object]:
    dependencies: dict[str, str] = {}
    for package in PACKAGES:
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError:
            dependencies[package] = "not-installed"
    dataset_payload = asdict(dataset)
    dataset_path = Path(dataset.path)
    if dataset_path.is_absolute():
        dataset_payload["path"] = str(dataset_path.relative_to(root)).replace("\\", "/")
    input_paths = sorted((root / "configs" / "paper").glob("*.yaml"))
    input_paths.extend(sorted((root / "results" / "paper" / "source_data").glob("*.csv")))
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": profile,
        "git_commit": git_commit(root),
        "seed": 42,
        "method": {
            "pareto_dispatch": "sequential 24-hour MILPs with storage-state continuity",
            "feasibility": "integrated sizing and dispatch MILP",
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "dependencies": dependencies,
        },
        "dataset": dataset_payload,
        "inputs": [
            {
                "path": str(path.relative_to(root)).replace("\\", "/"),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in input_paths
        ],
        "runs": runs,
        "duration_seconds": duration_seconds,
        "outputs": [
            {
                "path": str(path.relative_to(root)).replace("\\", "/"),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(outputs)
        ],
    }


def write_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
