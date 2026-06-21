from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from cbpe.data import validate_dataset
from cbpe.feasibility import solve_minimum_grid_peak
from cbpe.figures import generate_paper_artifacts
from cbpe.manifest import build_manifest, write_manifest


PAPER_CONFIGS = (
    "pv_bsv_6000.yaml",
    "pv_bsv_3000.yaml",
    "pv_bsv_1500.yaml",
    "pv_bsv_h2_1500.yaml",
)


def _run_full_pareto(root: Path) -> list[dict[str, object]]:
    runs: list[dict[str, object]] = []
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    for filename in PAPER_CONFIGS:
        config = root / "configs" / "paper" / filename
        started = time.perf_counter()
        completed = subprocess.run(
            [sys.executable, "-m", "src.optimization.ga_nsga2", "--config", str(config)],
            cwd=root,
            env=env,
            text=True,
            capture_output=True,
        )
        duration = time.perf_counter() - started
        log_dir = root / "results" / "release" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / f"{config.stem}.stdout.log").write_text(completed.stdout, encoding="utf-8")
        (log_dir / f"{config.stem}.stderr.log").write_text(completed.stderr, encoding="utf-8")
        runs.append({
            "scenario": config.stem,
            "stage": "nsga2",
            "returncode": completed.returncode,
            "duration_seconds": duration,
            "status": "COMPLETE" if completed.returncode == 0 else "EXECUTION_ERROR",
        })
        if completed.returncode != 0:
            raise RuntimeError(f"NSGA-II failed for {config.stem}; see results/release/logs")
        generated = root / "results" / "runs" / f"{config.stem}_with_h2" / "pareto_latest" / "pareto.csv"
        target = root / "results" / "paper" / "source_data" / f"pareto_{config.stem}.csv"
        if generated.is_file():
            shutil.copy2(generated, target)
    return runs


def reproduce(root: Path, profile: str) -> Path:
    started = time.perf_counter()
    data_path = root / "data" / "processed" / "fleet_demand_sp.csv"
    dataset = validate_dataset(data_path)
    runs: list[dict[str, object]] = []

    if profile == "paper":
        runs.extend(_run_full_pareto(root))

    horizon = 48 if profile == "smoke" else None
    limit = 20 if profile == "smoke" else None
    for scenario in ("pv_bsv_1500", "pv_bsv_h2_1500"):
        result = solve_minimum_grid_peak(
            root / "configs" / "paper" / f"{scenario}.yaml",
            data_path,
            horizon_hours=horizon,
            time_limit_seconds=limit,
        )
        runs.append({"scenario": scenario, "stage": "integrated_feasibility", **result.as_dict()})

    output_dir = root / "results" / "paper"
    artifacts = generate_paper_artifacts(output_dir / "source_data", output_dir)
    feasibility_path = output_dir / "tables" / f"feasibility_{profile}.json"
    feasibility_path.write_text(
        json.dumps([run for run in runs if run["stage"] == "integrated_feasibility"], indent=2) + "\n",
        encoding="utf-8",
    )
    artifacts.append(feasibility_path)
    annual_audit = output_dir / "tables" / "feasibility_annual.json"
    if annual_audit.is_file():
        artifacts.append(annual_audit)
    manifest_path = output_dir / "manifest.json"
    manifest = build_manifest(
        root,
        profile,
        dataset,
        artifacts,
        runs,
        time.perf_counter() - started,
    )
    write_manifest(manifest_path, manifest)
    return manifest_path
