from __future__ import annotations

import json
from pathlib import Path

from cbpe.data import sha256_file


def verify_manifest(root: Path, manifest_path: Path) -> list[str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    for item in [*manifest.get("inputs", []), *manifest.get("outputs", [])]:
        path = root / item["path"]
        if not path.is_file():
            errors.append(f"missing: {item['path']}")
        elif sha256_file(path) != item["sha256"]:
            errors.append(f"hash mismatch: {item['path']}")
    dataset = manifest.get("dataset", {})
    dataset_path = Path(dataset.get("path", ""))
    if not dataset_path.is_absolute():
        dataset_path = root / dataset_path
    if not dataset_path.is_file():
        errors.append(f"missing dataset: {dataset_path}")
    elif sha256_file(dataset_path) != dataset.get("sha256"):
        errors.append(f"dataset hash mismatch: {dataset_path}")
    return errors
