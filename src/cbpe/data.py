from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ("hour", "demand_kw", "pv_factor")


@dataclass(frozen=True)
class DatasetReport:
    path: str
    rows: int
    sha256: str
    demand_min_kw: float
    demand_max_kw: float
    pv_factor_min: float
    pv_factor_max: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dataset(path: Path, expected_rows: int = 8760) -> DatasetReport:
    frame = pd.read_csv(path)
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing dataset columns: {missing}")
    if len(frame) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, found {len(frame)}")
    numeric = frame.loc[:, REQUIRED_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError("Dataset contains missing or non-numeric required values")
    if (numeric["demand_kw"] < 0).any():
        raise ValueError("demand_kw must be non-negative")
    if not numeric["pv_factor"].between(0, 1).all():
        raise ValueError("pv_factor must be in [0, 1]")
    expected_hours = pd.Series(range(expected_rows), dtype="int64")
    if not numeric["hour"].reset_index(drop=True).astype("int64").equals(expected_hours):
        raise ValueError("hour must be a contiguous zero-based annual index")
    return DatasetReport(
        path=str(path),
        rows=len(frame),
        sha256=sha256_file(path),
        demand_min_kw=float(numeric["demand_kw"].min()),
        demand_max_kw=float(numeric["demand_kw"].max()),
        pv_factor_min=float(numeric["pv_factor"].min()),
        pv_factor_max=float(numeric["pv_factor"].max()),
    )

