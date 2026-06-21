from pathlib import Path

import pandas as pd
import pytest

from cbpe.data import validate_dataset


ROOT = Path(__file__).resolve().parents[1]


def test_canonical_dataset() -> None:
    report = validate_dataset(ROOT / "data" / "processed" / "fleet_demand_sp.csv")
    assert report.rows == 8760
    assert report.demand_min_kw >= 0
    assert 0 <= report.pv_factor_min <= report.pv_factor_max <= 1
    assert len(report.sha256) == 64


def test_invalid_pv_factor_is_rejected(tmp_path: Path) -> None:
    frame = pd.DataFrame({"hour": [0], "demand_kw": [1.0], "pv_factor": [1.1]})
    path = tmp_path / "invalid.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="pv_factor"):
        validate_dataset(path, expected_rows=1)

