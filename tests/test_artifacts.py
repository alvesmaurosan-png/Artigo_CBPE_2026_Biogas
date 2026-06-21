from pathlib import Path

import pandas as pd

from cbpe.figures import generate_paper_artifacts


ROOT = Path(__file__).resolve().parents[1]


def test_four_figures_and_table_are_generated(tmp_path: Path) -> None:
    artifacts = generate_paper_artifacts(ROOT / "results" / "paper" / "source_data", tmp_path)
    figures = [path for path in artifacts if path.suffix == ".png"]
    assert len(figures) == 4
    assert all(path.stat().st_size > 1000 for path in figures)
    table = pd.read_csv(tmp_path / "tables" / "table_2_hydrogen_effect.csv")
    assert table["non_dominated_solutions"].tolist() == [2, 47]
    assert table["minimum_grid_peak_kw"].round(2).tolist() == [468.25, 351.25]
    assert table["grid_dependency_at_minimum_peak_percent"].round(1).tolist() == [66.5, 44.4]
