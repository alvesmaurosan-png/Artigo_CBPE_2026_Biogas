from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


SCENARIOS = (
    ("PV+BSV | BSV <= 6000 kWh", "pareto_pv_bsv_6000.csv", "o"),
    ("PV+BSV | BSV <= 3000 kWh", "pareto_pv_bsv_3000.csv", "s"),
    ("PV+BSV | BSV <= 1500 kWh", "pareto_pv_bsv_1500.csv", "x"),
    ("PV+BSV+H2 | BSV <= 1500 kWh", "pareto_pv_bsv_h2_1500.csv", "^"),
)


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_paper_artifacts(source: Path, output: Path) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    artifacts: list[Path] = []

    frames: list[tuple[str, pd.DataFrame, str]] = []
    for label, filename, marker in SCENARIOS:
        frame = pd.read_csv(source / filename).dropna(subset=["lcoe_usd_kwh", "P_peak_grid_opt_kw"])
        frames.append((label, frame, marker))

    fig, ax = plt.subplots(figsize=(8, 4.8))
    minima = [(label, float(frame["P_peak_grid_opt_kw"].min())) for label, frame, _ in frames]
    ax.barh([item[0] for item in minima], [item[1] for item in minima], color="#4472C4")
    ax.axvline(468.25, color="black", linestyle="--", label="468.25 kW (observed PV+BSV minimum)")
    ax.set_xlabel("Minimum observed grid peak (kW)")
    ax.set_title("Observed power minima by technology scenario")
    ax.legend(fontsize=8)
    artifacts.append(_save(fig, figures / "figure_1_observed_power_minima.png"))

    dispatch = pd.read_csv(source / "weekly_typical_dispatch.csv")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(dispatch["hour_of_week"], dispatch["p_grid_kw"], label="Grid")
    ax.plot(dispatch["hour_of_week"], dispatch["p_pv_used_kw"], label="PV")
    ax.plot(dispatch["hour_of_week"], dispatch["p_bat_dis_kw"], label="BSV discharge")
    ax.plot(dispatch["hour_of_week"], dispatch["p_fc_kw"], label="Fuel cell")
    ax.set(xlabel="Hour of week", ylabel="Power (kW)", title="Representative weekly dispatch")
    ax.legend(ncol=4, fontsize=8)
    ax.grid(alpha=0.25)
    artifacts.append(_save(fig, figures / "figure_2_operational_profile.png"))

    contributions = pd.read_csv(source / "annual_vs_peak.csv")
    x = range(len(contributions))
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.bar([i - 0.18 for i in x], contributions["annual_kwh"] / 1000, width=0.36, label="Annual")
    ax.bar([i + 0.18 for i in x], contributions["peak_kwh"] / 1000, width=0.36, label="Peak window")
    ax.set_xticks(list(x), contributions["source"])
    ax.set(ylabel="Energy (MWh)", title="Annual and peak-window energy contributions")
    ax.legend()
    artifacts.append(_save(fig, figures / "figure_3_energy_contributions.png"))

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, frame, marker in frames:
        ax.scatter(frame["P_peak_grid_opt_kw"], frame["lcoe_usd_kwh"], label=f"{label} (n={len(frame)})", marker=marker, alpha=0.8)
    ax.set(xlabel="Grid peak (kW)", ylabel="LCOE (USD/kWh)", title="Pareto-front contraction and recovery")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    artifacts.append(_save(fig, figures / "figure_4_pareto_constraints.png"))

    no_h2 = frames[2][1]
    with_h2 = frames[3][1]
    rows = []
    for label, frame in (("PV + BSV", no_h2), ("PV + BSV + H2", with_h2)):
        minimum_peak_row = frame.loc[frame["P_peak_grid_opt_kw"].idxmin()]
        rows.append({
            "configuration": label,
            "non_dominated_solutions": len(frame),
            "minimum_lcoe_usd_kwh": float(frame["lcoe_usd_kwh"].min()),
            "minimum_grid_peak_kw": float(frame["P_peak_grid_opt_kw"].min()),
            "grid_dependency_at_minimum_peak_percent": float(minimum_peak_row["total_grid_dependency_ratio"] * 100),
        })
    table_path = tables / "table_2_hydrogen_effect.csv"
    pd.DataFrame(rows).to_csv(table_path, index=False)
    artifacts.append(table_path)
    return artifacts
