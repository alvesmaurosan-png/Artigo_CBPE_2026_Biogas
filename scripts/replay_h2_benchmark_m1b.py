from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

# ------------------------------------------------------------
# PROJECT ROOT
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.milp_dispatch import MILPDispatchOptimizer
from src.economics.lcoe import build_economics_summary


# ------------------------------------------------------------
# BENCHMARK DEFINITIVO H2
# Pareto global definitivo - extremo de mínimo Pgrid
# ------------------------------------------------------------
CAPACITIES = {
    "pv_kw": 990.0,
    "bsv_kwh": 1241.0,
    "electrolyzer_kw": 441.0,
    "h2_tank_kg": 200.0,
    "fuelcell_kw": 117.0,
}

REFERENCE = {
    "P_peak_grid_opt_kw": 351.2476001141024,
    "lcoe_usd_kwh": 0.1940992206737364,
    "capex_total_usd": 2417120.0,
    "annualized_capex_usd": 246189.010991,
    "fixed_opex_annual_usd": 38763.4,
    "E_grid_total_kwh": 955761.963848,
    "total_grid_dependency_ratio": 0.421394,
}


def main() -> None:

    config_path = (
        PROJECT_ROOT
        / "configs"
        / "paper"
        / "pv_bsv_h2_1500_m1b.yaml"
    )

    with config_path.open("r", encoding="utf-8-sig") as f:
        config = yaml.safe_load(f)

    data_path = PROJECT_ROOT / config["data"]["demand_profile_csv"]

    df = pd.read_csv(data_path)

    required = {"hour", "demand_kw", "pv_factor"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Dataset missing required columns: {sorted(missing)}"
        )

    if len(df) != 8760:
        raise ValueError(
            f"Expected 8760 rows, found {len(df)}"
        )

    dt = float(
        config["data"].get("timestep_hours", 1.0)
    )

    period_hours = int(
        config["optimization"].get(
            "pareto_period_hours",
            24,
        )
    )

    # --------------------------------------------------------
    # FIXED CAPACITY MILP REPLAY
    # --------------------------------------------------------
    optimizer = MILPDispatchOptimizer(
        config=config,
        capacities=CAPACITIES,
        degradation_model=None,
    )

    result = optimizer.run_annual_simulation(
        df=df,
        period_hours=period_hours,
    )

    if result.dispatch_df.empty:
        raise RuntimeError(
            f"MILP returned empty dispatch. "
            f"status={result.solver_status}"
        )

    dispatch = result.dispatch_df.copy()

    # --------------------------------------------------------
    # ORIGINAL ECONOMICS
    # Same function used inside NSGA-II
    # --------------------------------------------------------
    economics = build_economics_summary(
        config=config,
        capacities=CAPACITIES,
        dispatch_df=dispatch,
    )

    # --------------------------------------------------------
    # INDEPENDENT AUDIT METRICS
    # --------------------------------------------------------
    pgrid_peak = float(
        dispatch["p_grid_kw"].max()
    )

    e_grid = float(
        dispatch["p_grid_kw"].sum() * dt
    )

    e_load = float(
        dispatch["demand_kw"].sum() * dt
    )

    grid_dependency = (
        e_grid / e_load if e_load > 0 else float("nan")
    )

    unserved_energy = (
        float(dispatch["p_unserved_kw"].sum() * dt)
        if "p_unserved_kw" in dispatch.columns
        else 0.0
    )

    calculated = {
        "P_peak_grid_opt_kw": pgrid_peak,
        "lcoe_usd_kwh": float(
            economics["lcoe_usd_kwh"]
        ),
        "capex_total_usd": float(
            economics["capex_total_usd"]
        ),
        "annualized_capex_usd": float(
            economics["annualized_capex_usd"]
        ),
        "fixed_opex_annual_usd": float(
            economics["fixed_opex_annual_usd"]
        ),
        "grid_opex_annual_usd": float(
            economics["grid_opex_annual_usd"]
        ),
        "grid_peak_opex_annual_usd": float(
            economics["grid_peak_opex_annual_usd"]
        ),
        "E_grid_total_kwh": e_grid,
        "E_load_total_kwh": e_load,
        "total_grid_dependency_ratio": grid_dependency,
        "unserved_energy_kwh": unserved_energy,
        "final_battery_kwh": float(
            result.final_battery_kwh
        ),
        "final_h2_kg": float(
            result.final_h2_kg
        ),
    }

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------
    output_dir = (
        PROJECT_ROOT
        / "results"
        / "validation"
        / "h2_m1b_monthly_demand"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dispatch_path = output_dir / "dispatch_8760.csv"
    summary_path = output_dir / "summary.json"

    dispatch.to_csv(
        dispatch_path,
        index=False,
    )

    summary_path.write_text(
        json.dumps(
            {
                "capacities": CAPACITIES,
                "reference": REFERENCE,
                "calculated": calculated,
                "delta": {
                    key: (
                        calculated[key] - value
                        if key in calculated
                        else None
                    )
                    for key, value in REFERENCE.items()
                },
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # CONSOLE REPORT
    # --------------------------------------------------------
    print()
    print("=" * 72)
    print("H2 FIXED BENCHMARK REPLAY")
    print("=" * 72)

    print("\nCONFIGURATION")
    for key, value in CAPACITIES.items():
        print(f"{key:28s} = {value}")

    print(f"\nperiod_hours                = {period_hours}")
    print(f"solver_status               = {result.solver_status}")

    print("\nRESULTS")
    for key, value in calculated.items():
        print(f"{key:32s} = {value}")

    print("\nREFERENCE COMPARISON")

    for key, reference in REFERENCE.items():
        calc = calculated.get(key)

        if calc is None:
            continue

        delta = calc - reference
        pct = (
            delta / reference * 100
            if reference != 0
            else float("nan")
        )

        print(
            f"{key:32s} "
            f"ref={reference:.12f} "
            f"calc={calc:.12f} "
            f"delta={delta:.12f} "
            f"({pct:+.6f}%)"
        )

    print("\nFILES")
    print(dispatch_path)
    print(summary_path)
    print()


if __name__ == "__main__":
    main()
