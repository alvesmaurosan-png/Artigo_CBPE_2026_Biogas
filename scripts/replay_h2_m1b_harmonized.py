from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DISPATCH_PATH = (
    PROJECT_ROOT
    / "results"
    / "validation"
    / "h2_m1b_monthly_demand"
    / "dispatch_8760.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "validation"
    / "h2_m1b_harmonized"
)

DEMAND_CHARGE_USD_KW_MONTH = 30.0

# Valores reproduzidos no M0
ANNUALIZED_CAPEX_USD = 246189.01099061375
FIXED_OPEX_USD = 38763.4


def main() -> None:

    df = pd.read_csv(DISPATCH_PATH)

    if len(df) != 8760:
        raise ValueError(
            f"Expected 8760 rows, found {len(df)}"
        )

    required = {
        "t_global",
        "hour",
        "demand_kw",
        "p_grid_kw",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    # --------------------------------------------------------
    # CALENDAR
    # 2026 is not leap year
    # --------------------------------------------------------
    dt_index = pd.date_range(
        "2026-01-01 00:00:00",
        periods=8760,
        freq="h",
    )

    df = df.copy()
    df["datetime"] = dt_index
    df["month"] = df["datetime"].dt.month

    # --------------------------------------------------------
    # MONTHLY GRID PEAKS
    # --------------------------------------------------------
    monthly = (
        df.groupby("month", as_index=False)
        .agg(
            p_grid_peak_kw=("p_grid_kw", "max"),
            e_grid_kwh=("p_grid_kw", "sum"),
        )
    )

    monthly["demand_charge_usd"] = (
        monthly["p_grid_peak_kw"]
        * DEMAND_CHARGE_USD_KW_MONTH
    )

    annual_demand_charge = float(
        monthly["demand_charge_usd"].sum()
    )

    # --------------------------------------------------------
    # ORIGINAL GRID ENERGY COST
    # Reconstruct directly from TOU tariff
    # --------------------------------------------------------
    hod = df["datetime"].dt.hour

    tariff = pd.Series(
        0.15,
        index=df.index,
        dtype=float,
    )

    peak_mask = (hod >= 18) & (hod < 21)
    tariff.loc[peak_mask] = 0.50

    df["tariff_usd_kwh"] = tariff

    df["grid_energy_cost_usd"] = (
        df["p_grid_kw"]
        * df["tariff_usd_kwh"]
    )

    grid_energy_cost = float(
        df["grid_energy_cost_usd"].sum()
    )

    # --------------------------------------------------------
    # VARIABLE H2 OPEX
    # --------------------------------------------------------
    variable_h2_opex = 0.0

    if "p_elz_kw" in df.columns:
        variable_h2_opex += float(
            (df["p_elz_kw"] * 0.005).sum()
        )

    if "p_fc_kw" in df.columns:
        variable_h2_opex += float(
            (df["p_fc_kw"] * 0.010).sum()
        )

    # --------------------------------------------------------
    # ENERGY SERVED
    # --------------------------------------------------------
    unserved = (
        float(df["p_unserved_kw"].sum())
        if "p_unserved_kw" in df.columns
        else 0.0
    )

    energy_served = float(
        df["demand_kw"].sum() - unserved
    )

    # --------------------------------------------------------
    # HISTORICAL VS HARMONIZED
    # --------------------------------------------------------
    annual_cost_historical = (
        ANNUALIZED_CAPEX_USD
        + FIXED_OPEX_USD
        + variable_h2_opex
        + grid_energy_cost
    )

    annual_cost_harmonized = (
        annual_cost_historical
        + annual_demand_charge
    )

    lcoe_historical = (
        annual_cost_historical / energy_served
    )

    lcoe_harmonized = (
        annual_cost_harmonized / energy_served
    )

    result = {
        "demand_charge_usd_kw_month":
            DEMAND_CHARGE_USD_KW_MONTH,

        "annualized_capex_usd":
            ANNUALIZED_CAPEX_USD,

        "fixed_opex_usd":
            FIXED_OPEX_USD,

        "variable_h2_opex_usd":
            variable_h2_opex,

        "grid_energy_cost_usd":
            grid_energy_cost,

        "annual_demand_charge_usd":
            annual_demand_charge,

        "annual_cost_historical_usd":
            annual_cost_historical,

        "annual_cost_harmonized_usd":
            annual_cost_harmonized,

        "energy_served_kwh":
            energy_served,

        "lcoe_historical_usd_kwh":
            lcoe_historical,

        "lcoe_harmonized_usd_kwh":
            lcoe_harmonized,

        "lcoe_increase_absolute":
            lcoe_harmonized - lcoe_historical,

        "lcoe_increase_percent":
            (
                (lcoe_harmonized / lcoe_historical - 1)
                * 100
            ),

        "annual_pgrid_peak_kw":
            float(df["p_grid_kw"].max()),
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    monthly_path = (
        OUTPUT_DIR
        / "monthly_demand_charge.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "summary.json"
    )

    monthly.to_csv(
        monthly_path,
        index=False,
    )

    summary_path.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("H2 HARMONIZED EX-POST DEMAND CHARGE")
    print("=" * 72)

    print("\nMONTHLY PEAKS")
    print(
        monthly.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print("\nSUMMARY")
    for key, value in result.items():
        print(f"{key:36s} = {value}")

    print("\nFILES")
    print(monthly_path)
    print(summary_path)
    print()


if __name__ == "__main__":
    main()
