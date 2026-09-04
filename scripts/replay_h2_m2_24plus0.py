from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.models.milp_dispatch import MILPDispatchOptimizer
from src.economics.lcoe import build_economics_summary


# ============================================================
# FILES
# ============================================================

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "paper"
    / "pv_bsv_h2_1500_m1b.yaml"
)

LEGACY_DISPATCH_PATH = (
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
    / "h2_m2_24plus0"
)

M2_DISPATCH_PATH = (
    OUTPUT_DIR
    / "dispatch_8760.csv"
)

M2_MONTHLY_PATH = (
    OUTPUT_DIR
    / "monthly_demand_charge.csv"
)

COMPARISON_PATH = (
    OUTPUT_DIR
    / "comparison_m1b_vs_m2_24plus0.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "summary.json"
)


# ============================================================
# M2 TEST SETTINGS
# ============================================================

PERIOD_HOURS = 24
COMMIT_HOURS = 24
LOOKAHEAD_HOURS = 0


# ============================================================
# FIXED H2 BENCHMARK
# ============================================================

CAPACITIES = {
    "pv_kw": 990.0,
    "bsv_kwh": 1241.0,
    "electrolyzer_kw": 441.0,
    "h2_tank_kg": 200.0,
    "fuelcell_kw": 117.0,
}


# ============================================================
# NUMERICAL TOLERANCES
# ============================================================

ABS_TOL_POWER_KW = 1.0e-6
ABS_TOL_ENERGY_KWH = 1.0e-6
ABS_TOL_STATE = 1.0e-6

# Some solver-equivalent solutions can differ microscopically
# while remaining physically/economically identical.
REL_TOL = 1.0e-9


# ============================================================
# HELPERS
# ============================================================

def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(
            f"Invalid YAML root: {path}"
        )

    return cfg


def resolve_project_path(
    value: str,
) -> Path:
    path = Path(value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def load_input_data(
    cfg: dict[str, Any],
) -> pd.DataFrame:
    input_path = resolve_project_path(
        cfg["data"]["demand_profile_csv"]
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Demand profile not found: {input_path}"
        )

    df = pd.read_csv(input_path)

    required = {
        "hour",
        "demand_kw",
        "pv_factor",
    }

    missing = required.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            "Demand profile missing columns: "
            f"{sorted(missing)}"
        )

    if len(df) != 8760:
        raise ValueError(
            f"Expected 8760 rows, got {len(df)}"
        )

    return df.reset_index(drop=True)


def assign_month_2026(
    t_global: pd.Series,
) -> pd.Series:
    """
    Converts global hour [0..8759] into month [1..12].
    2026 is not a leap year.
    """

    month_days = np.array(
        [
            31,
            28,
            31,
            30,
            31,
            30,
            31,
            31,
            30,
            31,
            30,
            31,
        ],
        dtype=int,
    )

    cumulative_hours = np.cumsum(
        month_days * 24
    )

    months = (
        np.searchsorted(
            cumulative_hours,
            t_global.to_numpy(
                dtype=int
            ),
            side="right",
        )
        + 1
    )

    return pd.Series(
        months,
        index=t_global.index,
        dtype=int,
    )


def compute_monthly_demand_charge(
    dispatch: pd.DataFrame,
    cfg: dict[str, Any],
    dt: float,
) -> pd.DataFrame:
    d = dispatch.copy()

    if "t_global" not in d.columns:
        d["t_global"] = np.arange(
            len(d),
            dtype=int,
        )

    d["month"] = assign_month_2026(
        d["t_global"]
    )

    monthly = (
        d.groupby(
            "month",
            as_index=False,
        )
        .agg(
            p_grid_peak_kw=(
                "p_grid_kw",
                "max",
            ),
            e_grid_kwh=(
                "p_grid_kw",
                lambda s: float(
                    s.sum() * dt
                ),
            ),
        )
    )

    demand_rate = float(
        cfg["tariff"].get(
            "demand_charge_usd_kw_month",
            0.0,
        )
    )

    monthly[
        "demand_charge_usd"
    ] = (
        monthly["p_grid_peak_kw"]
        * demand_rate
    )

    return monthly


def get_variable_h2_opex(
    economics: dict[str, Any],
) -> float:
    """
    Preserve compatibility with current/legacy economics naming.
    """

    candidates = (
        "variable_h2_opex_annual_usd",
        "h2_variable_opex_annual_usd",
        "variable_opex_h2_annual_usd",
    )

    for key in candidates:
        if key in economics:
            return float(
                economics[key]
            )

    return 0.0


def compute_expost_summary(
    dispatch: pd.DataFrame,
    cfg: dict[str, Any],
    capacities: dict[str, float],
) -> dict[str, float]:
    dt = float(
        cfg["data"].get(
            "timestep_hours",
            1.0,
        )
    )

    economics = build_economics_summary(
        config=cfg,
        capacities=capacities,
        dispatch_df=dispatch,
    )

    monthly = compute_monthly_demand_charge(
        dispatch=dispatch,
        cfg=cfg,
        dt=dt,
    )

    annual_demand_charge_usd = float(
        monthly[
            "demand_charge_usd"
        ].sum()
    )

    annualized_capex_usd = float(
        economics.get(
            "annualized_capex_usd",
            0.0,
        )
    )

    fixed_opex_usd = float(
        economics.get(
            "fixed_opex_annual_usd",
            0.0,
        )
    )

    variable_h2_opex_usd = (
        get_variable_h2_opex(
            economics
        )
    )

    grid_energy_cost_usd = float(
        economics.get(
            "grid_opex_annual_usd",
            0.0,
        )
    )

    energy_served_kwh = float(
        economics.get(
            "energy_served_annual_kwh",
            (
                dispatch["demand_kw"]
                * dt
            ).sum(),
        )
    )

    grid_energy_kwh = float(
        (
            dispatch["p_grid_kw"]
            * dt
        ).sum()
    )

    unserved_energy_kwh = float(
        (
            dispatch["p_unserved_kw"]
            * dt
        ).sum()
    )

    annual_cost_historical_usd = (
        annualized_capex_usd
        + fixed_opex_usd
        + variable_h2_opex_usd
        + grid_energy_cost_usd
    )

    annual_cost_harmonized_usd = (
        annual_cost_historical_usd
        + annual_demand_charge_usd
    )

    lcoe_historical_usd_kwh = (
        annual_cost_historical_usd
        / energy_served_kwh
    )

    lcoe_harmonized_usd_kwh = (
        annual_cost_harmonized_usd
        / energy_served_kwh
    )

    p_peak_grid_kw = float(
        dispatch["p_grid_kw"].max()
    )

    total_grid_dependency_ratio = (
        grid_energy_kwh
        / energy_served_kwh
    )

    final_battery_kwh = float(
        dispatch[
            "soc_bat_kwh"
        ].iloc[-1]
    )

    if "h2_level_kg" not in dispatch.columns:
        raise KeyError(
            "Expected H2 dispatch column "
            "'h2_level_kg' not found."
        )

    final_h2_kg = float(
        dispatch[
            "h2_level_kg"
        ].iloc[-1]
    )

    return {
        "P_peak_grid_kw": (
            p_peak_grid_kw
        ),
        "E_grid_total_kwh": (
            grid_energy_kwh
        ),
        "E_load_total_kwh": (
            energy_served_kwh
        ),
        "total_grid_dependency_ratio": (
            total_grid_dependency_ratio
        ),
        "unserved_energy_kwh": (
            unserved_energy_kwh
        ),
        "final_battery_kwh": (
            final_battery_kwh
        ),
        "final_h2_kg": (
            final_h2_kg
        ),
        "capex_total_usd": float(
            economics.get(
                "capex_total_usd",
                0.0,
            )
        ),
        "annualized_capex_usd": (
            annualized_capex_usd
        ),
        "fixed_opex_usd": (
            fixed_opex_usd
        ),
        "variable_h2_opex_usd": (
            variable_h2_opex_usd
        ),
        "grid_energy_cost_usd": (
            grid_energy_cost_usd
        ),
        "annual_demand_charge_usd": (
            annual_demand_charge_usd
        ),
        "annual_cost_historical_usd": (
            annual_cost_historical_usd
        ),
        "annual_cost_harmonized_usd": (
            annual_cost_harmonized_usd
        ),
        "lcoe_historical_usd_kwh": (
            lcoe_historical_usd_kwh
        ),
        "lcoe_harmonized_usd_kwh": (
            lcoe_harmonized_usd_kwh
        ),
    }


def compare_value(
    legacy: float,
    m2: float,
) -> dict[str, float | bool]:
    delta = (
        m2 - legacy
    )

    if legacy != 0:
        delta_percent = (
            delta
            / legacy
            * 100.0
        )
    else:
        delta_percent = (
            0.0
            if delta == 0
            else float("nan")
        )

    equivalent = bool(
        np.isclose(
            legacy,
            m2,
            rtol=REL_TOL,
            atol=max(
                ABS_TOL_POWER_KW,
                ABS_TOL_ENERGY_KWH,
            ),
        )
    )

    return {
        "legacy": legacy,
        "m2_24plus0": m2,
        "delta": delta,
        "delta_percent": (
            delta_percent
        ),
        "equivalent": equivalent,
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # CONFIG / INPUT
    # --------------------------------------------------------

    cfg = load_yaml(
        CONFIG_PATH
    )

    route = str(
        cfg.get(
            "system",
            {},
        ).get(
            "route",
            "hydrogen",
        )
    ).strip().lower()

    if route != "hydrogen":
        raise ValueError(
            f"Expected hydrogen route, "
            f"got {route!r}"
        )

    df_h = load_input_data(
        cfg
    )

    dt = float(
        cfg["data"].get(
            "timestep_hours",
            1.0,
        )
    )

    # --------------------------------------------------------
    # LEGACY REFERENCE DISPATCH
    # --------------------------------------------------------

    if not LEGACY_DISPATCH_PATH.exists():
        raise FileNotFoundError(
            "Legacy M1b dispatch not found.\n"
            f"Expected: {LEGACY_DISPATCH_PATH}\n"
            "Run first:\n"
            "python .\\scripts\\replay_h2_benchmark_m1b.py"
        )

    legacy_dispatch = pd.read_csv(
        LEGACY_DISPATCH_PATH
    )

    if len(
        legacy_dispatch
    ) != 8760:
        raise ValueError(
            "Legacy dispatch must have "
            "8760 rows."
        )

    # --------------------------------------------------------
    # RUN M2 ENGINE WITH ZERO LOOK-AHEAD
    # --------------------------------------------------------

    optimizer = MILPDispatchOptimizer(
        config=cfg,
        capacities=CAPACITIES,
        degradation_model=None,
    )

    result = (
        optimizer
        .run_annual_simulation(
            df=df_h,
            period_hours=PERIOD_HOURS,
            commit_hours=COMMIT_HOURS,
            lookahead_hours=LOOKAHEAD_HOURS,
        )
    )

    if result.dispatch_df.empty:
        raise RuntimeError(
            "M2 24+0 returned empty dispatch. "
            f"solver_status="
            f"{result.solver_status}"
        )

    m2_dispatch = (
        result.dispatch_df
        .copy()
        .reset_index(
            drop=True
        )
    )

    if len(
        m2_dispatch
    ) != 8760:
        raise RuntimeError(
            "M2 committed dispatch must "
            f"contain 8760 rows; got "
            f"{len(m2_dispatch)}."
        )

    # --------------------------------------------------------
    # EX-POST ACCOUNTING
    # --------------------------------------------------------

    legacy_summary = (
        compute_expost_summary(
            dispatch=legacy_dispatch,
            cfg=cfg,
            capacities=CAPACITIES,
        )
    )

    m2_summary = (
        compute_expost_summary(
            dispatch=m2_dispatch,
            cfg=cfg,
            capacities=CAPACITIES,
        )
    )

    legacy_monthly = (
        compute_monthly_demand_charge(
            dispatch=legacy_dispatch,
            cfg=cfg,
            dt=dt,
        )
    )

    m2_monthly = (
        compute_monthly_demand_charge(
            dispatch=m2_dispatch,
            cfg=cfg,
            dt=dt,
        )
    )

    # --------------------------------------------------------
    # DIRECT DISPATCH DIFFERENCES
    # --------------------------------------------------------

    common_columns = [
        column
        for column in (
            "p_grid_kw",
            "p_pv_used_kw",
            "p_pv_curtail_kw",
            "p_bat_ch_kw",
            "p_bat_dis_kw",
            "soc_bat_kwh",
            "p_elz_kw",
            "p_fc_kw",
            "h2_level_kg",
            "p_unserved_kw",
        )
        if (
            column
            in legacy_dispatch.columns
            and column
            in m2_dispatch.columns
        )
    ]

    dispatch_max_abs_diff = {}

    for column in common_columns:
        dispatch_max_abs_diff[
            column
        ] = float(
            (
                m2_dispatch[column]
                - legacy_dispatch[column]
            )
            .abs()
            .max()
        )

    # --------------------------------------------------------
    # METRIC COMPARISON
    # --------------------------------------------------------

    metrics_to_compare = [
        "P_peak_grid_kw",
        "E_grid_total_kwh",
        "E_load_total_kwh",
        "total_grid_dependency_ratio",
        "unserved_energy_kwh",
        "final_battery_kwh",
        "final_h2_kg",
        "capex_total_usd",
        "annualized_capex_usd",
        "fixed_opex_usd",
        "variable_h2_opex_usd",
        "grid_energy_cost_usd",
        "annual_demand_charge_usd",
        "annual_cost_historical_usd",
        "annual_cost_harmonized_usd",
        "lcoe_historical_usd_kwh",
        "lcoe_harmonized_usd_kwh",
    ]

    comparison_rows = []

    all_metrics_equivalent = True

    for metric in metrics_to_compare:
        comparison = (
            compare_value(
                float(
                    legacy_summary[
                        metric
                    ]
                ),
                float(
                    m2_summary[
                        metric
                    ]
                ),
            )
        )

        comparison_rows.append(
            {
                "metric": metric,
                **comparison,
            }
        )

        all_metrics_equivalent = (
            all_metrics_equivalent
            and bool(
                comparison[
                    "equivalent"
                ]
            )
        )

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    # --------------------------------------------------------
    # MONTHLY PEAK COMPARISON
    # --------------------------------------------------------

    monthly_compare = (
        legacy_monthly[
            [
                "month",
                "p_grid_peak_kw",
                "demand_charge_usd",
            ]
        ]
        .rename(
            columns={
                "p_grid_peak_kw":
                    "legacy_peak_kw",
                "demand_charge_usd":
                    "legacy_demand_charge_usd",
            }
        )
        .merge(
            m2_monthly[
                [
                    "month",
                    "p_grid_peak_kw",
                    "demand_charge_usd",
                ]
            ].rename(
                columns={
                    "p_grid_peak_kw":
                        "m2_peak_kw",
                    "demand_charge_usd":
                        "m2_demand_charge_usd",
                }
            ),
            on="month",
            how="inner",
        )
    )

    monthly_compare[
        "delta_peak_kw"
    ] = (
        monthly_compare[
            "m2_peak_kw"
        ]
        - monthly_compare[
            "legacy_peak_kw"
        ]
    )

    monthly_compare[
        "delta_demand_charge_usd"
    ] = (
        monthly_compare[
            "m2_demand_charge_usd"
        ]
        - monthly_compare[
            "legacy_demand_charge_usd"
        ]
    )

    max_monthly_peak_diff_kw = float(
        monthly_compare[
            "delta_peak_kw"
        ].abs().max()
    )

    # --------------------------------------------------------
    # EQUIVALENCE GATES
    # --------------------------------------------------------

    solver_gate = (
        result.solver_status
        in {
            "OPTIMAL",
            "FEASIBLE",
        }
    )

    dispatch_length_gate = (
        len(m2_dispatch)
        == len(
            legacy_dispatch
        )
        == 8760
    )

    monthly_peak_gate = (
        max_monthly_peak_diff_kw
        <= ABS_TOL_POWER_KW
    )

    zero_eens_gate = (
        m2_summary[
            "unserved_energy_kwh"
        ]
        <= ABS_TOL_ENERGY_KWH
    )

    metrics_gate = (
        all_metrics_equivalent
    )

    all_gates_pass = all(
        [
            solver_gate,
            dispatch_length_gate,
            monthly_peak_gate,
            zero_eens_gate,
            metrics_gate,
        ]
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    m2_dispatch.to_csv(
        M2_DISPATCH_PATH,
        index=False,
    )

    m2_monthly.to_csv(
        M2_MONTHLY_PATH,
        index=False,
    )

    comparison_df.to_csv(
        COMPARISON_PATH,
        index=False,
    )

    summary = {
        "case": (
            "H2 M2 engine equivalence "
            "test â€” 24h commit + 0h "
            "lookahead"
        ),

        "configuration": {
            "period_hours": (
                PERIOD_HOURS
            ),
            "commit_hours": (
                COMMIT_HOURS
            ),
            "lookahead_hours": (
                LOOKAHEAD_HOURS
            ),
            "capacities": (
                CAPACITIES
            ),
        },

        "solver_status": (
            result.solver_status
        ),

        "legacy": (
            legacy_summary
        ),

        "m2_24plus0": (
            m2_summary
        ),

        "dispatch_max_abs_diff": (
            dispatch_max_abs_diff
        ),

        "max_monthly_peak_diff_kw": (
            max_monthly_peak_diff_kw
        ),

        "gates": {
            "solver": (
                solver_gate
            ),
            "dispatch_length": (
                dispatch_length_gate
            ),
            "monthly_peak_equivalence": (
                monthly_peak_gate
            ),
            "zero_eens": (
                zero_eens_gate
            ),
            "expost_metric_equivalence": (
                metrics_gate
            ),
            "all_gates": (
                all_gates_pass
            ),
        },
    }

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ========================================================
    # CONSOLE REPORT
    # ========================================================

    print()
    print("=" * 78)
    print(
        "H2 M2 ENGINE EQUIVALENCE "
        "TEST â€” 24h COMMIT + 0h LOOK-AHEAD"
    )
    print("=" * 78)

    print()
    print("CONFIGURATION")
    print(
        f"period_hours                   = "
        f"{PERIOD_HOURS}"
    )
    print(
        f"commit_hours                   = "
        f"{COMMIT_HOURS}"
    )
    print(
        f"lookahead_hours                = "
        f"{LOOKAHEAD_HOURS}"
    )

    for key, value in (
        CAPACITIES.items()
    ):
        print(
            f"{key:<30} = {value}"
        )

    print()
    print(
        f"solver_status                  = "
        f"{result.solver_status}"
    )

    print()
    print("EX-POST COMPARISON")
    print(
        comparison_df.to_string(
            index=False
        )
    )

    print()
    print("MONTHLY PEAK COMPARISON")
    print(
        monthly_compare.to_string(
            index=False
        )
    )

    print()
    print("MAX ABS DISPATCH DIFFERENCE")

    for (
        column,
        value,
    ) in dispatch_max_abs_diff.items():
        print(
            f"{column:<30} = "
            f"{value:.12e}"
        )

    print()
    print("GATES")
    print(
        f"solver                         = "
        f"{'PASS' if solver_gate else 'FAIL'}"
    )
    print(
        f"dispatch_length                = "
        f"{'PASS' if dispatch_length_gate else 'FAIL'}"
    )
    print(
        f"monthly_peak_equivalence       = "
        f"{'PASS' if monthly_peak_gate else 'FAIL'}"
    )
    print(
        f"zero_eens                      = "
        f"{'PASS' if zero_eens_gate else 'FAIL'}"
    )
    print(
        f"expost_metric_equivalence      = "
        f"{'PASS' if metrics_gate else 'FAIL'}"
    )
    print(
        f"all_gates                      = "
        f"{'PASS' if all_gates_pass else 'FAIL'}"
    )

    print()
    print("FILES")
    print(
        M2_DISPATCH_PATH
    )
    print(
        M2_MONTHLY_PATH
    )
    print(
        COMPARISON_PATH
    )
    print(
        SUMMARY_PATH
    )

    print()
    print("=" * 78)

    if not all_gates_pass:
        print(
            "H2 M2 24+0 EQUIVALENCE "
            "GATE: FAIL"
        )
        print(
            "Do not proceed to 24+24 "
            "before resolving the "
            "differences."
        )
        print("=" * 78)
        raise SystemExit(2)

    print(
        "H2 M2 24+0 EQUIVALENCE "
        "GATE: PASS"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()

