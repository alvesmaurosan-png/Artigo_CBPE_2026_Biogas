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
    / "h2_m2_24plus24"
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
    / "comparison_m1b_vs_m2_24plus24.csv"
)

MONTHLY_COMPARISON_PATH = (
    OUTPUT_DIR
    / "monthly_comparison_m1b_vs_m2_24plus24.csv"
)

HOURLY_PROFILE_PATH = (
    OUTPUT_DIR
    / "hourly_dispatch_profile.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "summary.json"
)


# ============================================================
# M2 SETTINGS
# ============================================================

PERIOD_HOURS = 24
COMMIT_HOURS = 24
LOOKAHEAD_HOURS = 24


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


# ============================================================
# HELPERS
# ============================================================

def load_yaml(
    path: Path,
) -> dict[str, Any]:

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
            e_fc_kwh=(
                "p_fc_kw",
                lambda s: float(
                    s.sum() * dt
                ),
            ),
            e_elz_kwh=(
                "p_elz_kw",
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

    if (
        "h2_level_kg"
        not in dispatch.columns
    ):
        raise KeyError(
            "Expected H2 dispatch column "
            "'h2_level_kg' not found."
        )

    final_h2_kg = float(
        dispatch[
            "h2_level_kg"
        ].iloc[-1]
    )

    fc_energy_kwh = float(
        (
            dispatch["p_fc_kw"]
            * dt
        ).sum()
    )

    elz_energy_kwh = float(
        (
            dispatch["p_elz_kw"]
            * dt
        ).sum()
    )

    battery_charge_kwh = float(
        (
            dispatch[
                "p_bat_ch_kw"
            ]
            * dt
        ).sum()
    )

    battery_discharge_kwh = float(
        (
            dispatch[
                "p_bat_dis_kw"
            ]
            * dt
        ).sum()
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
        "fc_energy_kwh": (
            fc_energy_kwh
        ),
        "elz_energy_kwh": (
            elz_energy_kwh
        ),
        "battery_charge_kwh": (
            battery_charge_kwh
        ),
        "battery_discharge_kwh": (
            battery_discharge_kwh
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


def comparison_row(
    metric: str,
    legacy: float,
    m2: float,
) -> dict[str, float | str]:

    delta = m2 - legacy

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

    return {
        "metric": metric,
        "legacy": legacy,
        "m2_24plus24": m2,
        "delta": delta,
        "delta_percent": delta_percent,
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
    # LEGACY REFERENCE
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
            "Legacy dispatch must "
            "contain 8760 rows."
        )

    # --------------------------------------------------------
    # RUN M2 — 24h COMMIT + 24h LOOK-AHEAD
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
            "M2 24+24 returned empty dispatch. "
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
            "M2 committed dispatch "
            f"contains {len(m2_dispatch)} "
            "rows; expected 8760."
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
    # METRIC COMPARISON
    # --------------------------------------------------------

    metrics = [
        "P_peak_grid_kw",
        "E_grid_total_kwh",
        "E_load_total_kwh",
        "total_grid_dependency_ratio",
        "unserved_energy_kwh",
        "final_battery_kwh",
        "final_h2_kg",
        "fc_energy_kwh",
        "elz_energy_kwh",
        "battery_charge_kwh",
        "battery_discharge_kwh",
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

    for metric in metrics:
        comparison_rows.append(
            comparison_row(
                metric=metric,
                legacy=float(
                    legacy_summary[
                        metric
                    ]
                ),
                m2=float(
                    m2_summary[
                        metric
                    ]
                ),
            )
        )

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    # --------------------------------------------------------
    # MONTHLY COMPARISON
    # --------------------------------------------------------

    monthly_compare = (
        legacy_monthly
        .rename(
            columns={
                "p_grid_peak_kw":
                    "legacy_peak_kw",
                "e_grid_kwh":
                    "legacy_grid_kwh",
                "e_fc_kwh":
                    "legacy_fc_kwh",
                "e_elz_kwh":
                    "legacy_elz_kwh",
                "demand_charge_usd":
                    "legacy_demand_charge_usd",
            }
        )
        .merge(
            m2_monthly.rename(
                columns={
                    "p_grid_peak_kw":
                        "m2_peak_kw",
                    "e_grid_kwh":
                        "m2_grid_kwh",
                    "e_fc_kwh":
                        "m2_fc_kwh",
                    "e_elz_kwh":
                        "m2_elz_kwh",
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
        "delta_grid_kwh"
    ] = (
        monthly_compare[
            "m2_grid_kwh"
        ]
        - monthly_compare[
            "legacy_grid_kwh"
        ]
    )

    monthly_compare[
        "delta_fc_kwh"
    ] = (
        monthly_compare[
            "m2_fc_kwh"
        ]
        - monthly_compare[
            "legacy_fc_kwh"
        ]
    )

    monthly_compare[
        "delta_elz_kwh"
    ] = (
        monthly_compare[
            "m2_elz_kwh"
        ]
        - monthly_compare[
            "legacy_elz_kwh"
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

    # --------------------------------------------------------
    # LOCAL-HOUR PROFILE
    # --------------------------------------------------------

    m2_dispatch[
        "local_hour"
    ] = (
        m2_dispatch[
            "t_global"
        ]
        % 24
    )

    hourly_profile = (
        m2_dispatch
        .groupby(
            "local_hour",
            as_index=False,
        )
        .agg(
            grid_kwh=(
                "p_grid_kw",
                lambda s: float(
                    s.sum() * dt
                ),
            ),
            fc_kwh=(
                "p_fc_kw",
                lambda s: float(
                    s.sum() * dt
                ),
            ),
            elz_kwh=(
                "p_elz_kw",
                lambda s: float(
                    s.sum() * dt
                ),
            ),
            battery_charge_kwh=(
                "p_bat_ch_kw",
                lambda s: float(
                    s.sum() * dt
                ),
            ),
            battery_discharge_kwh=(
                "p_bat_dis_kw",
                lambda s: float(
                    s.sum() * dt
                ),
            ),
            avg_h2_level_kg=(
                "h2_level_kg",
                "mean",
            ),
        )
    )

    # --------------------------------------------------------
    # PEAK-HOUR DIAGNOSTIC
    # --------------------------------------------------------

    m2_month_index = assign_month_2026(
        m2_dispatch[
            "t_global"
        ]
    )

    m2_dispatch[
        "month"
    ] = m2_month_index

    peak_idx = (
        m2_dispatch
        .groupby(
            "month"
        )[
            "p_grid_kw"
        ]
        .idxmax()
    )

    peak_hours = (
        m2_dispatch
        .loc[
            peak_idx,
            [
                "month",
                "t_global",
                "local_hour",
                "demand_kw",
                "p_grid_kw",
                "p_fc_kw",
                "p_elz_kw",
                "p_bat_ch_kw",
                "p_bat_dis_kw",
                "soc_bat_kwh",
                "h2_level_kg",
            ],
        ]
        .copy()
    )

    # --------------------------------------------------------
    # GATES
    #
    # Unlike 24+0, we DO NOT require equivalence with legacy.
    # 24+24 is expected to change the dispatch.
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
        == 8760
    )

    zero_eens_gate = (
        m2_summary[
            "unserved_energy_kwh"
        ]
        <= ABS_TOL_ENERGY_KWH
    )

    finite_metrics_gate = all(
        np.isfinite(
            float(
                m2_summary[
                    metric
                ]
            )
        )
        for metric in metrics
    )

    all_gates_pass = all(
        [
            solver_gate,
            dispatch_length_gate,
            zero_eens_gate,
            finite_metrics_gate,
        ]
    )

    # --------------------------------------------------------
    # SAVE FILES
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

    monthly_compare.to_csv(
        MONTHLY_COMPARISON_PATH,
        index=False,
    )

    hourly_profile.to_csv(
        HOURLY_PROFILE_PATH,
        index=False,
    )

    peak_hours_path = (
        OUTPUT_DIR
        / "monthly_peak_hours.csv"
    )

    peak_hours.to_csv(
        peak_hours_path,
        index=False,
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary = {
        "case": (
            "H2 M2 — 24h commit + "
            "24h lookahead"
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

        "legacy_m1b": (
            legacy_summary
        ),

        "m2_24plus24": (
            m2_summary
        ),

        "headline_changes": {
            "delta_peak_grid_kw": (
                m2_summary[
                    "P_peak_grid_kw"
                ]
                - legacy_summary[
                    "P_peak_grid_kw"
                ]
            ),
            "delta_grid_energy_kwh": (
                m2_summary[
                    "E_grid_total_kwh"
                ]
                - legacy_summary[
                    "E_grid_total_kwh"
                ]
            ),
            "delta_demand_charge_usd": (
                m2_summary[
                    "annual_demand_charge_usd"
                ]
                - legacy_summary[
                    "annual_demand_charge_usd"
                ]
            ),
            "delta_lcoe_harmonized_usd_kwh": (
                m2_summary[
                    "lcoe_harmonized_usd_kwh"
                ]
                - legacy_summary[
                    "lcoe_harmonized_usd_kwh"
                ]
            ),
        },

        "gates": {
            "solver": (
                solver_gate
            ),
            "dispatch_length": (
                dispatch_length_gate
            ),
            "zero_eens": (
                zero_eens_gate
            ),
            "finite_metrics": (
                finite_metrics_gate
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
    print("=" * 80)
    print(
        "H2 M2 — 24h COMMIT "
        "+ 24h LOOK-AHEAD"
    )
    print("=" * 80)

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
    print("EX-POST M1b LEGACY × M2 24+24")
    print(
        comparison_df.to_string(
            index=False
        )
    )

    print()
    print("MONTHLY COMPARISON")
    print(
        monthly_compare.to_string(
            index=False
        )
    )

    print()
    print("MONTHLY PEAK HOURS — M2")
    print(
        peak_hours.to_string(
            index=False
        )
    )

    print()
    print("HEADLINE RESULTS")
    print(
        f"P_peak legacy                  = "
        f"{legacy_summary['P_peak_grid_kw']}"
    )
    print(
        f"P_peak M2                      = "
        f"{m2_summary['P_peak_grid_kw']}"
    )
    print(
        f"delta P_peak                   = "
        f"{m2_summary['P_peak_grid_kw'] - legacy_summary['P_peak_grid_kw']}"
    )

    print(
        f"E_grid legacy                  = "
        f"{legacy_summary['E_grid_total_kwh']}"
    )
    print(
        f"E_grid M2                      = "
        f"{m2_summary['E_grid_total_kwh']}"
    )

    print(
        f"demand charge legacy           = "
        f"{legacy_summary['annual_demand_charge_usd']}"
    )
    print(
        f"demand charge M2               = "
        f"{m2_summary['annual_demand_charge_usd']}"
    )

    print(
        f"LCOE harmonized legacy         = "
        f"{legacy_summary['lcoe_harmonized_usd_kwh']}"
    )
    print(
        f"LCOE harmonized M2             = "
        f"{m2_summary['lcoe_harmonized_usd_kwh']}"
    )

    print(
        f"EENS M2                        = "
        f"{m2_summary['unserved_energy_kwh']}"
    )

    print(
        f"final battery M2               = "
        f"{m2_summary['final_battery_kwh']}"
    )
    print(
        f"final H2 M2                    = "
        f"{m2_summary['final_h2_kg']}"
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
        f"zero_eens                      = "
        f"{'PASS' if zero_eens_gate else 'FAIL'}"
    )
    print(
        f"finite_metrics                 = "
        f"{'PASS' if finite_metrics_gate else 'FAIL'}"
    )
    print(
        f"all_gates                      = "
        f"{'PASS' if all_gates_pass else 'FAIL'}"
    )

    print()
    print("FILES")
    print(M2_DISPATCH_PATH)
    print(M2_MONTHLY_PATH)
    print(COMPARISON_PATH)
    print(MONTHLY_COMPARISON_PATH)
    print(HOURLY_PROFILE_PATH)
    print(peak_hours_path)
    print(SUMMARY_PATH)

    print()
    print("=" * 80)

    if not all_gates_pass:
        print(
            "H2 M2 24+24 VALIDATION "
            "GATE: FAIL"
        )
        print("=" * 80)
        raise SystemExit(2)

    print(
        "H2 M2 24+24 VALIDATION "
        "GATE: PASS"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()