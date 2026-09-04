from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.milp_dispatch import MILPDispatchOptimizer


# ============================================================
# FILES
# ============================================================

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "paper"
    / "pv_bsv_biogas_b2_50.yaml"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "validation"
    / "b2_50_physical"
)

DISPATCH_PATH = OUTPUT_DIR / "dispatch_8760.csv"
MONTHLY_PATH = OUTPUT_DIR / "monthly_peaks.csv"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"


# ============================================================
# NUMERICAL TOLERANCES
# ============================================================

ABS_TOL_ELECTRIC_KW = 1.0e-6
ABS_TOL_BIOGAS_NM3 = 1.0e-6
ABS_TOL_STORAGE_NM3 = 1.0e-6
ABS_TOL_POWER_KW = 1.0e-6
ABS_TOL_EENS_KWH = 1.0e-6
ABS_TOL_TERMINAL_NM3 = 1.0e-6


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
            f"Invalid YAML root in {path}"
        )

    return cfg


def resolve_project_path(
    value: str,
) -> Path:
    path = Path(value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def load_dispatch_input(
    cfg: dict[str, Any],
) -> pd.DataFrame:
    data_cfg = cfg["data"]

    input_path = resolve_project_path(
        data_cfg["demand_profile_csv"]
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

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"Demand profile missing columns: "
            f"{sorted(missing)}"
        )

    expected_hours = int(
        data_cfg.get(
            "horizon_hours",
            len(df),
        )
    )

    if len(df) != expected_hours:
        raise ValueError(
            f"Unexpected horizon length: "
            f"{len(df)} rows; "
            f"expected {expected_hours}"
        )

    if expected_hours != 8760:
        raise ValueError(
            "B2-50 physical baseline is defined "
            "for the 8760-hour 2026 horizon."
        )

    return df.reset_index(drop=True)


def build_capacities(
    cfg: dict[str, Any],
) -> dict[str, float]:
    fixed = cfg.get("fixed_case", {})

    if not bool(fixed.get("enabled", False)):
        raise ValueError(
            "fixed_case.enabled must be true "
            "for B2-50 physical replay."
        )

    capacities = {
        "pv_kw": float(fixed["pv_kw"]),
        "bsv_kwh": float(fixed["bsv_kwh"]),
        "chp_kw": float(fixed["chp_kw"]),
        "biogas_storage_nm3": float(
            fixed["biogas_storage_nm3"]
        ),
    }

    for key, value in capacities.items():
        if value <= 0:
            raise ValueError(
                f"Invalid fixed capacity "
                f"{key}={value}"
            )

    return capacities


def assign_month_2026(
    t_global: pd.Series,
) -> pd.Series:
    month_days = [
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
    ]

    month_hours = np.array(
        month_days,
        dtype=int,
    ) * 24

    limits = np.cumsum(month_hours)

    values = t_global.to_numpy(
        dtype=int
    )

    months = (
        np.searchsorted(
            limits,
            values,
            side="right",
        )
        + 1
    )

    return pd.Series(
        months,
        index=t_global.index,
        dtype=int,
    )


def check_close(
    value: float,
    target: float,
    tolerance: float,
) -> bool:
    return abs(value - target) <= tolerance


def gate_label(
    condition: bool,
) -> str:
    return "PASS" if condition else "FAIL"


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    cfg = load_yaml(CONFIG_PATH)

    # --------------------------------------------------------
    # Route verification
    # --------------------------------------------------------

    route = str(
        cfg.get(
            "system",
            {},
        ).get(
            "route",
            "",
        )
    ).strip().lower()

    if route != "biogas":
        raise ValueError(
            f"Expected system.route='biogas', "
            f"got {route!r}"
        )

    # --------------------------------------------------------
    # Input data
    # --------------------------------------------------------

    df_h = load_dispatch_input(cfg)

    dt = float(
        cfg["data"].get(
            "timestep_hours",
            1.0,
        )
    )

    if dt <= 0:
        raise ValueError(
            "data.timestep_hours must be > 0"
        )

    # --------------------------------------------------------
    # Fixed capacities
    # --------------------------------------------------------

    capacities = build_capacities(cfg)

    # --------------------------------------------------------
    # Key B2 physical parameters
    # --------------------------------------------------------

    bg_cfg = cfg["technology"]["biogas"]
    storage_cfg = cfg["technology"]["biogas_storage"]
    chp_cfg = cfg["technology"]["chp_biogas"]

    substrate_t_day = float(
        bg_cfg["substrate_t_day"]
    )

    biogas_yield_nm3_per_t = float(
        bg_cfg["biogas_yield_nm3_per_t"]
    )

    biogas_lhv_kwh_per_nm3 = float(
        bg_cfg["lhv_kwh_per_nm3"]
    )

    parasitic_fraction = float(
        bg_cfg["parasitic_fraction"]
    )

    eta_el = float(
        chp_cfg["eta_el"]
    )

    eta_th = float(
        chp_cfg.get(
            "eta_th",
            0.0,
        )
    )

    min_load_fraction = float(
        chp_cfg["min_load_fraction"]
    )

    availability_fraction = float(
        chp_cfg.get(
            "availability_fraction",
            1.0,
        )
    )

    storage_nm3 = float(
        capacities["biogas_storage_nm3"]
    )

    storage_soc_init = float(
        storage_cfg["soc_init_fraction"]
    )

    storage_soc_min = float(
        storage_cfg["soc_min_fraction"]
    )

    storage_soc_max = float(
        storage_cfg["soc_max_fraction"]
    )

    terminal_cyclic = bool(
        storage_cfg.get(
            "enforce_terminal_cyclic_state",
            False,
        )
    )

    chp_kw = float(
        capacities["chp_kw"]
    )

    min_chp_kw = (
        chp_kw
        * min_load_fraction
    )

    initial_biogas_nm3 = (
        storage_nm3
        * storage_soc_init
    )

    storage_min_nm3 = (
        storage_nm3
        * storage_soc_min
    )

    storage_max_nm3 = (
        storage_nm3
        * storage_soc_max
    )

    expected_biogas_day_nm3 = (
        substrate_t_day
        * biogas_yield_nm3_per_t
    )

    expected_biogas_hour_nm3 = (
        expected_biogas_day_nm3
        / 24.0
    )

    expected_biogas_year_nm3 = (
        expected_biogas_day_nm3
        * 365.0
    )

    expected_substrate_year_t = (
        substrate_t_day
        * 365.0
    )

    expected_net_energy_per_t_kwh = (
        biogas_yield_nm3_per_t
        * biogas_lhv_kwh_per_nm3
        * eta_el
        * (1.0 - parasitic_fraction)
    )

    expected_net_energy_day_kwh = (
        substrate_t_day
        * expected_net_energy_per_t_kwh
    )

    expected_net_energy_year_kwh = (
        expected_net_energy_day_kwh
        * 365.0
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = MILPDispatchOptimizer(
        config=cfg,
        capacities=capacities,
        degradation_model=None,
    )

    period_hours = int(
        cfg["optimization"].get(
            "single_case_period_hours",
            24,
        )
    )

    result = optimizer.run_annual_simulation(
        df=df_h,
        period_hours=period_hours,
    )

    dispatch = result.dispatch_df.copy()

    if dispatch.empty:
        raise RuntimeError(
            "B2-50 dispatch is empty. "
            f"solver_status={result.solver_status}"
        )

    # --------------------------------------------------------
    # Required dispatch columns
    # --------------------------------------------------------

    required_dispatch_columns = {
        "t_global",
        "hour",
        "demand_kw",
        "pv_kw",
        "p_grid_kw",
        "p_pv_used_kw",
        "p_pv_curtail_kw",
        "p_bat_ch_kw",
        "p_bat_dis_kw",
        "soc_bat_kwh",
        "p_unserved_kw",
        "p_chp_net_kw",
        "p_chp_gross_equiv_kw",
        "p_biogas_aux_kw",
        "u_chp_on",
        "biogas_production_nm3",
        "biogas_use_nm3",
        "biogas_level_nm3",
    }

    missing_dispatch = (
        required_dispatch_columns
        .difference(dispatch.columns)
    )

    if missing_dispatch:
        raise RuntimeError(
            "B2 dispatch missing audit columns: "
            f"{sorted(missing_dispatch)}"
        )

    # ========================================================
    # PHYSICAL AUDIT
    # ========================================================

    # --------------------------------------------------------
    # 1. Electrical balance
    #
    # grid + PV + battery discharge + CHP net + unserved
    # =
    # demand + battery charging
    # --------------------------------------------------------

    dispatch["electric_balance_residual_kw"] = (
        dispatch["p_grid_kw"]
        + dispatch["p_pv_used_kw"]
        + dispatch["p_bat_dis_kw"]
        + dispatch["p_chp_net_kw"]
        + dispatch["p_unserved_kw"]
        - dispatch["demand_kw"]
        - dispatch["p_bat_ch_kw"]
    )

    max_abs_electric_residual_kw = float(
        dispatch[
            "electric_balance_residual_kw"
        ]
        .abs()
        .max()
    )

    # --------------------------------------------------------
    # 2. Biogas storage balance
    # --------------------------------------------------------

    previous_level = (
        dispatch["biogas_level_nm3"]
        .shift(1)
    )

    previous_level.iloc[0] = (
        initial_biogas_nm3
    )

    dispatch["biogas_balance_residual_nm3"] = (
        dispatch["biogas_level_nm3"]
        - previous_level
        - dispatch["biogas_production_nm3"]
        + dispatch["biogas_use_nm3"]
    )

    max_abs_biogas_residual_nm3 = float(
        dispatch[
            "biogas_balance_residual_nm3"
        ]
        .abs()
        .max()
    )

    # --------------------------------------------------------
    # 3. EENS
    # --------------------------------------------------------

    unserved_energy_kwh = float(
        (
            dispatch["p_unserved_kw"]
            * dt
        ).sum()
    )

    # --------------------------------------------------------
    # 4. Storage bounds
    # --------------------------------------------------------

    gasometer_min_nm3 = float(
        dispatch[
            "biogas_level_nm3"
        ].min()
    )

    gasometer_max_nm3 = float(
        dispatch[
            "biogas_level_nm3"
        ].max()
    )

    storage_lower_violation_nm3 = max(
        0.0,
        storage_min_nm3
        - gasometer_min_nm3,
    )

    storage_upper_violation_nm3 = max(
        0.0,
        gasometer_max_nm3
        - storage_max_nm3,
    )

    # --------------------------------------------------------
    # 5. CHP minimum-load logic
    # --------------------------------------------------------

    p_chp = dispatch[
        "p_chp_net_kw"
    ].to_numpy(dtype=float)

    u_chp = dispatch[
        "u_chp_on"
    ].to_numpy(dtype=int)

    off_power_violation = np.where(
        u_chp == 0,
        np.maximum(
            p_chp,
            0.0,
        ),
        0.0,
    )

    on_min_load_violation = np.where(
        u_chp == 1,
        np.maximum(
            min_chp_kw - p_chp,
            0.0,
        ),
        0.0,
    )

    max_chp_off_violation_kw = float(
        off_power_violation.max()
    )

    max_chp_min_load_violation_kw = float(
        on_min_load_violation.max()
    )

    max_chp_power_violation_kw = max(
        0.0,
        float(
            dispatch[
                "p_chp_net_kw"
            ].max()
        )
        - chp_kw,
    )

    # --------------------------------------------------------
    # 6. Annual resource conservation
    # --------------------------------------------------------

    biogas_production_annual_nm3 = float(
        dispatch[
            "biogas_production_nm3"
        ].sum()
    )

    biogas_use_annual_nm3 = float(
        dispatch[
            "biogas_use_nm3"
        ].sum()
    )

    biogas_inventory_change_nm3 = (
        float(
            dispatch[
                "biogas_level_nm3"
            ].iloc[-1]
        )
        - initial_biogas_nm3
    )

    annual_biogas_conservation_residual_nm3 = (
        biogas_production_annual_nm3
        - biogas_use_annual_nm3
        - biogas_inventory_change_nm3
    )

    # --------------------------------------------------------
    # 7. Terminal gas-holder condition
    # --------------------------------------------------------

    final_biogas_nm3 = float(
        result.final_biogas_nm3
    )

    terminal_state_delta_nm3 = (
        final_biogas_nm3
        - initial_biogas_nm3
    )

    # --------------------------------------------------------
    # 8. CHP energies
    # --------------------------------------------------------

    chp_net_energy_kwh = float(
        (
            dispatch[
                "p_chp_net_kw"
            ]
            * dt
        ).sum()
    )

    chp_gross_equiv_energy_kwh = float(
        (
            dispatch[
                "p_chp_gross_equiv_kw"
            ]
            * dt
        ).sum()
    )

    auxiliary_energy_kwh = float(
        (
            dispatch[
                "p_biogas_aux_kw"
            ]
            * dt
        ).sum()
    )

    # --------------------------------------------------------
    # 9. CHP utilization
    # --------------------------------------------------------

    chp_operating_hours = float(
        (
            dispatch["u_chp_on"]
            * dt
        ).sum()
    )

    chp_equivalent_full_load_hours_net = (
        chp_net_energy_kwh
        / chp_kw
        if chp_kw > 0
        else 0.0
    )

    chp_capacity_factor_net = (
        chp_net_energy_kwh
        / (
            chp_kw
            * len(dispatch)
            * dt
        )
        if chp_kw > 0
        else 0.0
    )

    # --------------------------------------------------------
    # 10. Grid / load
    # --------------------------------------------------------

    annual_pgrid_peak_kw = float(
        dispatch[
            "p_grid_kw"
        ].max()
    )

    grid_energy_annual_kwh = float(
        (
            dispatch[
                "p_grid_kw"
            ]
            * dt
        ).sum()
    )

    load_energy_annual_kwh = float(
        (
            dispatch[
                "demand_kw"
            ]
            * dt
        ).sum()
    )

    total_grid_dependency_ratio = (
        grid_energy_annual_kwh
        / load_energy_annual_kwh
        if load_energy_annual_kwh > 0
        else float("nan")
    )

    # --------------------------------------------------------
    # 11. Monthly grid peaks
    # --------------------------------------------------------

    dispatch["month"] = assign_month_2026(
        dispatch["t_global"]
    )

    monthly = (
        dispatch
        .groupby(
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
            e_chp_net_kwh=(
                "p_chp_net_kw",
                lambda s: float(
                    s.sum() * dt
                ),
            ),
            biogas_use_nm3=(
                "biogas_use_nm3",
                "sum",
            ),
        )
    )

    demand_charge_rate = float(
        cfg["tariff"].get(
            "demand_charge_usd_kw_month",
            0.0,
        )
    )

    monthly[
        "demand_charge_usd"
    ] = (
        monthly[
            "p_grid_peak_kw"
        ]
        * demand_charge_rate
    )

    annual_demand_charge_usd = float(
        monthly[
            "demand_charge_usd"
        ].sum()
    )

    # ========================================================
    # GATES
    # ========================================================

    solver_gate = (
        result.solver_status
        in {
            "OPTIMAL",
            "FEASIBLE",
        }
    )

    electric_balance_gate = (
        max_abs_electric_residual_kw
        <= ABS_TOL_ELECTRIC_KW
    )

    biogas_balance_gate = (
        max_abs_biogas_residual_nm3
        <= ABS_TOL_BIOGAS_NM3
    )

    eens_gate = (
        unserved_energy_kwh
        <= ABS_TOL_EENS_KWH
    )

    storage_gate = (
        storage_lower_violation_nm3
        <= ABS_TOL_STORAGE_NM3
        and storage_upper_violation_nm3
        <= ABS_TOL_STORAGE_NM3
    )

    chp_min_load_gate = (
        max_chp_off_violation_kw
        <= ABS_TOL_POWER_KW
        and max_chp_min_load_violation_kw
        <= ABS_TOL_POWER_KW
        and max_chp_power_violation_kw
        <= ABS_TOL_POWER_KW
    )

    annual_resource_gate = (
        abs(
            annual_biogas_conservation_residual_nm3
        )
        <= ABS_TOL_BIOGAS_NM3
    )

    production_gate = check_close(
        biogas_production_annual_nm3,
        expected_biogas_year_nm3,
        max(
            ABS_TOL_BIOGAS_NM3,
            expected_biogas_year_nm3
            * 1.0e-9,
        ),
    )

    if terminal_cyclic:
        terminal_gate = (
            abs(
                terminal_state_delta_nm3
            )
            <= ABS_TOL_TERMINAL_NM3
        )
    else:
        terminal_gate = True

    all_gates_pass = all(
        [
            solver_gate,
            electric_balance_gate,
            biogas_balance_gate,
            eens_gate,
            storage_gate,
            chp_min_load_gate,
            annual_resource_gate,
            production_gate,
            terminal_gate,
        ]
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {
        "case": "B2-50 physical validation",
        "route": route,
        "config_file": str(CONFIG_PATH),
        "input_file": str(
            resolve_project_path(
                cfg["data"][
                    "demand_profile_csv"
                ]
            )
        ),

        "period_hours": period_hours,
        "solver_status": result.solver_status,
        "solve_time_sec": float(
            result.solve_time_sec
        ),

        "capacities": {
            "pv_kw": capacities["pv_kw"],
            "bsv_kwh": capacities["bsv_kwh"],
            "chp_kw_net": chp_kw,
            "biogas_storage_nm3": (
                storage_nm3
            ),
        },

        "biogas_parameters": {
            "substrate_t_day": (
                substrate_t_day
            ),
            "substrate_t_year": (
                expected_substrate_year_t
            ),
            "biogas_yield_nm3_per_t": (
                biogas_yield_nm3_per_t
            ),
            "biogas_lhv_kwh_per_nm3": (
                biogas_lhv_kwh_per_nm3
            ),
            "eta_el": eta_el,
            "eta_th": eta_th,
            "parasitic_fraction": (
                parasitic_fraction
            ),
            "min_load_fraction": (
                min_load_fraction
            ),
            "availability_fraction": (
                availability_fraction
            ),
            "availability_applied_temporally": False,
        },

        "expected_resource": {
            "biogas_day_nm3": (
                expected_biogas_day_nm3
            ),
            "biogas_hour_nm3": (
                expected_biogas_hour_nm3
            ),
            "biogas_year_nm3": (
                expected_biogas_year_nm3
            ),
            "net_energy_per_t_kwh": (
                expected_net_energy_per_t_kwh
            ),
            "net_energy_day_kwh": (
                expected_net_energy_day_kwh
            ),
            "net_energy_year_kwh": (
                expected_net_energy_year_kwh
            ),
        },

        "physical_results": {
            "biogas_production_annual_nm3": (
                biogas_production_annual_nm3
            ),
            "biogas_use_annual_nm3": (
                biogas_use_annual_nm3
            ),
            "biogas_inventory_change_nm3": (
                biogas_inventory_change_nm3
            ),
            "annual_biogas_conservation_residual_nm3": (
                annual_biogas_conservation_residual_nm3
            ),

            "initial_biogas_nm3": (
                initial_biogas_nm3
            ),
            "final_biogas_nm3": (
                final_biogas_nm3
            ),
            "terminal_state_delta_nm3": (
                terminal_state_delta_nm3
            ),

            "gasometer_min_nm3": (
                gasometer_min_nm3
            ),
            "gasometer_max_nm3": (
                gasometer_max_nm3
            ),
            "gasometer_allowed_min_nm3": (
                storage_min_nm3
            ),
            "gasometer_allowed_max_nm3": (
                storage_max_nm3
            ),

            "chp_net_energy_kwh": (
                chp_net_energy_kwh
            ),
            "chp_gross_equiv_energy_kwh": (
                chp_gross_equiv_energy_kwh
            ),
            "auxiliary_energy_kwh": (
                auxiliary_energy_kwh
            ),

            "chp_operating_hours": (
                chp_operating_hours
            ),
            "chp_equivalent_full_load_hours_net": (
                chp_equivalent_full_load_hours_net
            ),
            "chp_capacity_factor_net": (
                chp_capacity_factor_net
            ),

            "grid_energy_annual_kwh": (
                grid_energy_annual_kwh
            ),
            "load_energy_annual_kwh": (
                load_energy_annual_kwh
            ),
            "total_grid_dependency_ratio": (
                total_grid_dependency_ratio
            ),
            "annual_pgrid_peak_kw": (
                annual_pgrid_peak_kw
            ),

            "unserved_energy_kwh": (
                unserved_energy_kwh
            ),

            "annual_demand_charge_usd": (
                annual_demand_charge_usd
            ),
        },

        "audit": {
            "max_abs_electric_balance_residual_kw": (
                max_abs_electric_residual_kw
            ),
            "max_abs_biogas_balance_residual_nm3": (
                max_abs_biogas_residual_nm3
            ),
            "storage_lower_violation_nm3": (
                storage_lower_violation_nm3
            ),
            "storage_upper_violation_nm3": (
                storage_upper_violation_nm3
            ),
            "max_chp_off_violation_kw": (
                max_chp_off_violation_kw
            ),
            "max_chp_min_load_violation_kw": (
                max_chp_min_load_violation_kw
            ),
            "max_chp_power_violation_kw": (
                max_chp_power_violation_kw
            ),
        },

        "gates": {
            "solver": {
                "pass": solver_gate,
                "status": gate_label(
                    solver_gate
                ),
            },
            "electric_balance": {
                "pass": electric_balance_gate,
                "status": gate_label(
                    electric_balance_gate
                ),
            },
            "biogas_balance": {
                "pass": biogas_balance_gate,
                "status": gate_label(
                    biogas_balance_gate
                ),
            },
            "zero_eens": {
                "pass": eens_gate,
                "status": gate_label(
                    eens_gate
                ),
            },
            "storage_bounds": {
                "pass": storage_gate,
                "status": gate_label(
                    storage_gate
                ),
            },
            "chp_min_load": {
                "pass": chp_min_load_gate,
                "status": gate_label(
                    chp_min_load_gate
                ),
            },
            "annual_biogas_conservation": {
                "pass": annual_resource_gate,
                "status": gate_label(
                    annual_resource_gate
                ),
            },
            "expected_biogas_production": {
                "pass": production_gate,
                "status": gate_label(
                    production_gate
                ),
            },
            "terminal_storage": {
                "pass": terminal_gate,
                "status": gate_label(
                    terminal_gate
                ),
            },
            "all_gates": {
                "pass": all_gates_pass,
                "status": gate_label(
                    all_gates_pass
                ),
            },
        },
    }

    # ========================================================
    # SAVE FILES
    # ========================================================

    dispatch.to_csv(
        DISPATCH_PATH,
        index=False,
    )

    monthly.to_csv(
        MONTHLY_PATH,
        index=False,
    )

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
    print("=" * 76)
    print("B2-50 PHYSICAL VALIDATION REPLAY")
    print("=" * 76)

    print()
    print("CONFIGURATION")
    print(
        f"route                         = "
        f"{route}"
    )
    print(
        f"pv_kw                         = "
        f"{capacities['pv_kw']}"
    )
    print(
        f"bsv_kwh                       = "
        f"{capacities['bsv_kwh']}"
    )
    print(
        f"chp_kw_net                    = "
        f"{chp_kw}"
    )
    print(
        f"biogas_storage_nm3            = "
        f"{storage_nm3}"
    )
    print(
        f"substrate_t_day               = "
        f"{substrate_t_day}"
    )
    print(
        f"biogas_yield_nm3_per_t        = "
        f"{biogas_yield_nm3_per_t}"
    )
    print(
        f"biogas_production_nm3_day     = "
        f"{expected_biogas_day_nm3}"
    )
    print(
        f"eta_el                         = "
        f"{eta_el}"
    )
    print(
        f"parasitic_fraction             = "
        f"{parasitic_fraction}"
    )
    print(
        f"min_load_fraction              = "
        f"{min_load_fraction}"
    )
    print(
        f"availability_fraction          = "
        f"{availability_fraction}"
    )
    print(
        "availability_applied          = "
        "False"
    )

    print()
    print(
        f"period_hours                   = "
        f"{period_hours}"
    )
    print(
        f"solver_status                  = "
        f"{result.solver_status}"
    )

    print()
    print("PHYSICAL RESULTS")
    print(
        f"biogas_production_annual_nm3   = "
        f"{biogas_production_annual_nm3}"
    )
    print(
        f"biogas_use_annual_nm3          = "
        f"{biogas_use_annual_nm3}"
    )
    print(
        f"initial_biogas_nm3             = "
        f"{initial_biogas_nm3}"
    )
    print(
        f"final_biogas_nm3               = "
        f"{final_biogas_nm3}"
    )
    print(
        f"gasometer_min_nm3              = "
        f"{gasometer_min_nm3}"
    )
    print(
        f"gasometer_max_nm3              = "
        f"{gasometer_max_nm3}"
    )

    print(
        f"chp_net_energy_kwh             = "
        f"{chp_net_energy_kwh}"
    )
    print(
        f"chp_gross_equiv_energy_kwh     = "
        f"{chp_gross_equiv_energy_kwh}"
    )
    print(
        f"auxiliary_energy_kwh           = "
        f"{auxiliary_energy_kwh}"
    )
    print(
        f"chp_operating_hours            = "
        f"{chp_operating_hours}"
    )
    print(
        f"chp_equiv_full_load_hours      = "
        f"{chp_equivalent_full_load_hours_net}"
    )

    print(
        f"E_grid_total_kwh               = "
        f"{grid_energy_annual_kwh}"
    )
    print(
        f"E_load_total_kwh               = "
        f"{load_energy_annual_kwh}"
    )
    print(
        f"total_grid_dependency_ratio    = "
        f"{total_grid_dependency_ratio}"
    )
    print(
        f"P_peak_grid_kw                 = "
        f"{annual_pgrid_peak_kw}"
    )
    print(
        f"unserved_energy_kwh            = "
        f"{unserved_energy_kwh}"
    )

    print()
    print("MONTHLY PEAKS")
    print(
        monthly.to_string(
            index=False
        )
    )

    print()
    print("AUDIT")
    print(
        f"max_electric_residual_kw       = "
        f"{max_abs_electric_residual_kw:.12e}"
    )
    print(
        f"max_biogas_residual_nm3        = "
        f"{max_abs_biogas_residual_nm3:.12e}"
    )
    print(
        f"annual_biogas_residual_nm3     = "
        f"{annual_biogas_conservation_residual_nm3:.12e}"
    )
    print(
        f"terminal_state_delta_nm3       = "
        f"{terminal_state_delta_nm3:.12e}"
    )
    print(
        f"storage_lower_violation_nm3    = "
        f"{storage_lower_violation_nm3:.12e}"
    )
    print(
        f"storage_upper_violation_nm3    = "
        f"{storage_upper_violation_nm3:.12e}"
    )
    print(
        f"max_chp_off_violation_kw       = "
        f"{max_chp_off_violation_kw:.12e}"
    )
    print(
        f"max_chp_min_load_violation_kw  = "
        f"{max_chp_min_load_violation_kw:.12e}"
    )
    print(
        f"max_chp_power_violation_kw     = "
        f"{max_chp_power_violation_kw:.12e}"
    )

    print()
    print("GATES")

    for gate_name, gate_data in (
        summary["gates"].items()
    ):
        print(
            f"{gate_name:<34} = "
            f"{gate_data['status']}"
        )

    print()
    print("FILES")
    print(DISPATCH_PATH)
    print(MONTHLY_PATH)
    print(SUMMARY_PATH)

    print()
    print("=" * 76)

    if not all_gates_pass:
        print(
            "B2-50 PHYSICAL GATE: FAIL"
        )
        print(
            "Inspect the audit metrics above "
            "before changing economics or NSGA-II."
        )
        print("=" * 76)

        raise SystemExit(2)

    print(
        "B2-50 PHYSICAL GATE: PASS"
    )
    print("=" * 76)


if __name__ == "__main__":
    main()