from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml


# ============================================================
# PROJECT ROOT / IMPORTS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.milp_dispatch import MILPDispatchOptimizer
from src.economics.lcoe import build_economics_summary


# ============================================================
# PATHS
# ============================================================

DATA_DIR = ROOT / "results" / "paper" / "source_data"

BASE_CONFIG = (
    ROOT
    / "configs"
    / "paper"
    / "pv_bsv_biomethane_1500.yaml"
)

THERMAL_CONFIG = (
    ROOT
    / "configs"
    / "paper"
    / "pv_bsv_biomethane_1500_thermal.yaml"
)

B1_FILE = (
    DATA_DIR
    / "pareto_b1_global_nondominated_with_knee.csv"
)

H2_FILE = (
    DATA_DIR
    / "pareto_h2_global_nondominated_with_knee.csv"
)

OUT_SUMMARY = (
    DATA_DIR
    / "b1_thermal_dispatch_T0_T1_T2_summary.csv"
)

OUT_HOURLY = (
    DATA_DIR
    / "b1_thermal_dispatch_hourly.csv.gz"
)


# ============================================================
# LOAD CONFIGS
# ============================================================

with open(BASE_CONFIG, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

with open(THERMAL_CONFIG, "r", encoding="utf-8") as f:
    thermal_cfg = yaml.safe_load(f)


route = str(
    cfg.get("system", {}).get("route", "")
).strip().lower()

if route != "biomethane":
    raise ValueError(
        f"Base config must be biomethane, found {route!r}"
    )


# ============================================================
# LOAD PARETO FRONTS
# ============================================================

b1 = pd.read_csv(B1_FILE)
h2 = pd.read_csv(H2_FILE)

print("=== INPUT ===")
print("B1 solutions =", len(b1))
print("H2 solutions =", len(h2))

if len(b1) != 34:
    raise ValueError(
        f"Expected 34 B1 solutions, found {len(b1)}"
    )


# ============================================================
# LOAD ORIGINAL ANNUAL INPUT EXACTLY LIKE GA
# ============================================================

data_path = Path(
    cfg["data"]["demand_profile_csv"]
)

if not data_path.is_absolute():
    data_path = ROOT / data_path

if not data_path.exists():
    raise FileNotFoundError(data_path)

df_h = pd.read_csv(data_path)

if "hour" not in df_h.columns:
    df_h["hour"] = range(len(df_h))

required_cols = [
    "hour",
    "demand_kw",
    "pv_factor",
]

missing = [
    c for c in required_cols
    if c not in df_h.columns
]

if missing:
    raise KeyError(
        "Missing annual input columns: "
        + ", ".join(missing)
    )

df_h = (
    df_h
    .sort_values("hour")
    .reset_index(drop=True)
)

df_h["t_global"] = range(len(df_h))

if len(df_h) != 8760:
    raise ValueError(
        f"Expected 8760 rows, found {len(df_h)}"
    )


# ============================================================
# MODEL PARAMETERS
# ============================================================

eta_el_model = float(
    cfg["technology"]["chp"]["eta_el"]
)

eta_el_thermal = float(
    thermal_cfg[
        "technology"
    ][
        "chp_thermal"
    ][
        "electrical_efficiency"
    ]
)

eta_th = float(
    thermal_cfg[
        "technology"
    ][
        "chp_thermal"
    ][
        "thermal_efficiency"
    ]
)

heat_value = float(
    thermal_cfg[
        "thermal_recovery"
    ][
        "heat_reference_cost_usd_per_kwh_th"
    ]
)

dt = float(
    cfg.get(
        "data",
        {}
    ).get(
        "timestep_hours",
        1.0,
    )
)

period_hours = int(
    cfg[
        "optimization"
    ].get(
        "pareto_period_hours",
        24,
    )
)


if abs(
    eta_el_model - eta_el_thermal
) > 1e-12:
    raise ValueError(
        "CHP electrical efficiency mismatch: "
        f"MILP={eta_el_model}, "
        f"thermal={eta_el_thermal}"
    )


print()
print("=== CHP PARAMETERS ===")
print("eta_el =", eta_el_model)
print("eta_th =", eta_th)
print("heat value USD/kWh_th =", heat_value)
print("dt =", dt)
print("period_hours =", period_hours)


# ============================================================
# THERMAL DEMAND PROFILES
# ============================================================

thermal_profiles = {
    "T0": pd.DataFrame({
        "t_global": np.arange(8760),
        "thermal_demand_kw": np.zeros(8760),
    })
}


for case in ["T1", "T2"]:

    rel_path = (
        thermal_cfg[
            "thermal_recovery"
        ][
            "cases"
        ][
            case
        ][
            "demand_profile"
        ]
    )

    path = ROOT / rel_path

    if not path.exists():
        raise FileNotFoundError(path)

    prof = pd.read_csv(path)

    if len(prof) != 8760:
        raise ValueError(
            f"{case}: expected 8760 rows, "
            f"found {len(prof)}"
        )

    for col in [
        "t_global",
        "thermal_demand_kw",
    ]:
        if col not in prof.columns:
            raise KeyError(
                f"{case}: missing column {col}"
            )

    thermal_profiles[case] = (
        prof[
            [
                "t_global",
                "thermal_demand_kw",
            ]
        ]
        .copy()
    )


# ============================================================
# HELPERS
# ============================================================

def dominance_against_h2(
    lcoe: float,
    peak_kw: float,
):

    mask = (
        (
            h2["lcoe_usd_kwh"]
            <= lcoe
        )
        &
        (
            h2["P_peak_grid_opt_kw"]
            <= peak_kw
        )
        &
        (
            (
                h2["lcoe_usd_kwh"]
                < lcoe
            )
            |
            (
                h2["P_peak_grid_opt_kw"]
                < peak_kw
            )
        )
    )

    dominators = h2.loc[mask]

    return (
        not dominators.empty,
        len(dominators),
    )


# ============================================================
# LOOP OVER 34 FIXED B1 SOLUTIONS
# ============================================================

summary_records = []
hourly_records = []


for idx, row in b1.iterrows():

    run_name = str(
        row["source_run"]
    )

    source_solution_id = int(
        row["source_solution_id"]
    )

    print()
    print(
        f"[{idx + 1:02d}/{len(b1)}] "
        f"{run_name} / "
        f"{source_solution_id}"
    )


    capacities = {
        "pv_kw":
            float(row["pv_kw"]),

        "bsv_kwh":
            float(row["bsv_kwh"]),

        "biomethane_storage_nm3":
            float(
                row[
                    "biomethane_storage_nm3"
                ]
            ),

        "chp_kw":
            float(row["chp_kw"]),
    }


    # --------------------------------------------------------
    # RUN FIXED-CAPACITY MILP
    # --------------------------------------------------------

    optimizer = MILPDispatchOptimizer(
        config=cfg,
        capacities=capacities,
        degradation_model=None,
    )

    result = optimizer.run_annual_simulation(
        df=df_h,
        period_hours=period_hours,
    )

    dispatch = result.dispatch_df


    if dispatch.empty:
        raise RuntimeError(
            f"Empty dispatch for "
            f"{run_name}/{source_solution_id}"
        )


    if result.solver_status not in (
        "OPTIMAL",
        "FEASIBLE",
    ):
        raise RuntimeError(
            f"MILP failed for "
            f"{run_name}/{source_solution_id}: "
            f"{result.solver_status}"
        )


    if len(dispatch) != 8760:
        raise ValueError(
            f"{run_name}/{source_solution_id}: "
            f"dispatch rows={len(dispatch)}"
        )


    if "p_chp_kw" not in dispatch.columns:
        raise KeyError(
            "p_chp_kw missing from dispatch_df"
        )


    # --------------------------------------------------------
    # ECONOMIC AUDIT
    # --------------------------------------------------------

    economics = build_economics_summary(
        config=cfg,
        capacities=capacities,
        dispatch_df=dispatch,
    )

    lcoe_recomputed = float(
        economics["lcoe_usd_kwh"]
    )

    lcoe_original = float(
        row["lcoe_usd_kwh"]
    )

    lcoe_error = (
        lcoe_recomputed
        - lcoe_original
    )


    if abs(lcoe_error) > 1e-8:
        raise RuntimeError(
            "LCOE audit failed: "
            f"{run_name}/{source_solution_id} "
            f"stored={lcoe_original:.12f} "
            f"recomputed={lcoe_recomputed:.12f} "
            f"error={lcoe_error:.12e}"
        )


    # --------------------------------------------------------
    # ACTUAL CHP DISPATCH
    # --------------------------------------------------------

    dispatch = dispatch.copy()

    dispatch[
        "chp_el_energy_kwh"
    ] = (
        dispatch["p_chp_kw"]
        * dt
    )


    dispatch[
        "chp_fuel_energy_kwh"
    ] = (
        dispatch[
            "chp_el_energy_kwh"
        ]
        / eta_el_model
    )


    dispatch[
        "heat_available_kwh_th"
    ] = (
        dispatch[
            "chp_fuel_energy_kwh"
        ]
        * eta_th
    )


    chp_el_energy = float(
        dispatch[
            "chp_el_energy_kwh"
        ].sum()
    )

    heat_available = float(
        dispatch[
            "heat_available_kwh_th"
        ].sum()
    )


    chp_kw = capacities["chp_kw"]

    chp_capacity_factor = (
        chp_el_energy
        /
        (
            chp_kw
            * 8760.0
        )
        if chp_kw > 0
        else 0.0
    )


    # --------------------------------------------------------
    # T0 / T1 / T2
    # --------------------------------------------------------

    for case in [
        "T0",
        "T1",
        "T2",
    ]:

        profile = (
            thermal_profiles[case]
            .set_index("t_global")
        )


        tmp = dispatch[
            [
                "t_global",
                "hour",
                "p_chp_kw",
                "biomethane_use_nm3",
                "biomethane_delivery_nm3",
                "biomethane_level_nm3",
                "chp_el_energy_kwh",
                "chp_fuel_energy_kwh",
                "heat_available_kwh_th",
            ]
        ].copy()


        tmp = tmp.join(
            profile,
            on="t_global",
            how="left",
        )


        if (
            tmp[
                "thermal_demand_kw"
            ].isna().any()
        ):
            raise RuntimeError(
                f"{case}: thermal profile "
                "alignment failed"
            )


        tmp[
            "thermal_demand_kwh_th"
        ] = (
            tmp[
                "thermal_demand_kw"
            ]
            * dt
        )


        tmp[
            "heat_useful_kwh_th"
        ] = np.minimum(
            tmp[
                "heat_available_kwh_th"
            ],
            tmp[
                "thermal_demand_kwh_th"
            ],
        )


        tmp[
            "heat_dumped_kwh_th"
        ] = (
            tmp[
                "heat_available_kwh_th"
            ]
            -
            tmp[
                "heat_useful_kwh_th"
            ]
        )


        thermal_demand = float(
            tmp[
                "thermal_demand_kwh_th"
            ].sum()
        )

        heat_useful = float(
            tmp[
                "heat_useful_kwh_th"
            ].sum()
        )

        heat_dumped = float(
            tmp[
                "heat_dumped_kwh_th"
            ].sum()
        )


        heat_utilization_ratio = (
            heat_useful
            / heat_available
            if heat_available > 0
            else 0.0
        )


        demand_coverage_ratio = (
            heat_useful
            / thermal_demand
            if thermal_demand > 0
            else 0.0
        )


        heat_credit = (
            heat_useful
            * heat_value
        )


        e_load = float(
            row["E_load_total_kwh"]
        )


        adjusted_lcoe = (
            lcoe_original
            - heat_credit / e_load
        )


        dominated, n_dominators = (
            dominance_against_h2(
                lcoe=adjusted_lcoe,
                peak_kw=float(
                    row[
                        "P_peak_grid_opt_kw"
                    ]
                ),
            )
        )


        summary_records.append({

            "thermal_case":
                case,

            "source_run":
                run_name,

            "source_solution_id":
                source_solution_id,

            "pv_kw":
                capacities["pv_kw"],

            "bsv_kwh":
                capacities["bsv_kwh"],

            "biomethane_storage_nm3":
                capacities[
                    "biomethane_storage_nm3"
                ],

            "chp_kw":
                chp_kw,

            "P_peak_grid_opt_kw":
                float(
                    row[
                        "P_peak_grid_opt_kw"
                    ]
                ),

            "base_lcoe_usd_kwh":
                lcoe_original,

            "recomputed_lcoe_usd_kwh":
                lcoe_recomputed,

            "lcoe_audit_error":
                lcoe_error,

            "chp_electric_energy_kwh":
                chp_el_energy,

            "chp_capacity_factor":
                chp_capacity_factor,

            "heat_available_kwh_th":
                heat_available,

            "thermal_demand_kwh_th":
                thermal_demand,

            "heat_useful_kwh_th":
                heat_useful,

            "heat_dumped_kwh_th":
                heat_dumped,

            "heat_utilization_ratio":
                heat_utilization_ratio,

            "thermal_demand_coverage_ratio":
                demand_coverage_ratio,

            "heat_credit_usd_per_year":
                heat_credit,

            "adjusted_lcoe_usd_kwh":
                adjusted_lcoe,

            "lcoe_reduction_usd_kwh":
                (
                    lcoe_original
                    - adjusted_lcoe
                ),

            "lcoe_reduction_percent":
                (
                    (
                        lcoe_original
                        - adjusted_lcoe
                    )
                    /
                    lcoe_original
                    * 100.0
                ),

            "dominated_by_h2":
                dominated,

            "n_h2_dominators":
                n_dominators,

            "solver_status":
                result.solver_status,

            "solve_time_sec":
                float(
                    result.solve_time_sec
                ),
        })


        tmp[
            "thermal_case"
        ] = case

        tmp[
            "source_run"
        ] = run_name

        tmp[
            "source_solution_id"
        ] = source_solution_id

        hourly_records.append(tmp)


# ============================================================
# SAVE
# ============================================================

summary = pd.DataFrame(
    summary_records
)

hourly = pd.concat(
    hourly_records,
    ignore_index=True,
)


summary.to_csv(
    OUT_SUMMARY,
    index=False,
)

hourly.to_csv(
    OUT_HOURLY,
    index=False,
    compression="gzip",
)


# ============================================================
# REPORT
# ============================================================

print()
print("=" * 70)
print("THERMAL DISPATCH ANALYSIS COMPLETE")
print("=" * 70)


for case in [
    "T0",
    "T1",
    "T2",
]:

    s = summary[
        summary[
            "thermal_case"
        ] == case
    ]

    print()
    print(f"=== {case} ===")

    print(
        "mean CHP capacity factor =",
        s[
            "chp_capacity_factor"
        ].mean()
    )

    print(
        "mean CHP electric energy MWh =",
        s[
            "chp_electric_energy_kwh"
        ].mean()
        / 1000.0
    )

    print(
        "mean heat available MWh_th =",
        s[
            "heat_available_kwh_th"
        ].mean()
        / 1000.0
    )

    print(
        "mean heat useful MWh_th =",
        s[
            "heat_useful_kwh_th"
        ].mean()
        / 1000.0
    )

    print(
        "mean heat utilization % =",
        s[
            "heat_utilization_ratio"
        ].mean()
        * 100.0
    )

    print(
        "mean thermal demand coverage % =",
        s[
            "thermal_demand_coverage_ratio"
        ].mean()
        * 100.0
    )

    print(
        "mean heat credit USD/year =",
        s[
            "heat_credit_usd_per_year"
        ].mean()
    )

    print(
        "mean adjusted LCOE =",
        s[
            "adjusted_lcoe_usd_kwh"
        ].mean()
    )

    print(
        "mean LCOE reduction % =",
        s[
            "lcoe_reduction_percent"
        ].mean()
    )

    print(
        "B1 still dominated by H2 =",
        int(
            s[
                "dominated_by_h2"
            ].sum()
        ),
        "/",
        len(s),
    )


print()
print("=== LCOE AUDIT ===")

print(
    "max abs error =",
    summary[
        "lcoe_audit_error"
    ].abs().max()
)


print()
print("=== OUTPUT FILES ===")

print(
    OUT_SUMMARY.relative_to(ROOT)
)

print(
    OUT_HOURLY.relative_to(ROOT)
)