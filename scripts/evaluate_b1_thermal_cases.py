from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "results" / "paper" / "source_data"

B1_FILE = (
    DATA_DIR
    / "pareto_b1_global_nondominated_with_knee.csv"
)

H2_FILE = (
    DATA_DIR
    / "pareto_h2_global_nondominated_with_knee.csv"
)

CONFIG_FILE = (
    ROOT
    / "configs"
    / "paper"
    / "pv_bsv_biomethane_1500_thermal.yaml"
)

OUT_DETAIL = (
    DATA_DIR
    / "b1_thermal_cases_T0_T1_T2.csv"
)

OUT_BREAK_EVEN = (
    DATA_DIR
    / "b1_thermal_break_even_vs_h2.csv"
)


# ============================================================
# LOAD
# ============================================================

b1 = pd.read_csv(B1_FILE)
h2 = pd.read_csv(H2_FILE)

cfg = yaml.safe_load(
    CONFIG_FILE.read_text(
        encoding="utf-8"
    )
)

thermal_cfg = cfg["thermal_recovery"]
chp_cfg = cfg["technology"]["chp_thermal"]
demand_cfg = cfg["thermal_demand"]

heat_value = float(
    thermal_cfg[
        "heat_reference_cost_usd_per_kwh_th"
    ]
)

eta_el = float(
    chp_cfg["electrical_efficiency"]
)

eta_th = float(
    chp_cfg["thermal_efficiency"]
)


# ============================================================
# VALIDATION
# ============================================================

required_b1 = [
    "source_run",
    "source_solution_id",
    "lcoe_usd_kwh",
    "P_peak_grid_opt_kw",
    "E_load_total_kwh",
    "annualized_capex_usd",
    "fixed_opex_annual_usd",
    "grid_opex_annual_usd",
    "variable_biomethane_opex_annual_usd",
    "degradation_opex_annual_usd",
    "chp_kw",
]

missing = [
    c for c in required_b1
    if c not in b1.columns
]

if missing:
    raise KeyError(
        "Missing B1 columns: "
        + ", ".join(missing)
    )


# ============================================================
# BASE ANNUAL COST
# ============================================================

b1["base_annual_cost_usd"] = (
    b1["annualized_capex_usd"]
    + b1["fixed_opex_annual_usd"]
    + b1["grid_opex_annual_usd"]
    + b1["variable_biomethane_opex_annual_usd"]
    + b1["degradation_opex_annual_usd"]
)


# ============================================================
# CHP THERMAL GENERATION
#
# Approximation:
#
# E_th / E_el = eta_th / eta_el
#
# We do not have hourly CHP energy in the consolidated Pareto
# CSV, therefore first-pass thermal output is inferred from
# CHP electrical capacity using a utilization proxy.
#
# This is a screening calculation, not yet dispatch-integrated.
# ============================================================

# proxy based on annual biomethane variable cost / CHP sizing
# fallback: assumed capacity factor if needed

DEFAULT_CHP_CAPACITY_FACTOR = 0.25


def estimate_chp_electric_energy_kwh(row):

    # First screening approximation
    return (
        float(row["chp_kw"])
        * 8760.0
        * DEFAULT_CHP_CAPACITY_FACTOR
    )


# ============================================================
# THERMAL CASES
# ============================================================

records = []

for _, row in b1.iterrows():

    e_chp_el = estimate_chp_electric_energy_kwh(
        row
    )

    heat_generated = (
        e_chp_el
        * eta_th
        / eta_el
    )

    for case_name, case_cfg in (
        thermal_cfg["cases"].items()
    ):

        recovery_fraction = float(
            case_cfg["recovery_fraction"]
        )

        demand_case = (
            case_cfg["thermal_demand_case"]
        )

        annual_demand = float(
            demand_cfg[demand_case][
                "annual_energy_kwh_th"
            ]
        )

        heat_recoverable = (
            heat_generated
            * recovery_fraction
        )

        heat_useful = min(
            heat_recoverable,
            annual_demand,
        )

        heat_credit = (
            heat_useful
            * heat_value
        )

        adjusted_cost = (
            row["base_annual_cost_usd"]
            - heat_credit
        )

        adjusted_lcoe = (
            adjusted_cost
            / row["E_load_total_kwh"]
        )

        records.append({

            "thermal_case":
                case_name,

            "source_run":
                row["source_run"],

            "source_solution_id":
                row["source_solution_id"],

            "pv_kw":
                row["pv_kw"],

            "bsv_kwh":
                row["bsv_kwh"],

            "biomethane_storage_nm3":
                row[
                    "biomethane_storage_nm3"
                ],

            "chp_kw":
                row["chp_kw"],

            "P_peak_grid_opt_kw":
                row[
                    "P_peak_grid_opt_kw"
                ],

            "lcoe_T0_original_usd_kwh":
                row["lcoe_usd_kwh"],

            "recovery_fraction":
                recovery_fraction,

            "thermal_demand_kwh_th":
                annual_demand,

            "estimated_chp_el_energy_kwh":
                e_chp_el,

            "heat_generated_kwh_th":
                heat_generated,

            "heat_recoverable_kwh_th":
                heat_recoverable,

            "heat_useful_kwh_th":
                heat_useful,

            "heat_utilization_ratio":
                (
                    heat_useful
                    / heat_generated
                    if heat_generated > 0
                    else 0.0
                ),

            "heat_credit_usd_per_year":
                heat_credit,

            "adjusted_annual_cost_usd":
                adjusted_cost,

            "adjusted_lcoe_usd_kwh":
                adjusted_lcoe,

            "lcoe_reduction_usd_kwh":
                (
                    row["lcoe_usd_kwh"]
                    - adjusted_lcoe
                ),

            "lcoe_reduction_percent":
                (
                    (
                        row["lcoe_usd_kwh"]
                        - adjusted_lcoe
                    )
                    / row["lcoe_usd_kwh"]
                    * 100.0
                ),
        })


thermal = pd.DataFrame(records)


# ============================================================
# CROSS-ROUTE DOMINANCE AFTER HEAT CREDIT
# ============================================================

thermal["dominated_by_h2"] = False
thermal["n_h2_dominators"] = 0


for i, r in thermal.iterrows():

    dominators = h2[
        (
            h2["lcoe_usd_kwh"]
            <= r["adjusted_lcoe_usd_kwh"]
        )
        &
        (
            h2["P_peak_grid_opt_kw"]
            <= r["P_peak_grid_opt_kw"]
        )
        &
        (
            (
                h2["lcoe_usd_kwh"]
                < r["adjusted_lcoe_usd_kwh"]
            )
            |
            (
                h2["P_peak_grid_opt_kw"]
                < r["P_peak_grid_opt_kw"]
            )
        )
    ]

    thermal.loc[
        i,
        "dominated_by_h2"
    ] = not dominators.empty

    thermal.loc[
        i,
        "n_h2_dominators"
    ] = len(dominators)


# ============================================================
# BREAK-EVEN CREDIT AGAINST H2 DOMINATOR
# ============================================================

break_even_records = []


for _, b in b1.iterrows():

    dominators = h2[
        (
            h2["lcoe_usd_kwh"]
            <= b["lcoe_usd_kwh"]
        )
        &
        (
            h2["P_peak_grid_opt_kw"]
            <= b["P_peak_grid_opt_kw"]
        )
    ].copy()

    if dominators.empty:
        continue

    # Closest H2 in peak while still dominating
    dominator = (
        dominators
        .sort_values(
            [
                "P_peak_grid_opt_kw",
                "lcoe_usd_kwh",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .iloc[0]
    )

    lcoe_gap = (
        b["lcoe_usd_kwh"]
        - dominator["lcoe_usd_kwh"]
    )

    annual_credit_needed = (
        lcoe_gap
        * b["E_load_total_kwh"]
    )

    heat_needed = (
        annual_credit_needed
        / heat_value
        if heat_value > 0
        else np.nan
    )

    break_even_records.append({

        "b1_source_run":
            b["source_run"],

        "b1_source_solution_id":
            b["source_solution_id"],

        "b1_peak_kw":
            b["P_peak_grid_opt_kw"],

        "b1_lcoe_usd_kwh":
            b["lcoe_usd_kwh"],

        "h2_global_solution_id":
            dominator[
                "global_solution_id"
            ],

        "h2_peak_kw":
            dominator[
                "P_peak_grid_opt_kw"
            ],

        "h2_lcoe_usd_kwh":
            dominator[
                "lcoe_usd_kwh"
            ],

        "lcoe_gap_usd_kwh":
            lcoe_gap,

        "annual_heat_credit_break_even_usd":
            annual_credit_needed,

        "heat_useful_break_even_kwh_th":
            heat_needed,
    })


break_even = pd.DataFrame(
    break_even_records
)


# ============================================================
# SAVE
# ============================================================

thermal.to_csv(
    OUT_DETAIL,
    index=False,
)

break_even.to_csv(
    OUT_BREAK_EVEN,
    index=False,
)


# ============================================================
# REPORT
# ============================================================

print()
print("=== THERMAL CASES ===")

for case in ["T0", "T1", "T2"]:

    s = thermal[
        thermal["thermal_case"] == case
    ]

    print()
    print(case)

    print(
        "mean heat useful kWh_th =",
        s["heat_useful_kwh_th"].mean()
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
        "B1 solutions still dominated by H2 =",
        int(
            s[
                "dominated_by_h2"
            ].sum()
        ),
        "/",
        len(s),
    )


print()
print("=== BREAK-EVEN HEAT ===")

print(
    break_even[
        "annual_heat_credit_break_even_usd"
    ].describe()
)

print()

print(
    break_even[
        "heat_useful_break_even_kwh_th"
    ].describe()
)


print()
print("=== OUTPUT FILES ===")
print(
    OUT_DETAIL.relative_to(ROOT)
)
print(
    OUT_BREAK_EVEN.relative_to(ROOT)
)