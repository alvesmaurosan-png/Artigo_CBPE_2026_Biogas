from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    ROOT
    / "results"
    / "paper"
    / "source_data"
)

INPUT = (
    DATA_DIR
    / "b1_thermal_dispatch_T0_T1_T2_summary.csv"
)

H2_FILE = (
    DATA_DIR
    / "pareto_h2_global_nondominated_with_knee.csv"
)

OUTPUT = (
    DATA_DIR
    / "b1_thermal_TMAX.csv"
)


HEAT_VALUE_USD_PER_KWH_TH = 0.08


# ============================================================
# LOAD
# ============================================================

thermal = pd.read_csv(INPUT)
h2 = pd.read_csv(H2_FILE)


# Use one row per B1 solution.
# T0 contains the same electrical dispatch and heat availability
# as T1/T2, but zero useful heat.
base = (
    thermal[
        thermal["thermal_case"] == "T0"
    ]
    .copy()
    .reset_index(drop=True)
)


# ============================================================
# TMAX
#
# All recoverable CHP heat is considered useful.
# Electrical dispatch remains unchanged.
# ============================================================

base["thermal_case"] = "TMAX"

base["heat_useful_kwh_th"] = (
    base["heat_available_kwh_th"]
)

base["heat_dumped_kwh_th"] = 0.0

base["heat_utilization_ratio"] = 1.0

base["heat_credit_usd_per_year"] = (
    base["heat_available_kwh_th"]
    * HEAT_VALUE_USD_PER_KWH_TH
)

base["adjusted_lcoe_usd_kwh"] = (
    base["base_lcoe_usd_kwh"]
    -
    (
        base["heat_credit_usd_per_year"]
        / 2268096.045006
    )
)

base["lcoe_reduction_usd_kwh"] = (
    base["base_lcoe_usd_kwh"]
    - base["adjusted_lcoe_usd_kwh"]
)

base["lcoe_reduction_percent"] = (
    base["lcoe_reduction_usd_kwh"]
    / base["base_lcoe_usd_kwh"]
    * 100.0
)


# ============================================================
# DOMINANCE AGAINST H2
# ============================================================

dominated = []
n_dominators = []


for _, row in base.iterrows():

    mask = (
        (
            h2["lcoe_usd_kwh"]
            <= row["adjusted_lcoe_usd_kwh"]
        )
        &
        (
            h2["P_peak_grid_opt_kw"]
            <= row["P_peak_grid_opt_kw"]
        )
        &
        (
            (
                h2["lcoe_usd_kwh"]
                < row["adjusted_lcoe_usd_kwh"]
            )
            |
            (
                h2["P_peak_grid_opt_kw"]
                < row["P_peak_grid_opt_kw"]
            )
        )
    )

    d = h2.loc[mask]

    dominated.append(
        not d.empty
    )

    n_dominators.append(
        len(d)
    )


base["dominated_by_h2"] = dominated
base["n_h2_dominators"] = n_dominators


# ============================================================
# SAVE
# ============================================================

base.to_csv(
    OUTPUT,
    index=False,
)


# ============================================================
# REPORT
# ============================================================

print()
print("=== TMAX — 100% OF AVAILABLE HEAT VALUED ===")

print(
    "solutions =",
    len(base),
)

print(
    "mean heat available MWh_th =",
    base["heat_available_kwh_th"].mean()
    / 1000.0,
)

print(
    "mean heat credit USD/year =",
    base["heat_credit_usd_per_year"].mean(),
)

print(
    "mean adjusted LCOE =",
    base["adjusted_lcoe_usd_kwh"].mean(),
)

print(
    "mean LCOE reduction % =",
    base["lcoe_reduction_percent"].mean(),
)

print(
    "B1 still dominated by H2 =",
    int(
        base["dominated_by_h2"].sum()
    ),
    "/",
    len(base),
)

print()
print("=== NONDOMINATED B1 UNDER TMAX ===")

survivors = base[
    ~base["dominated_by_h2"]
]

if survivors.empty:
    print("NONE")
else:
    print(
        survivors[
            [
                "source_run",
                "source_solution_id",
                "chp_kw",
                "P_peak_grid_opt_kw",
                "base_lcoe_usd_kwh",
                "heat_available_kwh_th",
                "heat_credit_usd_per_year",
                "adjusted_lcoe_usd_kwh",
            ]
        ].to_string(
            index=False
        )
    )

print()
print("Saved:")
print(
    OUTPUT.relative_to(ROOT)
)