from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path("results/paper/source_data")

H2_FILE = ROOT / "pareto_h2_global_nondominated_with_knee.csv"
B1_FILE = ROOT / "pareto_b1_global_nondominated_with_knee.csv"
PAIR_FILE = ROOT / "h2_b1_dominance_decomposition.csv"

OUT_DETAIL = ROOT / "h2_b1_annual_cost_decomposition_34_pairs.csv"
OUT_SUMMARY = ROOT / "h2_b1_annual_cost_decomposition_summary.csv"
OUT_MEAN = ROOT / "h2_b1_annual_cost_mean_components.csv"


# ============================================================
# LOAD
# ============================================================

h2 = pd.read_csv(H2_FILE)
b1 = pd.read_csv(B1_FILE)
pairs = pd.read_csv(PAIR_FILE)


# ============================================================
# HELPERS
# ============================================================

def value(row, column, default=0.0):
    if column not in row.index:
        return default

    v = row[column]

    if pd.isna(v):
        return default

    return float(v)


def get_h2_dispatchable_cost(row):
    """
    High-level H2 variable cost.
    Avoids double counting.
    """
    if "variable_dispatchable_opex_annual_usd" in row.index:
        v = row["variable_dispatchable_opex_annual_usd"]

        if not pd.isna(v):
            return float(v)

    return value(
        row,
        "variable_h2_opex_annual_usd",
        0.0,
    )


def get_b1_dispatchable_cost(row):
    """
    Prefer already aggregated biomethane variable cost.
    Otherwise reconstruct fuel + CHP variable O&M.
    """

    if "variable_dispatchable_opex_annual_usd" in row.index:
        v = row["variable_dispatchable_opex_annual_usd"]

        if not pd.isna(v):
            return float(v)

    if "variable_biomethane_opex_annual_usd" in row.index:
        v = row["variable_biomethane_opex_annual_usd"]

        if not pd.isna(v):
            return float(v)

    return (
        value(
            row,
            "biomethane_fuel_opex_annual_usd",
            0.0,
        )
        +
        value(
            row,
            "chp_variable_opex_annual_usd",
            0.0,
        )
    )


# ============================================================
# TRACEABILITY INDEXES
# ============================================================

h2_lookup = {}

for _, row in h2.iterrows():
    key = (
        str(row["source_run"]),
        int(row["source_solution_id"]),
    )
    h2_lookup[key] = row


b1_lookup = {}

for _, row in b1.iterrows():
    key = (
        str(row["source_run"]),
        int(row["source_solution_id"]),
    )
    b1_lookup[key] = row


# ============================================================
# DECOMPOSE 34 DOMINATOR–DOMINATED PAIRS
# ============================================================

records = []

for _, p in pairs.iterrows():

    b1_key = (
        str(p["b1_source_run"]),
        int(p["b1_source_solution_id"]),
    )

    h2_key = (
        str(p["h2_source_run"]),
        int(p["h2_source_solution_id"]),
    )

    if b1_key not in b1_lookup:
        raise KeyError(
            f"B1 pair not found: {b1_key}"
        )

    if h2_key not in h2_lookup:
        raise KeyError(
            f"H2 pair not found: {h2_key}"
        )

    br = b1_lookup[b1_key]
    hr = h2_lookup[h2_key]

    # --------------------------------------------------------
    # H2 annual components
    # --------------------------------------------------------

    h2_capex_ann = value(
        hr,
        "annualized_capex_usd",
    )

    h2_fixed = value(
        hr,
        "fixed_opex_annual_usd",
    )

    h2_grid = value(
        hr,
        "grid_opex_annual_usd",
    )

    h2_peak = value(
        hr,
        "grid_peak_opex_annual_usd",
    )

    h2_dispatch = get_h2_dispatchable_cost(hr)

    h2_degradation = value(
        hr,
        "degradation_opex_annual_usd",
    )

    h2_total = (
        h2_capex_ann
        + h2_fixed
        + h2_grid
        + h2_peak
        + h2_dispatch
        + h2_degradation
    )

    # --------------------------------------------------------
    # B1 annual components
    # --------------------------------------------------------

    b1_capex_ann = value(
        br,
        "annualized_capex_usd",
    )

    b1_fixed = value(
        br,
        "fixed_opex_annual_usd",
    )

    b1_grid = value(
        br,
        "grid_opex_annual_usd",
    )

    b1_peak = value(
        br,
        "grid_peak_opex_annual_usd",
    )

    b1_dispatch = get_b1_dispatchable_cost(br)

    b1_degradation = value(
        br,
        "degradation_opex_annual_usd",
    )

    b1_total = (
        b1_capex_ann
        + b1_fixed
        + b1_grid
        + b1_peak
        + b1_dispatch
        + b1_degradation
    )

    # --------------------------------------------------------
    # Differences
    #
    # Positive delta = H2 advantage
    # i.e. B1 cost - H2 cost
    # --------------------------------------------------------

    rec = {
        # Traceability
        "b1_source_run":
            p["b1_source_run"],

        "b1_source_solution_id":
            p["b1_source_solution_id"],

        "h2_global_solution_id":
            p["h2_global_solution_id"],

        "h2_source_run":
            p["h2_source_run"],

        "h2_source_solution_id":
            p["h2_source_solution_id"],

        # Objectives
        "b1_lcoe_usd_kwh":
            br["lcoe_usd_kwh"],

        "h2_lcoe_usd_kwh":
            hr["lcoe_usd_kwh"],

        "delta_lcoe_b1_minus_h2":
            br["lcoe_usd_kwh"]
            - hr["lcoe_usd_kwh"],

        "b1_peak_kw":
            br["P_peak_grid_opt_kw"],

        "h2_peak_kw":
            hr["P_peak_grid_opt_kw"],

        "delta_peak_b1_minus_h2_kw":
            br["P_peak_grid_opt_kw"]
            - hr["P_peak_grid_opt_kw"],

        # Raw CAPEX
        "b1_capex_total_usd":
            value(br, "capex_total_usd"),

        "h2_capex_total_usd":
            value(hr, "capex_total_usd"),

        "delta_capex_total_b1_minus_h2_usd":
            value(br, "capex_total_usd")
            - value(hr, "capex_total_usd"),

        # Annualized CAPEX
        "b1_annualized_capex_usd":
            b1_capex_ann,

        "h2_annualized_capex_usd":
            h2_capex_ann,

        "delta_annualized_capex_b1_minus_h2_usd":
            b1_capex_ann - h2_capex_ann,

        # Fixed OPEX
        "b1_fixed_opex_usd":
            b1_fixed,

        "h2_fixed_opex_usd":
            h2_fixed,

        "delta_fixed_opex_b1_minus_h2_usd":
            b1_fixed - h2_fixed,

        # Grid energy cost
        "b1_grid_opex_usd":
            b1_grid,

        "h2_grid_opex_usd":
            h2_grid,

        "delta_grid_opex_b1_minus_h2_usd":
            b1_grid - h2_grid,

        # Peak cost
        "b1_peak_opex_usd":
            b1_peak,

        "h2_peak_opex_usd":
            h2_peak,

        "delta_peak_opex_b1_minus_h2_usd":
            b1_peak - h2_peak,

        # Dispatchable / fuel
        "b1_dispatchable_opex_usd":
            b1_dispatch,

        "h2_dispatchable_opex_usd":
            h2_dispatch,

        "delta_dispatchable_opex_b1_minus_h2_usd":
            b1_dispatch - h2_dispatch,

        # B1 subcomponents for audit
        "b1_biomethane_fuel_opex_usd":
            value(
                br,
                "biomethane_fuel_opex_annual_usd",
            ),

        "b1_chp_variable_opex_usd":
            value(
                br,
                "chp_variable_opex_annual_usd",
            ),

        # H2 subcomponent for audit
        "h2_variable_h2_opex_usd":
            value(
                hr,
                "variable_h2_opex_annual_usd",
            ),

        # Degradation
        "b1_degradation_opex_usd":
            b1_degradation,

        "h2_degradation_opex_usd":
            h2_degradation,

        "delta_degradation_b1_minus_h2_usd":
            b1_degradation
            - h2_degradation,

        # Total annual cost reconstructed
        "b1_total_annual_cost_reconstructed_usd":
            b1_total,

        "h2_total_annual_cost_reconstructed_usd":
            h2_total,

        "delta_total_annual_cost_b1_minus_h2_usd":
            b1_total - h2_total,

        "h2_total_annual_cost_advantage_percent":
            (
                (b1_total - h2_total)
                / b1_total
            ) * 100.0,

        # Energy quantities
        "b1_E_grid_kwh":
            value(
                br,
                "E_grid_total_kwh",
            ),

        "h2_E_grid_kwh":
            value(
                hr,
                "E_grid_total_kwh",
            ),

        "delta_E_grid_b1_minus_h2_kwh":
            value(br, "E_grid_total_kwh")
            - value(hr, "E_grid_total_kwh"),

        "b1_grid_dependency_ratio":
            value(
                br,
                "total_grid_dependency_ratio",
            ),

        "h2_grid_dependency_ratio":
            value(
                hr,
                "total_grid_dependency_ratio",
            ),

        "delta_grid_dependency_b1_minus_h2_pp":
            (
                value(
                    br,
                    "total_grid_dependency_ratio",
                )
                -
                value(
                    hr,
                    "total_grid_dependency_ratio",
                )
            ) * 100.0,

        # Technical sizing
        "b1_pv_kw":
            value(br, "pv_kw"),

        "h2_pv_kw":
            value(hr, "pv_kw"),

        "b1_bsv_kwh":
            value(br, "bsv_kwh"),

        "h2_bsv_kwh":
            value(hr, "bsv_kwh"),

        "b1_chp_kw":
            value(br, "chp_kw"),

        "b1_biomethane_storage_nm3":
            value(
                br,
                "biomethane_storage_nm3",
            ),

        "h2_fuelcell_kw":
            value(hr, "fuelcell_kw"),

        "h2_electrolyzer_kw":
            value(hr, "electrolyzer_kw"),

        "h2_tank_kg":
            value(hr, "h2_tank_kg"),
    }

    records.append(rec)


detail = pd.DataFrame(records)

detail.to_csv(
    OUT_DETAIL,
    index=False,
)


# ============================================================
# SUMMARY OF DIFFERENCES
# ============================================================

delta_columns = [
    "delta_annualized_capex_b1_minus_h2_usd",
    "delta_fixed_opex_b1_minus_h2_usd",
    "delta_grid_opex_b1_minus_h2_usd",
    "delta_peak_opex_b1_minus_h2_usd",
    "delta_dispatchable_opex_b1_minus_h2_usd",
    "delta_degradation_b1_minus_h2_usd",
    "delta_total_annual_cost_b1_minus_h2_usd",
]

summary_records = []

for col in delta_columns:

    s = detail[col]

    summary_records.append({
        "component": col,
        "min_usd_per_year": s.min(),
        "mean_usd_per_year": s.mean(),
        "median_usd_per_year": s.median(),
        "max_usd_per_year": s.max(),
        "positive_pairs": int((s > 0).sum()),
        "negative_pairs": int((s < 0).sum()),
        "zero_pairs": int((s == 0).sum()),
    })


summary = pd.DataFrame(summary_records)

summary.to_csv(
    OUT_SUMMARY,
    index=False,
)


# ============================================================
# MEAN ANNUAL COST COMPONENTS BY ROUTE
# ============================================================

mean_components = pd.DataFrame({
    "component": [
        "Annualized CAPEX",
        "Fixed OPEX",
        "Grid energy",
        "Grid peak",
        "Dispatchable/fuel",
        "Degradation",
        "Total annual",
    ],

    "H2_mean_usd_per_year": [
        detail["h2_annualized_capex_usd"].mean(),
        detail["h2_fixed_opex_usd"].mean(),
        detail["h2_grid_opex_usd"].mean(),
        detail["h2_peak_opex_usd"].mean(),
        detail["h2_dispatchable_opex_usd"].mean(),
        detail["h2_degradation_opex_usd"].mean(),
        detail[
            "h2_total_annual_cost_reconstructed_usd"
        ].mean(),
    ],

    "B1_mean_usd_per_year": [
        detail["b1_annualized_capex_usd"].mean(),
        detail["b1_fixed_opex_usd"].mean(),
        detail["b1_grid_opex_usd"].mean(),
        detail["b1_peak_opex_usd"].mean(),
        detail["b1_dispatchable_opex_usd"].mean(),
        detail["b1_degradation_opex_usd"].mean(),
        detail[
            "b1_total_annual_cost_reconstructed_usd"
        ].mean(),
    ],
})

mean_components[
    "delta_B1_minus_H2_usd_per_year"
] = (
    mean_components["B1_mean_usd_per_year"]
    -
    mean_components["H2_mean_usd_per_year"]
)

mean_components.to_csv(
    OUT_MEAN,
    index=False,
)


# ============================================================
# INTERNAL CONSISTENCY CHECK
#
# LCOE reconstructed from annual cost / load
# ============================================================

if "E_load_total_kwh" in h2.columns:
    h2_load = float(
        h2["E_load_total_kwh"].iloc[0]
    )
else:
    h2_load = np.nan

if "E_load_total_kwh" in b1.columns:
    b1_load = float(
        b1["E_load_total_kwh"].iloc[0]
    )
else:
    b1_load = np.nan


if not np.isnan(h2_load):

    detail[
        "h2_lcoe_reconstructed"
    ] = (
        detail[
            "h2_total_annual_cost_reconstructed_usd"
        ]
        / h2_load
    )

    detail[
        "h2_lcoe_reconstruction_error"
    ] = (
        detail["h2_lcoe_reconstructed"]
        - detail["h2_lcoe_usd_kwh"]
    )


if not np.isnan(b1_load):

    detail[
        "b1_lcoe_reconstructed"
    ] = (
        detail[
            "b1_total_annual_cost_reconstructed_usd"
        ]
        / b1_load
    )

    detail[
        "b1_lcoe_reconstruction_error"
    ] = (
        detail["b1_lcoe_reconstructed"]
        - detail["b1_lcoe_usd_kwh"]
    )


# rewrite with consistency columns
detail.to_csv(
    OUT_DETAIL,
    index=False,
)


# ============================================================
# REPORT
# ============================================================

print()
print("=== ANNUAL COST DECOMPOSITION ===")
print("pairs =", len(detail))

print()
print("=== MEAN ANNUAL COMPONENTS ===")
print(
    mean_components.to_string(
        index=False
    )
)

print()
print("=== MEAN B1 - H2 DIFFERENCES ===")

for _, row in mean_components.iterrows():

    print(
        f"{row['component']:20s} "
        f"{row['delta_B1_minus_H2_usd_per_year']:12.2f} USD/year"
    )


print()
print("=== TOTAL ANNUAL COST ADVANTAGE ===")

print(
    detail[
        "h2_total_annual_cost_advantage_percent"
    ].describe().to_string()
)


print()
print("=== LCOE RECONSTRUCTION CHECK ===")

if (
    "h2_lcoe_reconstruction_error"
    in detail.columns
):

    print(
        "H2 max abs error =",
        detail[
            "h2_lcoe_reconstruction_error"
        ].abs().max()
    )

if (
    "b1_lcoe_reconstruction_error"
    in detail.columns
):

    print(
        "B1 max abs error =",
        detail[
            "b1_lcoe_reconstruction_error"
        ].abs().max()
    )


print()
print("=== OUTPUT FILES ===")

print(OUT_DETAIL)
print(OUT_SUMMARY)
print(OUT_MEAN)