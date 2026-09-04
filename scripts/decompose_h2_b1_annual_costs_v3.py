from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    ROOT
    / "results"
    / "paper"
    / "source_data"
)

PAIR_FILE = (
    DATA_DIR
    / "h2_b1_dominance_decomposition.csv"
)

H2_FILE = (
    DATA_DIR
    / "pareto_h2_global_nondominated_with_knee.csv"
)

B1_FILE = (
    DATA_DIR
    / "pareto_b1_global_nondominated_with_knee.csv"
)


OUT_DETAIL = (
    DATA_DIR
    / "h2_b1_annual_cost_decomposition_34_pairs_v3.csv"
)

OUT_SUMMARY = (
    DATA_DIR
    / "h2_b1_annual_cost_decomposition_summary_v3.csv"
)

OUT_CONTRIBUTION = (
    DATA_DIR
    / "h2_b1_annual_cost_contributions_v3.csv"
)


# ============================================================
# LOAD
# ============================================================

pairs = pd.read_csv(PAIR_FILE)
h2 = pd.read_csv(H2_FILE)
b1 = pd.read_csv(B1_FILE)


print("=== INPUT ===")
print("pairs =", len(pairs))
print("H2 solutions =", len(h2))
print("B1 solutions =", len(b1))


# ============================================================
# BASIC VALIDATION
# ============================================================

if len(pairs) != 34:
    raise ValueError(
        f"Expected 34 dominance pairs, found {len(pairs)}"
    )


required_h2 = [
    "source_run",
    "source_solution_id",
    "lcoe_usd_kwh",
    "P_peak_grid_opt_kw",
    "annualized_capex_usd",
    "fixed_opex_annual_usd",
    "grid_opex_annual_usd",
    "variable_h2_opex_annual_usd",
    "degradation_opex_annual_usd",
    "E_load_total_kwh",
    "E_grid_total_kwh",
    "total_grid_dependency_ratio",
    "pv_kw",
    "bsv_kwh",
    "electrolyzer_kw",
    "h2_tank_kg",
    "fuelcell_kw",
]

required_b1 = [
    "source_run",
    "source_solution_id",
    "lcoe_usd_kwh",
    "P_peak_grid_opt_kw",
    "annualized_capex_usd",
    "fixed_opex_annual_usd",
    "grid_opex_annual_usd",
    "degradation_opex_annual_usd",
    "E_load_total_kwh",
    "E_grid_total_kwh",
    "total_grid_dependency_ratio",
    "pv_kw",
    "bsv_kwh",
    "biomethane_storage_nm3",
    "chp_kw",
]


missing_h2 = [
    c for c in required_h2
    if c not in h2.columns
]

missing_b1 = [
    c for c in required_b1
    if c not in b1.columns
]

if missing_h2:
    raise KeyError(
        "Missing H2 columns: "
        + ", ".join(missing_h2)
    )

if missing_b1:
    raise KeyError(
        "Missing B1 columns: "
        + ", ".join(missing_b1)
    )


# ============================================================
# B1 DISPATCHABLE OPEX
#
# Prefer aggregate route-neutral field.
# If unavailable, use variable_biomethane.
# Last fallback = fuel + CHP variable.
# ============================================================

def b1_dispatchable_opex(row):

    if (
        "variable_dispatchable_opex_annual_usd"
        in row.index
        and pd.notna(
            row["variable_dispatchable_opex_annual_usd"]
        )
    ):
        return float(
            row["variable_dispatchable_opex_annual_usd"]
        )

    if (
        "variable_biomethane_opex_annual_usd"
        in row.index
        and pd.notna(
            row["variable_biomethane_opex_annual_usd"]
        )
    ):
        return float(
            row["variable_biomethane_opex_annual_usd"]
        )

    fuel = float(
        row.get(
            "biomethane_fuel_opex_annual_usd",
            0.0,
        )
    )

    chp = float(
        row.get(
            "chp_variable_opex_annual_usd",
            0.0,
        )
    )

    return fuel + chp


# ============================================================
# INDEXES FOR EXACT TRACEABILITY
# ============================================================

h2_index = {
    (
        str(row["source_run"]),
        int(row["source_solution_id"]),
    ): row
    for _, row in h2.iterrows()
}

b1_index = {
    (
        str(row["source_run"]),
        int(row["source_solution_id"]),
    ): row
    for _, row in b1.iterrows()
}


# ============================================================
# RECONSTRUCT THE 34 EXACT PAIRS
# ============================================================

records = []

for _, pair in pairs.iterrows():

    h2_key = (
        str(pair["h2_source_run"]),
        int(pair["h2_source_solution_id"]),
    )

    b1_key = (
        str(pair["b1_source_run"]),
        int(pair["b1_source_solution_id"]),
    )

    if h2_key not in h2_index:
        raise KeyError(
            f"H2 solution not found: {h2_key}"
        )

    if b1_key not in b1_index:
        raise KeyError(
            f"B1 solution not found: {b1_key}"
        )

    hr = h2_index[h2_key]
    br = b1_index[b1_key]

    # --------------------------------------------------------
    # H2 additive annual cost components
    # --------------------------------------------------------

    h2_capex_ann = float(
        hr["annualized_capex_usd"]
    )

    h2_fixed = float(
        hr["fixed_opex_annual_usd"]
    )

    h2_grid = float(
        hr["grid_opex_annual_usd"]
    )

    h2_dispatch = float(
        hr["variable_h2_opex_annual_usd"]
    )

    h2_degradation = float(
        hr["degradation_opex_annual_usd"]
    )

    h2_total = (
        h2_capex_ann
        + h2_fixed
        + h2_grid
        + h2_dispatch
        + h2_degradation
    )


    # --------------------------------------------------------
    # B1 additive annual cost components
    # --------------------------------------------------------

    b1_capex_ann = float(
        br["annualized_capex_usd"]
    )

    b1_fixed = float(
        br["fixed_opex_annual_usd"]
    )

    b1_grid = float(
        br["grid_opex_annual_usd"]
    )

    b1_dispatch = b1_dispatchable_opex(br)

    b1_degradation = float(
        br["degradation_opex_annual_usd"]
    )

    b1_total = (
        b1_capex_ann
        + b1_fixed
        + b1_grid
        + b1_dispatch
        + b1_degradation
    )


    # --------------------------------------------------------
    # LCOE reconstruction
    # --------------------------------------------------------

    h2_energy = float(
        hr["E_load_total_kwh"]
    )

    b1_energy = float(
        br["E_load_total_kwh"]
    )

    h2_lcoe_reconstructed = (
        h2_total / h2_energy
    )

    b1_lcoe_reconstructed = (
        b1_total / b1_energy
    )


    # --------------------------------------------------------
    # Record
    #
    # delta = B1 - H2
    #
    # Positive:
    #     H2 has lower cost for that component.
    #
    # Negative:
    #     B1 has lower cost.
    # --------------------------------------------------------

    records.append({

        # Traceability
        "b1_source_run":
            pair["b1_source_run"],

        "b1_source_solution_id":
            int(pair["b1_source_solution_id"]),

        "h2_global_solution_id":
            pair["h2_global_solution_id"],

        "h2_source_run":
            pair["h2_source_run"],

        "h2_source_solution_id":
            int(pair["h2_source_solution_id"]),

        "n_h2_dominators":
            pair["n_h2_dominators"],

        "objective_distance":
            pair["objective_distance"],


        # Objectives
        "b1_lcoe_usd_kwh":
            float(br["lcoe_usd_kwh"]),

        "h2_lcoe_usd_kwh":
            float(hr["lcoe_usd_kwh"]),

        "delta_lcoe_b1_minus_h2_usd_kwh":
            float(br["lcoe_usd_kwh"])
            - float(hr["lcoe_usd_kwh"]),

        "b1_peak_kw":
            float(br["P_peak_grid_opt_kw"]),

        "h2_peak_kw":
            float(hr["P_peak_grid_opt_kw"]),

        "delta_peak_b1_minus_h2_kw":
            float(br["P_peak_grid_opt_kw"])
            - float(hr["P_peak_grid_opt_kw"]),


        # Annualized CAPEX
        "b1_annualized_capex_usd":
            b1_capex_ann,

        "h2_annualized_capex_usd":
            h2_capex_ann,

        "delta_capex_ann_b1_minus_h2_usd":
            b1_capex_ann - h2_capex_ann,


        # Fixed OPEX
        "b1_fixed_opex_usd":
            b1_fixed,

        "h2_fixed_opex_usd":
            h2_fixed,

        "delta_fixed_b1_minus_h2_usd":
            b1_fixed - h2_fixed,


        # Grid
        "b1_grid_opex_usd":
            b1_grid,

        "h2_grid_opex_usd":
            h2_grid,

        "delta_grid_b1_minus_h2_usd":
            b1_grid - h2_grid,


        # Dispatchable
        "b1_dispatchable_opex_usd":
            b1_dispatch,

        "h2_dispatchable_opex_usd":
            h2_dispatch,

        "delta_dispatchable_b1_minus_h2_usd":
            b1_dispatch - h2_dispatch,


        # Degradation
        "b1_degradation_opex_usd":
            b1_degradation,

        "h2_degradation_opex_usd":
            h2_degradation,

        "delta_degradation_b1_minus_h2_usd":
            b1_degradation - h2_degradation,


        # Total
        "b1_total_annual_cost_usd":
            b1_total,

        "h2_total_annual_cost_usd":
            h2_total,

        "delta_total_b1_minus_h2_usd":
            b1_total - h2_total,

        "h2_total_cost_advantage_percent":
            (
                (b1_total - h2_total)
                / b1_total
                * 100.0
            ),


        # LCOE audit
        "b1_lcoe_reconstructed":
            b1_lcoe_reconstructed,

        "h2_lcoe_reconstructed":
            h2_lcoe_reconstructed,

        "b1_lcoe_reconstruction_error":
            (
                b1_lcoe_reconstructed
                - float(br["lcoe_usd_kwh"])
            ),

        "h2_lcoe_reconstruction_error":
            (
                h2_lcoe_reconstructed
                - float(hr["lcoe_usd_kwh"])
            ),


        # Grid energy / dependency
        "b1_E_grid_kwh":
            float(br["E_grid_total_kwh"]),

        "h2_E_grid_kwh":
            float(hr["E_grid_total_kwh"]),

        "delta_E_grid_b1_minus_h2_kwh":
            (
                float(br["E_grid_total_kwh"])
                - float(hr["E_grid_total_kwh"])
            ),

        "b1_grid_dependency_ratio":
            float(
                br["total_grid_dependency_ratio"]
            ),

        "h2_grid_dependency_ratio":
            float(
                hr["total_grid_dependency_ratio"]
            ),

        "delta_grid_dependency_pp":
            (
                float(
                    br["total_grid_dependency_ratio"]
                )
                -
                float(
                    hr["total_grid_dependency_ratio"]
                )
            ) * 100.0,


        # Technical sizing
        "b1_pv_kw":
            float(br["pv_kw"]),

        "h2_pv_kw":
            float(hr["pv_kw"]),

        "delta_pv_h2_minus_b1_kw":
            float(hr["pv_kw"])
            - float(br["pv_kw"]),

        "b1_bsv_kwh":
            float(br["bsv_kwh"]),

        "h2_bsv_kwh":
            float(hr["bsv_kwh"]),

        "delta_bsv_h2_minus_b1_kwh":
            float(hr["bsv_kwh"])
            - float(br["bsv_kwh"]),

        "b1_chp_kw":
            float(br["chp_kw"]),

        "b1_biomethane_storage_nm3":
            float(
                br["biomethane_storage_nm3"]
            ),

        "h2_fuelcell_kw":
            float(hr["fuelcell_kw"]),

        "h2_electrolyzer_kw":
            float(hr["electrolyzer_kw"]),

        "h2_tank_kg":
            float(hr["h2_tank_kg"]),
    })


detail = pd.DataFrame(records)


# ============================================================
# AUDIT
# ============================================================

max_h2_error = (
    detail["h2_lcoe_reconstruction_error"]
    .abs()
    .max()
)

max_b1_error = (
    detail["b1_lcoe_reconstruction_error"]
    .abs()
    .max()
)

if max_h2_error > 1e-10:
    raise RuntimeError(
        f"H2 LCOE reconstruction error too large: "
        f"{max_h2_error}"
    )

if max_b1_error > 1e-10:
    raise RuntimeError(
        f"B1 LCOE reconstruction error too large: "
        f"{max_b1_error}"
    )


# ============================================================
# SUMMARY
# ============================================================

components = [
    (
        "Annualized CAPEX",
        "b1_annualized_capex_usd",
        "h2_annualized_capex_usd",
        "delta_capex_ann_b1_minus_h2_usd",
    ),
    (
        "Fixed OPEX",
        "b1_fixed_opex_usd",
        "h2_fixed_opex_usd",
        "delta_fixed_b1_minus_h2_usd",
    ),
    (
        "Grid energy",
        "b1_grid_opex_usd",
        "h2_grid_opex_usd",
        "delta_grid_b1_minus_h2_usd",
    ),
    (
        "Dispatchable/fuel",
        "b1_dispatchable_opex_usd",
        "h2_dispatchable_opex_usd",
        "delta_dispatchable_b1_minus_h2_usd",
    ),
    (
        "Degradation",
        "b1_degradation_opex_usd",
        "h2_degradation_opex_usd",
        "delta_degradation_b1_minus_h2_usd",
    ),
    (
        "Total annual",
        "b1_total_annual_cost_usd",
        "h2_total_annual_cost_usd",
        "delta_total_b1_minus_h2_usd",
    ),
]


summary_records = []

for (
    label,
    b1_col,
    h2_col,
    delta_col,
) in components:

    summary_records.append({

        "component": label,

        "H2_mean_usd_per_year":
            detail[h2_col].mean(),

        "B1_mean_usd_per_year":
            detail[b1_col].mean(),

        "delta_B1_minus_H2_mean_usd_per_year":
            detail[delta_col].mean(),

        "delta_median_usd_per_year":
            detail[delta_col].median(),

        "delta_min_usd_per_year":
            detail[delta_col].min(),

        "delta_max_usd_per_year":
            detail[delta_col].max(),

        "pairs_H2_lower_cost":
            int(
                (detail[delta_col] > 0).sum()
            ),

        "pairs_B1_lower_cost":
            int(
                (detail[delta_col] < 0).sum()
            ),
    })


summary = pd.DataFrame(
    summary_records
)


# ============================================================
# CONTRIBUTION TO NET DIFFERENCE
# ============================================================

mean_total_delta = (
    detail[
        "delta_total_b1_minus_h2_usd"
    ].mean()
)

contribution_records = []

for (
    label,
    _,
    _,
    delta_col,
) in components[:-1]:

    mean_delta = (
        detail[delta_col].mean()
    )

    contribution_records.append({

        "component":
            label,

        "mean_delta_B1_minus_H2_usd_per_year":
            mean_delta,

        "share_of_net_difference_percent":
            (
                mean_delta
                / mean_total_delta
                * 100.0
            ),
    })


contribution = pd.DataFrame(
    contribution_records
)


# ============================================================
# SAVE
# ============================================================

detail.to_csv(
    OUT_DETAIL,
    index=False,
)

summary.to_csv(
    OUT_SUMMARY,
    index=False,
)

contribution.to_csv(
    OUT_CONTRIBUTION,
    index=False,
)


# ============================================================
# REPORT
# ============================================================

print()
print("=== LCOE AUDIT ===")
print(
    "H2 max abs reconstruction error =",
    max_h2_error,
)
print(
    "B1 max abs reconstruction error =",
    max_b1_error,
)


print()
print(
    "=== CORRECTED MEAN ANNUAL COST COMPONENTS ==="
)

print(
    summary[
        [
            "component",
            "H2_mean_usd_per_year",
            "B1_mean_usd_per_year",
            "delta_B1_minus_H2_mean_usd_per_year",
        ]
    ].to_string(index=False)
)


print()
print(
    "=== CONTRIBUTION TO NET ANNUAL COST DIFFERENCE ==="
)

print(
    contribution.to_string(index=False)
)


print()
print(
    "=== H2 NET ANNUAL COST ADVANTAGE (%) ==="
)

print(
    detail[
        "h2_total_cost_advantage_percent"
    ].describe().to_string()
)


print()
print(
    "=== PAIRS WITH H2 LOWER TOTAL ANNUAL COST ==="
)

print(
    int(
        (
            detail[
                "delta_total_b1_minus_h2_usd"
            ]
            > 0
        ).sum()
    ),
    "/",
    len(detail),
)


print()
print(
    "=== PAIRS WITH B1 LOWER TOTAL ANNUAL COST ==="
)

print(
    int(
        (
            detail[
                "delta_total_b1_minus_h2_usd"
            ]
            < 0
        ).sum()
    ),
    "/",
    len(detail),
)


print()
print("=== OUTPUT FILES ===")
print(
    OUT_DETAIL.relative_to(ROOT)
)
print(
    OUT_SUMMARY.relative_to(ROOT)
)
print(
    OUT_CONTRIBUTION.relative_to(ROOT)
)