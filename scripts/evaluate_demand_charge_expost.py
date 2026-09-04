from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    ROOT
    / "results"
    / "paper"
    / "source_data"
)

H2_FILE = (
    DATA_DIR
    / "pareto_h2_global_nondominated_with_knee.csv"
)

B1_FILE = (
    DATA_DIR
    / "pareto_b1_global_nondominated_with_knee.csv"
)

OUT_H2 = (
    DATA_DIR
    / "pareto_h2_with_demand_charge_expost.csv"
)

OUT_B1 = (
    DATA_DIR
    / "pareto_b1_with_demand_charge_expost.csv"
)

OUT_COMBINED = (
    DATA_DIR
    / "pareto_h2_b1_with_demand_charge_expost.csv"
)

OUT_GLOBAL = (
    DATA_DIR
    / "pareto_h2_b1_global_nondominated_with_demand_charge_expost.csv"
)

OUT_DOMINANCE = (
    DATA_DIR
    / "b1_dominance_by_h2_with_demand_charge_expost.csv"
)


# ============================================================
# PARAMETERS
# ============================================================

DEMAND_CHARGE_USD_KW_MONTH = 30.0
MONTHS_PER_YEAR = 12.0


# ============================================================
# LOAD
# ============================================================

h2 = pd.read_csv(H2_FILE).copy()
b1 = pd.read_csv(B1_FILE).copy()

print("=== INPUT ===")
print("H2 solutions =", len(h2))
print("B1 solutions =", len(b1))


# ============================================================
# VALIDATION
# ============================================================

required = [
    "lcoe_usd_kwh",
    "P_peak_grid_opt_kw",
    "E_load_total_kwh",
]

for name, df in [
    ("H2", h2),
    ("B1", b1),
]:
    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:
        raise KeyError(
            f"{name}: missing columns: "
            + ", ".join(missing)
        )


# ============================================================
# EX-POST DEMAND CHARGE
# ============================================================

def add_demand_charge(df, route):

    out = df.copy()

    out["route"] = route

    out[
        "demand_charge_usd_kw_month"
    ] = DEMAND_CHARGE_USD_KW_MONTH

    out[
        "demand_charge_annual_usd"
    ] = (
        out["P_peak_grid_opt_kw"]
        * DEMAND_CHARGE_USD_KW_MONTH
        * MONTHS_PER_YEAR
    )

    out[
        "demand_charge_lcoe_increment_usd_kwh"
    ] = (
        out["demand_charge_annual_usd"]
        / out["E_load_total_kwh"]
    )

    out[
        "lcoe_with_demand_charge_usd_kwh"
    ] = (
        out["lcoe_usd_kwh"]
        + out[
            "demand_charge_lcoe_increment_usd_kwh"
        ]
    )

    out[
        "lcoe_increase_percent"
    ] = (
        (
            out[
                "lcoe_with_demand_charge_usd_kwh"
            ]
            -
            out["lcoe_usd_kwh"]
        )
        /
        out["lcoe_usd_kwh"]
        * 100.0
    )

    return out


h2x = add_demand_charge(
    h2,
    "H2",
)

b1x = add_demand_charge(
    b1,
    "B1",
)


# ============================================================
# NONDOMINANCE FUNCTION
# ============================================================

def is_dominated(df, i):

    xi = df.loc[
        i,
        "P_peak_grid_opt_kw",
    ]

    yi = df.loc[
        i,
        "lcoe_with_demand_charge_usd_kwh",
    ]

    better_or_equal = (
        (
            df["P_peak_grid_opt_kw"]
            <= xi
        )
        &
        (
            df[
                "lcoe_with_demand_charge_usd_kwh"
            ]
            <= yi
        )
    )

    strictly_better = (
        (
            df["P_peak_grid_opt_kw"]
            < xi
        )
        |
        (
            df[
                "lcoe_with_demand_charge_usd_kwh"
            ]
            < yi
        )
    )

    mask = (
        better_or_equal
        &
        strictly_better
    )

    mask.loc[i] = False

    return bool(
        mask.any()
    )


def add_nondominated_flag(df):

    out = df.copy().reset_index(drop=True)

    dominated = []

    for i in out.index:

        dominated.append(
            is_dominated(
                out,
                i,
            )
        )

    out[
        "globally_nondominated_expost"
    ] = [
        not x
        for x in dominated
    ]

    return out


# ============================================================
# COMBINED GLOBAL FRONT
# ============================================================

combined = pd.concat(
    [
        h2x,
        b1x,
    ],
    ignore_index=True,
    sort=False,
)

combined = add_nondominated_flag(
    combined
)

global_front = combined[
    combined[
        "globally_nondominated_expost"
    ]
].copy()

global_front = (
    global_front
    .sort_values(
        [
            "P_peak_grid_opt_kw",
            "lcoe_with_demand_charge_usd_kwh",
        ]
    )
    .reset_index(drop=True)
)


# ============================================================
# B1 DOMINANCE BY H2
# ============================================================

rows = []

for i, rb in b1x.iterrows():

    bp = float(
        rb[
            "P_peak_grid_opt_kw"
        ]
    )

    bl = float(
        rb[
            "lcoe_with_demand_charge_usd_kwh"
        ]
    )

    candidates = h2x[
        (
            h2x[
                "P_peak_grid_opt_kw"
            ]
            <= bp
        )
        &
        (
            h2x[
                "lcoe_with_demand_charge_usd_kwh"
            ]
            <= bl
        )
    ].copy()

    dominated = (
        len(candidates)
        > 0
    )

    if dominated:

        candidates[
            "objective_distance"
        ] = (
            (
                candidates[
                    "P_peak_grid_opt_kw"
                ]
                - bp
            ) ** 2
            +
            (
                (
                    candidates[
                        "lcoe_with_demand_charge_usd_kwh"
                    ]
                    - bl
                )
                * 1000.0
            ) ** 2
        ) ** 0.5

        best = candidates.loc[
            candidates[
                "objective_distance"
            ].idxmin()
        ]

        h2_id = best.get(
            "global_solution_id",
            best.get(
                "solution_id",
                None,
            ),
        )

        h2_peak = float(
            best[
                "P_peak_grid_opt_kw"
            ]
        )

        h2_lcoe = float(
            best[
                "lcoe_with_demand_charge_usd_kwh"
            ]
        )

    else:

        h2_id = None
        h2_peak = None
        h2_lcoe = None


    rows.append(
        {
            "b1_index": i,

            "b1_source_run":
                rb.get(
                    "source_run",
                    None,
                ),

            "b1_source_solution_id":
                rb.get(
                    "source_solution_id",
                    rb.get(
                        "solution_id",
                        None,
                    ),
                ),

            "b1_peak_kw":
                bp,

            "b1_lcoe_original_usd_kwh":
                float(
                    rb[
                        "lcoe_usd_kwh"
                    ]
                ),

            "b1_lcoe_expost_usd_kwh":
                bl,

            "dominated_by_h2_expost":
                dominated,

            "h2_dominator_id":
                h2_id,

            "h2_peak_kw":
                h2_peak,

            "h2_lcoe_expost_usd_kwh":
                h2_lcoe,
        }
    )


dominance = pd.DataFrame(
    rows
)


# ============================================================
# SAVE
# ============================================================

h2x.to_csv(
    OUT_H2,
    index=False,
)

b1x.to_csv(
    OUT_B1,
    index=False,
)

combined.to_csv(
    OUT_COMBINED,
    index=False,
)

global_front.to_csv(
    OUT_GLOBAL,
    index=False,
)

dominance.to_csv(
    OUT_DOMINANCE,
    index=False,
)


# ============================================================
# REPORT
# ============================================================

print()
print("=== H2 ===")

print(
    "mean demand charge USD/year =",
    h2x[
        "demand_charge_annual_usd"
    ].mean(),
)

print(
    "mean LCOE increase % =",
    h2x[
        "lcoe_increase_percent"
    ].mean(),
)

print(
    "LCOE range original =",
    h2x[
        "lcoe_usd_kwh"
    ].min(),
    "to",
    h2x[
        "lcoe_usd_kwh"
    ].max(),
)

print(
    "LCOE range expost =",
    h2x[
        "lcoe_with_demand_charge_usd_kwh"
    ].min(),
    "to",
    h2x[
        "lcoe_with_demand_charge_usd_kwh"
    ].max(),
)


print()
print("=== B1 ===")

print(
    "mean demand charge USD/year =",
    b1x[
        "demand_charge_annual_usd"
    ].mean(),
)

print(
    "mean LCOE increase % =",
    b1x[
        "lcoe_increase_percent"
    ].mean(),
)

print(
    "LCOE range original =",
    b1x[
        "lcoe_usd_kwh"
    ].min(),
    "to",
    b1x[
        "lcoe_usd_kwh"
    ].max(),
)

print(
    "LCOE range expost =",
    b1x[
        "lcoe_with_demand_charge_usd_kwh"
    ].min(),
    "to",
    b1x[
        "lcoe_with_demand_charge_usd_kwh"
    ].max(),
)


print()
print("=== GLOBAL NONDOMINATED FRONT ===")

print(
    "solutions =",
    len(global_front),
)

print(
    global_front[
        "route"
    ].value_counts().to_string()
)


print()
print("=== B1 DOMINANCE TEST ===")

n_dom = int(
    dominance[
        "dominated_by_h2_expost"
    ].sum()
)

print(
    "B1 dominated by H2 =",
    n_dom,
    "/",
    len(dominance),
)

print(
    "B1 survivors =",
    len(dominance)
    - n_dom,
)


print()
print("=== OUTPUT FILES ===")

for path in [
    OUT_H2,
    OUT_B1,
    OUT_COMBINED,
    OUT_GLOBAL,
    OUT_DOMINANCE,
]:

    print(
        path.relative_to(ROOT)
    )