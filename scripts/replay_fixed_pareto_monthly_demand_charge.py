from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from src.models.milp_dispatch import MILPDispatchOptimizer
from src.economics.lcoe import build_economics_summary


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

SOURCE_DIR = (
    ROOT
    / "results"
    / "paper"
    / "source_data"
)

DATA_FILE = (
    ROOT
    / "data"
    / "processed"
    / "fleet_demand_sp.csv"
)

H2_CONFIG_FILE = (
    ROOT
    / "configs"
    / "paper"
    / "pv_bsv_h2_1500.yaml"
)

B1_CONFIG_FILE = (
    ROOT
    / "configs"
    / "paper"
    / "pv_bsv_biomethane_1500.yaml"
)

H2_PARETO_FILE = (
    SOURCE_DIR
    / "pareto_h2_global_nondominated_with_knee.csv"
)

B1_PARETO_FILE = (
    SOURCE_DIR
    / "pareto_b1_global_nondominated_with_knee.csv"
)


# ============================================================
# OUTPUTS
# ============================================================

OUT_MONTHLY = (
    SOURCE_DIR
    / "demand_charge_monthly_fixed_dispatch.csv"
)

OUT_SUMMARY = (
    SOURCE_DIR
    / "pareto_h2_b1_demand_charge_monthly_expost.csv"
)

OUT_GLOBAL = (
    SOURCE_DIR
    / "pareto_h2_b1_global_nondominated_monthly_demand_charge.csv"
)

OUT_B1_DOMINANCE = (
    SOURCE_DIR
    / "b1_dominance_by_h2_monthly_demand_charge.csv"
)

OUT_REPLAY_AUDIT = (
    SOURCE_DIR
    / "fixed_dispatch_replay_audit.csv"
)


# ============================================================
# PARAMETERS
# ============================================================

DEMAND_CHARGE_USD_KW_MONTH = 30.0

# Baseline 2026: 8760 h = non-leap year.
CALENDAR_START = "2026-01-01 00:00:00"

# Tolerances only for auditing the replay.
PEAK_TOLERANCE_KW = 0.10
LCOE_TOLERANCE_USD_KWH = 1e-5


# ============================================================
# UTILITIES
# ============================================================

def load_yaml(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return yaml.safe_load(f)


def prepare_input_data(path: Path) -> pd.DataFrame:

    df = pd.read_csv(path).copy()

    required = [
        "hour",
        "demand_kw",
        "pv_factor",
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:
        raise KeyError(
            "Input data missing columns: "
            + ", ".join(missing)
        )

    df["hour"] = pd.to_numeric(
        df["hour"],
        errors="coerce",
    )

    if df["hour"].isna().any():
        raise ValueError(
            "Invalid values in hour column"
        )

    df = (
        df
        .sort_values("hour")
        .reset_index(drop=True)
    )

    # Reproduce ga_nsga2.py preparation.
    df["t_global"] = range(
        len(df)
    )

    return df


def get_capacities(
    row: pd.Series,
    route: str,
) -> dict[str, float]:

    if route == "H2":

        cols = [
            "pv_kw",
            "bsv_kwh",
            "electrolyzer_kw",
            "h2_tank_kg",
            "fuelcell_kw",
        ]

    elif route == "B1":

        cols = [
            "pv_kw",
            "bsv_kwh",
            "biomethane_storage_nm3",
            "chp_kw",
        ]

    else:

        raise ValueError(
            f"Unknown route: {route}"
        )

    missing = [
        c
        for c in cols
        if c not in row.index
    ]

    if missing:
        raise KeyError(
            f"{route}: missing capacity columns: "
            + ", ".join(missing)
        )

    return {
        c: float(row[c])
        for c in cols
    }


def add_calendar(
    dispatch: pd.DataFrame,
) -> pd.DataFrame:

    out = dispatch.copy()

    if len(out) != 8760:
        raise ValueError(
            "Expected 8760 hourly rows, "
            f"received {len(out)}"
        )

    timestamps = pd.date_range(
        start=CALENDAR_START,
        periods=len(out),
        freq="h",
    )

    out["timestamp"] = timestamps
    out["month"] = timestamps.month

    return out


def monthly_peaks(
    dispatch: pd.DataFrame,
) -> pd.DataFrame:

    if "p_grid_kw" not in dispatch.columns:
        raise KeyError(
            "dispatch does not contain p_grid_kw"
        )

    d = add_calendar(
        dispatch
    )

    result = (
        d
        .groupby(
            "month",
            as_index=False,
        )["p_grid_kw"]
        .max()
        .rename(
            columns={
                "p_grid_kw":
                    "monthly_peak_grid_kw"
            }
        )
    )

    if len(result) != 12:
        raise ValueError(
            "Expected 12 monthly peaks, "
            f"received {len(result)}"
        )

    result[
        "demand_charge_usd"
    ] = (
        result[
            "monthly_peak_grid_kw"
        ]
        * DEMAND_CHARGE_USD_KW_MONTH
    )

    return result


def nondominated_mask(
    df: pd.DataFrame,
    peak_col: str,
    lcoe_col: str,
) -> pd.Series:

    values = df[
        [
            peak_col,
            lcoe_col,
        ]
    ].to_numpy()

    keep = []

    for i in range(
        len(values)
    ):

        peak_i = values[i, 0]
        lcoe_i = values[i, 1]

        dominated = False

        for j in range(
            len(values)
        ):

            if i == j:
                continue

            peak_j = values[j, 0]
            lcoe_j = values[j, 1]

            no_worse = (
                peak_j <= peak_i
                and
                lcoe_j <= lcoe_i
            )

            strictly_better = (
                peak_j < peak_i
                or
                lcoe_j < lcoe_i
            )

            if (
                no_worse
                and
                strictly_better
            ):
                dominated = True
                break

        keep.append(
            not dominated
        )

    return pd.Series(
        keep,
        index=df.index,
    )


# ============================================================
# LOAD INPUTS
# ============================================================

df_h = prepare_input_data(
    DATA_FILE
)

h2_cfg = load_yaml(
    H2_CONFIG_FILE
)

b1_cfg = load_yaml(
    B1_CONFIG_FILE
)

h2 = pd.read_csv(
    H2_PARETO_FILE
).copy()

b1 = pd.read_csv(
    B1_PARETO_FILE
).copy()


print("=" * 72)
print("FIXED-PARETO MONTHLY DEMAND CHARGE REPLAY")
print("=" * 72)

print()
print("=== INPUT ===")
print("Hourly rows =", len(df_h))
print("H2 solutions =", len(h2))
print("B1 solutions =", len(b1))

print()
print("=== CONFIGURATION ===")

print(
    "H2 period_hours =",
    h2_cfg[
        "optimization"
    ][
        "pareto_period_hours"
    ],
)

print(
    "B1 period_hours =",
    b1_cfg[
        "optimization"
    ][
        "pareto_period_hours"
    ],
)

print(
    "Demand charge =",
    DEMAND_CHARGE_USD_KW_MONTH,
    "USD/kW.month",
)


# ============================================================
# REPLAY
# ============================================================

summary_rows = []
monthly_rows = []
audit_rows = []


def replay_route(
    df_pareto: pd.DataFrame,
    config: dict,
    route: str,
):

    period_hours = int(
        config[
            "optimization"
        ][
            "pareto_period_hours"
        ]
    )

    n = len(
        df_pareto
    )

    for count, (
        idx,
        row,
    ) in enumerate(
        df_pareto.iterrows(),
        start=1,
    ):

        source_run = row.get(
            "source_run",
            "",
        )

        source_solution_id = row.get(
            "source_solution_id",
            row.get(
                "solution_id",
                idx,
            ),
        )

        global_solution_id = row.get(
            "global_solution_id",
            idx,
        )

        print()
        print(
            f"[{route} {count:02d}/{n:02d}] "
            f"global={global_solution_id} "
            f"source={source_run}/{source_solution_id}"
        )

        capacities = get_capacities(
            row,
            route,
        )

        optimizer = (
            MILPDispatchOptimizer(
                config=config,
                capacities=capacities,
                degradation_model=None,
            )
        )

        result = (
            optimizer
            .run_annual_simulation(
                df=df_h,
                period_hours=period_hours,
            )
        )

        dispatch = (
            result.dispatch_df
        )

        if (
            dispatch.empty
            or
            result.solver_status
            not in (
                "OPTIMAL",
                "FEASIBLE",
            )
        ):

            print(
                "  ERROR: replay failed:",
                result.solver_status,
            )

            audit_rows.append(
                {
                    "route": route,
                    "global_solution_id":
                        global_solution_id,
                    "source_run":
                        source_run,
                    "source_solution_id":
                        source_solution_id,
                    "status":
                        result.solver_status,
                    "replay_valid":
                        False,
                }
            )

            continue

        # ---------------------------------------------
        # Replay economics
        # ---------------------------------------------

        economics = (
            build_economics_summary(
                config=config,
                capacities=capacities,
                dispatch_df=dispatch,
            )
        )

        replay_peak = float(
            dispatch[
                "p_grid_kw"
            ].max()
        )

        replay_lcoe = float(
            economics[
                "lcoe_usd_kwh"
            ]
        )

        stored_peak = float(
            row[
                "P_peak_grid_opt_kw"
            ]
        )

        stored_lcoe = float(
            row[
                "lcoe_usd_kwh"
            ]
        )

        delta_peak = (
            replay_peak
            - stored_peak
        )

        delta_lcoe = (
            replay_lcoe
            - stored_lcoe
        )

        replay_valid = (
            abs(delta_peak)
            <= PEAK_TOLERANCE_KW
            and
            abs(delta_lcoe)
            <= LCOE_TOLERANCE_USD_KWH
        )

        audit_rows.append(
            {
                "route":
                    route,

                "global_solution_id":
                    global_solution_id,

                "source_run":
                    source_run,

                "source_solution_id":
                    source_solution_id,

                "status":
                    result.solver_status,

                "stored_peak_kw":
                    stored_peak,

                "replay_peak_kw":
                    replay_peak,

                "delta_peak_kw":
                    delta_peak,

                "stored_lcoe_usd_kwh":
                    stored_lcoe,

                "replay_lcoe_usd_kwh":
                    replay_lcoe,

                "delta_lcoe_usd_kwh":
                    delta_lcoe,

                "replay_valid":
                    replay_valid,

                "solve_time_sec":
                    result.solve_time_sec,
            }
        )

        print(
            "  peak stored/replay =",
            f"{stored_peak:.6f}",
            "/",
            f"{replay_peak:.6f}",
        )

        print(
            "  LCOE stored/replay =",
            f"{stored_lcoe:.9f}",
            "/",
            f"{replay_lcoe:.9f}",
        )

        # ---------------------------------------------
        # Monthly demand charge
        # ---------------------------------------------

        mp = monthly_peaks(
            dispatch
        )

        annual_demand_charge = float(
            mp[
                "demand_charge_usd"
            ].sum()
        )

        # D1 stress-test value:
        d1_annual_charge = (
            replay_peak
            * DEMAND_CHARGE_USD_KW_MONTH
            * 12.0
        )

        # Critical invariant:
        if (
            annual_demand_charge
            >
            d1_annual_charge
            + 1e-6
        ):
            raise RuntimeError(
                "Consistency error: "
                "D2 monthly charge > D1 annual-peak charge"
            )

        energy_load = float(
            row[
                "E_load_total_kwh"
            ]
        )

        lcoe_increment = (
            annual_demand_charge
            / energy_load
        )

        adjusted_lcoe = (
            stored_lcoe
            + lcoe_increment
        )

        for _, mr in (
            mp.iterrows()
        ):

            monthly_rows.append(
                {
                    "route":
                        route,

                    "global_solution_id":
                        global_solution_id,

                    "source_run":
                        source_run,

                    "source_solution_id":
                        source_solution_id,

                    "month":
                        int(
                            mr["month"]
                        ),

                    "monthly_peak_grid_kw":
                        float(
                            mr[
                                "monthly_peak_grid_kw"
                            ]
                        ),

                    "demand_charge_usd_kw_month":
                        DEMAND_CHARGE_USD_KW_MONTH,

                    "monthly_demand_charge_usd":
                        float(
                            mr[
                                "demand_charge_usd"
                            ]
                        ),
                }
            )

        summary = {
            "route":
                route,

            "global_solution_id":
                global_solution_id,

            "source_run":
                source_run,

            "source_solution_id":
                source_solution_id,

            "P_peak_grid_opt_kw":
                stored_peak,

            "replay_peak_kw":
                replay_peak,

            "lcoe_original_usd_kwh":
                stored_lcoe,

            "replay_lcoe_usd_kwh":
                replay_lcoe,

            "E_load_total_kwh":
                energy_load,

            "demand_charge_usd_kw_month":
                DEMAND_CHARGE_USD_KW_MONTH,

            "d1_annual_peak_charge_usd":
                d1_annual_charge,

            "d2_monthly_demand_charge_usd":
                annual_demand_charge,

            "d2_minus_d1_usd":
                (
                    annual_demand_charge
                    - d1_annual_charge
                ),

            "d2_vs_d1_percent":
                (
                    annual_demand_charge
                    /
                    d1_annual_charge
                    * 100.0
                ),

            "d2_lcoe_increment_usd_kwh":
                lcoe_increment,

            "lcoe_D2_usd_kwh":
                adjusted_lcoe,

            "lcoe_D2_increase_percent":
                (
                    lcoe_increment
                    /
                    stored_lcoe
                    * 100.0
                ),

            "replay_valid":
                replay_valid,

            "solve_time_sec":
                result.solve_time_sec,
        }

        # Preserve capacities for audit.
        summary.update(
            capacities
        )

        summary_rows.append(
            summary
        )


# ============================================================
# RUN H2 + B1
# ============================================================

replay_route(
    h2,
    h2_cfg,
    "H2",
)

replay_route(
    b1,
    b1_cfg,
    "B1",
)


# ============================================================
# DATAFRAMES
# ============================================================

summary = pd.DataFrame(
    summary_rows
)

monthly = pd.DataFrame(
    monthly_rows
)

audit = pd.DataFrame(
    audit_rows
)


# ============================================================
# REPLAY AUDIT
# ============================================================

print()
print("=" * 72)
print("REPLAY AUDIT")
print("=" * 72)

if audit.empty:
    raise RuntimeError(
        "Replay audit is empty"
    )

successful = audit[
    audit["replay_valid"].notna()
].copy()

print(
    "valid replays =",
    int(
        successful[
            "replay_valid"
        ].sum()
    ),
    "/",
    len(successful),
)

print(
    "max abs peak error kW =",
    successful[
        "delta_peak_kw"
    ].abs().max(),
)

print(
    "max abs LCOE error =",
    successful[
        "delta_lcoe_usd_kwh"
    ].abs().max(),
)


# ============================================================
# GLOBAL NONDOMINANCE D2
# ============================================================

summary[
    "globally_nondominated_D2"
] = nondominated_mask(
    summary,
    peak_col=
        "P_peak_grid_opt_kw",
    lcoe_col=
        "lcoe_D2_usd_kwh",
)

global_front = (
    summary[
        summary[
            "globally_nondominated_D2"
        ]
    ]
    .sort_values(
        [
            "P_peak_grid_opt_kw",
            "lcoe_D2_usd_kwh",
        ]
    )
    .reset_index(drop=True)
)


# ============================================================
# B1 DOMINANCE BY H2
# ============================================================

h2_D2 = summary[
    summary["route"] == "H2"
].copy()

b1_D2 = summary[
    summary["route"] == "B1"
].copy()

dominance_rows = []


for _, rb in (
    b1_D2.iterrows()
):

    candidates = h2_D2[
        (
            h2_D2[
                "P_peak_grid_opt_kw"
            ]
            <=
            rb[
                "P_peak_grid_opt_kw"
            ]
        )
        &
        (
            h2_D2[
                "lcoe_D2_usd_kwh"
            ]
            <=
            rb[
                "lcoe_D2_usd_kwh"
            ]
        )
    ].copy()

    dominated = (
        len(candidates)
        > 0
    )

    if dominated:

        # Representative dominator:
        candidates[
            "distance"
        ] = (
            (
                candidates[
                    "P_peak_grid_opt_kw"
                ]
                -
                rb[
                    "P_peak_grid_opt_kw"
                ]
            ) ** 2
            +
            (
                (
                    candidates[
                        "lcoe_D2_usd_kwh"
                    ]
                    -
                    rb[
                        "lcoe_D2_usd_kwh"
                    ]
                )
                * 1000.0
            ) ** 2
        ) ** 0.5

        best = candidates.loc[
            candidates[
                "distance"
            ].idxmin()
        ]

        h2_id = best[
            "global_solution_id"
        ]

        h2_peak = best[
            "P_peak_grid_opt_kw"
        ]

        h2_lcoe = best[
            "lcoe_D2_usd_kwh"
        ]

    else:

        h2_id = None
        h2_peak = None
        h2_lcoe = None

    dominance_rows.append(
        {
            "b1_global_solution_id":
                rb[
                    "global_solution_id"
                ],

            "b1_source_run":
                rb[
                    "source_run"
                ],

            "b1_source_solution_id":
                rb[
                    "source_solution_id"
                ],

            "b1_peak_kw":
                rb[
                    "P_peak_grid_opt_kw"
                ],

            "b1_lcoe_D2_usd_kwh":
                rb[
                    "lcoe_D2_usd_kwh"
                ],

            "dominated_by_h2_D2":
                dominated,

            "h2_dominator_id":
                h2_id,

            "h2_peak_kw":
                h2_peak,

            "h2_lcoe_D2_usd_kwh":
                h2_lcoe,
        }
    )


dominance = pd.DataFrame(
    dominance_rows
)


# ============================================================
# SAVE
# ============================================================

monthly.to_csv(
    OUT_MONTHLY,
    index=False,
)

summary.to_csv(
    OUT_SUMMARY,
    index=False,
)

global_front.to_csv(
    OUT_GLOBAL,
    index=False,
)

dominance.to_csv(
    OUT_B1_DOMINANCE,
    index=False,
)

audit.to_csv(
    OUT_REPLAY_AUDIT,
    index=False,
)


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 72)
print("D2 MONTHLY DEMAND CHARGE RESULTS")
print("=" * 72)


for route in [
    "H2",
    "B1",
]:

    q = summary[
        summary["route"] == route
    ]

    print()
    print(
        f"=== {route} ==="
    )

    print(
        "solutions =",
        len(q),
    )

    print(
        "mean D1 charge USD/year =",
        q[
            "d1_annual_peak_charge_usd"
        ].mean(),
    )

    print(
        "mean D2 charge USD/year =",
        q[
            "d2_monthly_demand_charge_usd"
        ].mean(),
    )

    print(
        "mean D2 / D1 % =",
        q[
            "d2_vs_d1_percent"
        ].mean(),
    )

    print(
        "mean D2 LCOE increase % =",
        q[
            "lcoe_D2_increase_percent"
        ].mean(),
    )

    print(
        "D2 LCOE range =",
        q[
            "lcoe_D2_usd_kwh"
        ].min(),
        "to",
        q[
            "lcoe_D2_usd_kwh"
        ].max(),
    )


print()
print("=== GLOBAL NONDOMINATED D2 ===")

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
print("=== B1 DOMINANCE D2 ===")

n_dom = int(
    dominance[
        "dominated_by_h2_D2"
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

for f in [
    OUT_REPLAY_AUDIT,
    OUT_MONTHLY,
    OUT_SUMMARY,
    OUT_GLOBAL,
    OUT_B1_DOMINANCE,
]:

    print(
        f.relative_to(ROOT)
    )