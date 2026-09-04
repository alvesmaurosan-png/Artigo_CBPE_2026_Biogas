from pathlib import Path
import pandas as pd

H2_FILE = Path(
    "results/paper/source_data/"
    "pareto_h2_global_nondominated_with_knee.csv"
)

B1_FILE = Path(
    "results/paper/source_data/"
    "pareto_b1_global_nondominated_with_knee.csv"
)

OUT = Path("results/paper/source_data")
OUT.mkdir(parents=True, exist_ok=True)

h2 = pd.read_csv(H2_FILE)
b1 = pd.read_csv(B1_FILE)

h2 = h2.copy()
b1 = b1.copy()

h2["route"] = "H2"
b1["route"] = "B1_biomethane"

# Preserve traceability.
# H2 already has global_solution_id.
if "global_solution_id" not in b1.columns:
    b1["global_solution_id"] = range(len(b1))

objectives = [
    "lcoe_usd_kwh",
    "P_peak_grid_opt_kw",
]

combined = pd.concat(
    [h2, b1],
    ignore_index=True,
    sort=False,
)

# ---------------------------------------------------------
# Global nondominance test
# Both objectives are minimized.
# ---------------------------------------------------------

def is_dominated(i, df):
    x = df.loc[i, objectives]

    for j in df.index:
        if i == j:
            continue

        y = df.loc[j, objectives]

        no_worse = (
            (y["lcoe_usd_kwh"] <= x["lcoe_usd_kwh"])
            and
            (y["P_peak_grid_opt_kw"] <= x["P_peak_grid_opt_kw"])
        )

        strictly_better = (
            (y["lcoe_usd_kwh"] < x["lcoe_usd_kwh"])
            or
            (y["P_peak_grid_opt_kw"] < x["P_peak_grid_opt_kw"])
        )

        if no_worse and strictly_better:
            return True

    return False


combined["globally_dominated_cross_route"] = [
    is_dominated(i, combined)
    for i in combined.index
]

global_front = combined[
    ~combined["globally_dominated_cross_route"]
].copy()

# ---------------------------------------------------------
# For every B1 solution, test specifically whether H2
# contains a solution that dominates it.
# ---------------------------------------------------------

b1_rows = []

for _, b in b1.iterrows():

    mask = (
        (h2["lcoe_usd_kwh"] <= b["lcoe_usd_kwh"])
        &
        (h2["P_peak_grid_opt_kw"] <= b["P_peak_grid_opt_kw"])
        &
        (
            (h2["lcoe_usd_kwh"] < b["lcoe_usd_kwh"])
            |
            (h2["P_peak_grid_opt_kw"] < b["P_peak_grid_opt_kw"])
        )
    )

    dominators = h2.loc[mask].copy()

    record = {
        "source_run": b["source_run"],
        "source_solution_id": b["source_solution_id"],
        "lcoe_usd_kwh": b["lcoe_usd_kwh"],
        "P_peak_grid_opt_kw": b["P_peak_grid_opt_kw"],
        "dominated_by_h2": not dominators.empty,
        "n_h2_dominators": len(dominators),
    }

    if not dominators.empty:
        # One representative H2 dominator:
        # closest in peak among the valid dominators.
        d = dominators.sort_values(
            ["P_peak_grid_opt_kw", "lcoe_usd_kwh"],
            ascending=[False, True],
        ).iloc[0]

        record.update({
            "h2_dominator_global_solution_id":
                d.get("global_solution_id"),
            "h2_dominator_source_run":
                d.get("source_run"),
            "h2_dominator_source_solution_id":
                d.get("source_solution_id"),
            "h2_dominator_lcoe_usd_kwh":
                d["lcoe_usd_kwh"],
            "h2_dominator_peak_kw":
                d["P_peak_grid_opt_kw"],
        })

    b1_rows.append(record)

b1_test = pd.DataFrame(b1_rows)

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

combined.to_csv(
    OUT / "pareto_h2_b1_all_cross_route.csv",
    index=False,
)

global_front.to_csv(
    OUT / "pareto_h2_b1_global_nondominated.csv",
    index=False,
)

b1_test.to_csv(
    OUT / "b1_dominance_by_h2.csv",
    index=False,
)

# ---------------------------------------------------------
# Report
# ---------------------------------------------------------

print()
print("=== CROSS-ROUTE PARETO ANALYSIS ===")
print("H2 solutions =", len(h2))
print("B1 solutions =", len(b1))
print("Combined =", len(combined))

print()
print("=== GLOBAL NONDOMINATED FRONT ===")
print("solutions =", len(global_front))
print(global_front["route"].value_counts().to_string())

print()
print("=== B1 DOMINANCE TEST ===")
print(
    b1_test["dominated_by_h2"]
    .value_counts()
    .rename(index={True: "dominated", False: "not dominated"})
    .to_string()
)

print()
print("B1 dominated by H2 =",
      int(b1_test["dominated_by_h2"].sum()),
      "/", len(b1_test))

survivors = b1_test[
    ~b1_test["dominated_by_h2"]
]

print()
print("=== B1 SURVIVORS ===")

if survivors.empty:
    print("NONE")
else:
    print(
        survivors[
            [
                "source_run",
                "source_solution_id",
                "lcoe_usd_kwh",
                "P_peak_grid_opt_kw",
            ]
        ].to_string(index=False)
    )

print()
print("=== FILES ===")
print(OUT / "pareto_h2_b1_all_cross_route.csv")
print(OUT / "pareto_h2_b1_global_nondominated.csv")
print(OUT / "b1_dominance_by_h2.csv")