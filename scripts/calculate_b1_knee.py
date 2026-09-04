from pathlib import Path
import numpy as np
import pandas as pd

INPUT = Path(
    "results/paper/source_data/"
    "pareto_b1_global_nondominated.csv"
)

OUTPUT_DIR = Path("results/paper/source_data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

required = [
    "source_run",
    "source_solution_id",
    "pv_kw",
    "bsv_kwh",
    "biomethane_storage_nm3",
    "chp_kw",
    "lcoe_usd_kwh",
    "P_peak_grid_opt_kw",
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

# ------------------------------------------------------------
# 1. Normalize the two minimization objectives to [0, 1]
# ------------------------------------------------------------

x = df["P_peak_grid_opt_kw"].astype(float)
y = df["lcoe_usd_kwh"].astype(float)

x_norm = (x - x.min()) / (x.max() - x.min())
y_norm = (y - y.min()) / (y.max() - y.min())

df["peak_norm"] = x_norm
df["lcoe_norm"] = y_norm

# ------------------------------------------------------------
# 2. Define the line connecting the two Pareto extremes
# ------------------------------------------------------------

idx_min_peak = df["P_peak_grid_opt_kw"].idxmin()
idx_min_lcoe = df["lcoe_usd_kwh"].idxmin()

A = np.array([
    df.loc[idx_min_peak, "peak_norm"],
    df.loc[idx_min_peak, "lcoe_norm"],
])

B = np.array([
    df.loc[idx_min_lcoe, "peak_norm"],
    df.loc[idx_min_lcoe, "lcoe_norm"],
])

AB = B - A
AB_norm = np.linalg.norm(AB)

if AB_norm == 0:
    raise ValueError("Pareto extremes are coincident.")

# ------------------------------------------------------------
# 3. Perpendicular distance of every solution to the chord
# ------------------------------------------------------------

distances = []

for _, row in df.iterrows():
    P = np.array([
        row["peak_norm"],
        row["lcoe_norm"],
    ])

    AP = P - A

    distance = abs(
        AB[0] * AP[1] - AB[1] * AP[0]
    ) / AB_norm

    distances.append(distance)

df["knee_distance"] = distances

# ------------------------------------------------------------
# 4. Knee = maximum perpendicular distance
# ------------------------------------------------------------

idx_knee = df["knee_distance"].idxmax()
knee = df.loc[idx_knee]

# ------------------------------------------------------------
# 5. Save complete frontier and selected knee
# ------------------------------------------------------------

front_path = OUTPUT_DIR / "pareto_b1_global_nondominated_with_knee.csv"
knee_path = OUTPUT_DIR / "pareto_b1_knee_point.csv"

df.to_csv(front_path, index=False)
df.loc[[idx_knee]].to_csv(knee_path, index=False)

# ------------------------------------------------------------
# 6. Report
# ------------------------------------------------------------

print()
print("=== B1 CONSOLIDATED PARETO ===")
print("solutions =", len(df))

print()
print("=== EXTREME: MINIMUM GRID PEAK ===")
print(
    df.loc[
        idx_min_peak,
        [
            "source_run",
            "source_solution_id",
            "pv_kw",
            "bsv_kwh",
            "biomethane_storage_nm3",
            "chp_kw",
            "lcoe_usd_kwh",
            "P_peak_grid_opt_kw",
        ],
    ].to_string()
)

print()
print("=== KNEE POINT ===")
print(
    knee[
        [
            "source_run",
            "source_solution_id",
            "pv_kw",
            "bsv_kwh",
            "biomethane_storage_nm3",
            "chp_kw",
            "lcoe_usd_kwh",
            "P_peak_grid_opt_kw",
            "peak_norm",
            "lcoe_norm",
            "knee_distance",
        ]
    ].to_string()
)

print()
print("=== EXTREME: MINIMUM LCOE ===")
print(
    df.loc[
        idx_min_lcoe,
        [
            "source_run",
            "source_solution_id",
            "pv_kw",
            "bsv_kwh",
            "biomethane_storage_nm3",
            "chp_kw",
            "lcoe_usd_kwh",
            "P_peak_grid_opt_kw",
        ],
    ].to_string()
)

print()
print("=== OUTPUT FILES ===")
print(front_path)
print(knee_path)