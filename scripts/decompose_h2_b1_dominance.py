from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ROOT = Path("results/paper/source_data")

H2_FILE = ROOT / "pareto_h2_global_nondominated_with_knee.csv"
B1_FILE = ROOT / "pareto_b1_global_nondominated_with_knee.csv"
B1_KNEE_FILE = ROOT / "pareto_b1_knee_point.csv"

OUT_DECOMP = ROOT / "h2_b1_dominance_decomposition.csv"
OUT_BUDGET = ROOT / "h2_b1_peak_budget_advantage.csv"
OUT_SUMMARY = ROOT / "h2_b1_dominance_summary.csv"

FIG_DIR = Path("results/paper/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD
# ============================================================

h2 = pd.read_csv(H2_FILE)
b1 = pd.read_csv(B1_FILE)
b1_knee = pd.read_csv(B1_KNEE_FILE).iloc[0]


# ============================================================
# REQUIRED COLUMNS
# ============================================================

common_required = [
    "lcoe_usd_kwh",
    "P_peak_grid_opt_kw",
    "pv_kw",
    "bsv_kwh",
    "capex_total_usd",
    "E_grid_total_kwh",
    "total_grid_dependency_ratio",
]

for name, df in [("H2", h2), ("B1", b1)]:
    missing = [c for c in common_required if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns: {missing}")


# ============================================================
# KNEES AND EXTREMES
# ============================================================

# H2 knee already frozen as global_solution_id 41
h2_knee = h2.loc[h2["global_solution_id"] == 41].iloc[0]

h2_min_peak = h2.loc[h2["P_peak_grid_opt_kw"].idxmin()]
h2_min_lcoe = h2.loc[h2["lcoe_usd_kwh"].idxmin()]

b1_min_peak = b1.loc[b1["P_peak_grid_opt_kw"].idxmin()]
b1_min_lcoe = b1.loc[b1["lcoe_usd_kwh"].idxmin()]


# ============================================================
# NORMALIZATION FOR REPRESENTATIVE DOMINATOR
# ============================================================

all_lcoe = pd.concat(
    [h2["lcoe_usd_kwh"], b1["lcoe_usd_kwh"]],
    ignore_index=True,
)

all_peak = pd.concat(
    [h2["P_peak_grid_opt_kw"], b1["P_peak_grid_opt_kw"]],
    ignore_index=True,
)

lcoe_min = all_lcoe.min()
lcoe_max = all_lcoe.max()

peak_min = all_peak.min()
peak_max = all_peak.max()


def norm_lcoe(x):
    return (x - lcoe_min) / (lcoe_max - lcoe_min)


def norm_peak(x):
    return (x - peak_min) / (peak_max - peak_min)


# ============================================================
# DOMINANCE DECOMPOSITION
# ============================================================

records = []

for _, b in b1.iterrows():

    dominators = h2[
        (h2["lcoe_usd_kwh"] <= b["lcoe_usd_kwh"])
        &
        (h2["P_peak_grid_opt_kw"] <= b["P_peak_grid_opt_kw"])
        &
        (
            (h2["lcoe_usd_kwh"] < b["lcoe_usd_kwh"])
            |
            (h2["P_peak_grid_opt_kw"] < b["P_peak_grid_opt_kw"])
        )
    ].copy()

    if dominators.empty:
        raise RuntimeError(
            "B1 solution unexpectedly has no H2 dominator: "
            f"{b.get('source_run')} / {b.get('source_solution_id')}"
        )

    # --------------------------------------------------------
    # Representative dominator:
    # minimum Euclidean distance in normalized objective space
    # --------------------------------------------------------

    dominators["objective_distance"] = np.sqrt(
        (
            norm_lcoe(dominators["lcoe_usd_kwh"])
            - norm_lcoe(b["lcoe_usd_kwh"])
        ) ** 2
        +
        (
            norm_peak(dominators["P_peak_grid_opt_kw"])
            - norm_peak(b["P_peak_grid_opt_kw"])
        ) ** 2
    )

    d = dominators.loc[
        dominators["objective_distance"].idxmin()
    ]

    rec = {
        # ----------------------------------------------------
        # Traceability
        # ----------------------------------------------------
        "b1_source_run": b.get("source_run"),
        "b1_source_solution_id": b.get("source_solution_id"),

        "h2_global_solution_id": d.get("global_solution_id"),
        "h2_source_run": d.get("source_run"),
        "h2_source_solution_id": d.get("source_solution_id"),

        "n_h2_dominators": len(dominators),
        "objective_distance": d["objective_distance"],

        # ----------------------------------------------------
        # Objectives
        # ----------------------------------------------------
        "b1_lcoe_usd_kwh": b["lcoe_usd_kwh"],
        "h2_lcoe_usd_kwh": d["lcoe_usd_kwh"],

        # Positive = H2 advantage
        "h2_lcoe_advantage_usd_kwh":
            b["lcoe_usd_kwh"] - d["lcoe_usd_kwh"],

        "h2_lcoe_advantage_percent":
            (
                (b["lcoe_usd_kwh"] - d["lcoe_usd_kwh"])
                / b["lcoe_usd_kwh"]
            ) * 100.0,

        "b1_peak_kw": b["P_peak_grid_opt_kw"],
        "h2_peak_kw": d["P_peak_grid_opt_kw"],

        # Positive = H2 lower peak
        "h2_peak_advantage_kw":
            b["P_peak_grid_opt_kw"]
            - d["P_peak_grid_opt_kw"],

        "h2_peak_advantage_percent":
            (
                (
                    b["P_peak_grid_opt_kw"]
                    - d["P_peak_grid_opt_kw"]
                )
                / b["P_peak_grid_opt_kw"]
            ) * 100.0,

        # ----------------------------------------------------
        # PV and BSV
        # Positive here means H2 uses MORE capacity
        # ----------------------------------------------------
        "b1_pv_kw": b["pv_kw"],
        "h2_pv_kw": d["pv_kw"],
        "delta_pv_h2_minus_b1_kw":
            d["pv_kw"] - b["pv_kw"],

        "b1_bsv_kwh": b["bsv_kwh"],
        "h2_bsv_kwh": d["bsv_kwh"],
        "delta_bsv_h2_minus_b1_kwh":
            d["bsv_kwh"] - b["bsv_kwh"],

        # ----------------------------------------------------
        # CAPEX
        # Positive advantage = H2 lower CAPEX
        # ----------------------------------------------------
        "b1_capex_usd": b["capex_total_usd"],
        "h2_capex_usd": d["capex_total_usd"],

        "h2_capex_advantage_usd":
            b["capex_total_usd"]
            - d["capex_total_usd"],

        "h2_capex_advantage_percent":
            (
                (
                    b["capex_total_usd"]
                    - d["capex_total_usd"]
                )
                / b["capex_total_usd"]
            ) * 100.0,

        # ----------------------------------------------------
        # GRID ENERGY
        # Positive advantage = H2 imports less
        # ----------------------------------------------------
        "b1_E_grid_kwh": b["E_grid_total_kwh"],
        "h2_E_grid_kwh": d["E_grid_total_kwh"],

        "h2_grid_energy_advantage_kwh":
            b["E_grid_total_kwh"]
            - d["E_grid_total_kwh"],

        "h2_grid_energy_advantage_percent":
            (
                (
                    b["E_grid_total_kwh"]
                    - d["E_grid_total_kwh"]
                )
                / b["E_grid_total_kwh"]
            ) * 100.0,

        # ----------------------------------------------------
        # GRID SHARE
        # ----------------------------------------------------
        "b1_grid_dependency_ratio":
            b["total_grid_dependency_ratio"],

        "h2_grid_dependency_ratio":
            d["total_grid_dependency_ratio"],

        "h2_grid_dependency_advantage_pp":
            (
                b["total_grid_dependency_ratio"]
                - d["total_grid_dependency_ratio"]
            ) * 100.0,

        # ----------------------------------------------------
        # DISPATCHABLE TECHNOLOGY
        # Not directly comparable dimensionally:
        # FC vs CHP power is comparable;
        # storage media are reported separately.
        # ----------------------------------------------------
        "b1_chp_kw": b.get("chp_kw", np.nan),

        "b1_biomethane_storage_nm3":
            b.get("biomethane_storage_nm3", np.nan),

        "h2_fuelcell_kw":
            d.get("fuelcell_kw", np.nan),

        "h2_electrolyzer_kw":
            d.get("electrolyzer_kw", np.nan),

        "h2_tank_kg":
            d.get("h2_tank_kg", np.nan),

        "delta_dispatchable_electric_kw_h2_minus_b1":
            d.get("fuelcell_kw", np.nan)
            - b.get("chp_kw", np.nan),
    }

    records.append(rec)


decomp = pd.DataFrame(records)
decomp.to_csv(OUT_DECOMP, index=False)


# ============================================================
# SUMMARY STATISTICS
# ============================================================

summary_rows = []

metrics = [
    "h2_lcoe_advantage_usd_kwh",
    "h2_lcoe_advantage_percent",
    "h2_peak_advantage_kw",
    "h2_peak_advantage_percent",
    "h2_capex_advantage_usd",
    "h2_capex_advantage_percent",
    "h2_grid_energy_advantage_kwh",
    "h2_grid_energy_advantage_percent",
    "h2_grid_dependency_advantage_pp",
    "delta_pv_h2_minus_b1_kw",
    "delta_bsv_h2_minus_b1_kwh",
    "delta_dispatchable_electric_kw_h2_minus_b1",
]

for metric in metrics:
    s = decomp[metric].dropna()

    summary_rows.append({
        "metric": metric,
        "min": s.min(),
        "mean": s.mean(),
        "median": s.median(),
        "max": s.max(),
    })

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_SUMMARY, index=False)


# ============================================================
# PEAK-BUDGET ADVANTAGE MAP
#
# For each allowed grid peak:
# choose lowest LCOE solution whose actual peak <= budget.
# ============================================================

budget_start = int(
    np.ceil(
        max(
            h2["P_peak_grid_opt_kw"].min(),
            b1["P_peak_grid_opt_kw"].min(),
        )
    )
)

budget_end = int(
    np.ceil(
        max(
            h2["P_peak_grid_opt_kw"].max(),
            b1["P_peak_grid_opt_kw"].max(),
        )
    )
)

budget_records = []

for budget in np.arange(
    budget_start,
    budget_end + 0.001,
    1.0,
):

    h2_feasible = h2[
        h2["P_peak_grid_opt_kw"] <= budget
    ]

    b1_feasible = b1[
        b1["P_peak_grid_opt_kw"] <= budget
    ]

    if h2_feasible.empty or b1_feasible.empty:
        continue

    h2_best = h2_feasible.loc[
        h2_feasible["lcoe_usd_kwh"].idxmin()
    ]

    b1_best = b1_feasible.loc[
        b1_feasible["lcoe_usd_kwh"].idxmin()
    ]

    delta = (
        b1_best["lcoe_usd_kwh"]
        - h2_best["lcoe_usd_kwh"]
    )

    budget_records.append({
        "peak_budget_kw": budget,

        "h2_best_lcoe_usd_kwh":
            h2_best["lcoe_usd_kwh"],

        "h2_actual_peak_kw":
            h2_best["P_peak_grid_opt_kw"],

        "h2_global_solution_id":
            h2_best.get("global_solution_id"),

        "b1_best_lcoe_usd_kwh":
            b1_best["lcoe_usd_kwh"],

        "b1_actual_peak_kw":
            b1_best["P_peak_grid_opt_kw"],

        "b1_source_run":
            b1_best.get("source_run"),

        "b1_source_solution_id":
            b1_best.get("source_solution_id"),

        # Positive means H2 is cheaper.
        "h2_lcoe_advantage_usd_kwh":
            delta,

        "h2_lcoe_advantage_percent":
            (delta / b1_best["lcoe_usd_kwh"]) * 100.0,
    })


budget_df = pd.DataFrame(budget_records)
budget_df.to_csv(OUT_BUDGET, index=False)


# ============================================================
# FIGURE 1 — OVERLAID PARETO FRONTS
# ============================================================

fig, ax = plt.subplots(figsize=(9, 6))

ax.plot(
    h2["P_peak_grid_opt_kw"],
    h2["lcoe_usd_kwh"],
    marker="o",
    markersize=3,
    linewidth=1.2,
    label="H₂ consolidado",
)

ax.plot(
    b1["P_peak_grid_opt_kw"],
    b1["lcoe_usd_kwh"],
    marker="s",
    markersize=3,
    linewidth=1.2,
    label="B1 — Biometano–CHP",
)

# H2 characteristic points
ax.scatter(
    h2_min_peak["P_peak_grid_opt_kw"],
    h2_min_peak["lcoe_usd_kwh"],
    s=80,
    marker="^",
)

ax.scatter(
    h2_knee["P_peak_grid_opt_kw"],
    h2_knee["lcoe_usd_kwh"],
    s=110,
    marker="*",
)

ax.scatter(
    h2_min_lcoe["P_peak_grid_opt_kw"],
    h2_min_lcoe["lcoe_usd_kwh"],
    s=80,
    marker="v",
)

# B1 characteristic points
ax.scatter(
    b1_min_peak["P_peak_grid_opt_kw"],
    b1_min_peak["lcoe_usd_kwh"],
    s=80,
    marker="^",
)

ax.scatter(
    b1_knee["P_peak_grid_opt_kw"],
    b1_knee["lcoe_usd_kwh"],
    s=110,
    marker="*",
)

ax.scatter(
    b1_min_lcoe["P_peak_grid_opt_kw"],
    b1_min_lcoe["lcoe_usd_kwh"],
    s=80,
    marker="v",
)

ax.annotate(
    "Knee H₂",
    (
        h2_knee["P_peak_grid_opt_kw"],
        h2_knee["lcoe_usd_kwh"],
    ),
    xytext=(8, -18),
    textcoords="offset points",
)

ax.annotate(
    "Knee B1",
    (
        b1_knee["P_peak_grid_opt_kw"],
        b1_knee["lcoe_usd_kwh"],
    ),
    xytext=(-55, 12),
    textcoords="offset points",
)

ax.set_xlabel("Pico de potência da rede (kW)")
ax.set_ylabel("LCOE (USD/kWh)")
ax.set_title(
    "Fronteiras de Pareto consolidadas — H₂ × Biometano–CHP"
)

ax.grid(True, alpha=0.25)
ax.legend()
fig.tight_layout()

fig.savefig(
    FIG_DIR / "pareto_h2_vs_b1_overlay.png",
    dpi=300,
)

fig.savefig(
    FIG_DIR / "pareto_h2_vs_b1_overlay.pdf",
)

plt.close(fig)


# ============================================================
# FIGURE 2 — RELATIVE LCOE ADVANTAGE BY PEAK BUDGET
# ============================================================

fig, ax = plt.subplots(figsize=(9, 5.5))

ax.plot(
    budget_df["peak_budget_kw"],
    budget_df["h2_lcoe_advantage_usd_kwh"],
    linewidth=1.8,
)

ax.axhline(
    0.0,
    linewidth=1.0,
    linestyle="--",
)

ax.set_xlabel(
    "Limite admissível de pico da rede (kW)"
)

ax.set_ylabel(
    "Vantagem H₂ em LCOE (USD/kWh)\n"
    "B1 − H₂"
)

ax.set_title(
    "Vantagem econômica relativa da rota H₂ "
    "para o mesmo orçamento de potência"
)

ax.grid(True, alpha=0.25)

fig.tight_layout()

fig.savefig(
    FIG_DIR / "h2_lcoe_advantage_by_peak_budget.png",
    dpi=300,
)

fig.savefig(
    FIG_DIR / "h2_lcoe_advantage_by_peak_budget.pdf",
)

plt.close(fig)


# ============================================================
# REPORT
# ============================================================

print()
print("=== DOMINANCE DECOMPOSITION ===")
print("B1 solutions =", len(decomp))
print(
    "All B1 dominated =",
    len(decomp) == len(b1),
)

print()
print("=== H2 ADVANTAGE — REPRESENTATIVE DOMINATORS ===")

for col in [
    "h2_lcoe_advantage_percent",
    "h2_peak_advantage_kw",
    "h2_capex_advantage_percent",
    "h2_grid_energy_advantage_percent",
    "h2_grid_dependency_advantage_pp",
]:
    print()
    print(col)
    print(
        decomp[col]
        .describe()
        .to_string()
    )

print()
print("=== PEAK-BUDGET MAP ===")
print(
    "Budget range:",
    budget_df["peak_budget_kw"].min(),
    "to",
    budget_df["peak_budget_kw"].max(),
    "kW",
)

print(
    "H2 LCOE advantage min =",
    budget_df["h2_lcoe_advantage_usd_kwh"].min(),
)

print(
    "H2 LCOE advantage max =",
    budget_df["h2_lcoe_advantage_usd_kwh"].max(),
)

print(
    "H2 LCOE advantage mean =",
    budget_df["h2_lcoe_advantage_usd_kwh"].mean(),
)

print()
print("=== OUTPUT FILES ===")
print(OUT_DECOMP)
print(OUT_SUMMARY)
print(OUT_BUDGET)

print(
    FIG_DIR / "pareto_h2_vs_b1_overlay.png"
)

print(
    FIG_DIR / "h2_lcoe_advantage_by_peak_budget.png"
)