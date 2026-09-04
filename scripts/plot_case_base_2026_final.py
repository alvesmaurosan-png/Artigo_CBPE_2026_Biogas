from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "results" / "paper" / "source_data"
FIG_DIR = ROOT / "results" / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

H2_FILE = (
    DATA_DIR
    / "pareto_h2_global_nondominated_with_knee.csv"
)

B1_FILE = (
    DATA_DIR
    / "pareto_b1_global_nondominated_with_knee.csv"
)

OUT_PNG = (
    FIG_DIR
    / "pareto_case_base_2026_h2_vs_b1_final.png"
)

OUT_PDF = (
    FIG_DIR
    / "pareto_case_base_2026_h2_vs_b1_final.pdf"
)

OUT_SOURCE = (
    DATA_DIR
    / "pareto_case_base_2026_figure_source.csv"
)


# ============================================================
# LOAD
# ============================================================

h2 = pd.read_csv(H2_FILE)
b1 = pd.read_csv(B1_FILE)


print("=== INPUT ===")
print("H2 solutions =", len(h2))
print("B1 solutions =", len(b1))


if len(h2) != 61:
    raise ValueError(
        f"Expected 61 H2 solutions, found {len(h2)}"
    )

if len(b1) != 34:
    raise ValueError(
        f"Expected 34 B1 solutions, found {len(b1)}"
    )


# ============================================================
# CHARACTERISTIC POINTS
# ============================================================

# H2
h2_min_peak = h2.loc[
    h2["P_peak_grid_opt_kw"].idxmin()
]

h2_min_lcoe = h2.loc[
    h2["lcoe_usd_kwh"].idxmin()
]

# Frozen H2 knee
h2_knee = h2.loc[
    h2["global_solution_id"] == 41
].iloc[0]


# B1
b1_min_peak = b1.loc[
    b1["P_peak_grid_opt_kw"].idxmin()
]

b1_min_lcoe = b1.loc[
    b1["lcoe_usd_kwh"].idxmin()
]

# B1 knee = max distance
b1_knee = b1.loc[
    b1["knee_distance"].idxmax()
]


# ============================================================
# SORT FOR CLEAN LINES
# ============================================================

h2_plot = h2.sort_values(
    "P_peak_grid_opt_kw"
).reset_index(drop=True)

b1_plot = b1.sort_values(
    "P_peak_grid_opt_kw"
).reset_index(drop=True)


# ============================================================
# FIGURE
# ============================================================

fig, ax = plt.subplots(figsize=(10.5, 6.8))


# Pareto fronts
ax.plot(
    h2_plot["P_peak_grid_opt_kw"],
    h2_plot["lcoe_usd_kwh"],
    marker="o",
    markersize=3.5,
    linewidth=1.4,
    label="H₂ consolidado — 61 soluções",
)

ax.plot(
    b1_plot["P_peak_grid_opt_kw"],
    b1_plot["lcoe_usd_kwh"],
    marker="s",
    markersize=3.5,
    linewidth=1.4,
    label="B1 — Biometano–CHP T0 — 34 soluções",
)


# ------------------------------------------------------------
# H2 characteristic points
# ------------------------------------------------------------

ax.scatter(
    h2_min_peak["P_peak_grid_opt_kw"],
    h2_min_peak["lcoe_usd_kwh"],
    s=90,
    marker="^",
    zorder=5,
)

ax.scatter(
    h2_knee["P_peak_grid_opt_kw"],
    h2_knee["lcoe_usd_kwh"],
    s=140,
    marker="*",
    zorder=6,
)

ax.scatter(
    h2_min_lcoe["P_peak_grid_opt_kw"],
    h2_min_lcoe["lcoe_usd_kwh"],
    s=90,
    marker="v",
    zorder=5,
)


# ------------------------------------------------------------
# B1 characteristic points
# ------------------------------------------------------------

ax.scatter(
    b1_min_peak["P_peak_grid_opt_kw"],
    b1_min_peak["lcoe_usd_kwh"],
    s=90,
    marker="^",
    zorder=5,
)

ax.scatter(
    b1_knee["P_peak_grid_opt_kw"],
    b1_knee["lcoe_usd_kwh"],
    s=140,
    marker="*",
    zorder=6,
)

ax.scatter(
    b1_min_lcoe["P_peak_grid_opt_kw"],
    b1_min_lcoe["lcoe_usd_kwh"],
    s=90,
    marker="v",
    zorder=5,
)


# ============================================================
# ANNOTATIONS
# ============================================================

ax.annotate(
    (
        "H₂ — mínimo pico\n"
        f"{h2_min_peak['P_peak_grid_opt_kw']:.2f} kW\n"
        f"{h2_min_peak['lcoe_usd_kwh']:.5f} USD/kWh"
    ),
    (
        h2_min_peak["P_peak_grid_opt_kw"],
        h2_min_peak["lcoe_usd_kwh"],
    ),
    xytext=(12, -8),
    textcoords="offset points",
    fontsize=8.5,
)

ax.annotate(
    (
        "H₂ — knee\n"
        f"{h2_knee['P_peak_grid_opt_kw']:.2f} kW\n"
        f"{h2_knee['lcoe_usd_kwh']:.5f} USD/kWh"
    ),
    (
        h2_knee["P_peak_grid_opt_kw"],
        h2_knee["lcoe_usd_kwh"],
    ),
    xytext=(-74, -36),
    textcoords="offset points",
    fontsize=8.5,
)

ax.annotate(
    (
        "H₂ — mínimo LCOE\n"
        f"{h2_min_lcoe['P_peak_grid_opt_kw']:.2f} kW\n"
        f"{h2_min_lcoe['lcoe_usd_kwh']:.5f} USD/kWh"
    ),
    (
        h2_min_lcoe["P_peak_grid_opt_kw"],
        h2_min_lcoe["lcoe_usd_kwh"],
    ),
    xytext=(-110, -28),
    textcoords="offset points",
    fontsize=8.5,
)


ax.annotate(
    (
        "B1 — mínimo pico\n"
        f"{b1_min_peak['P_peak_grid_opt_kw']:.2f} kW\n"
        f"{b1_min_peak['lcoe_usd_kwh']:.5f} USD/kWh"
    ),
    (
        b1_min_peak["P_peak_grid_opt_kw"],
        b1_min_peak["lcoe_usd_kwh"],
    ),
    xytext=(10, 10),
    textcoords="offset points",
    fontsize=8.5,
)

ax.annotate(
    (
        "B1 — knee\n"
        f"{b1_knee['P_peak_grid_opt_kw']:.2f} kW\n"
        f"{b1_knee['lcoe_usd_kwh']:.5f} USD/kWh"
    ),
    (
        b1_knee["P_peak_grid_opt_kw"],
        b1_knee["lcoe_usd_kwh"],
    ),
    xytext=(-48, 16),
    textcoords="offset points",
    fontsize=8.5,
)

ax.annotate(
    (
        "B1 — mínimo LCOE\n"
        f"{b1_min_lcoe['P_peak_grid_opt_kw']:.2f} kW\n"
        f"{b1_min_lcoe['lcoe_usd_kwh']:.5f} USD/kWh"
    ),
    (
        b1_min_lcoe["P_peak_grid_opt_kw"],
        b1_min_lcoe["lcoe_usd_kwh"],
    ),
    xytext=(10, -20),
    textcoords="offset points",
    fontsize=8.5,
)


# ============================================================
# GLOBAL DOMINANCE MESSAGE
# ============================================================

ax.text(
    0.02,
    0.03,
    (
        "Fronteira global combinada: "
        "61 soluções não dominadas, todas H₂.\n"
        "As 34 soluções B1-T0 são dominadas no plano "
        "LCOE × pico de potência da rede."
    ),
    transform=ax.transAxes,
    fontsize=9,
    va="bottom",
    bbox=dict(
        boxstyle="round,pad=0.4",
        alpha=0.10,
    ),
)


# ============================================================
# AXES
# ============================================================

ax.set_xlabel(
    "Pico máximo de potência importada da rede (kW)"
)

ax.set_ylabel(
    "LCOE (USD/kWh)"
)

ax.set_title(
    "Caso-base 2026 — Fronteiras de Pareto consolidadas\n"
    "H₂ verde × Biometano–CHP"
)

ax.grid(
    alpha=0.25,
)

ax.legend(
    loc="upper right",
)

fig.tight_layout()


# ============================================================
# SAVE
# ============================================================

fig.savefig(
    OUT_PNG,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    OUT_PDF,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# SAVE FIGURE SOURCE TABLE
# ============================================================

h2_source = h2.copy()
h2_source["route"] = "H2"

b1_source = b1.copy()
b1_source["route"] = "B1_T0"

source = pd.concat(
    [
        h2_source,
        b1_source,
    ],
    ignore_index=True,
    sort=False,
)

source.to_csv(
    OUT_SOURCE,
    index=False,
)


# ============================================================
# REPORT
# ============================================================

print()
print("=== CHARACTERISTIC POINTS ===")

print()
print("H2 MIN PEAK")
print(
    h2_min_peak[
        [
            "P_peak_grid_opt_kw",
            "lcoe_usd_kwh",
        ]
    ].to_string()
)

print()
print("H2 KNEE")
print(
    h2_knee[
        [
            "P_peak_grid_opt_kw",
            "lcoe_usd_kwh",
        ]
    ].to_string()
)

print()
print("H2 MIN LCOE")
print(
    h2_min_lcoe[
        [
            "P_peak_grid_opt_kw",
            "lcoe_usd_kwh",
        ]
    ].to_string()
)


print()
print("B1 MIN PEAK")
print(
    b1_min_peak[
        [
            "P_peak_grid_opt_kw",
            "lcoe_usd_kwh",
        ]
    ].to_string()
)

print()
print("B1 KNEE")
print(
    b1_knee[
        [
            "P_peak_grid_opt_kw",
            "lcoe_usd_kwh",
        ]
    ].to_string()
)

print()
print("B1 MIN LCOE")
print(
    b1_min_lcoe[
        [
            "P_peak_grid_opt_kw",
            "lcoe_usd_kwh",
        ]
    ].to_string()
)


print()
print("=== FILES GENERATED ===")
print(OUT_PNG.relative_to(ROOT))
print(OUT_PDF.relative_to(ROOT))
print(OUT_SOURCE.relative_to(ROOT))