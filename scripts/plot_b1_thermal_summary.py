from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "results" / "paper" / "source_data"
FIG_DIR = ROOT / "results" / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


THERMAL_SUMMARY = (
    DATA_DIR
    / "b1_thermal_dispatch_T0_T1_T2_summary.csv"
)

TMAX_FILE = (
    DATA_DIR
    / "b1_thermal_TMAX.csv"
)

BREAK_EVEN_FILE = (
    DATA_DIR
    / "b1_thermal_break_even_vs_h2.csv"
)


thermal = pd.read_csv(THERMAL_SUMMARY)
tmax = pd.read_csv(TMAX_FILE)
break_even = pd.read_csv(BREAK_EVEN_FILE)


# ============================================================
# FIGURE 1
# T0 × T1 × T2 × TMAX
# ============================================================

cases = ["T0", "T1", "T2", "TMAX"]

mean_lcoe_reduction = []
n_nondominated = []
mean_heat_useful_mwh = []
mean_heat_credit = []


for case in ["T0", "T1", "T2"]:

    s = thermal[
        thermal["thermal_case"] == case
    ]

    mean_lcoe_reduction.append(
        s["lcoe_reduction_percent"].mean()
    )

    n_nondominated.append(
        int((~s["dominated_by_h2"]).sum())
    )

    mean_heat_useful_mwh.append(
        s["heat_useful_kwh_th"].mean()
        / 1000.0
    )

    mean_heat_credit.append(
        s["heat_credit_usd_per_year"].mean()
    )


# TMAX
mean_lcoe_reduction.append(
    tmax["lcoe_reduction_percent"].mean()
)

n_nondominated.append(
    int((~tmax["dominated_by_h2"]).sum())
)

mean_heat_useful_mwh.append(
    tmax["heat_useful_kwh_th"].mean()
    / 1000.0
)

mean_heat_credit.append(
    tmax["heat_credit_usd_per_year"].mean()
)


x = np.arange(len(cases))


fig, ax1 = plt.subplots(figsize=(10, 6))

bars = ax1.bar(
    x,
    mean_lcoe_reduction,
    width=0.58,
)

ax1.set_ylabel(
    "Redução média do LCOE (%)"
)

ax1.set_xticks(x)

ax1.set_xticklabels([
    "T0\nsem calor",
    "T1\n200 MWh$_{th}$/ano",
    "T2\n400 MWh$_{th}$/ano",
    "TMAX\n100% do calor",
])

ax1.set_title(
    "Impacto da valorização térmica sobre a rota B1"
)

ax1.grid(
    axis="y",
    alpha=0.25,
)


for bar, value in zip(
    bars,
    mean_lcoe_reduction,
):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.3f}%",
        ha="center",
        va="bottom",
        fontsize=9,
    )


ax2 = ax1.twinx()

ax2.plot(
    x,
    n_nondominated,
    marker="o",
    linewidth=1.8,
)

ax2.set_ylabel(
    "Número de soluções B1 não dominadas"
)

ax2.set_ylim(
    0,
    max(2, max(n_nondominated) + 1),
)

for xi, value in zip(
    x,
    n_nondominated,
):
    ax2.text(
        xi,
        value + 0.05,
        str(value),
        ha="center",
        va="bottom",
        fontsize=9,
    )


fig.tight_layout()


fig.savefig(
    FIG_DIR
    / "thermal_T0_T1_T2_TMAX_lcoe_dominance.png",
    dpi=300,
)

fig.savefig(
    FIG_DIR
    / "thermal_T0_T1_T2_TMAX_lcoe_dominance.pdf",
)

plt.close(fig)


# ============================================================
# OPTIONAL TABLE SOURCE FOR FIGURE 1
# ============================================================

fig1_table = pd.DataFrame({
    "case": cases,
    "mean_heat_useful_MWh_th_year":
        mean_heat_useful_mwh,
    "mean_heat_credit_USD_year":
        mean_heat_credit,
    "mean_lcoe_reduction_percent":
        mean_lcoe_reduction,
    "B1_nondominated_solutions":
        n_nondominated,
})

fig1_table.to_csv(
    DATA_DIR
    / "thermal_T0_T1_T2_TMAX_figure_source.csv",
    index=False,
)


# ============================================================
# FIGURE 2
# POTENCIAL TÉRMICO × BREAK-EVEN
#
# Compare:
# available heat credit under TMAX
# versus required heat credit for break-even with H2
# ============================================================

available = (
    tmax[
        [
            "source_run",
            "source_solution_id",
            "heat_credit_usd_per_year",
        ]
    ]
    .rename(
        columns={
            "heat_credit_usd_per_year":
                "available_heat_credit_usd_year"
        }
    )
)


be = break_even[
    [
        "b1_source_run",
        "b1_source_solution_id",
        "annual_heat_credit_break_even_usd",
    ]
].copy()


merged = available.merge(
    be,
    left_on=[
        "source_run",
        "source_solution_id",
    ],
    right_on=[
        "b1_source_run",
        "b1_source_solution_id",
    ],
    how="inner",
)


merged["margin_usd_year"] = (
    merged[
        "available_heat_credit_usd_year"
    ]
    -
    merged[
        "annual_heat_credit_break_even_usd"
    ]
)


merged = merged.sort_values(
    "annual_heat_credit_break_even_usd"
).reset_index(drop=True)


x = np.arange(1, len(merged) + 1)


fig, ax = plt.subplots(figsize=(10, 6))


ax.scatter(
    x,
    merged[
        "available_heat_credit_usd_year"
    ],
    s=28,
    label="Crédito térmico disponível — TMAX",
)

ax.scatter(
    x,
    merged[
        "annual_heat_credit_break_even_usd"
    ],
    s=28,
    marker="x",
    label="Crédito térmico necessário — break-even",
)


ax.set_yscale("log")


ax.set_xlabel(
    "Soluções B1 ordenadas por crédito térmico de break-even"
)

ax.set_ylabel(
    "Crédito térmico anual (USD/ano)"
)

ax.set_title(
    "Potencial térmico disponível × necessidade de break-even"
)

ax.grid(
    alpha=0.25,
    which="both",
)

ax.legend()


# Highlight the single solution where available >= required

survivors = merged[
    merged[
        "available_heat_credit_usd_year"
    ]
    >=
    merged[
        "annual_heat_credit_break_even_usd"
    ]
]


for idx, row in survivors.iterrows():

    xi = idx + 1

    ax.annotate(
        (
            f"{row['source_run']} / "
            f"{int(row['source_solution_id'])}"
        ),
        (
            xi,
            row[
                "available_heat_credit_usd_year"
            ],
        ),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=8,
    )


fig.tight_layout()


fig.savefig(
    FIG_DIR
    / "thermal_potential_vs_break_even.png",
    dpi=300,
)

fig.savefig(
    FIG_DIR
    / "thermal_potential_vs_break_even.pdf",
)

plt.close(fig)


# ============================================================
# SAVE SOURCE DATA
# ============================================================

merged.to_csv(
    DATA_DIR
    / "thermal_potential_vs_break_even_figure_source.csv",
    index=False,
)


# ============================================================
# REPORT
# ============================================================

print()
print("=== FIGURE 1 ===")
print(fig1_table.to_string(index=False))

print()
print("=== FIGURE 2 ===")

print(
    "solutions compared =",
    len(merged),
)

print(
    "available credit mean USD/year =",
    merged[
        "available_heat_credit_usd_year"
    ].mean(),
)

print(
    "break-even credit mean USD/year =",
    merged[
        "annual_heat_credit_break_even_usd"
    ].mean(),
)

print(
    "solutions with available >= break-even =",
    int(
        (
            merged[
                "available_heat_credit_usd_year"
            ]
            >=
            merged[
                "annual_heat_credit_break_even_usd"
            ]
        ).sum()
    ),
    "/",
    len(merged),
)

print()
print("=== FILES GENERATED ===")

for path in [
    FIG_DIR
    / "thermal_T0_T1_T2_TMAX_lcoe_dominance.png",

    FIG_DIR
    / "thermal_T0_T1_T2_TMAX_lcoe_dominance.pdf",

    FIG_DIR
    / "thermal_potential_vs_break_even.png",

    FIG_DIR
    / "thermal_potential_vs_break_even.pdf",
]:
    print(path.relative_to(ROOT))