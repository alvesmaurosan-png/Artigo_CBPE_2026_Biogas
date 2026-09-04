from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "results" / "paper" / "source_data"
FIG_DIR = ROOT / "results" / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

summary_file = DATA_DIR / "h2_b1_annual_cost_decomposition_summary_v3.csv"
detail_file = DATA_DIR / "h2_b1_annual_cost_decomposition_34_pairs_v3.csv"

summary = pd.read_csv(summary_file)
detail = pd.read_csv(detail_file)

# ============================================================
# FIGURE 1 — WATERFALL ECONÔMICO
# delta = B1 - H2
# positivo = vantagem H2
# negativo = vantagem B1
# ============================================================

components = [
    "Annualized CAPEX",
    "Fixed OPEX",
    "Grid energy",
    "Dispatchable/fuel",
    "Degradation",
]

vals = []

for comp in components:
    row = summary.loc[
        summary["component"] == comp
    ].iloc[0]

    vals.append(
        row["delta_B1_minus_H2_mean_usd_per_year"]
    )

total = summary.loc[
    summary["component"] == "Total annual"
].iloc[0]["delta_B1_minus_H2_mean_usd_per_year"]

cum = [0]

for v in vals:
    cum.append(cum[-1] + v)

starts = cum[:-1]

fig, ax = plt.subplots(figsize=(10, 6))

for i, (label, value, start) in enumerate(
    zip(components, vals, starts)
):
    ax.bar(
        i,
        value,
        bottom=start,
    )

    y = start + value

    ax.text(
        i,
        y,
        f"{value:,.0f}",
        ha="center",
        va="bottom" if value >= 0 else "top",
        fontsize=9,
    )

ax.bar(
    len(components),
    total,
)

ax.text(
    len(components),
    total,
    f"{total:,.0f}",
    ha="center",
    va="bottom",
    fontsize=10,
    fontweight="bold",
)

labels = [
    "CAPEX\nanualizado",
    "OPEX\nfixo",
    "Energia\nrede",
    "Despachável/\ncombustível",
    "Degradação",
    "Total\nlíquido",
]

ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels)

ax.axhline(0, linewidth=1)

ax.set_ylabel("Diferença média B1 − H₂ (USD/ano)")
ax.set_title(
    "Decomposição da vantagem econômica média do H₂\n"
    "34 pares dominador–dominado"
)

ax.grid(axis="y", alpha=0.25)

fig.tight_layout()

fig.savefig(
    FIG_DIR / "waterfall_h2_vs_b1_annual_cost.png",
    dpi=300,
)

fig.savefig(
    FIG_DIR / "waterfall_h2_vs_b1_annual_cost.pdf",
)

plt.close(fig)


# ============================================================
# FIGURE 2 — DIFERENÇAS FÍSICAS MÉDIAS
# ============================================================

mean_grid_b1 = detail["b1_E_grid_kwh"].mean()
mean_grid_h2 = detail["h2_E_grid_kwh"].mean()

mean_dep_b1 = (
    detail["b1_grid_dependency_ratio"].mean()
)
mean_dep_h2 = (
    detail["h2_grid_dependency_ratio"].mean()
)

mean_peak_b1 = detail["b1_peak_kw"].mean()
mean_peak_h2 = detail["h2_peak_kw"].mean()


metrics = [
    "Energia importada\n(MWh/ano)",
    "Dependência da rede\n(%)",
    "Pico da rede\n(kW)",
]

b1_values = [
    mean_grid_b1 / 1000.0,
    mean_dep_b1 * 100.0,
    mean_peak_b1,
]

h2_values = [
    mean_grid_h2 / 1000.0,
    mean_dep_h2 * 100.0,
    mean_peak_h2,
]

x = range(len(metrics))
width = 0.36

fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(
    [i - width / 2 for i in x],
    b1_values,
    width=width,
    label="B1 — Biometano–CHP",
)

ax.bar(
    [i + width / 2 for i in x],
    h2_values,
    width=width,
    label="H₂",
)

for i, (b1v, h2v) in enumerate(
    zip(b1_values, h2_values)
):
    ax.text(
        i - width / 2,
        b1v,
        f"{b1v:,.1f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

    ax.text(
        i + width / 2,
        h2v,
        f"{h2v:,.1f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

ax.set_xticks(list(x))
ax.set_xticklabels(metrics)

ax.set_title(
    "Diferenças físicas médias — H₂ × Biometano–CHP\n"
    "34 pares dominador–dominado"
)

ax.legend()
ax.grid(axis="y", alpha=0.25)

fig.tight_layout()

fig.savefig(
    FIG_DIR / "physical_differences_h2_vs_b1.png",
    dpi=300,
)

fig.savefig(
    FIG_DIR / "physical_differences_h2_vs_b1.pdf",
)

plt.close(fig)


print()
print("=== FIGURES GENERATED ===")

for p in [
    FIG_DIR / "waterfall_h2_vs_b1_annual_cost.png",
    FIG_DIR / "waterfall_h2_vs_b1_annual_cost.pdf",
    FIG_DIR / "physical_differences_h2_vs_b1.png",
    FIG_DIR / "physical_differences_h2_vs_b1.pdf",
]:
    print(p.relative_to(ROOT))