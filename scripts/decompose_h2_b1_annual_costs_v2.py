from pathlib import Path
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "results"
    / "paper"
    / "source_data"
    / "h2_b1_dominance_decomposition.csv"
)

OUT_DIR = (
    ROOT
    / "results"
    / "paper"
    / "source_data"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)


df = pd.read_csv(INPUT)


def find_col(frame, candidates):
    for c in candidates:
        if c in frame.columns:
            return c
    raise KeyError(
        "Nenhuma destas colunas foi encontrada: "
        + ", ".join(candidates)
    )


# ---------------------------------------------------------
# Identificar colunas H2 / B1
# ---------------------------------------------------------

h2_annualized_capex = find_col(
    df,
    [
        "h2_annualized_capex_usd",
        "H2_annualized_capex_usd",
        "annualized_capex_usd_h2",
    ],
)

b1_annualized_capex = find_col(
    df,
    [
        "b1_annualized_capex_usd",
        "B1_annualized_capex_usd",
        "annualized_capex_usd_b1",
    ],
)

h2_fixed = find_col(
    df,
    [
        "h2_fixed_opex_annual_usd",
        "H2_fixed_opex_annual_usd",
        "fixed_opex_annual_usd_h2",
    ],
)

b1_fixed = find_col(
    df,
    [
        "b1_fixed_opex_annual_usd",
        "B1_fixed_opex_annual_usd",
        "fixed_opex_annual_usd_b1",
    ],
)

h2_grid = find_col(
    df,
    [
        "h2_grid_opex_annual_usd",
        "H2_grid_opex_annual_usd",
        "grid_opex_annual_usd_h2",
    ],
)

b1_grid = find_col(
    df,
    [
        "b1_grid_opex_annual_usd",
        "B1_grid_opex_annual_usd",
        "grid_opex_annual_usd_b1",
    ],
)

h2_dispatch = find_col(
    df,
    [
        "h2_variable_dispatchable_opex_annual_usd",
        "h2_variable_h2_opex_annual_usd",
        "variable_dispatchable_opex_annual_usd_h2",
        "variable_h2_opex_annual_usd_h2",
    ],
)

b1_dispatch = find_col(
    df,
    [
        "b1_variable_dispatchable_opex_annual_usd",
        "b1_variable_biomethane_opex_annual_usd",
        "variable_dispatchable_opex_annual_usd_b1",
        "variable_biomethane_opex_annual_usd_b1",
    ],
)


def optional_col(frame, candidates):
    for c in candidates:
        if c in frame.columns:
            return frame[c].fillna(0.0)
    return pd.Series(0.0, index=frame.index)


h2_deg = optional_col(
    df,
    [
        "h2_degradation_opex_annual_usd",
        "degradation_opex_annual_usd_h2",
    ],
)

b1_deg = optional_col(
    df,
    [
        "b1_degradation_opex_annual_usd",
        "degradation_opex_annual_usd_b1",
    ],
)


# ---------------------------------------------------------
# Componentes economicamente ADITIVOS do LCOE
# ---------------------------------------------------------

df["h2_cost_annualized_capex"] = df[h2_annualized_capex]
df["b1_cost_annualized_capex"] = df[b1_annualized_capex]

df["h2_cost_fixed_opex"] = df[h2_fixed]
df["b1_cost_fixed_opex"] = df[b1_fixed]

df["h2_cost_grid"] = df[h2_grid]
df["b1_cost_grid"] = df[b1_grid]

df["h2_cost_dispatchable"] = df[h2_dispatch]
df["b1_cost_dispatchable"] = df[b1_dispatch]

df["h2_cost_degradation"] = h2_deg
df["b1_cost_degradation"] = b1_deg


# ---------------------------------------------------------
# Custo anual total correto
# ---------------------------------------------------------

components = [
    "annualized_capex",
    "fixed_opex",
    "grid",
    "dispatchable",
    "degradation",
]

df["h2_total_annual_cost_usd"] = sum(
    df[f"h2_cost_{c}"]
    for c in components
)

df["b1_total_annual_cost_usd"] = sum(
    df[f"b1_cost_{c}"]
    for c in components
)

df["delta_b1_minus_h2_total_usd"] = (
    df["b1_total_annual_cost_usd"]
    - df["h2_total_annual_cost_usd"]
)

df["h2_total_cost_advantage_percent"] = (
    df["delta_b1_minus_h2_total_usd"]
    / df["b1_total_annual_cost_usd"]
    * 100.0
)


# ---------------------------------------------------------
# Diferenças por componente
# positivo = B1 custa mais
# negativo = H2 custa mais
# ---------------------------------------------------------

for c in components:
    df[f"delta_b1_minus_h2_{c}_usd"] = (
        df[f"b1_cost_{c}"]
        - df[f"h2_cost_{c}"]
    )


# ---------------------------------------------------------
# Estatística dos componentes
# ---------------------------------------------------------

rows = []

labels = {
    "annualized_capex": "Annualized CAPEX",
    "fixed_opex": "Fixed OPEX",
    "grid": "Grid energy",
    "dispatchable": "Dispatchable/fuel",
    "degradation": "Degradation",
}

for c in components:
    rows.append(
        {
            "component": labels[c],
            "H2_mean_usd_per_year":
                df[f"h2_cost_{c}"].mean(),
            "B1_mean_usd_per_year":
                df[f"b1_cost_{c}"].mean(),
            "delta_B1_minus_H2_usd_per_year":
                df[f"delta_b1_minus_h2_{c}_usd"].mean(),
        }
    )

rows.append(
    {
        "component": "Total annual",
        "H2_mean_usd_per_year":
            df["h2_total_annual_cost_usd"].mean(),
        "B1_mean_usd_per_year":
            df["b1_total_annual_cost_usd"].mean(),
        "delta_B1_minus_H2_usd_per_year":
            df["delta_b1_minus_h2_total_usd"].mean(),
    }
)

summary = pd.DataFrame(rows)


# ---------------------------------------------------------
# Contribuição de cada componente para a diferença total
# ---------------------------------------------------------

mean_total_delta = (
    df["delta_b1_minus_h2_total_usd"].mean()
)

contribution_rows = []

for c in components:
    delta = df[
        f"delta_b1_minus_h2_{c}_usd"
    ].mean()

    contribution_rows.append(
        {
            "component": labels[c],
            "mean_delta_B1_minus_H2_usd_per_year": delta,
            "share_of_net_difference_percent":
                (
                    delta / mean_total_delta * 100.0
                    if mean_total_delta != 0
                    else np.nan
                ),
        }
    )

contribution = pd.DataFrame(contribution_rows)


# ---------------------------------------------------------
# Saídas
# ---------------------------------------------------------

pairs_file = (
    OUT_DIR
    / "h2_b1_annual_cost_decomposition_34_pairs_v2.csv"
)

summary_file = (
    OUT_DIR
    / "h2_b1_annual_cost_decomposition_summary_v2.csv"
)

contribution_file = (
    OUT_DIR
    / "h2_b1_annual_cost_difference_contributions_v2.csv"
)

df.to_csv(pairs_file, index=False)
summary.to_csv(summary_file, index=False)
contribution.to_csv(contribution_file, index=False)


# ---------------------------------------------------------
# Console
# ---------------------------------------------------------

print()
print("=== CORRECTED ANNUAL COST DECOMPOSITION ===")
print("pairs =", len(df))

print()
print("=== MEAN ANNUAL COMPONENTS ===")
print(summary.to_string(index=False))

print()
print("=== MEAN B1 - H2 DIFFERENCES ===")

for _, row in summary.iterrows():
    print(
        f"{row['component']:20s} "
        f"{row['delta_B1_minus_H2_usd_per_year']:12.2f} USD/year"
    )

print()
print("=== COMPONENT CONTRIBUTIONS TO NET DIFFERENCE ===")
print(contribution.to_string(index=False))

print()
print("=== H2 TOTAL ANNUAL COST ADVANTAGE (%) ===")
print(
    df["h2_total_cost_advantage_percent"].describe()
)

print()
print("=== OUTPUT FILES ===")
print(pairs_file.relative_to(ROOT))
print(summary_file.relative_to(ROOT))
print(contribution_file.relative_to(ROOT))