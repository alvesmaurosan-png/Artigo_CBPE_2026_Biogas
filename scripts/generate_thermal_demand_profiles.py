from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)


N_HOURS = 8760


# ============================================================
# PERFIS DIÁRIOS ADIMENSIONAIS
#
# São perfis exploratórios para screening.
# Posteriormente devem ser substituídos por um perfil de
# demanda térmica documentado ou medido na garagem.
# ============================================================

# T1:
# demanda moderada, com maior uso pela manhã e fim do dia.
T1_WEIGHTS = np.array([
    0.15,  # 00
    0.10,  # 01
    0.10,  # 02
    0.10,  # 03
    0.15,  # 04
    0.40,  # 05
    0.80,  # 06
    1.00,  # 07
    0.90,  # 08
    0.55,  # 09
    0.40,  # 10
    0.35,  # 11
    0.35,  # 12
    0.35,  # 13
    0.40,  # 14
    0.50,  # 15
    0.70,  # 16
    0.90,  # 17
    1.00,  # 18
    1.00,  # 19
    0.90,  # 20
    0.70,  # 21
    0.45,  # 22
    0.25,  # 23
], dtype=float)


# T2:
# demanda elevada e mais distribuída ao longo do dia.
T2_WEIGHTS = np.array([
    0.30,  # 00
    0.25,  # 01
    0.25,  # 02
    0.25,  # 03
    0.30,  # 04
    0.55,  # 05
    0.85,  # 06
    1.00,  # 07
    1.00,  # 08
    0.85,  # 09
    0.75,  # 10
    0.70,  # 11
    0.70,  # 12
    0.70,  # 13
    0.75,  # 14
    0.80,  # 15
    0.90,  # 16
    1.00,  # 17
    1.00,  # 18
    1.00,  # 19
    1.00,  # 20
    0.90,  # 21
    0.70,  # 22
    0.45,  # 23
], dtype=float)


TARGETS = {
    "T1": {
        "annual_energy_kwh_th": 200000.0,
        "weights": T1_WEIGHTS,
    },
    "T2": {
        "annual_energy_kwh_th": 400000.0,
        "weights": T2_WEIGHTS,
    },
}


def build_profile(
    annual_energy_kwh_th: float,
    weights: np.ndarray,
) -> pd.DataFrame:

    if len(weights) != 24:
        raise ValueError("Daily thermal profile must contain 24 values.")

    if np.any(weights < 0):
        raise ValueError("Thermal profile weights cannot be negative.")

    t_global = np.arange(N_HOURS, dtype=int)

    local_hour = t_global % 24

    raw = weights[local_hour]

    # Scale exactly to requested annual thermal energy.
    scale = annual_energy_kwh_th / raw.sum()

    thermal_demand_kw = raw * scale

    df = pd.DataFrame({
        "t_global": t_global,
        "hour": t_global,
        "local_hour": local_hour,
        "thermal_demand_kw": thermal_demand_kw,
    })

    return df


for case, cfg in TARGETS.items():

    df = build_profile(
        annual_energy_kwh_th=cfg["annual_energy_kwh_th"],
        weights=cfg["weights"],
    )

    path = OUT_DIR / f"thermal_demand_{case}.csv"

    df.to_csv(path, index=False)

    annual_energy = df["thermal_demand_kw"].sum()

    print()
    print(f"=== {case} ===")
    print("rows =", len(df))
    print("annual energy kWh_th =", annual_energy)
    print(
        "mean thermal load kW =",
        df["thermal_demand_kw"].mean(),
    )
    print(
        "peak thermal load kW =",
        df["thermal_demand_kw"].max(),
    )
    print("saved =", path.relative_to(ROOT))