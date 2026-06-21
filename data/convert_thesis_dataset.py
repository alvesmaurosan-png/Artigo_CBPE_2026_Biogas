from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# Paths robustos
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "fleet_demand_sp_thesis.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "fleet_demand_sp.csv"


def normalize_pv_factor(series: pd.Series) -> pd.Series:
    """
    Converte irradiância solar em pv_factor normalizado entre 0 e 1.
    Se a série já estiver entre 0 e 1, apenas limita os valores.
    """
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)

    s_min = float(s.min())
    s_max = float(s.max())

    if s_max <= 1.0 and s_min >= 0.0:
        return s.clip(lower=0.0, upper=1.0)

    if s_max == s_min:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)

    return ((s - s_min) / (s_max - s_min)).clip(lower=0.0, upper=1.0)


def build_hour_column(df: pd.DataFrame) -> pd.Series:
    """
    Extrai 'hour' do timestamp. Se não houver timestamp, tenta usar coluna hour existente.
    """
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        if ts.isna().any():
            raise ValueError("A coluna 'timestamp' existe, mas contém valores inválidos.")
        return ts.dt.hour.astype(int)

    if "hour" in df.columns:
        return pd.to_numeric(df["hour"], errors="coerce").fillna(0).astype(int)

    raise ValueError("O arquivo precisa ter 'timestamp' ou 'hour'.")


def build_demand_column(df: pd.DataFrame) -> pd.Series:
    """
    Converte demanda do dataset da tese para demand_kw.
    Prioridade:
      1) bus_demand
      2) demand_kw
    """
    if "bus_demand" in df.columns:
        demand = pd.to_numeric(df["bus_demand"], errors="coerce")
    elif "demand_kw" in df.columns:
        demand = pd.to_numeric(df["demand_kw"], errors="coerce")
    else:
        raise ValueError("O arquivo precisa ter 'bus_demand' ou 'demand_kw'.")

    demand = demand.fillna(0.0).clip(lower=0.0)
    return demand.astype(float)


def build_pv_factor_column(df: pd.DataFrame) -> pd.Series:
    """
    Converte solar_irradiance em pv_factor.
    Prioridade:
      1) solar_irradiance
      2) pv_factor
    """
    if "solar_irradiance" in df.columns:
        return normalize_pv_factor(df["solar_irradiance"]).astype(float)

    if "pv_factor" in df.columns:
        pv = pd.to_numeric(df["pv_factor"], errors="coerce").fillna(0.0)
        return pv.clip(lower=0.0, upper=1.0).astype(float)

    raise ValueError("O arquivo precisa ter 'solar_irradiance' ou 'pv_factor'.")


def convert_dataset(input_path: Path, output_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Arquivo de entrada não encontrado: {input_path}")

    df = pd.read_csv(input_path)

    out = pd.DataFrame({
        "hour": build_hour_column(df),
        "demand_kw": build_demand_column(df),
        "pv_factor": build_pv_factor_column(df),
    })

    # Ordenação opcional por timestamp, se existir
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        out = out.loc[ts.sort_values().index].reset_index(drop=True)

    # Sanidade mínima
    if len(out) == 0:
        raise ValueError("O dataset convertido ficou vazio.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    return out


def main() -> None:
    print("Convertendo dataset da tese para o formato do artigo...")
    print(f"Entrada: {INPUT_PATH}")
    print(f"Saída:   {OUTPUT_PATH}")

    out = convert_dataset(INPUT_PATH, OUTPUT_PATH)

    print("\nConversão concluída com sucesso.")
    print("\nPrimeiras linhas:")
    print(out.head())

    print("\nResumo:")
    print(out.describe())

    print(f"\nNúmero de linhas: {len(out)}")


if __name__ == "__main__":
    main()