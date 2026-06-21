from __future__ import annotations

from typing import Any
import pandas as pd


# ============================================================
# VALIDAÇÕES BÁSICAS
# ============================================================

def _require_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"df_dispatch is missing required columns: {missing}")


def _sanitize_nonnegative_series(
    series: pd.Series,
    label: str,
    tolerance: float = 1e-6,
) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")

    if s.isna().any():
        raise ValueError(f"{label} contains non-numeric or NaN values")

    min_val = float(s.min())
    if min_val < -tolerance:
        raise ValueError(f"{label} contains materially negative values (min={min_val})")

    return s.clip(lower=0.0)


def _build_hour_of_day(df: pd.DataFrame) -> pd.Series:
    _require_columns(df, ["hour"])

    hour_series = pd.to_numeric(df["hour"], errors="coerce")

    if hour_series.isna().any():
        raise ValueError("hour contains non-numeric or NaN values")

    return hour_series.astype(int) % 24


# ============================================================
# MÉTRICAS DE ENERGIA NA PONTA
# ============================================================

def compute_peak_energy_kwh(
    df_dispatch: pd.DataFrame,
    start_hour: int,
    end_hour: int,
    col_kw: str,
    dt_hours: float = 1.0,
) -> float:
    """
    Energia total na janela de ponta (kWh)
    """
    if dt_hours <= 0:
        raise ValueError("dt_hours must be > 0")

    _require_columns(df_dispatch, ["hour", col_kw])

    hour_of_day = _build_hour_of_day(df_dispatch)
    power_series = _sanitize_nonnegative_series(df_dispatch[col_kw], col_kw)

    peak_mask = (hour_of_day >= int(start_hour)) & (hour_of_day < int(end_hour))

    return float((power_series.loc[peak_mask] * float(dt_hours)).sum())


# ============================================================
# DEPENDÊNCIA DE REDE NA PONTA
# ============================================================

def compute_peak_grid_dependency_ratio(
    e_peak_opt: float,
    e_peak_ref: float,
    tolerance: float = 1e-6,
) -> float:
    """
    Calcula:
        ratio = E_opt / E_ref

    Agora recebe diretamente os valores → evita recalcular e inconsistência.
    """

    if e_peak_ref < 0:
        raise ValueError("E_grid_peak_ref_kwh cannot be negative")

    if e_peak_ref <= tolerance:
        # ERRO ESTRUTURAL → não mascarar
        raise ValueError(
            f"E_grid_peak_ref_kwh too small ({e_peak_ref}). "
            "Reference scenario inválido."
        )

    return float(e_peak_opt / e_peak_ref)


def compute_r_peak(
    ratio: float,
) -> float:
    """
    Redução percentual de energia na ponta
    """
    return float((1.0 - ratio) * 100.0)


# ============================================================
# MÉTRICAS ECONÔMICAS
# ============================================================

def compute_r_cost(
    lcoe_usd_kwh: float,
    exchange_rate_brl_per_usd: float,
    reference_cost_brl_kwh: float,
) -> float:

    if reference_cost_brl_kwh <= 0:
        raise ValueError("reference_cost_brl_kwh must be > 0")

    if exchange_rate_brl_per_usd <= 0:
        raise ValueError("exchange_rate_brl_per_usd must be > 0")

    if lcoe_usd_kwh < 0:
        raise ValueError("lcoe_usd_kwh cannot be negative")

    lcoe_brl_kwh = lcoe_usd_kwh * exchange_rate_brl_per_usd

    return float((1.0 - (lcoe_brl_kwh / reference_cost_brl_kwh)) * 100.0)


# ============================================================
# FUNÇÃO PRINCIPAL
# ============================================================

def build_peak_metrics_summary(
    df_dispatch_opt: pd.DataFrame,
    df_dispatch_ref: pd.DataFrame,
    tariff_cfg: dict[str, Any],
    economics_opt: dict[str, Any],
    exchange_rate_brl_per_usd: float | None = None,
    reference_cost_brl_kwh: float | None = None,
    dt_hours: float = 1.0,
) -> dict[str, float]:

    if "peak_window" not in tariff_cfg:
        raise ValueError("tariff_cfg must contain 'peak_window'")

    start_hour = int(tariff_cfg["peak_window"]["start_hour"])
    end_hour = int(tariff_cfg["peak_window"]["end_hour"])

    # --------------------------------------------------------
    # Energia ponta
    # --------------------------------------------------------
    e_peak_opt = compute_peak_energy_kwh(
        df_dispatch_opt, start_hour, end_hour, "p_grid_kw", dt_hours
    )

    e_peak_ref = compute_peak_energy_kwh(
        df_dispatch_ref, start_hour, end_hour, "p_grid_kw", dt_hours
    )

    # --------------------------------------------------------
    # Dependência
    # --------------------------------------------------------
    ratio = compute_peak_grid_dependency_ratio(
        e_peak_opt=e_peak_opt,
        e_peak_ref=e_peak_ref,
    )

    peak_percent = 100.0 * ratio
    r_peak = compute_r_peak(ratio)

    metrics: dict[str, float] = {
        "E_grid_peak_opt_kwh": float(e_peak_opt),
        "E_grid_peak_ref_kwh": float(e_peak_ref),
        "peak_grid_dependency_ratio": float(ratio),
        "peak_grid_dependency_percent": float(peak_percent),
        "R_peak_percent": float(r_peak),
    }

    # --------------------------------------------------------
    # LCOE
    # --------------------------------------------------------
    if "lcoe_usd_kwh" in economics_opt:
        lcoe_usd = float(economics_opt["lcoe_usd_kwh"])
        metrics["lcoe_usd_kwh"] = lcoe_usd

        if exchange_rate_brl_per_usd is not None:
            lcoe_brl = lcoe_usd * float(exchange_rate_brl_per_usd)
            metrics["LCOE_opt_BRL_kWh"] = float(lcoe_brl)

            if reference_cost_brl_kwh is not None:
                metrics["R_cost_percent"] = compute_r_cost(
                    lcoe_usd_kwh=lcoe_usd,
                    exchange_rate_brl_per_usd=exchange_rate_brl_per_usd,
                    reference_cost_brl_kwh=reference_cost_brl_kwh,
                )

    return metrics