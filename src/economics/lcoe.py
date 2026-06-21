from __future__ import annotations

from typing import Any, Dict
import pandas as pd


# ------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------
def _require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def _sanitize(series: pd.Series, name: str) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.isna().any():
        raise ValueError(f"{name} has NaN")
    if (s < -1e-6).any():
        raise ValueError(f"{name} has negative values")
    return s.clip(lower=0)


# ------------------------------------------------------------
# FINANCE
# ------------------------------------------------------------
def capital_recovery_factor(wacc: float, n: int) -> float:
    if n <= 0:
        raise ValueError("Invalid horizon")

    if wacc == 0:
        return 1.0 / n

    return (wacc * (1 + wacc) * n) / ((1 + wacc) * n - 1)


# ------------------------------------------------------------
# CAPEX / OPEX
# ------------------------------------------------------------
def _bsv_total(cfg: Dict[str, float]) -> float:
    return (
        cfg["bsv_module_usd_per_kwh"]
        + cfg["bsv_repurpose_usd_per_kwh"]
        + cfg["bsv_integration_usd_per_kwh"]
    )


def compute_capex_total(cap: Dict[str, float], cfg: Dict[str, float]) -> float:
    return float(
        cap["pv_kw"] * cfg["pv_usd_per_kw"]
        + cap["bsv_kwh"] * _bsv_total(cfg)
        + cap["electrolyzer_kw"] * cfg["electrolyzer_usd_per_kw"]
        + cap["h2_tank_kg"] * cfg["h2_tank_usd_per_kg"]
        + cap["fuelcell_kw"] * cfg["fuelcell_usd_per_kw"]
    )


def compute_fixed_opex_annual(cap: Dict[str, float], eco: Dict[str, Any]) -> float:
    capex_cfg = eco["capex"]
    frac = eco["opex_fixed"]

    pv = cap["pv_kw"] * capex_cfg["pv_usd_per_kw"]
    bsv = cap["bsv_kwh"] * _bsv_total(capex_cfg)
    elz = cap["electrolyzer_kw"] * capex_cfg["electrolyzer_usd_per_kw"]
    fc = cap["fuelcell_kw"] * capex_cfg["fuelcell_usd_per_kw"]
    tank = cap["h2_tank_kg"] * capex_cfg["h2_tank_usd_per_kg"]

    return float(
        pv * frac["pv_fraction_of_capex_per_year"]
        + bsv * frac["bsv_fraction_of_capex_per_year"]
        + elz * frac["electrolyzer_fraction_of_capex_per_year"]
        + fc * frac["fuelcell_fraction_of_capex_per_year"]
        + tank * frac["h2_tank_fraction_of_capex_per_year"]
    )


# ------------------------------------------------------------
# GRID
# ------------------------------------------------------------
def _build_hour(df: pd.DataFrame) -> pd.Series:
    return df["hour"].astype(int) % 24


def compute_grid_opex_annual(df: pd.DataFrame, tariff: Dict[str, Any], dt: float) -> float:
    _require_columns(df, ["p_grid_kw", "hour"])

    grid = _sanitize(df["p_grid_kw"], "p_grid")
    hour = _build_hour(df)

    peak = (hour >= tariff["peak_window"]["start_hour"]) & (
        hour < tariff["peak_window"]["end_hour"]
    )

    price = pd.Series(tariff["offpeak_price_usd_kwh"], index=df.index)
    price.loc[peak] = tariff["peak_price_usd_kwh"]

    return float((grid * dt * price).sum())


def compute_grid_peak_opex_annual(df: pd.DataFrame, tariff: Dict[str, Any], dt: float) -> float:
    _require_columns(df, ["p_grid_kw", "hour"])

    grid = _sanitize(df["p_grid_kw"], "p_grid")
    hour = _build_hour(df)

    peak = (hour >= tariff["peak_window"]["start_hour"]) & (
        hour < tariff["peak_window"]["end_hour"]
    )

    return float((grid[peak] * dt * tariff["peak_price_usd_kwh"]).sum())


def compute_grid_peak_energy_annual(df: pd.DataFrame, tariff: Dict[str, Any], dt: float) -> float:
    _require_columns(df, ["p_grid_kw", "hour"])

    grid = _sanitize(df["p_grid_kw"], "p_grid")
    hour = _build_hour(df)

    peak = (hour >= tariff["peak_window"]["start_hour"]) & (
        hour < tariff["peak_window"]["end_hour"]
    )

    return float((grid[peak] * dt).sum())


def compute_total_grid_energy_annual(df: pd.DataFrame, dt: float) -> float:
    grid = _sanitize(df["p_grid_kw"], "p_grid")
    return float((grid * dt).sum())


# ------------------------------------------------------------
# ENERGY
# ------------------------------------------------------------
def compute_annual_energy_served(df: pd.DataFrame, dt: float) -> float:
    _require_columns(df, ["demand_kw"])
    demand = _sanitize(df["demand_kw"], "demand")
    return float((demand * dt).sum())


# ------------------------------------------------------------
# H2 OPEX
# ------------------------------------------------------------
def compute_variable_h2_opex_annual(
    df: pd.DataFrame, eco: Dict[str, Any], dt: float
) -> float:
    cfg = eco.get("opex_variable_h2", {})

    cost_elz = float(cfg.get("electrolyzer_usd_kwh", 0))
    cost_fc = float(cfg.get("fuelcell_usd_kwh", 0))

    total = 0.0

    if "p_elz_kw" in df:
        total += float((_sanitize(df["p_elz_kw"], "elz") * dt * cost_elz).sum())

    if "p_fc_kw" in df:
        total += float((_sanitize(df["p_fc_kw"], "fc") * dt * cost_fc).sum())

    return total


# ------------------------------------------------------------
# LCOE
# ------------------------------------------------------------
def compute_lcoe(
    capex: float,
    fixed_opex: float,
    grid_cost: float,
    energy: float,
    crf: float,
    variable_h2: float = 0.0,
    degradation: float = 0.0,
) -> float:

    if energy <= 0:
        raise ValueError("Energy must be > 0")

    total = (
        capex * crf
        + fixed_opex
        + grid_cost
        + variable_h2
        + degradation
    )

    return float(total / energy)


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def build_economics_summary(
    config: dict[str, Any],
    capacities: dict[str, float],
    dispatch_df: pd.DataFrame,
) -> dict[str, float]:

    eco = config["economics"]
    tariff = config["tariff"]
    dt = float(config["data"].get("timestep_hours", 1.0))

    crf = capital_recovery_factor(
        eco["wacc_real"], eco["analysis_horizon_years"]
    )

    capex = compute_capex_total(capacities, eco["capex"])
    fixed = compute_fixed_opex_annual(capacities, eco)

    grid = compute_grid_opex_annual(dispatch_df, tariff, dt)
    grid_peak = compute_grid_peak_opex_annual(dispatch_df, tariff, dt)
    grid_peak_energy = compute_grid_peak_energy_annual(dispatch_df, tariff, dt)
    grid_total_energy = compute_total_grid_energy_annual(dispatch_df, dt)

    energy = compute_annual_energy_served(dispatch_df, dt)

    h2 = compute_variable_h2_opex_annual(dispatch_df, eco, dt)

    lcoe = compute_lcoe(capex, fixed, grid, energy, crf, h2)

    return {
        "capex_total_usd": capex,
        "annualized_capex_usd": capex * crf,
        "fixed_opex_annual_usd": fixed,
        "grid_opex_annual_usd": grid,
        "grid_peak_opex_annual_usd": grid_peak,
        "grid_peak_energy_annual_kwh": grid_peak_energy,
        "grid_total_energy_annual_kwh": grid_total_energy,
        "energy_served_annual_kwh": energy,
        "variable_h2_opex_annual_usd": h2,
        "lcoe_usd_kwh": lcoe,
    }