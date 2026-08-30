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



def compute_capex_total(
    cap: Dict[str, float],
    cfg: Dict[str, float],
) -> float:

    common = (
        cap["pv_kw"] * cfg["pv_usd_per_kw"]
        + cap["bsv_kwh"] * _bsv_total(cfg)
    )

    if (
        "electrolyzer_kw" in cap
        or "h2_tank_kg" in cap
        or "fuelcell_kw" in cap
    ):
        dispatchable = (
            cap.get("electrolyzer_kw", 0.0)
            * cfg.get("electrolyzer_usd_per_kw", 0.0)
            + cap.get("h2_tank_kg", 0.0)
            * cfg.get("h2_tank_usd_per_kg", 0.0)
            + cap.get("fuelcell_kw", 0.0)
            * cfg.get("fuelcell_usd_per_kw", 0.0)
        )

    elif (
        "biomethane_storage_nm3" in cap
        or "chp_kw" in cap
    ):
        dispatchable = (
            cap.get("biomethane_storage_nm3", 0.0)
            * cfg.get(
                "biomethane_storage_usd_per_nm3",
                0.0,
            )
            + cap.get("chp_kw", 0.0)
            * cfg.get("chp_usd_per_kw", 0.0)
        )

    else:
        dispatchable = 0.0

    return float(common + dispatchable)



def compute_fixed_opex_annual(
    cap: Dict[str, float],
    eco: Dict[str, Any],
) -> float:

    capex_cfg = eco["capex"]
    frac = eco["opex_fixed"]

    pv = (
        cap["pv_kw"]
        * capex_cfg["pv_usd_per_kw"]
    )

    bsv = (
        cap["bsv_kwh"]
        * _bsv_total(capex_cfg)
    )

    common = (
        pv
        * frac["pv_fraction_of_capex_per_year"]
        + bsv
        * frac["bsv_fraction_of_capex_per_year"]
    )

    if (
        "electrolyzer_kw" in cap
        or "h2_tank_kg" in cap
        or "fuelcell_kw" in cap
    ):
        elz = (
            cap.get("electrolyzer_kw", 0.0)
            * capex_cfg.get(
                "electrolyzer_usd_per_kw",
                0.0,
            )
        )

        fc = (
            cap.get("fuelcell_kw", 0.0)
            * capex_cfg.get(
                "fuelcell_usd_per_kw",
                0.0,
            )
        )

        tank = (
            cap.get("h2_tank_kg", 0.0)
            * capex_cfg.get(
                "h2_tank_usd_per_kg",
                0.0,
            )
        )

        dispatchable = (
            elz
            * frac.get(
                "electrolyzer_fraction_of_capex_per_year",
                0.0,
            )
            + fc
            * frac.get(
                "fuelcell_fraction_of_capex_per_year",
                0.0,
            )
            + tank
            * frac.get(
                "h2_tank_fraction_of_capex_per_year",
                0.0,
            )
        )

    elif (
        "biomethane_storage_nm3" in cap
        or "chp_kw" in cap
    ):
        storage = (
            cap.get("biomethane_storage_nm3", 0.0)
            * capex_cfg.get(
                "biomethane_storage_usd_per_nm3",
                0.0,
            )
        )

        chp = (
            cap.get("chp_kw", 0.0)
            * capex_cfg.get(
                "chp_usd_per_kw",
                0.0,
            )
        )

        dispatchable = (
            storage
            * frac.get(
                "biomethane_storage_fraction_of_capex_per_year",
                0.0,
            )
            + chp
            * frac.get(
                "chp_fraction_of_capex_per_year",
                0.0,
            )
        )

    else:
        dispatchable = 0.0

    return float(common + dispatchable)


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

def compute_variable_biomethane_opex_annual(
    df: pd.DataFrame,
    eco: Dict[str, Any],
    dt: float,
) -> dict[str, float]:

    cfg = eco.get(
        "opex_variable_biomethane",
        {},
    )

    biomethane_price = float(
        cfg.get(
            "biomethane_usd_per_nm3",
            0.0,
        )
    )

    chp_variable_cost = float(
        cfg.get(
            "chp_usd_per_kwh",
            0.0,
        )
    )

    biomethane_fuel = 0.0
    chp_variable = 0.0

    if "biomethane_use_nm3" in df:
        biomethane_fuel = float(
            (
                _sanitize(
                    df["biomethane_use_nm3"],
                    "biomethane_use",
                )
                * biomethane_price
            ).sum()
        )

    if "p_chp_kw" in df:
        chp_variable = float(
            (
                _sanitize(
                    df["p_chp_kw"],
                    "p_chp",
                )
                * dt
                * chp_variable_cost
            ).sum()
        )

    return {
        "biomethane_fuel_opex_annual_usd":
            biomethane_fuel,
        "chp_variable_opex_annual_usd":
            chp_variable,
        "variable_biomethane_opex_annual_usd":
            biomethane_fuel + chp_variable,
    }


def compute_lcoe(
    capex: float,
    fixed_opex: float,
    grid_cost: float,
    energy: float,
    crf: float,
    variable_dispatchable_opex: float = 0.0,
    degradation: float = 0.0,
) -> float:

    if energy <= 0:
        raise ValueError("Energy must be > 0")

    total = (
        capex * crf
        + fixed_opex
        + grid_cost
        + variable_dispatchable_opex
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

    route = str(
        config.get("system", {}).get("route", "hydrogen")
    ).strip().lower()

    if route not in {"hydrogen", "biomethane"}:
        raise ValueError(
            f"Unsupported system.route: {route!r}. "
            "Expected 'hydrogen' or 'biomethane'."
        )

    crf = capital_recovery_factor(
        eco["wacc_real"],
        eco["analysis_horizon_years"],
    )

    # ---------------------------------------------------------
    # CAPEX / OPEX fixo
    # ---------------------------------------------------------
    capex = compute_capex_total(
        capacities,
        eco["capex"],
    )

    fixed = compute_fixed_opex_annual(
        capacities,
        eco,
    )

    # ---------------------------------------------------------
    # Rede
    # ---------------------------------------------------------
    grid = compute_grid_opex_annual(
        dispatch_df,
        tariff,
        dt,
    )

    grid_peak = compute_grid_peak_opex_annual(
        dispatch_df,
        tariff,
        dt,
    )

    grid_peak_energy = compute_grid_peak_energy_annual(
        dispatch_df,
        tariff,
        dt,
    )

    grid_total_energy = compute_total_grid_energy_annual(
        dispatch_df,
        dt,
    )

    # ---------------------------------------------------------
    # Energia atendida
    # ---------------------------------------------------------
    energy = compute_annual_energy_served(
        dispatch_df,
        dt,
    )

    # ---------------------------------------------------------
    # OPEX variavel da tecnologia despachavel
    # ---------------------------------------------------------
    variable_h2 = 0.0

    biomethane_fuel = 0.0
    chp_variable = 0.0
    variable_biomethane = 0.0

    if route == "hydrogen":
        variable_h2 = compute_variable_h2_opex_annual(
            dispatch_df,
            eco,
            dt,
        )

        variable_dispatchable = variable_h2

    elif route == "biomethane":
        bm_opex = compute_variable_biomethane_opex_annual(
            dispatch_df,
            eco,
            dt,
        )

        biomethane_fuel = float(
            bm_opex["biomethane_fuel_opex_annual_usd"]
        )

        chp_variable = float(
            bm_opex["chp_variable_opex_annual_usd"]
        )

        variable_biomethane = float(
            bm_opex["variable_biomethane_opex_annual_usd"]
        )

        variable_dispatchable = variable_biomethane

    # ---------------------------------------------------------
    # LCOE
    #
    # Mesma definicao economica para ambas as rotas:
    #
    # annualized CAPEX
    # + fixed OPEX
    # + grid OPEX
    # + variable dispatchable OPEX
    # --------------------------------
    # annual energy served
    # ---------------------------------------------------------
    lcoe = compute_lcoe(
        capex,
        fixed,
        grid,
        energy,
        crf,
        variable_dispatchable,
    )

    return {
        "capex_total_usd": capex,
        "annualized_capex_usd": capex * crf,
        "fixed_opex_annual_usd": fixed,

        "grid_opex_annual_usd": grid,
        "grid_peak_opex_annual_usd": grid_peak,
        "grid_peak_energy_annual_kwh": grid_peak_energy,
        "grid_total_energy_annual_kwh": grid_total_energy,

        "energy_served_annual_kwh": energy,

        # Legacy H2 field preserved for backward compatibility.
        "variable_h2_opex_annual_usd": variable_h2,

        # Biomethane-specific audit fields.
        "biomethane_fuel_opex_annual_usd": biomethane_fuel,
        "chp_variable_opex_annual_usd": chp_variable,
        "variable_biomethane_opex_annual_usd": variable_biomethane,

        # Route-neutral field for new downstream code.
        "variable_dispatchable_opex_annual_usd": variable_dispatchable,

        "lcoe_usd_kwh": lcoe,
    }