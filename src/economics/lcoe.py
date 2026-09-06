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

    return (wacc * (1 + wacc) ** n) / ((1 + wacc) ** n - 1)


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


# ------------------------------------------------------------
# BIOGAS B2 ECONOMICS
# ------------------------------------------------------------
def compute_biogas_capex_total(
    capacities: Dict[str, float],
    config: Dict[str, Any],
) -> dict[str, float]:
    eco = config["economics"]
    capex_cfg = eco["capex"]
    bg_cfg = config.get("technology", {}).get("biogas", {})

    substrate_t_day = float(bg_cfg.get("substrate_t_day", 0.0))
    if substrate_t_day <= 0.0:
        raise ValueError(
            "technology.biogas.substrate_t_day "
            "must be > 0 for route='biogas'."
        )

    substrate_t_year = substrate_t_day * 365.0

    pv_capex = (
        float(capacities["pv_kw"])
        * float(capex_cfg["pv_usd_per_kw"])
    )

    bsv_capex = (
        float(capacities["bsv_kwh"])
        * _bsv_total(capex_cfg)
    )

    ad_capex = (
        substrate_t_year
        * float(
            capex_cfg.get(
                "anaerobic_digestion_usd_per_t_year_capacity",
                0.0,
            )
        )
    )

    storage_capex = (
        float(capacities.get("biogas_storage_nm3", 0.0))
        * float(
            capex_cfg.get(
                "biogas_storage_usd_per_m3",
                0.0,
            )
        )
    )

    chp_capex = (
        float(capacities.get("chp_kw", 0.0))
        * float(
            capex_cfg.get(
                "chp_biogas_usd_per_kw",
                0.0,
            )
        )
    )

    total = (
        pv_capex
        + bsv_capex
        + ad_capex
        + storage_capex
        + chp_capex
    )

    return {
        "pv_capex_usd": float(pv_capex),
        "bsv_capex_usd": float(bsv_capex),
        "anaerobic_digestion_capex_usd": float(ad_capex),
        "biogas_storage_capex_usd": float(storage_capex),
        "chp_biogas_capex_usd": float(chp_capex),
        "substrate_capacity_t_year": float(substrate_t_year),
        "capex_total_usd": float(total),
    }


def compute_biogas_fixed_opex_annual(
    capacities: Dict[str, float],
    config: Dict[str, Any],
) -> dict[str, float]:
    eco = config["economics"]
    capex_cfg = eco["capex"]
    frac = eco["opex_fixed"]

    pv_capex = (
        float(capacities["pv_kw"])
        * float(capex_cfg["pv_usd_per_kw"])
    )

    bsv_capex = (
        float(capacities["bsv_kwh"])
        * _bsv_total(capex_cfg)
    )

    chp_capex = (
        float(capacities.get("chp_kw", 0.0))
        * float(
            capex_cfg.get(
                "chp_biogas_usd_per_kw",
                0.0,
            )
        )
    )

    pv_fixed = (
        pv_capex
        * float(frac.get("pv_fraction_of_capex_per_year", 0.0))
    )

    bsv_fixed = (
        bsv_capex
        * float(frac.get("bsv_fraction_of_capex_per_year", 0.0))
    )

    chp_fixed = (
        chp_capex
        * float(
            frac.get(
                "chp_biogas_fraction_of_capex_per_year",
                0.0,
            )
        )
    )

    total = pv_fixed + bsv_fixed + chp_fixed

    return {
        "pv_fixed_opex_annual_usd": float(pv_fixed),
        "bsv_fixed_opex_annual_usd": float(bsv_fixed),
        "chp_fixed_opex_annual_usd": float(chp_fixed),
        "fixed_opex_annual_usd": float(total),
    }


def compute_variable_biogas_opex_annual(
    df: pd.DataFrame,
    config: Dict[str, Any],
    dt: float,
) -> dict[str, Any]:
    eco = config["economics"]
    cfg = eco.get("opex_variable_biogas", {})
    bg_cfg = config.get("technology", {}).get("biogas", {})

    substrate_t_day = float(bg_cfg.get("substrate_t_day", 0.0))
    if substrate_t_day <= 0.0:
        raise ValueError(
            "technology.biogas.substrate_t_day "
            "must be > 0 for route='biogas'."
        )

    substrate_t_year = substrate_t_day * 365.0

    ad_cost_per_t = float(
        cfg.get("anaerobic_digestion_usd_per_t_substrate", 0.0)
    )
    ad_opex = substrate_t_year * ad_cost_per_t

    chp_cost_gross = float(
        cfg.get("chp_usd_per_kwh_gross", 0.0)
    )

    if "p_chp_gross_equiv_kw" in df.columns:
        chp_gross_energy = float(
            (
                _sanitize(
                    df["p_chp_gross_equiv_kw"],
                    "p_chp_gross_equiv",
                )
                * dt
            ).sum()
        )
    elif "p_chp_net_kw" in df.columns:
        parasitic_fraction = float(
            bg_cfg.get("parasitic_fraction", 0.0)
        )

        if not 0.0 <= parasitic_fraction < 1.0:
            raise ValueError(
                "Invalid biogas parasitic_fraction."
            )

        chp_net_energy = float(
            (
                _sanitize(
                    df["p_chp_net_kw"],
                    "p_chp_net",
                )
                * dt
            ).sum()
        )
        chp_gross_energy = (
            chp_net_energy
            / (1.0 - parasitic_fraction)
        )
    else:
        chp_gross_energy = 0.0

    chp_variable_opex = (
        chp_gross_energy
        * chp_cost_gross
    )

    substrate_cost = (
        substrate_t_year
        * float(cfg.get("substrate_usd_per_t", 0.0))
    )

    gate_fee_credit = (
        substrate_t_year
        * float(cfg.get("gate_fee_usd_per_t", 0.0))
    )

    if "substrate_logistics_usd_per_t" in cfg:
        logistics_usd_per_t = float(
            cfg["substrate_logistics_usd_per_t"]
        )
        logistics_method = "explicit_usd_per_t"
    else:
        logistics_usd_per_t = (
            float(cfg.get("transport_usd_per_t_km", 0.0))
            * float(cfg.get("transport_distance_km", 0.0))
        )
        logistics_method = "legacy_usd_per_t_km"

    logistics_opex = (
        substrate_t_year
        * logistics_usd_per_t
    )

    total = (
        ad_opex
        + chp_variable_opex
        + substrate_cost
        + logistics_opex
        - gate_fee_credit
    )

    return {
        "substrate_t_year": float(substrate_t_year),
        "anaerobic_digestion_opex_annual_usd": float(ad_opex),
        "chp_gross_energy_annual_kwh": float(chp_gross_energy),
        "chp_variable_opex_annual_usd": float(chp_variable_opex),
        "substrate_purchase_opex_annual_usd": float(substrate_cost),
        "substrate_logistics_usd_per_t": float(logistics_usd_per_t),
        "substrate_logistics_opex_annual_usd": float(logistics_opex),
        "gate_fee_credit_annual_usd": float(gate_fee_credit),
        "logistics_method": logistics_method,
        "variable_biogas_opex_annual_usd": float(total),
    }


def compute_annual_demand_charge(
    df: pd.DataFrame,
    tariff: Dict[str, Any],
) -> dict[str, Any]:
    demand_rate = float(
        tariff.get(
            "demand_charge_usd_kw_month",
            0.0,
        )
    )

    if not bool(
        tariff.get(
            "include_demand_charge",
            False,
        )
    ):
        return {
            "annual_demand_charge_usd": 0.0,
            "monthly_peak_kw": {},
        }

    _require_columns(df, ["p_grid_kw"])
    grid = _sanitize(df["p_grid_kw"], "p_grid")

    if "month" in df.columns:
        month = pd.to_numeric(
            df["month"],
            errors="coerce",
        )
        if month.isna().any():
            raise ValueError("month has NaN")
        month = month.astype(int)
    else:
        if len(df) != 8760:
            raise ValueError(
                "Demand-charge month reconstruction "
                "requires either a 'month' column or "
                "an 8760-hour annual dataframe."
            )

        month_days = [
            31, 28, 31, 30, 31, 30,
            31, 31, 30, 31, 30, 31,
        ]

        month_values: list[int] = []
        for month_number, days in enumerate(
            month_days,
            start=1,
        ):
            month_values.extend(
                [month_number] * days * 24
            )

        month = pd.Series(
            month_values,
            index=df.index,
            dtype=int,
        )

    monthly_peak = (
        pd.DataFrame(
            {
                "month": month,
                "p_grid_kw": grid,
            }
        )
        .groupby("month")["p_grid_kw"]
        .max()
    )

    annual_charge = float(
        monthly_peak.sum()
        * demand_rate
    )

    return {
        "annual_demand_charge_usd": annual_charge,
        "monthly_peak_kw": {
            str(int(k)): float(v)
            for k, v in monthly_peak.items()
        },
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
) -> dict[str, Any]:

    eco = config["economics"]
    tariff = config["tariff"]
    dt = float(
        config["data"].get(
            "timestep_hours",
            1.0,
        )
    )

    route = str(
        config
        .get("system", {})
        .get("route", "hydrogen")
    ).strip().lower()

    if route not in {
        "hydrogen",
        "biomethane",
        "biogas",
    }:
        raise ValueError(
            f"Unsupported system.route: {route!r}. "
            "Expected 'hydrogen', 'biomethane' or 'biogas'."
        )

    crf = capital_recovery_factor(
        eco["wacc_real"],
        eco["analysis_horizon_years"],
    )

    biogas_capex_breakdown: dict[str, float] = {}
    biogas_fixed_breakdown: dict[str, float] = {}

    if route == "biogas":
        biogas_capex_breakdown = compute_biogas_capex_total(
            capacities,
            config,
        )
        capex = float(
            biogas_capex_breakdown["capex_total_usd"]
        )

        biogas_fixed_breakdown = (
            compute_biogas_fixed_opex_annual(
                capacities,
                config,
            )
        )
        fixed = float(
            biogas_fixed_breakdown["fixed_opex_annual_usd"]
        )
    else:
        capex = compute_capex_total(
            capacities,
            eco["capex"],
        )
        fixed = compute_fixed_opex_annual(
            capacities,
            eco,
        )

    annualized_capex = capex * crf

    grid_energy_cost = compute_grid_opex_annual(
        dispatch_df,
        tariff,
        dt,
    )

    grid_peak_energy_cost = (
        compute_grid_peak_opex_annual(
            dispatch_df,
            tariff,
            dt,
        )
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

    demand_result = compute_annual_demand_charge(
        dispatch_df,
        tariff,
    )
    annual_demand_charge = float(
        demand_result["annual_demand_charge_usd"]
    )

    energy = compute_annual_energy_served(
        dispatch_df,
        dt,
    )

    variable_h2 = 0.0
    biomethane_fuel = 0.0
    variable_biomethane = 0.0
    variable_biogas = 0.0
    biogas_opex_breakdown: dict[str, Any] = {}
    chp_variable = 0.0

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

    else:
        biogas_opex_breakdown = (
            compute_variable_biogas_opex_annual(
                dispatch_df,
                config,
                dt,
            )
        )

        chp_variable = float(
            biogas_opex_breakdown["chp_variable_opex_annual_usd"]
        )
        variable_biogas = float(
            biogas_opex_breakdown["variable_biogas_opex_annual_usd"]
        )
        variable_dispatchable = variable_biogas

    lcoe_legacy = compute_lcoe(
        capex,
        fixed,
        grid_energy_cost,
        energy,
        crf,
        variable_dispatchable,
    )

    annual_cost_legacy = (
        annualized_capex
        + fixed
        + grid_energy_cost
        + variable_dispatchable
    )

    annual_cost_harmonized = (
        annual_cost_legacy
        + annual_demand_charge
    )

    lcoe_harmonized = (
        annual_cost_harmonized
        / energy
    )

    result: dict[str, Any] = {
        "route": route,
        "wacc_real": float(eco["wacc_real"]),
        "analysis_horizon_years": int(
            eco["analysis_horizon_years"]
        ),
        "capital_recovery_factor": float(crf),

        "capex_total_usd": float(capex),
        "annualized_capex_usd": float(annualized_capex),
        "fixed_opex_annual_usd": float(fixed),

        "grid_opex_annual_usd": float(grid_energy_cost),
        "grid_peak_opex_annual_usd": float(
            grid_peak_energy_cost
        ),
        "grid_peak_energy_opex_annual_usd": float(
            grid_peak_energy_cost
        ),
        "grid_peak_energy_annual_kwh": float(
            grid_peak_energy
        ),
        "grid_total_energy_annual_kwh": float(
            grid_total_energy
        ),

        "annual_demand_charge_usd": float(
            annual_demand_charge
        ),
        "monthly_peak_kw": demand_result["monthly_peak_kw"],

        "energy_served_annual_kwh": float(energy),

        "variable_h2_opex_annual_usd": float(variable_h2),
        "biomethane_fuel_opex_annual_usd": float(
            biomethane_fuel
        ),
        "variable_biomethane_opex_annual_usd": float(
            variable_biomethane
        ),
        "chp_variable_opex_annual_usd": float(
            chp_variable
        ),
        "variable_biogas_opex_annual_usd": float(
            variable_biogas
        ),
        "variable_dispatchable_opex_annual_usd": float(
            variable_dispatchable
        ),

        "annual_cost_legacy_usd": float(
            annual_cost_legacy
        ),
        "annual_cost_harmonized_usd": float(
            annual_cost_harmonized
        ),

        "lcoe_usd_kwh": float(lcoe_legacy),
        "lcoe_legacy_usd_kwh": float(lcoe_legacy),
        "lcoe_harmonized_usd_kwh": float(
            lcoe_harmonized
        ),
    }

    if route == "biogas":
        result.update(
            {
                "biogas_capex_breakdown":
                    biogas_capex_breakdown,
                "biogas_fixed_opex_breakdown":
                    biogas_fixed_breakdown,
                "biogas_variable_opex_breakdown":
                    biogas_opex_breakdown,
            }
        )

    return result
