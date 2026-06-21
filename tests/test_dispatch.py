from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.models.milp_dispatch import MILPDispatchOptimizer


ROOT = Path(__file__).resolve().parents[1]


def run_dispatch():
    config = yaml.safe_load((ROOT / "configs" / "paper" / "base.yaml").read_text(encoding="utf-8-sig"))
    config["reproducibility"]["solver_time_limit_sec"] = 20
    data = pd.read_csv(ROOT / "data" / "processed" / "fleet_demand_sp.csv").iloc[:48]
    capacities = {
        "pv_kw": 722.0,
        "bsv_kwh": 854.0,
        "electrolyzer_kw": 355.0,
        "h2_tank_kg": 358.0,
        "fuelcell_kw": 87.0,
    }
    optimizer = MILPDispatchOptimizer(config, capacities)
    return optimizer, optimizer.run_annual_simulation(data, period_hours=24)


def test_dispatch_balance_limits_and_soc() -> None:
    optimizer, result = run_dispatch()
    assert result.solver_status in {"OPTIMAL", "FEASIBLE"}
    frame = result.dispatch_df
    supply = frame["p_grid_kw"] + frame["p_pv_used_kw"] + frame["p_bat_dis_kw"] + frame["p_fc_kw"] + frame["p_unserved_kw"]
    use = frame["demand_kw"] + frame["p_bat_ch_kw"] + frame["p_elz_kw"]
    assert np.allclose(supply, use, atol=1e-6)
    assert frame["p_grid_kw"].max() <= config_grid_limit(optimizer) + 1e-6
    assert frame["soc_bat_kwh"].between(optimizer.battery_soc_min_kwh - 1e-6, optimizer.usable_battery_kwh + 1e-6).all()
    assert (frame["p_bat_ch_kw"] * frame["p_bat_dis_kw"] < 1e-7).all()
    assert (frame["p_elz_kw"] * frame["p_fc_kw"] < 1e-7).all()


def config_grid_limit(optimizer: MILPDispatchOptimizer) -> float:
    return optimizer.grid_power_max_kw


def test_reduced_dispatch_is_deterministic() -> None:
    _, first = run_dispatch()
    _, second = run_dispatch()
    columns = ["p_grid_kw", "p_bat_ch_kw", "p_bat_dis_kw", "p_elz_kw", "p_fc_kw"]
    assert np.allclose(first.dispatch_df[columns], second.dispatch_df[columns], atol=1e-6)

