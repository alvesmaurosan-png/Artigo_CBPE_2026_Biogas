from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import yaml
from ortools.linear_solver import pywraplp

from cbpe.status import SolverStatus


@dataclass(frozen=True)
class FeasibilityResult:
    status: str
    objective_peak_kw: float | None
    best_bound_kw: float | None
    relative_gap: float | None
    solve_time_seconds: float
    horizon_hours: int
    capacities: dict[str, float]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _status(status: int, hit_time_limit: bool) -> SolverStatus:
    if status == pywraplp.Solver.OPTIMAL:
        return SolverStatus.OPTIMAL
    if status == pywraplp.Solver.INFEASIBLE:
        return SolverStatus.INFEASIBLE
    if hit_time_limit:
        return SolverStatus.TIME_LIMIT
    if status == pywraplp.Solver.FEASIBLE:
        return SolverStatus.FEASIBLE
    if status == pywraplp.Solver.NOT_SOLVED:
        return SolverStatus.TIME_LIMIT
    return SolverStatus.EXECUTION_ERROR


def solve_minimum_grid_peak(
    config_path: Path,
    data_path: Path,
    *,
    horizon_hours: int | None = None,
    time_limit_seconds: int | None = None,
) -> FeasibilityResult:
    """Jointly size assets and dispatch them while minimizing grid peak.

    This model is intentionally separate from NSGA-II. It is the evidentiary
    model for a structural power threshold and therefore reports solver bound
    and gap rather than treating absence of a GA candidate as infeasibility.
    """

    config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    frame = pd.read_csv(data_path)
    if horizon_hours is not None:
        frame = frame.iloc[:horizon_hours].copy()
    frame = frame.reset_index(drop=True)
    limits = config["optimization"]["constraints"]
    battery = config["technology"]["battery"]
    efficiencies = config["technology"]["efficiencies"]
    h2_lhv = float(config["technology"]["hydrogen"]["lhv_kwh_per_kg"])
    dt = float(config["data"].get("timestep_hours", 1.0))

    solver = pywraplp.Solver.CreateSolver(str(config["reproducibility"].get("solver_name", "SCIP")))
    if solver is None:
        raise RuntimeError("Configured MILP solver is unavailable")
    limit = int(time_limit_seconds or config["reproducibility"].get("solver_time_limit_sec", 90))
    solver.SetTimeLimit(limit * 1000)
    gap_target = float(config["reproducibility"].get("solver_mip_gap", 0.001))
    solver.SetSolverSpecificParametersAsString(f"limits/gap = {gap_target}")

    def capacity(name: str):
        bounds = limits[name]
        return solver.NumVar(float(bounds["min"]), float(bounds["max"]), name)

    pv_cap = capacity("pv_kw")
    bsv_cap = capacity("bsv_kwh")
    elz_cap = capacity("electrolyzer_kw")
    tank_cap = capacity("h2_tank_kg")
    fc_cap = capacity("fuelcell_kw")
    peak = solver.NumVar(0, solver.infinity(), "grid_peak_kw")

    usable_factor = float(battery["soh_initial"]) * float(battery["usable_soc_window"])
    soc_min_fraction = float(battery["min_soc_fraction"])
    soc_init_fraction = float(battery["soc_init_fraction"])
    c_rate = float(battery["c_rate_max"])
    eta_ch = float(battery["battery_roundtrip"]) ** 0.5
    eta_dis = eta_ch
    eta_elz = float(efficiencies["electrolyzer"])
    eta_fc = float(efficiencies["fuelcell"])
    h2_init_fraction = float(config["technology"]["h2_storage"]["soc_init_fraction"])

    max_usable = float(limits["bsv_kwh"]["max"]) * usable_factor
    max_battery_power = max_usable * c_rate
    max_elz = float(limits["electrolyzer_kw"]["max"])
    max_fc = float(limits["fuelcell_kw"]["max"])
    max_tank = float(limits["h2_tank_kg"]["max"])

    previous_soc = None
    previous_h2 = None
    last_soc = None
    last_h2 = None
    initial_soc_expression = soc_init_fraction * usable_factor * bsv_cap
    initial_h2_expression = h2_init_fraction * tank_cap

    for t, row in frame.iterrows():
        grid = solver.NumVar(0, solver.infinity(), f"grid_{t}")
        pv_used = solver.NumVar(0, solver.infinity(), f"pv_used_{t}")
        pv_curtail = solver.NumVar(0, solver.infinity(), f"pv_curtail_{t}")
        charge = solver.NumVar(0, max_battery_power, f"charge_{t}")
        discharge = solver.NumVar(0, max_battery_power, f"discharge_{t}")
        elz = solver.NumVar(0, max_elz, f"elz_{t}")
        fc = solver.NumVar(0, max_fc, f"fc_{t}")
        soc = solver.NumVar(0, max_usable, f"soc_{t}")
        h2 = solver.NumVar(0, max_tank, f"h2_{t}")
        battery_mode = solver.BoolVar(f"battery_mode_{t}")
        h2_mode = solver.BoolVar(f"h2_mode_{t}")

        solver.Add(grid <= peak)
        solver.Add(pv_used + pv_curtail == float(row["pv_factor"]) * pv_cap)
        solver.Add(grid + pv_used + discharge + fc == float(row["demand_kw"]) + charge + elz)

        solver.Add(charge <= usable_factor * c_rate * bsv_cap)
        solver.Add(discharge <= usable_factor * c_rate * bsv_cap)
        solver.Add(charge <= max_battery_power * (1 - battery_mode))
        solver.Add(discharge <= max_battery_power * battery_mode)
        solver.Add(soc >= soc_min_fraction * usable_factor * bsv_cap)
        solver.Add(soc <= usable_factor * bsv_cap)
        delta_soc = charge * eta_ch * dt - discharge * dt / eta_dis
        solver.Add(soc == (initial_soc_expression if previous_soc is None else previous_soc) + delta_soc)

        solver.Add(elz <= elz_cap)
        solver.Add(fc <= fc_cap)
        solver.Add(elz <= max_elz * (1 - h2_mode))
        solver.Add(fc <= max_fc * h2_mode)
        solver.Add(h2 <= tank_cap)
        delta_h2 = elz * eta_elz * dt / h2_lhv - fc * dt / eta_fc / h2_lhv
        solver.Add(h2 == (initial_h2_expression if previous_h2 is None else previous_h2) + delta_h2)
        solver.Add(fc <= float(row["demand_kw"]))

        previous_soc, previous_h2 = soc, h2
        last_soc, last_h2 = soc, h2

    if last_soc is not None:
        solver.Add(last_soc >= initial_soc_expression)
    if last_h2 is not None:
        solver.Add(last_h2 >= initial_h2_expression)
    solver.Minimize(peak)

    started = time.perf_counter()
    raw_status = solver.Solve()
    elapsed = time.perf_counter() - started
    scientific_status = _status(raw_status, elapsed >= limit * 0.99)
    has_solution = raw_status in {pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE}
    objective = float(peak.solution_value()) if has_solution else None
    try:
        bound = float(solver.Objective().BestBound()) if has_solution else None
    except AttributeError:
        bound = None
    relative_gap = None
    if objective is not None and bound is not None and abs(objective) > 1e-12:
        relative_gap = max(0.0, (objective - bound) / abs(objective))

    capacities = {}
    if has_solution:
        capacities = {
            "pv_kw": float(pv_cap.solution_value()),
            "bsv_kwh": float(bsv_cap.solution_value()),
            "electrolyzer_kw": float(elz_cap.solution_value()),
            "h2_tank_kg": float(tank_cap.solution_value()),
            "fuelcell_kw": float(fc_cap.solution_value()),
        }
    return FeasibilityResult(
        status=scientific_status.value,
        objective_peak_kw=objective,
        best_bound_kw=bound,
        relative_gap=relative_gap,
        solve_time_seconds=elapsed,
        horizon_hours=len(frame),
        capacities=capacities,
    )
