from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
from ortools.linear_solver import pywraplp


@dataclass
class DispatchResult:
    dispatch_df: pd.DataFrame
    final_battery_kwh: float
    final_h2_kg: float
    objective_value: float
    solver_status: str
    solve_time_sec: float
    milp_gap: Optional[float] = None
    final_biomethane_nm3: float = 0.0
    final_biogas_nm3: float = 0.0


class MILPDispatchOptimizer:
    def __init__(self, config, capacities, degradation_model=None):
        self.config = config

        # ---------------------------------------------------------
        # SYSTEM ROUTE
        # ---------------------------------------------------------
        # Backward compatibility:
        # legacy configurations without system.route remain hydrogen.
        self.route = str(
            self.config.get("system", {}).get("route", "hydrogen")
        ).strip().lower()

        if self.route not in {"hydrogen", "biomethane", "biogas"}:
            raise ValueError(
                f"Unsupported system.route: {self.route!r}. "
                "Expected 'hydrogen', 'biomethane' or 'biogas'."
            )
        self.capacities = capacities
        self.degradation_model = degradation_model

        # ---------------------------------------------------------
        # REPRODUTIBILIDADE / SOLVER
        # ---------------------------------------------------------
        repro_cfg = self.config.get("reproducibility", {})
        self.solver_name = str(repro_cfg.get("solver_name", "SCIP"))
        self.solver_time_limit_sec = float(repro_cfg.get("solver_time_limit_sec", 90))
        self.solver_mip_gap = float(repro_cfg.get("solver_mip_gap", 0.001))

        # ---------------------------------------------------------
        # DADOS
        # ---------------------------------------------------------
        self.timestep_hours = float(self.config["data"].get("timestep_hours", 1.0))
        if self.timestep_hours <= 0:
            raise ValueError("data.timestep_hours must be > 0")

        # ---------------------------------------------------------
        # CAPACIDADES
        # ---------------------------------------------------------
        self.pv_kw = float(capacities["pv_kw"])
        self.bsv_nominal_kwh = float(capacities["bsv_kwh"])

        if self.route == "hydrogen":
            self.electrolyzer_kw = float(capacities["electrolyzer_kw"])
            self.h2_tank_kg = float(capacities["h2_tank_kg"])
            self.fuelcell_kw = float(capacities["fuelcell_kw"])

        elif self.route == "biomethane":
            self.biomethane_storage_nm3 = float(
                capacities["biomethane_storage_nm3"]
            )
            self.chp_kw = float(capacities["chp_kw"])

        elif self.route == "biogas":
            self.biogas_storage_nm3 = float(
                capacities["biogas_storage_nm3"]
            )
            self.chp_kw = float(capacities["chp_kw"])

            if self.biogas_storage_nm3 <= 0:
                raise ValueError(
                    "capacities.biogas_storage_nm3 must be > 0"
                )

            if self.chp_kw <= 0:
                raise ValueError(
                    "capacities.chp_kw must be > 0"
                )

        # ---------------------------------------------------------
        # BATERIA
        # ---------------------------------------------------------
        bcfg = self.config["technology"]["battery"]

        self.usable_battery_kwh = (
            self.bsv_nominal_kwh
            * float(bcfg["soh_initial"])
            * float(bcfg["usable_soc_window"])
        )

        self.battery_power_max_kw = self.usable_battery_kwh * float(bcfg["c_rate_max"])
        self.battery_soc_min_kwh = self.usable_battery_kwh * float(bcfg["min_soc_fraction"])
        self.battery_soc_init_fraction = float(bcfg["soc_init_fraction"])

        eff_rt = float(bcfg["battery_roundtrip"])
        if eff_rt <= 0 or eff_rt > 1:
            raise ValueError("technology.battery.battery_roundtrip must be in (0, 1]")
        self.eff_battery_ch = eff_rt ** 0.5
        self.eff_battery_dis = eff_rt ** 0.5

        # ---------------------------------------------------------
        # DISPATCHABLE TECHNOLOGY ROUTE
        # ---------------------------------------------------------
        if self.route == "hydrogen":
            eff = self.config["technology"]["efficiencies"]

            self.eff_electrolyzer = float(eff["electrolyzer"])
            self.eff_fuelcell = float(eff["fuelcell"])

            if self.eff_electrolyzer <= 0 or self.eff_fuelcell <= 0:
                raise ValueError("Hydrogen efficiencies must be > 0")

            self.h2_lhv = float(
                self.config["technology"]["hydrogen"]["lhv_kwh_per_kg"]
            )
            if self.h2_lhv <= 0:
                raise ValueError(
                    "technology.hydrogen.lhv_kwh_per_kg must be > 0"
                )

            self.h2_soc_init_fraction = float(
                self.config["technology"]["h2_storage"]["soc_init_fraction"]
            )

            if not 0.0 <= self.h2_soc_init_fraction <= 1.0:
                raise ValueError(
                    "technology.h2_storage.soc_init_fraction must be in [0, 1]"
                )

        elif self.route == "biomethane":
            bm_cfg = self.config["technology"]["biomethane"]
            bm_storage_cfg = self.config["technology"]["biomethane_storage"]
            chp_cfg = self.config["technology"]["chp"]

            self.biomethane_lhv_kwh_per_nm3 = float(
                bm_cfg["lhv_kwh_per_nm3"]
            )
            self.biomethane_methane_fraction = float(
                bm_cfg["methane_fraction"]
            )
            self.biomethane_max_supply_nm3_day = float(
                bm_cfg["max_supply_nm3_day"]
            )

            self.biomethane_soc_init_fraction = float(
                bm_storage_cfg["soc_init_fraction"]
            )
            self.biomethane_soc_min_fraction = float(
                bm_storage_cfg.get("soc_min_fraction", 0.0)
            )
            self.biomethane_terminal_cyclic = bool(
                bm_storage_cfg.get("enforce_terminal_cyclic_state", False)
            )

            self.chp_efficiency_el = float(chp_cfg["eta_el"])
            self.chp_availability_fraction = float(
                chp_cfg.get("availability_fraction", 1.0)
            )

            if self.biomethane_lhv_kwh_per_nm3 <= 0:
                raise ValueError(
                    "technology.biomethane.lhv_kwh_per_nm3 must be > 0"
                )

            if not 0.0 < self.biomethane_methane_fraction <= 1.0:
                raise ValueError(
                    "technology.biomethane.methane_fraction must be in (0, 1]"
                )

            if self.biomethane_max_supply_nm3_day < 0:
                raise ValueError(
                    "technology.biomethane.max_supply_nm3_day must be >= 0"
                )

            if not 0.0 <= self.biomethane_soc_init_fraction <= 1.0:
                raise ValueError(
                    "technology.biomethane_storage.soc_init_fraction must be in [0, 1]"
                )

            if not 0.0 <= self.biomethane_soc_min_fraction <= 1.0:
                raise ValueError(
                    "technology.biomethane_storage.soc_min_fraction must be in [0, 1]"
                )

            if not 0.0 < self.chp_efficiency_el <= 1.0:
                raise ValueError(
                    "technology.chp.eta_el must be in (0, 1]"
                )

            if not 0.0 <= self.chp_availability_fraction <= 1.0:
                raise ValueError(
                    "technology.chp.availability_fraction must be in [0, 1]"
                )

            delivery_cfg = self.config["delivery"]

            self.biomethane_delivery_enabled = bool(
                delivery_cfg.get("enabled", True)
            )
            self.biomethane_delivery_hour = int(
                delivery_cfg["hour"]
            )
            self.biomethane_delivery_max_nm3_day = float(
                delivery_cfg["max_nm3_per_day"]
            )
            self.biomethane_optimize_delivery_volume = bool(
                delivery_cfg.get("optimize_delivery_volume", True)
            )
            self.biomethane_max_deliveries_per_day = int(
                delivery_cfg.get("max_deliveries_per_day", 1)
            )

            if not 0 <= self.biomethane_delivery_hour <= 23:
                raise ValueError(
                    "delivery.hour must be between 0 and 23"
                )

            if self.biomethane_delivery_max_nm3_day < 0:
                raise ValueError(
                    "delivery.max_nm3_per_day must be >= 0"
                )

        elif self.route == "biogas":
            bg_cfg = self.config["technology"]["biogas"]
            bg_storage_cfg = self.config["technology"]["biogas_storage"]
            chp_bg_cfg = self.config["technology"]["chp_biogas"]

            # -----------------------------------------------------
            # ProduÃ§Ã£o agregada de biogÃ¡s
            # -----------------------------------------------------
            self.biogas_substrate_t_day = float(
                bg_cfg["substrate_t_day"]
            )
            self.biogas_yield_nm3_per_t = float(
                bg_cfg["biogas_yield_nm3_per_t"]
            )
            self.biogas_lhv_kwh_per_nm3 = float(
                bg_cfg["lhv_kwh_per_nm3"]
            )
            self.biogas_parasitic_fraction = float(
                bg_cfg.get("parasitic_fraction", 0.0)
            )
            self.biogas_production_mode = str(
                bg_cfg.get(
                    "production_mode",
                    "constant_daily_average",
                )
            ).strip().lower()

            if self.biogas_substrate_t_day < 0:
                raise ValueError(
                    "technology.biogas.substrate_t_day must be >= 0"
                )

            if self.biogas_yield_nm3_per_t <= 0:
                raise ValueError(
                    "technology.biogas.biogas_yield_nm3_per_t must be > 0"
                )

            if self.biogas_lhv_kwh_per_nm3 <= 0:
                raise ValueError(
                    "technology.biogas.lhv_kwh_per_nm3 must be > 0"
                )

            if not 0.0 <= self.biogas_parasitic_fraction < 1.0:
                raise ValueError(
                    "technology.biogas.parasitic_fraction "
                    "must be in [0, 1)"
                )

            if self.biogas_production_mode != "constant_daily_average":
                raise ValueError(
                    "B2 physical baseline currently supports only "
                    "technology.biogas.production_mode="
                    "'constant_daily_average'"
                )

            self.biogas_production_nm3_hour = (
                self.biogas_substrate_t_day
                * self.biogas_yield_nm3_per_t
                / 24.0
            )

            # -----------------------------------------------------
            # GasÃ´metro
            # -----------------------------------------------------
            self.biogas_soc_init_fraction = float(
                bg_storage_cfg["soc_init_fraction"]
            )
            self.biogas_soc_min_fraction = float(
                bg_storage_cfg["soc_min_fraction"]
            )
            self.biogas_soc_max_fraction = float(
                bg_storage_cfg["soc_max_fraction"]
            )
            self.biogas_terminal_cyclic = bool(
                bg_storage_cfg.get(
                    "enforce_terminal_cyclic_state",
                    False,
                )
            )

            if not (
                0.0
                <= self.biogas_soc_min_fraction
                < self.biogas_soc_max_fraction
                <= 1.0
            ):
                raise ValueError(
                    "biogas storage requires "
                    "0 <= SOCmin < SOCmax <= 1"
                )

            if not (
                self.biogas_soc_min_fraction
                <= self.biogas_soc_init_fraction
                <= self.biogas_soc_max_fraction
            ):
                raise ValueError(
                    "biogas initial SOC must lie between "
                    "SOCmin and SOCmax"
                )

            # -----------------------------------------------------
            # CHP B2
            # -----------------------------------------------------
            self.biogas_chp_efficiency_el = float(
                chp_bg_cfg["eta_el"]
            )
            self.biogas_chp_efficiency_th = float(
                chp_bg_cfg.get("eta_th", 0.0)
            )
            self.biogas_chp_min_load_fraction = float(
                chp_bg_cfg["min_load_fraction"]
            )

            # Valor consolidado, mas sua traduÃ§Ã£o temporal ainda
            # NÃƒO Ã© aplicada como derating contÃ­nuo.
            self.biogas_chp_availability_fraction = float(
                chp_bg_cfg.get("availability_fraction", 1.0)
            )

            if not 0.0 < self.biogas_chp_efficiency_el <= 1.0:
                raise ValueError(
                    "technology.chp_biogas.eta_el must be in (0, 1]"
                )

            if not 0.0 <= self.biogas_chp_efficiency_th <= 1.0:
                raise ValueError(
                    "technology.chp_biogas.eta_th must be in [0, 1]"
                )

            if not 0.0 <= self.biogas_chp_min_load_fraction <= 1.0:
                raise ValueError(
                    "technology.chp_biogas.min_load_fraction "
                    "must be in [0, 1]"
                )

            if not 0.0 <= self.biogas_chp_availability_fraction <= 1.0:
                raise ValueError(
                    "technology.chp_biogas.availability_fraction "
                    "must be in [0, 1]"
                )

        # ---------------------------------------------------------
        # TARIFA
        # ---------------------------------------------------------
        tariff = self.config["tariff"]
        self.used_in_objective = bool(tariff.get("used_in_milp_objective", True))
        self.offpeak_price = float(tariff.get("offpeak_price_usd_kwh", 0.0))
        self.peak_price = float(tariff.get("peak_price_usd_kwh", 0.0))
        self.peak_start = int(tariff["peak_window"]["start_hour"])
        self.peak_end = int(tariff["peak_window"]["end_hour"])

        # ---------------------------------------------------------
        # DEMAND CHARGE â€” M1b
        # Disabled by default to preserve M0 behavior.
        # ---------------------------------------------------------
        self.include_demand_charge = bool(
            tariff.get("include_demand_charge", False)
        )
        self.demand_charge_usd_kw_month = float(
            tariff.get("demand_charge_usd_kw_month", 0.0)
        )
        self.demand_charge_in_milp_objective = bool(
            tariff.get("demand_charge_in_milp_objective", False)
        )

        if self.demand_charge_usd_kw_month < 0:
            raise ValueError(
                "tariff.demand_charge_usd_kw_month must be >= 0"
            )

        # ---------------------------------------------------------
        # GRID
        # ---------------------------------------------------------
        self.grid_power_max_kw = float(
            self.config["optimization"].get("grid_power_max_kw", 5000.0)
        )
        if self.grid_power_max_kw < 0:
            raise ValueError("optimization.grid_power_max_kw must be >= 0")

        # ---------------------------------------------------------
        # BLACKOUT
        # ---------------------------------------------------------
        blk = self.config.get("resilience", {}).get("blackout", {})
        self.blackout_enabled = bool(blk.get("enabled", False))
        self.blackout_mode = str(blk.get("mode", "single_event"))
        self.blackout_start_global = blk.get("start_global_hour", None)
        self.blackout_start_hour = blk.get("start_hour", None)
        self.blackout_duration = int(blk.get("duration_hours", 0))

        # ---------------------------------------------------------
        # OBJETIVO / PENALIDADES
        # ---------------------------------------------------------
        obj_cfg = self.config.get("objective", {})
        self.peak_penalty_multiplier = float(obj_cfg.get("peak_penalty_multiplier", 1.0))
        self.curtailment_penalty_usd_kwh = float(obj_cfg.get("curtailment_penalty_usd_kwh", 0.0))
        self.global_peak_penalty_weight = float(obj_cfg.get("global_peak_penalty_weight", 0.0))

        eco_cfg = self.config.get("economics", {})

        if self.route == "hydrogen":
            h2_var_cfg = eco_cfg.get("opex_variable_h2", {})

            self.elz_variable_cost_usd_kwh = float(
                h2_var_cfg.get("electrolyzer_usd_kwh", 0.0)
            )
            self.fc_variable_cost_usd_kwh = float(
                h2_var_cfg.get("fuelcell_usd_kwh", 0.0)
            )

        elif self.route == "biomethane":
            bm_var_cfg = eco_cfg.get(
                "opex_variable_biomethane",
                {}
            )

            self.biomethane_price_usd_nm3 = float(
                bm_var_cfg.get("biomethane_usd_per_nm3", 0.0)
            )
            self.chp_variable_cost_usd_kwh = float(
                bm_var_cfg.get("chp_usd_per_kwh", 0.0)
            )

            if self.biomethane_price_usd_nm3 < 0:
                raise ValueError(
                    "economics.opex_variable_biomethane."
                    "biomethane_usd_per_nm3 must be >= 0"
                )

            if self.chp_variable_cost_usd_kwh < 0:
                raise ValueError(
                    "economics.opex_variable_biomethane."
                    "chp_usd_per_kwh must be >= 0"
                )

        elif self.route == "biogas":
            bg_var_cfg = eco_cfg.get(
                "opex_variable_biogas",
                {},
            )

            self.biogas_chp_variable_cost_usd_kwh_gross = float(
                bg_var_cfg.get(
                    "chp_usd_per_kwh_gross",
                    0.0,
                )
            )

            if self.biogas_chp_variable_cost_usd_kwh_gross < 0:
                raise ValueError(
                    "economics.opex_variable_biogas."
                    "chp_usd_per_kwh_gross must be >= 0"
                )

        # penalidade leve de throughput da bateria para evitar cycling artificial
        self.battery_discharge_penalty_usd_kwh = float(
            obj_cfg.get("battery_discharge_penalty_usd_kwh", 0.005)
        )

        # confiabilidade
        rel_cfg = self.config.get("reliability", {})
        self.allow_unserved = bool(rel_cfg.get("allow_unserved", False))
        self.unserved_penalty_usd_kwh = float(rel_cfg.get("unserved_penalty_usd_kwh", 10000.0))

    # ---------------------------------------------------------

    def _is_blackout_hour(self, global_hour: int, local_hour: int) -> bool:
        if not self.blackout_enabled:
            return False

        if self.blackout_mode == "single_event":
            if self.blackout_start_global is None:
                raise ValueError("resilience.blackout.start_global_hour is required for single_event")
            return (
                int(self.blackout_start_global)
                <= global_hour
                < int(self.blackout_start_global) + self.blackout_duration
            )

        if self.blackout_mode == "recurring_daily":
            if self.blackout_start_hour is None:
                raise ValueError("resilience.blackout.start_hour is required for recurring_daily")
            return (
                int(self.blackout_start_hour)
                <= local_hour
                < int(self.blackout_start_hour) + self.blackout_duration
            )

        raise ValueError(f"Unsupported blackout mode: {self.blackout_mode}")

    # ---------------------------------------------------------

    def _build_solver(self) -> pywraplp.Solver:
        solver = pywraplp.Solver.CreateSolver(self.solver_name)
        if solver is None:
            raise RuntimeError(f"Could not create solver: {self.solver_name}")

        solver.SetTimeLimit(int(self.solver_time_limit_sec * 1000))
        return solver

    # ---------------------------------------------------------

    def _status_to_str(self, status: int) -> str:
        mapping = {
            pywraplp.Solver.OPTIMAL: "OPTIMAL",
            pywraplp.Solver.FEASIBLE: "FEASIBLE",
            pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
            pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
            pywraplp.Solver.ABNORMAL: "ABNORMAL",
            pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
        }
        return mapping.get(status, f"UNKNOWN_{status}")

    # ---------------------------------------------------------

    def _create_variables(self, solver: pywraplp.Solver, T: int) -> dict[str, object]:
        if self.allow_unserved:
            unserved_vars = [
                solver.NumVar(0, solver.infinity(), f"un_{t}")
                for t in range(T)
            ]
        else:
            unserved_vars = [
                solver.NumVar(0, 0, f"un_{t}")
                for t in range(T)
            ]

        # -----------------------------------------------------
        # Variaveis comuns a todas as rotas
        # -----------------------------------------------------
        variables: dict[str, object] = {
            "p_grid": [
                solver.NumVar(0, self.grid_power_max_kw, f"grid_{t}")
                for t in range(T)
            ],
            "p_pv_used": [
                solver.NumVar(0, self.pv_kw, f"pv_used_{t}")
                for t in range(T)
            ],
            "p_pv_curtail": [
                solver.NumVar(0, self.pv_kw, f"pv_cur_{t}")
                for t in range(T)
            ],
            "p_bat_ch": [
                solver.NumVar(
                    0,
                    self.battery_power_max_kw,
                    f"bat_ch_{t}",
                )
                for t in range(T)
            ],
            "p_bat_dis": [
                solver.NumVar(
                    0,
                    self.battery_power_max_kw,
                    f"bat_dis_{t}",
                )
                for t in range(T)
            ],
            "soc": [
                solver.NumVar(
                    self.battery_soc_min_kwh,
                    self.usable_battery_kwh,
                    f"soc_{t}",
                )
                for t in range(T)
            ],
            "unserved": unserved_vars,
            "u_bat_mode": [
                solver.BoolVar(f"u_bat_mode_{t}")
                for t in range(T)
            ],
            "P_peak": solver.NumVar(
                0,
                self.grid_power_max_kw,
                "P_peak",
            ),
        }

        # -----------------------------------------------------
        # Rota hydrogen
        # -----------------------------------------------------
        if self.route == "hydrogen":
            variables.update(
                {
                    "p_elz": [
                        solver.NumVar(
                            0,
                            self.electrolyzer_kw,
                            f"elz_{t}",
                        )
                        for t in range(T)
                    ],
                    "p_fc": [
                        solver.NumVar(
                            0,
                            self.fuelcell_kw,
                            f"fc_{t}",
                        )
                        for t in range(T)
                    ],
                    "h2": [
                        solver.NumVar(
                            0,
                            self.h2_tank_kg,
                            f"h2_{t}",
                        )
                        for t in range(T)
                    ],
                    "u_h2_mode": [
                        solver.BoolVar(f"u_h2_mode_{t}")
                        for t in range(T)
                    ],
                }
            )

        # -----------------------------------------------------
        # Rota biomethane
        # -----------------------------------------------------
        elif self.route == "biomethane":
            chp_power_available_kw = (
                self.chp_kw * self.chp_availability_fraction
            )

            variables.update(
                {
                    "p_chp": [
                        solver.NumVar(
                            0,
                            chp_power_available_kw,
                            f"chp_{t}",
                        )
                        for t in range(T)
                    ],
                    "biomethane_use": [
                        solver.NumVar(
                            0,
                            solver.infinity(),
                            f"bm_use_{t}",
                        )
                        for t in range(T)
                    ],
                    "biomethane_delivery": [
                        solver.NumVar(
                            0,
                            self.biomethane_delivery_max_nm3_day,
                            f"bm_delivery_{t}",
                        )
                        for t in range(T)
                    ],
                    "biomethane_level": [
                        solver.NumVar(
                            0,
                            self.biomethane_storage_nm3,
                            f"bm_level_{t}",
                        )
                        for t in range(T)
                    ],
                }
            )

        # -----------------------------------------------------
        # MONTHLY DEMAND PEAK â€” M1b
        # -----------------------------------------------------
        # -----------------------------------------------------
        # Rota biogas â€” B2
        # -----------------------------------------------------
        elif self.route == "biogas":
            biogas_level_min = (
                self.biogas_storage_nm3
                * self.biogas_soc_min_fraction
            )
            biogas_level_max = (
                self.biogas_storage_nm3
                * self.biogas_soc_max_fraction
            )

            variables.update(
                {
                    # PotÃªncia elÃ©trica lÃ­quida entregue pela
                    # rota B2 ao barramento da microrrede.
                    "p_chp_net": [
                        solver.NumVar(
                            0,
                            self.chp_kw,
                            f"bg_chp_net_{t}",
                        )
                        for t in range(T)
                    ],

                    # Estado ligado/desligado da CHP.
                    "u_chp_on": [
                        solver.BoolVar(
                            f"bg_chp_on_{t}"
                        )
                        for t in range(T)
                    ],

                    # Consumo horÃ¡rio de biogÃ¡s.
                    "biogas_use": [
                        solver.NumVar(
                            0,
                            solver.infinity(),
                            f"bg_use_{t}",
                        )
                        for t in range(T)
                    ],

                    # Estoque no gasÃ´metro limitado diretamente
                    # entre SOCmin e SOCmax.
                    "biogas_level": [
                        solver.NumVar(
                            biogas_level_min,
                            biogas_level_max,
                            f"bg_level_{t}",
                        )
                        for t in range(T)
                    ],
                }
            )

        if self.demand_charge_in_milp_objective:
            variables["P_peak_month"] = solver.NumVar(
                0,
                self.grid_power_max_kw,
                "P_peak_month",
            )
            variables["P_demand_increment"] = solver.NumVar(
                0,
                self.grid_power_max_kw,
                "P_demand_increment",
            )

        return variables
    # ---------------------------------------------------------

    def _add_constraints(
            self,
            solver: pywraplp.Solver,
            v: dict[str, object],
            df: pd.DataFrame,
            init_bat: float,
            init_dispatchable_state: float,
            prev_peak: float,
            prev_monthly_peak: float = 0.0,
            terminal_dispatchable_target: float | None = None,
    ) -> None:

        solver.Add(v["P_peak"] >= prev_peak)

        if self.demand_charge_in_milp_objective:
            solver.Add(
                v["P_peak_month"] >= prev_monthly_peak
            )
            solver.Add(
                v["P_demand_increment"]
                >= v["P_peak_month"] - prev_monthly_peak
            )

        dt = self.timestep_hours
        T = len(df)

        # -----------------------------------------------------
        # Limite diario efetivo de fornecimento de biometano
        # -----------------------------------------------------
        if self.route == "biomethane":
            biomethane_daily_delivery_limit = min(
                self.biomethane_delivery_max_nm3_day,
                self.biomethane_max_supply_nm3_day,
            )

        for t in range(T):
            pv = self.pv_kw * float(df.iloc[t]["pv_factor"])
            demand = float(df.iloc[t]["demand_kw"])

            local_hour = int(df.iloc[t]["hour"]) % 24
            global_hour = int(df.iloc[t]["t_global"])

            # -------------------------------------------------
            # Balanco eletrico por rota
            # -------------------------------------------------
            if self.route == "hydrogen":
                solver.Add(
                    v["p_grid"][t]
                    + v["p_pv_used"][t]
                    + v["p_bat_dis"][t]
                    + v["p_fc"][t]
                    + v["unserved"][t]
                    == demand
                    + v["p_bat_ch"][t]
                    + v["p_elz"][t]
                )

            elif self.route == "biomethane":
                solver.Add(
                    v["p_grid"][t]
                    + v["p_pv_used"][t]
                    + v["p_bat_dis"][t]
                    + v["p_chp"][t]
                    + v["unserved"][t]
                    == demand
                    + v["p_bat_ch"][t]
                )

            elif self.route == "biogas":
                solver.Add(
                    v["p_grid"][t]
                    + v["p_pv_used"][t]
                    + v["p_bat_dis"][t]
                    + v["p_chp_net"][t]
                    + v["unserved"][t]
                    == demand
                    + v["p_bat_ch"][t]
                )

            # -------------------------------------------------
            # PV
            # -------------------------------------------------
            solver.Add(
                v["p_pv_used"][t]
                + v["p_pv_curtail"][t]
                == pv
            )

            # -------------------------------------------------
            # Grid
            # -------------------------------------------------
            solver.Add(v["p_grid"][t] <= v["P_peak"])

            if self.demand_charge_in_milp_objective:
                solver.Add(
                    v["p_grid"][t] <= v["P_peak_month"]
                )

            if self._is_blackout_hour(global_hour, local_hour):
                solver.Add(v["p_grid"][t] == 0)

            # -------------------------------------------------
            # Nao simultaneidade bateria
            # -------------------------------------------------
            solver.Add(
                v["p_bat_dis"][t]
                <= v["u_bat_mode"][t]
                * self.battery_power_max_kw
            )

            solver.Add(
                v["p_bat_ch"][t]
                <= (1 - v["u_bat_mode"][t])
                * self.battery_power_max_kw
            )

            # -------------------------------------------------
            # Dinamica bateria
            # -------------------------------------------------
            delta_soc = (
                v["p_bat_ch"][t]
                * self.eff_battery_ch
                * dt
                - v["p_bat_dis"][t]
                * dt
                / self.eff_battery_dis
            )

            if t == 0:
                solver.Add(
                    v["soc"][t]
                    == init_bat + delta_soc
                )
            else:
                solver.Add(
                    v["soc"][t]
                    == v["soc"][t - 1] + delta_soc
                )

            # =================================================
            # ROTA HYDROGEN
            # =================================================
            if self.route == "hydrogen":

                # ---------------------------------------------
                # Nao simultaneidade ELZ / FC
                # ---------------------------------------------
                solver.Add(
                    v["p_fc"][t]
                    <= v["u_h2_mode"][t]
                    * self.fuelcell_kw
                )

                solver.Add(
                    v["p_elz"][t]
                    <= (1 - v["u_h2_mode"][t])
                    * self.electrolyzer_kw
                )

                # ---------------------------------------------
                # Dinamica H2
                # ---------------------------------------------
                h2_prod = (
                    v["p_elz"][t]
                    * self.eff_electrolyzer
                    * dt
                    / self.h2_lhv
                )

                h2_cons = (
                    v["p_fc"][t]
                    * dt
                    / self.eff_fuelcell
                    / self.h2_lhv
                )

                if t == 0:
                    solver.Add(
                        v["h2"][t]
                        == init_dispatchable_state
                        + h2_prod
                        - h2_cons
                    )
                else:
                    solver.Add(
                        v["h2"][t]
                        == v["h2"][t - 1]
                        + h2_prod
                        - h2_cons
                    )

                # FC nao pode exceder demanda
                solver.Add(v["p_fc"][t] <= demand)

            # =================================================
            # ROTA BIOMETHANE
            # =================================================
            elif self.route == "biomethane":

                # ---------------------------------------------
                # CHP -> consumo de biometano
                #
                # V_BM = P_CHP * dt / (eta_el * PCI_BM)
                # ---------------------------------------------
                solver.Add(
                    v["biomethane_use"][t]
                    * self.chp_efficiency_el
                    * self.biomethane_lhv_kwh_per_nm3
                    == v["p_chp"][t] * dt
                )

                # ---------------------------------------------
                # Entrega
                # ---------------------------------------------
                if (
                    not self.biomethane_delivery_enabled
                    or local_hour != self.biomethane_delivery_hour
                ):
                    solver.Add(
                        v["biomethane_delivery"][t] == 0
                    )

                # ---------------------------------------------
                # Dinamica do estoque de biometano
                # ---------------------------------------------
                if t == 0:
                    solver.Add(
                        v["biomethane_level"][t]
                        == init_dispatchable_state
                        + v["biomethane_delivery"][t]
                        - v["biomethane_use"][t]
                    )
                else:
                    solver.Add(
                        v["biomethane_level"][t]
                        == v["biomethane_level"][t - 1]
                        + v["biomethane_delivery"][t]
                        - v["biomethane_use"][t]
                    )

                # ---------------------------------------------
                # Estoque minimo
                # ---------------------------------------------
                solver.Add(
                    v["biomethane_level"][t]
                    >= (
                        self.biomethane_storage_nm3
                        * self.biomethane_soc_min_fraction
                    )
                )

            # =================================================
            # ROTA BIOGAS â€” B2
            # =================================================
            elif self.route == "biogas":

                # ---------------------------------------------
                # CHP ligado/desligado + carga mÃ­nima
                #
                # chp_kw representa potÃªncia elÃ©trica lÃ­quida
                # da rota B2 entregue ao barramento.
                # ---------------------------------------------
                solver.Add(
                    v["p_chp_net"][t]
                    <= self.chp_kw
                    * v["u_chp_on"][t]
                )

                solver.Add(
                    v["p_chp_net"][t]
                    >= self.chp_kw
                    * self.biogas_chp_min_load_fraction
                    * v["u_chp_on"][t]
                )

                # ---------------------------------------------
                # BiogÃ¡s -> eletricidade lÃ­quida
                #
                # E_net =
                # V_BG * PCI_BG * eta_el * (1 - parasitic)
                #
                # Esta equaÃ§Ã£o reproduz a convenÃ§Ã£o consolidada
                # da planilha B2-2026:
                # 196,812 kWh/t bruto -> 177,1308 kWh/t lÃ­quido.
                # ---------------------------------------------
                solver.Add(
                    v["biogas_use"][t]
                    * self.biogas_lhv_kwh_per_nm3
                    * self.biogas_chp_efficiency_el
                    * (1.0 - self.biogas_parasitic_fraction)
                    == v["p_chp_net"][t] * dt
                )

                # ---------------------------------------------
                # ProduÃ§Ã£o contÃ­nua agregada de biogÃ¡s
                # ---------------------------------------------
                biogas_prod = (
                    self.biogas_production_nm3_hour
                    * dt
                )

                # ---------------------------------------------
                # DinÃ¢mica do gasÃ´metro
                # ---------------------------------------------
                if t == 0:
                    solver.Add(
                        v["biogas_level"][t]
                        == init_dispatchable_state
                        + biogas_prod
                        - v["biogas_use"][t]
                    )
                else:
                    solver.Add(
                        v["biogas_level"][t]
                        == v["biogas_level"][t - 1]
                        + biogas_prod
                        - v["biogas_use"][t]
                    )

        # =====================================================
        # RESTRICOES GLOBAIS POR ROTA
        # =====================================================

        if self.route == "hydrogen":

            # -------------------------------------------------
            # Conservacao global H2
            # -------------------------------------------------
            solver.Add(
                sum(
                    v["p_fc"][t]
                    * dt
                    / self.eff_fuelcell
                    / self.h2_lhv
                    for t in range(T)
                )
                <= init_dispatchable_state
                + sum(
                    v["p_elz"][t]
                    * self.eff_electrolyzer
                    * dt
                    / self.h2_lhv
                    for t in range(T)
                )
            )

        elif self.route == "biomethane":

            # -------------------------------------------------
            # Limite de entrega por dia calendario
            # -------------------------------------------------
            day_groups: dict[int, list[int]] = {}

            for t in range(T):
                global_hour = int(df.iloc[t]["t_global"])
                day_index = global_hour // 24
                day_groups.setdefault(day_index, []).append(t)

            for indices in day_groups.values():
                solver.Add(
                    sum(
                        v["biomethane_delivery"][t]
                        for t in indices
                    )
                    <= biomethane_daily_delivery_limit
                )

            # A condicao terminal ciclica nao e aplicada aqui.
            # Ela deve ser imposta somente no fim do horizonte
            # anual, e nao em cada bloco do rolling horizon.

        elif self.route == "biogas":
            # NÃ£o existe entrega externa em B2.
            # O recurso entra exclusivamente pela produÃ§Ã£o
            # contÃ­nua derivada do substrato.

            if terminal_dispatchable_target is not None:
                solver.Add(
                    v["biogas_level"][T - 1]
                    == terminal_dispatchable_target
                )

    # ---------------------------------------------------------

    def _set_objective(self, solver: pywraplp.Solver, v: dict[str, object], df: pd.DataFrame) -> None:
        objective = solver.Objective()
        dt = self.timestep_hours

        for t in range(len(df)):
            hour = int(df.iloc[t]["hour"]) % 24

            price = 0.0
            if self.used_in_objective:
                price = self.offpeak_price
                if self.peak_start <= hour < self.peak_end:
                    price = self.peak_price * self.peak_penalty_multiplier

            objective.SetCoefficient(v["p_grid"][t], price * dt)
            objective.SetCoefficient(v["p_pv_curtail"][t], self.curtailment_penalty_usd_kwh * dt)
            if self.route == "hydrogen":
                objective.SetCoefficient(
                    v["p_elz"][t],
                    self.elz_variable_cost_usd_kwh * dt,
                )
                objective.SetCoefficient(
                    v["p_fc"][t],
                    self.fc_variable_cost_usd_kwh * dt,
                )

            elif self.route == "biomethane":
                # Combustivel: USD/Nm3 consumido
                objective.SetCoefficient(
                    v["biomethane_use"][t],
                    self.biomethane_price_usd_nm3,
                )

                # OPEX variavel do CHP: USD/kWh eletrico gerado
                objective.SetCoefficient(
                    v["p_chp"][t],
                    self.chp_variable_cost_usd_kwh * dt,
                )

            elif self.route == "biogas":
                # OPEX CHP foi consolidado em USD/kWh bruto.
                # Como p_chp_net Ã© lÃ­quido:
                #
                # E_gross = E_net / (1 - parasitic)
                #
                gross_cost_per_net_kwh = (
                    self.biogas_chp_variable_cost_usd_kwh_gross
                    / (1.0 - self.biogas_parasitic_fraction)
                )

                objective.SetCoefficient(
                    v["p_chp_net"][t],
                    gross_cost_per_net_kwh * dt,
                )

            objective.SetCoefficient(v["p_bat_dis"][t], self.battery_discharge_penalty_usd_kwh * dt)
            objective.SetCoefficient(v["unserved"][t], self.unserved_penalty_usd_kwh * dt)

        if self.global_peak_penalty_weight > 0:
            objective.SetCoefficient(
                v["P_peak"],
                self.global_peak_penalty_weight,
            )

        if (
            self.demand_charge_in_milp_objective
            and self.include_demand_charge
            and self.demand_charge_usd_kw_month > 0
        ):
            objective.SetCoefficient(
                v["P_demand_increment"],
                self.demand_charge_usd_kw_month,
            )

        objective.SetMinimization()

    # ---------------------------------------------------------

    def solve_period(
        self,
        df: pd.DataFrame,
        init_bat: float,
        init_dispatchable_state: float,
        prev_peak: float = 0,
        prev_monthly_peak: float = 0.0,
        terminal_dispatchable_target: float | None = None,
    ) -> DispatchResult:
        solver = self._build_solver()
        v = self._create_variables(solver, len(df))

        self._add_constraints(
            solver,
            v,
            df,
            init_bat,
            init_dispatchable_state,
            prev_peak,
            prev_monthly_peak,
            terminal_dispatchable_target,
        )
        self._set_objective(solver, v, df)

        status = solver.Solve()
        status_str = self._status_to_str(status)
        solve_time_sec = float(solver.WallTime()) / 1000.0

        if status not in (
            pywraplp.Solver.OPTIMAL,
            pywraplp.Solver.FEASIBLE,
        ):
            if self.route == "hydrogen":
                final_h2_kg = init_dispatchable_state
                final_biomethane_nm3 = 0.0
                final_biogas_nm3 = 0.0

            elif self.route == "biomethane":
                final_h2_kg = 0.0
                final_biomethane_nm3 = init_dispatchable_state
                final_biogas_nm3 = 0.0

            else:  # biogas
                final_h2_kg = 0.0
                final_biomethane_nm3 = 0.0
                final_biogas_nm3 = init_dispatchable_state

            return DispatchResult(
                dispatch_df=pd.DataFrame(),
                final_battery_kwh=init_bat,
                final_h2_kg=final_h2_kg,
                objective_value=0.0,
                solver_status=status_str,
                solve_time_sec=solve_time_sec,
                milp_gap=None,
                final_biomethane_nm3=final_biomethane_nm3,
                final_biogas_nm3=final_biogas_nm3,
            )

        rows: list[dict[str, float | int]] = []

        for t in range(len(df)):
            local_hour = int(df.iloc[t]["hour"]) % 24
            global_hour = int(df.iloc[t]["t_global"])

            row: dict[str, float | int] = {
                "t_global": global_hour,
                "hour": int(df.iloc[t]["hour"]),
                "demand_kw": float(df.iloc[t]["demand_kw"]),
                "pv_kw": (
                    self.pv_kw
                    * float(df.iloc[t]["pv_factor"])
                ),
                "p_grid_kw": float(
                    v["p_grid"][t].solution_value()
                ),
                "p_pv_used_kw": float(
                    v["p_pv_used"][t].solution_value()
                ),
                "p_pv_curtail_kw": float(
                    v["p_pv_curtail"][t].solution_value()
                ),
                "p_bat_ch_kw": float(
                    v["p_bat_ch"][t].solution_value()
                ),
                "p_bat_dis_kw": float(
                    v["p_bat_dis"][t].solution_value()
                ),
                "soc_bat_kwh": float(
                    v["soc"][t].solution_value()
                ),
                "p_unserved_kw": float(
                    v["unserved"][t].solution_value()
                ),
                "blackout_flag": int(
                    self._is_blackout_hour(
                        global_hour,
                        local_hour,
                    )
                ),
            }

            if self.route == "hydrogen":
                row.update(
                    {
                        "p_elz_kw": float(
                            v["p_elz"][t].solution_value()
                        ),
                        "p_fc_kw": float(
                            v["p_fc"][t].solution_value()
                        ),
                        "h2_level_kg": float(
                            v["h2"][t].solution_value()
                        ),
                    }
                )

            elif self.route == "biomethane":
                row.update(
                    {
                        "p_chp_kw": float(
                            v["p_chp"][t].solution_value()
                        ),
                        "biomethane_use_nm3": float(
                            v["biomethane_use"][t].solution_value()
                        ),
                        "biomethane_delivery_nm3": float(
                            v["biomethane_delivery"][t].solution_value()
                        ),
                        "biomethane_level_nm3": float(
                            v["biomethane_level"][t].solution_value()
                        ),
                    }
                )

            elif self.route == "biogas":
                p_net = float(
                    v["p_chp_net"][t].solution_value()
                )

                p_gross_equiv = (
                    p_net
                    / (1.0 - self.biogas_parasitic_fraction)
                )

                row.update(
                    {
                        "p_chp_net_kw": p_net,
                        "p_chp_gross_equiv_kw": p_gross_equiv,
                        "p_biogas_aux_kw": (
                            p_gross_equiv - p_net
                        ),
                        "u_chp_on": int(
                            round(
                                v["u_chp_on"][t].solution_value()
                            )
                        ),
                        "biogas_production_nm3": (
                            self.biogas_production_nm3_hour
                            * self.timestep_hours
                        ),
                        "biogas_use_nm3": float(
                            v["biogas_use"][t].solution_value()
                        ),
                        "biogas_level_nm3": float(
                            v["biogas_level"][t].solution_value()
                        ),
                    }
                )

            rows.append(row)

        final_battery_kwh = float(
            v["soc"][len(df) - 1].solution_value()
        )

        if self.route == "hydrogen":
            final_h2_kg = float(
                v["h2"][len(df) - 1].solution_value()
            )
            final_biomethane_nm3 = 0.0
            final_biogas_nm3 = 0.0

        elif self.route == "biomethane":
            final_h2_kg = 0.0
            final_biomethane_nm3 = float(
                v["biomethane_level"][len(df) - 1].solution_value()
            )
            final_biogas_nm3 = 0.0

        else:  # biogas
            final_h2_kg = 0.0
            final_biomethane_nm3 = 0.0
            final_biogas_nm3 = float(
                v["biogas_level"][len(df) - 1].solution_value()
            )

        return DispatchResult(
            dispatch_df=pd.DataFrame(rows),
            final_battery_kwh=final_battery_kwh,
            final_h2_kg=final_h2_kg,
            objective_value=float(
                solver.Objective().Value()
            ),
            solver_status=status_str,
            solve_time_sec=solve_time_sec,
            milp_gap=None,
            final_biomethane_nm3=final_biomethane_nm3,
            final_biogas_nm3=final_biogas_nm3,
        )
    # ---------------------------------------------------------

    @staticmethod
    def _billing_month_from_global_hour(global_hour: int) -> int:
        """
        Return billing month 1..12 for the 8760-hour 2026 baseline.
        """
        month_hours = (
            31 * 24,
            28 * 24,
            31 * 24,
            30 * 24,
            31 * 24,
            30 * 24,
            31 * 24,
            31 * 24,
            30 * 24,
            31 * 24,
            30 * 24,
            31 * 24,
        )

        cumulative = 0
        for month, hours in enumerate(month_hours, start=1):
            cumulative += hours
            if global_hour < cumulative:
                return month

        raise ValueError(
            f"global_hour outside 8760-hour baseline: {global_hour}"
        )

    def run_annual_simulation(
        self,
        df: pd.DataFrame,
        period_hours: int = 168,
        commit_hours: int | None = None,
        lookahead_hours: int = 0,
    ) -> DispatchResult:

        # -----------------------------------------------------
        # OPTIONAL LOOK-AHEAD MODE â€” M2
        #
        # Backward compatibility:
        # commit_hours=None and lookahead_hours=0 execute the
        # historical rolling-horizon implementation unchanged.
        # -----------------------------------------------------
        if (
            commit_hours is not None
            or lookahead_hours != 0
        ):
            effective_commit_hours = (
                period_hours
                if commit_hours is None
                else int(commit_hours)
            )

            return self._run_annual_simulation_lookahead(
                df=df,
                commit_hours=effective_commit_hours,
                lookahead_hours=int(lookahead_hours),
            )

        if period_hours <= 0:
            raise ValueError("period_hours must be > 0")

        if (
            "hour" not in df.columns
            or "demand_kw" not in df.columns
            or "pv_factor" not in df.columns
        ):
            raise ValueError(
                "Input df must contain "
                "['hour', 'demand_kw', 'pv_factor']"
            )

        df = df.copy().reset_index(drop=True)

        # -----------------------------------------------------
        # Estado inicial comum da bateria
        # -----------------------------------------------------
        bat = (
            self.usable_battery_kwh
            * self.battery_soc_init_fraction
        )

        # -----------------------------------------------------
        # Estado inicial da tecnologia despachavel
        # -----------------------------------------------------
        if self.route == "hydrogen":
            dispatchable_state = (
                self.h2_tank_kg
                * self.h2_soc_init_fraction
            )

        elif self.route == "biomethane":
            dispatchable_state = (
                self.biomethane_storage_nm3
                * self.biomethane_soc_init_fraction
            )

        elif self.route == "biogas":
            dispatchable_state = (
                self.biogas_storage_nm3
                * self.biogas_soc_init_fraction
            )

        annual_initial_dispatchable_state = dispatchable_state

        peak = 0.0

        # Monthly demand-charge state (M1b)
        monthly_peak = 0.0
        current_billing_month: int | None = None

        out: list[pd.DataFrame] = []
        total_objective = 0.0
        total_solve_time = 0.0

        for i in range(0, len(df), period_hours):
            part = df.iloc[i : i + period_hours].copy()
            part["t_global"] = range(i, i + len(part))

            if self.demand_charge_in_milp_objective:
                start_month = self._billing_month_from_global_hour(i)
                end_month = self._billing_month_from_global_hour(
                    i + len(part) - 1
                )

                if start_month != end_month:
                    raise ValueError(
                        "A rolling-horizon block crosses a billing-month "
                        "boundary. Use period_hours aligned with billing days."
                    )

                if current_billing_month != start_month:
                    current_billing_month = start_month
                    monthly_peak = 0.0

            terminal_target = None

            if (
                self.route == "biogas"
                and self.biogas_terminal_cyclic
                and i + len(part) == len(df)
            ):
                terminal_target = (
                    annual_initial_dispatchable_state
                )

            result = self.solve_period(
                part,
                bat,
                dispatchable_state,
                peak,
                monthly_peak,
                terminal_target,
            )

            total_solve_time += result.solve_time_sec

            # -------------------------------------------------
            # Caso inviavel / sem despacho
            # -------------------------------------------------
            if result.dispatch_df.empty:

                if self.route == "hydrogen":
                    final_h2_kg = dispatchable_state
                    final_biomethane_nm3 = 0.0
                    final_biogas_nm3 = 0.0

                elif self.route == "biomethane":
                    final_h2_kg = 0.0
                    final_biomethane_nm3 = dispatchable_state
                    final_biogas_nm3 = 0.0

                else:  # biogas
                    final_h2_kg = 0.0
                    final_biomethane_nm3 = 0.0
                    final_biogas_nm3 = dispatchable_state

                return DispatchResult(
                    dispatch_df=pd.DataFrame(),
                    final_battery_kwh=bat,
                    final_h2_kg=final_h2_kg,
                    objective_value=total_objective,
                    solver_status=result.solver_status,
                    solve_time_sec=total_solve_time,
                    milp_gap=None,
                    final_biomethane_nm3=final_biomethane_nm3,
                    final_biogas_nm3=final_biogas_nm3,
                )

            # -------------------------------------------------
            # Atualizacao dos estados comuns
            # -------------------------------------------------
            block_grid_peak = float(
                result.dispatch_df["p_grid_kw"].max()
            )

            peak = max(
                peak,
                block_grid_peak,
            )

            if self.demand_charge_in_milp_objective:
                monthly_peak = max(
                    monthly_peak,
                    block_grid_peak,
                )

            bat = result.final_battery_kwh

            # -------------------------------------------------
            # Atualizacao do estado despachavel por rota
            # -------------------------------------------------
            if self.route == "hydrogen":
                dispatchable_state = result.final_h2_kg

            elif self.route == "biomethane":
                dispatchable_state = (
                    result.final_biomethane_nm3
                )

            elif self.route == "biogas":
                dispatchable_state = (
                    result.final_biogas_nm3
                )

            total_objective += result.objective_value
            out.append(result.dispatch_df)

        # -----------------------------------------------------
        # Estado final por rota
        # -----------------------------------------------------
        if self.route == "hydrogen":
            final_h2_kg = dispatchable_state
            final_biomethane_nm3 = 0.0
            final_biogas_nm3 = 0.0

        elif self.route == "biomethane":
            final_h2_kg = 0.0
            final_biomethane_nm3 = dispatchable_state
            final_biogas_nm3 = 0.0

        else:  # biogas
            final_h2_kg = 0.0
            final_biomethane_nm3 = 0.0
            final_biogas_nm3 = dispatchable_state

        return DispatchResult(
            dispatch_df=pd.concat(
                out,
                ignore_index=True,
            ),
            final_battery_kwh=bat,
            final_h2_kg=final_h2_kg,
            objective_value=total_objective,
            solver_status="OPTIMAL",
            solve_time_sec=total_solve_time,
            milp_gap=None,
            final_biomethane_nm3=final_biomethane_nm3,
            final_biogas_nm3=final_biogas_nm3,
        )
    def _run_annual_simulation_lookahead(
        self,
        df: pd.DataFrame,
        commit_hours: int,
        lookahead_hours: int,
    ) -> DispatchResult:
        """
        Annual receding-horizon simulation with optional look-ahead.

        At each iteration, the optimizer solves:

            commit_hours + lookahead_hours

        but only the first ``commit_hours`` are committed to the
        annual dispatch. The look-ahead portion is discarded and
        re-optimized at the next iteration.

        State propagation therefore uses the state at the end of
        the committed interval, not the state at the end of the
        complete optimization window.

        This method is intentionally separate from the historical
        ``run_annual_simulation`` implementation so that the legacy
        rolling-horizon behavior remains unchanged.
        """

        # -----------------------------------------------------
        # VALIDATION
        # -----------------------------------------------------
        if commit_hours <= 0:
            raise ValueError("commit_hours must be > 0")

        if lookahead_hours < 0:
            raise ValueError("lookahead_hours must be >= 0")

        if len(df) == 0:
            raise ValueError("df must not be empty")

        if "hour" not in df.columns:
            raise ValueError("df must contain column 'hour'")

        # -----------------------------------------------------
        # INITIAL BATTERY STATE
        # -----------------------------------------------------
        bat = (
            self.usable_battery_kwh
            * self.battery_soc_init_fraction
        )

        # -----------------------------------------------------
        # INITIAL DISPATCHABLE STATE
        # -----------------------------------------------------
        if self.route == "hydrogen":
            dispatchable_state = (
                self.h2_tank_kg
                * self.h2_soc_init_fraction
            )

        elif self.route == "biomethane":
            dispatchable_state = (
                self.biomethane_storage_nm3
                * self.biomethane_soc_init_fraction
            )

        elif self.route == "biogas":
            dispatchable_state = (
                self.biogas_storage_nm3
                * self.biogas_soc_init_fraction
            )

        else:
            raise ValueError(
                f"Unsupported system.route: {self.route!r}"
            )

        annual_initial_dispatchable_state = (
            dispatchable_state
        )

        # -----------------------------------------------------
        # ACCUMULATORS
        # -----------------------------------------------------
        out: list[pd.DataFrame] = []

        total_objective = 0.0
        total_solve_time = 0.0

        peak = 0.0
        monthly_peak = 0.0
        current_billing_month: int | None = None

        n_hours = len(df)

        # -----------------------------------------------------
        # RECEDING HORIZON
        # -----------------------------------------------------
        for start in range(
            0,
            n_hours,
            commit_hours,
        ):
            commit_end = min(
                start + commit_hours,
                n_hours,
            )

            optimization_end = min(
                commit_end + lookahead_hours,
                n_hours,
            )

            # -------------------------------------------------
            # M1b MONTH-BOUNDARY CLIPPING
            #
            # solve_period() currently contains only one
            # P_peak_month variable. Therefore a single
            # optimization window must not contain hours from
            # two different billing months.
            #
            # The committed interval remains unchanged. Only
            # the optional look-ahead is shortened when it
            # would cross a billing-month boundary.
            # -------------------------------------------------
            if (
                self.demand_charge_in_milp_objective
                and optimization_end > commit_end
            ):
                commit_month = (
                    self._billing_month_from_global_hour(
                        start
                    )
                )

                clipped_end = optimization_end

                for global_hour in range(
                    commit_end,
                    optimization_end,
                ):
                    lookahead_month = (
                        self._billing_month_from_global_hour(
                            global_hour
                        )
                    )

                    if lookahead_month != commit_month:
                        clipped_end = global_hour
                        break

                optimization_end = clipped_end

            # -------------------------------------------------
            # Optimization window
            # -------------------------------------------------
            part = df.iloc[
                start:optimization_end
            ].copy()

            part["t_global"] = range(
                start,
                optimization_end,
            )

            committed_length = (
                commit_end - start
            )

            # -------------------------------------------------
            # BILLING MONTH â€” M1b
            #
            # The committed interval determines the billing
            # state that is propagated.
            # -------------------------------------------------
            if self.demand_charge_in_milp_objective:
                start_month = (
                    self._billing_month_from_global_hour(
                        start
                    )
                )

                commit_end_month = (
                    self._billing_month_from_global_hour(
                        commit_end - 1
                    )
                )

                if start_month != commit_end_month:
                    raise ValueError(
                        "A committed rolling-horizon block "
                        "crosses a billing-month boundary. "
                        "Use commit_hours aligned with billing days."
                    )

                if current_billing_month != start_month:
                    current_billing_month = start_month
                    monthly_peak = 0.0

            # -------------------------------------------------
            # TERMINAL TARGET
            #
            # Only impose the annual cyclic dispatchable-state
            # target when the optimization window actually
            # reaches the end of the annual horizon.
            # -------------------------------------------------
            terminal_target = None

            if (
                self.route in {"biomethane", "biogas"}
                and optimization_end == n_hours
            ):
                terminal_target = (
                    annual_initial_dispatchable_state
                )

            # -------------------------------------------------
            # SOLVE COMPLETE COMMIT + LOOK-AHEAD WINDOW
            # -------------------------------------------------
            result = self.solve_period(
                part,
                bat,
                dispatchable_state,
                peak,
                monthly_peak,
                terminal_target,
            )

            total_solve_time += (
                result.solve_time_sec
            )

            # -------------------------------------------------
            # INFEASIBLE / EMPTY RESULT
            # -------------------------------------------------
            if result.dispatch_df.empty:

                if self.route == "hydrogen":
                    final_h2_kg = (
                        dispatchable_state
                    )
                    final_biomethane_nm3 = 0.0
                    final_biogas_nm3 = 0.0

                elif self.route == "biomethane":
                    final_h2_kg = 0.0
                    final_biomethane_nm3 = (
                        dispatchable_state
                    )
                    final_biogas_nm3 = 0.0

                else:  # biogas
                    final_h2_kg = 0.0
                    final_biomethane_nm3 = 0.0
                    final_biogas_nm3 = (
                        dispatchable_state
                    )

                return DispatchResult(
                    dispatch_df=pd.DataFrame(),
                    final_battery_kwh=bat,
                    final_h2_kg=final_h2_kg,
                    objective_value=total_objective,
                    solver_status=result.solver_status,
                    solve_time_sec=total_solve_time,
                    milp_gap=None,
                    final_biomethane_nm3=(
                        final_biomethane_nm3
                    ),
                    final_biogas_nm3=(
                        final_biogas_nm3
                    ),
                )

            # -------------------------------------------------
            # KEEP ONLY COMMITTED HOURS
            # -------------------------------------------------
            committed = (
                result.dispatch_df
                .iloc[:committed_length]
                .copy()
            )

            if len(committed) != committed_length:
                raise RuntimeError(
                    "Solver returned fewer rows than the "
                    "requested committed interval."
                )

            # -------------------------------------------------
            # GRID PEAK PROPAGATION
            #
            # Only realized/committed dispatch may update the
            # historical annual and monthly peaks.
            # -------------------------------------------------
            block_grid_peak = float(
                committed["p_grid_kw"].max()
            )

            peak = max(
                peak,
                block_grid_peak,
            )

            if self.demand_charge_in_milp_objective:
                monthly_peak = max(
                    monthly_peak,
                    block_grid_peak,
                )

            # -------------------------------------------------
            # PROPAGATE BATTERY STATE AT END OF COMMIT
            # -------------------------------------------------
            bat = float(
                committed.iloc[-1][
                    "soc_bat_kwh"
                ]
            )

            # -------------------------------------------------
            # PROPAGATE DISPATCHABLE STATE AT END OF COMMIT
            # -------------------------------------------------
            if self.route == "hydrogen":
                dispatchable_state = float(
                    committed.iloc[-1][
                        "h2_level_kg"
                    ]
                )

            elif self.route == "biomethane":
                dispatchable_state = float(
                    committed.iloc[-1][
                        "biomethane_level_nm3"
                    ]
                )

            elif self.route == "biogas":
                dispatchable_state = float(
                    committed.iloc[-1][
                        "biogas_level_nm3"
                    ]
                )

            # -------------------------------------------------
            # OBJECTIVE ACCOUNTING
            #
            # The optimization objective contains look-ahead
            # costs and therefore must not be accumulated
            # directly as realized annual cost.
            #
            # Keep this accumulator neutral in M2. Economic
            # replay must be calculated ex post from committed
            # dispatch only.
            # -------------------------------------------------
            total_objective += 0.0

            # -------------------------------------------------
            # STORE ONLY REALIZED / COMMITTED DISPATCH
            # -------------------------------------------------
            out.append(committed)

        # -----------------------------------------------------
        # FINAL ROUTE STATE
        # -----------------------------------------------------
        if self.route == "hydrogen":
            final_h2_kg = dispatchable_state
            final_biomethane_nm3 = 0.0
            final_biogas_nm3 = 0.0

        elif self.route == "biomethane":
            final_h2_kg = 0.0
            final_biomethane_nm3 = (
                dispatchable_state
            )
            final_biogas_nm3 = 0.0

        else:  # biogas
            final_h2_kg = 0.0
            final_biomethane_nm3 = 0.0
            final_biogas_nm3 = (
                dispatchable_state
            )

        dispatch_df = pd.concat(
            out,
            ignore_index=True,
        )

        # -----------------------------------------------------
        # ANNUAL OUTPUT INTEGRITY
        # -----------------------------------------------------
        if len(dispatch_df) != n_hours:
            raise RuntimeError(
                "Look-ahead annual simulation produced "
                f"{len(dispatch_df)} committed rows; "
                f"expected {n_hours}."
            )

        return DispatchResult(
            dispatch_df=dispatch_df,
            final_battery_kwh=bat,
            final_h2_kg=final_h2_kg,
            objective_value=total_objective,
            solver_status="OPTIMAL",
            solve_time_sec=total_solve_time,
            milp_gap=None,
            final_biomethane_nm3=(
                final_biomethane_nm3
            ),
            final_biogas_nm3=(
                final_biogas_nm3
            ),
        )


