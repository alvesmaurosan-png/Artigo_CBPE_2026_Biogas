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

        if self.route not in {"hydrogen", "biomethane"}:
            raise ValueError(
                f"Unsupported system.route: {self.route!r}. "
                "Expected 'hydrogen' or 'biomethane'."
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
    ) -> None:

        solver.Add(v["P_peak"] >= prev_peak)

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
            objective.SetCoefficient(v["p_bat_dis"][t], self.battery_discharge_penalty_usd_kwh * dt)
            objective.SetCoefficient(v["unserved"][t], self.unserved_penalty_usd_kwh * dt)

        if self.global_peak_penalty_weight > 0:
            objective.SetCoefficient(v["P_peak"], self.global_peak_penalty_weight)

        objective.SetMinimization()

    # ---------------------------------------------------------

    def solve_period(
        self,
        df: pd.DataFrame,
        init_bat: float,
        init_dispatchable_state: float,
        prev_peak: float = 0,
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
            else:
                final_h2_kg = 0.0
                final_biomethane_nm3 = init_dispatchable_state

            return DispatchResult(
                dispatch_df=pd.DataFrame(),
                final_battery_kwh=init_bat,
                final_h2_kg=final_h2_kg,
                objective_value=0.0,
                solver_status=status_str,
                solve_time_sec=solve_time_sec,
                milp_gap=None,
                final_biomethane_nm3=final_biomethane_nm3,
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

            rows.append(row)

        final_battery_kwh = float(
            v["soc"][len(df) - 1].solution_value()
        )

        if self.route == "hydrogen":
            final_h2_kg = float(
                v["h2"][len(df) - 1].solution_value()
            )
            final_biomethane_nm3 = 0.0

        else:
            final_h2_kg = 0.0
            final_biomethane_nm3 = float(
                v["biomethane_level"][len(df) - 1].solution_value()
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
        )
    # ---------------------------------------------------------

    def run_annual_simulation(
        self,
        df: pd.DataFrame,
        period_hours: int = 168,
    ) -> DispatchResult:

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

        peak = 0.0
        out: list[pd.DataFrame] = []
        total_objective = 0.0
        total_solve_time = 0.0

        for i in range(0, len(df), period_hours):
            part = df.iloc[i : i + period_hours].copy()
            part["t_global"] = range(i, i + len(part))

            result = self.solve_period(
                part,
                bat,
                dispatchable_state,
                peak,
            )

            total_solve_time += result.solve_time_sec

            # -------------------------------------------------
            # Caso inviavel / sem despacho
            # -------------------------------------------------
            if result.dispatch_df.empty:

                if self.route == "hydrogen":
                    final_h2_kg = dispatchable_state
                    final_biomethane_nm3 = 0.0
                else:
                    final_h2_kg = 0.0
                    final_biomethane_nm3 = dispatchable_state

                return DispatchResult(
                    dispatch_df=pd.DataFrame(),
                    final_battery_kwh=bat,
                    final_h2_kg=final_h2_kg,
                    objective_value=total_objective,
                    solver_status=result.solver_status,
                    solve_time_sec=total_solve_time,
                    milp_gap=None,
                    final_biomethane_nm3=final_biomethane_nm3,
                )

            # -------------------------------------------------
            # Atualizacao dos estados comuns
            # -------------------------------------------------
            peak = max(
                peak,
                float(result.dispatch_df["p_grid_kw"].max()),
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

            total_objective += result.objective_value
            out.append(result.dispatch_df)

        # -----------------------------------------------------
        # Estado final por rota
        # -----------------------------------------------------
        if self.route == "hydrogen":
            final_h2_kg = dispatchable_state
            final_biomethane_nm3 = 0.0

        else:
            final_h2_kg = 0.0
            final_biomethane_nm3 = dispatchable_state

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
        )