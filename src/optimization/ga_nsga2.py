from __future__ import annotations

import argparse
import copy
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------
# PATH
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.milp_dispatch import MILPDispatchOptimizer
from src.economics.lcoe import build_economics_summary
from src.metrics.peak_metrics import build_peak_metrics_summary


@dataclass
class Individual:
    capacities: Dict[str, float]
    objectives: Tuple[float, float] | None = None
    metrics: Dict[str, Any] | None = None
    rank: int | None = None
    crowding_distance: float = 0.0


class NSGA2Optimizer:
    """
    NSGA-II integrado ao pipeline MILP.

    Suporta dois regimes:
      1) regime antigo:
         - objetivo 1: lcoe_usd_kwh
         - objetivo 2: peak_grid_dependency_ratio ou grid_dependency_objective

      2) regime novo:
         - objetivo 1: lcoe_usd_kwh
         - objetivo 2: P_peak_grid_opt_kw

    O objetivo secundário é definido por:
      config["optimization"]["objective"][1]
    """

    def __init__(self, config: Dict[str, Any], df_h: pd.DataFrame):
        # --------------------------------------------------
        # ATRIBUIÇÕES BÁSICAS
        # --------------------------------------------------
        self.config = config

        # System route. Legacy configurations without system.route
        # remain hydrogen by default.
        self.route = str(
            self.config.get("system", {}).get("route", "hydrogen")
        ).strip().lower()

        if self.route not in {"hydrogen", "biomethane"}:
            raise ValueError(
                f"Unsupported system.route: {self.route!r}. "
                "Expected 'hydrogen' or 'biomethane'."
            )
        self.df_h = df_h

        # --------------------------------------------------
        # VALIDAÇÃO DO DATAFRAME
        # --------------------------------------------------
        required_cols = ["hour", "demand_kw", "pv_factor"]
        missing = [c for c in required_cols if c not in self.df_h.columns]
        if missing:
            raise ValueError(f"df_h inválido. Colunas ausentes: {missing}")

        # garantir ordenação e tipo numérico consistente
        self.df_h = self.df_h.copy()
        self.df_h["hour"] = pd.to_numeric(self.df_h["hour"], errors="coerce")
        if self.df_h["hour"].isna().any():
            raise ValueError("Coluna 'hour' contém valores inválidos")

        self.df_h = self.df_h.sort_values("hour").reset_index(drop=True)

        # --------------------------------------------------
        # REPRODUTIBILIDADE
        # --------------------------------------------------
        repro_cfg = self.config.get("reproducibility", {})
        seed = int(repro_cfg.get("seed", 42))

        random.seed(seed)
        np.random.seed(seed)

        # --------------------------------------------------
        # CONFIGURAÇÃO DO GA (COM FAIL-SAFE)
        # --------------------------------------------------
        ga_cfg = self.config.get("ga", {})

        self.population_size = int(ga_cfg.get("population", 20))
        self.generations = int(ga_cfg.get("generations", 10))
        self.crossover_prob = float(ga_cfg.get("crossover_prob", 0.9))
        self.mutation_prob = float(ga_cfg.get("mutation_prob", 0.1))
        self.tournament_size = int(ga_cfg.get("tournament_size", 2))
        self.integer_variables = bool(ga_cfg.get("integer_variables", True))

        if self.population_size <= 0:
            raise ValueError("ga.population must be > 0")
        if self.generations <= 0:
            raise ValueError("ga.generations must be > 0")
        if self.tournament_size <= 0:
            raise ValueError("ga.tournament_size must be > 0")

        # --------------------------------------------------
        # VARIÁVEIS DE DECISÃO
        # --------------------------------------------------
        if "decision_variables" not in ga_cfg:
            raise ValueError("config['ga']['decision_variables'] não definido")

        self.decision_variables = list(ga_cfg["decision_variables"])
        if len(self.decision_variables) == 0:
            raise ValueError("Lista de decision_variables não pode ser vazia")

        self.bounds = self._build_bounds()

        # --------------------------------------------------
        # DADOS TEMPORAIS
        # --------------------------------------------------
        data_cfg = self.config.get("data", {})

        self.dt_hours = float(data_cfg.get("timestep_hours", 1.0))
        self.evaluation_horizon = int(
            data_cfg.get("horizon_hours", len(self.df_h))
        )

        if self.dt_hours <= 0:
            raise ValueError("timestep_hours must be > 0")

        # --------------------------------------------------
        # OTIMIZAÇÃO
        # --------------------------------------------------
        opt_cfg = self.config.get("optimization", {})

        self.pareto_period_hours = int(opt_cfg.get("pareto_period_hours", 24))
        if self.pareto_period_hours <= 0:
            raise ValueError("optimization.pareto_period_hours must be > 0")

        self.total_grid_dependency_weight = float(
            opt_cfg.get("total_grid_dependency_weight", 0.15)
        )

        self.objective_list = list(
            opt_cfg.get(
                "objective",
                ["lcoe_usd_kwh", "peak_grid_dependency_ratio"]
            )
        )

        if len(self.objective_list) < 2:
            raise ValueError(
                "config['optimization']['objective'] deve conter pelo menos dois objetivos."
            )

        self.second_objective_name = self.objective_list[1]

        # --------------------------------------------------
        # CACHE DE AVALIAÇÃO
        # --------------------------------------------------
        self._evaluation_cache: Dict[
            Tuple[float, ...],
            Tuple[Tuple[float, float], Dict[str, Any]]
        ] = {}

    # -----------------------------------------------------------------
    # Helpers de configuração
    # -----------------------------------------------------------------
    def _build_bounds(self) -> Dict[str, Tuple[float, float]]:
        constraints = self.config["optimization"]["constraints"]
        bounds: Dict[str, Tuple[float, float]] = {}

        for var in self.decision_variables:
            if var not in constraints:
                raise KeyError(f"Variável '{var}' ausente em optimization.constraints.")
            bounds[var] = (
                float(constraints[var]["min"]),
                float(constraints[var]["max"]),
            )

        return bounds

    def _repair_numeric_bounds(self, capacities: Dict[str, float]) -> Dict[str, float]:
        repaired: Dict[str, float] = {}

        for var, value in capacities.items():
            low, high = self.bounds[var]
            v = max(low, min(high, float(value)))

            if self.integer_variables:
                v = int(round(v))

            repaired[var] = float(v)

        return repaired

    def _repair_logical_consistency(
        self,
        capacities: Dict[str, float],
    ) -> Dict[str, float]:
        cap = capacities.copy()

        if self.route == "hydrogen":
            if cap.get("h2_tank_kg", 0.0) <= 0.0:
                cap["electrolyzer_kw"] = 0.0
                cap["fuelcell_kw"] = 0.0

            if cap.get("electrolyzer_kw", 0.0) <= 0.0:
                cap["h2_tank_kg"] = 0.0
                cap["fuelcell_kw"] = 0.0

            if cap.get("fuelcell_kw", 0.0) <= 0.0:
                cap["h2_tank_kg"] = 0.0

        elif self.route == "biomethane":
            pass

        return cap

    def _repair(self, capacities: Dict[str, float]) -> Dict[str, float]:
        cap = self._repair_numeric_bounds(capacities)
        cap = self._repair_logical_consistency(cap)
        cap = self._repair_numeric_bounds(cap)
        return cap

    def _random_individual(self) -> Individual:
        capacities: Dict[str, float] = {}

        for var in self.decision_variables:
            low, high = self.bounds[var]

            if self.integer_variables:
                capacities[var] = float(random.randint(int(low), int(high)))
            else:
                capacities[var] = float(random.uniform(low, high))

        return Individual(capacities=self._repair(capacities))

    def _initialize_population(self) -> List[Individual]:
        warm_cfg = self.config.get("warm_start", {})
        enabled = bool(warm_cfg.get("enabled", False))

        if not enabled:
            return [
                self._random_individual()
                for _ in range(self.population_size)
            ]

        seeds = warm_cfg.get("seeds", [])

        if not isinstance(seeds, list):
            raise ValueError(
                "warm_start.seeds must be a list"
            )

        population: List[Individual] = []
        seen = set()

        for seed in seeds:
            if len(population) >= self.population_size:
                break

            if not isinstance(seed, dict):
                raise ValueError(
                    "Each warm_start seed must be a mapping"
                )

            missing = [
                var
                for var in self.decision_variables
                if var not in seed
            ]

            if missing:
                raise ValueError(
                    "Warm-start seed missing variables: "
                    + ", ".join(missing)
                )

            capacities = {
                var: float(seed[var])
                for var in self.decision_variables
            }

            capacities = self._repair(capacities)
            key = self._capacities_key(capacities)

            if key in seen:
                continue

            seen.add(key)

            population.append(
                Individual(
                    capacities=capacities,
                )
            )

        while len(population) < self.population_size:
            ind = self._random_individual()
            key = self._capacities_key(ind.capacities)

            if key in seen:
                continue

            seen.add(key)
            population.append(ind)

        return population
    def _capacities_key(self, capacities: Dict[str, float]) -> Tuple[float, ...]:
        return tuple(round(float(capacities[var]), 6) for var in self.decision_variables)

    def _build_reference_dispatch(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "hour": self.df_h["hour"].astype(int).values,
                "demand_kw": self.df_h["demand_kw"].astype(float).values,
                "p_grid_kw": self.df_h["demand_kw"].astype(float).values,
            }
        )

    # -----------------------------------------------------------------
    # Avaliação
    # -----------------------------------------------------------------
    def evaluate(self, individual: Individual) -> None:
        key = self._capacities_key(individual.capacities)

        if key in self._evaluation_cache:
            individual.objectives, individual.metrics = self._evaluation_cache[key]
            return

        optimizer = MILPDispatchOptimizer(
            config=self.config,
            capacities=individual.capacities,
            degradation_model=None,
        )

        result = optimizer.run_annual_simulation(
            df=self.df_h,
            period_hours=self.pareto_period_hours,
        )

        dispatch = result.dispatch_df

        # --------------------------------------------------
        # FAIL SAFE (MILP)
        # --------------------------------------------------

        if dispatch.empty or result.solver_status not in ("OPTIMAL", "FEASIBLE"):
            objectives = (1e9, 1e9)
            metrics = {
                "milp_status": result.solver_status,
                "solve_time_sec": float(result.solve_time_sec),
                "lcoe_usd_kwh": 1e9,
                "grid_opex_annual_usd": 1e9,
                "grid_peak_opex_annual_usd": 1e9,
                "peak_grid_dependency_ratio": 1e9,
                "peak_grid_dependency_percent": 1e9,
                "total_grid_dependency_ratio": 1e9,
                "grid_dependency_objective": 1e9,
                "P_peak_grid_opt_kw": 1e9,
                "E_grid_peak_opt_kwh": 1e9,
                "E_grid_peak_ref_kwh": 1e9,
                "R_peak_percent": -1.0,
                "E_grid_total_kwh": 1e9,
                "E_load_total_kwh": 1e9,
                "capex_total_usd": 1e9,
                "annualized_capex_usd": 1e9,
                "fixed_opex_annual_usd": 1e9,
                "variable_h2_opex_annual_usd": 1e9,
                "degradation_opex_annual_usd": 1e9,
            }

            individual.objectives = objectives
            individual.metrics = metrics
            self._evaluation_cache[key] = (objectives, metrics)
            return

        # --------------------------------------------------
        # ECONOMICS
        # --------------------------------------------------

        economics = build_economics_summary(
            config=self.config,
            capacities=individual.capacities,
            dispatch_df=dispatch,
        )

        # --------------------------------------------------
        # REFERÊNCIA
        # --------------------------------------------------

        df_ref = self._build_reference_dispatch()

        # --------------------------------------------------
        # PEAK METRICS
        # --------------------------------------------------

        peak_metrics = build_peak_metrics_summary(
            df_dispatch_opt=dispatch,
            df_dispatch_ref=df_ref,
            tariff_cfg=self.config["tariff"],
            economics_opt=economics,
            exchange_rate_brl_per_usd=None,
            reference_cost_brl_kwh=None,
            dt_hours=self.dt_hours,
        )

        # --------------------------------------------------
        # EXTRAÇÃO DE VARIÁVEIS
        # --------------------------------------------------

        lcoe = float(economics["lcoe_usd_kwh"])
        grid_opex = float(economics["grid_opex_annual_usd"])
        grid_peak_opex = float(economics["grid_peak_opex_annual_usd"])

        e_grid_peak_opt = float(peak_metrics["E_grid_peak_opt_kwh"])
        e_grid_peak_ref = float(peak_metrics["E_grid_peak_ref_kwh"])

        e_grid_total = float(economics.get("grid_total_energy_annual_kwh", 0.0))
        e_load_total = float(economics.get("energy_served_annual_kwh", 0.0))

        p_peak_grid_opt_kw = float(dispatch["p_grid_kw"].max())
        p_peak_grid_ref_kw = float(df_ref["p_grid_kw"].max())

        # --------------------------------------------------
        # DEPENDÊNCIAS
        # --------------------------------------------------

        if e_grid_peak_ref <= 0:
            peak_grid_dependency_ratio = 1e9
        else:
            peak_grid_dependency_ratio = e_grid_peak_opt / e_grid_peak_ref

        if e_load_total <= 0:
            total_grid_dependency_ratio = 1e9
        else:
            total_grid_dependency_ratio = e_grid_total / e_load_total

        peak_grid_dependency_percent = 100.0 * peak_grid_dependency_ratio

        # --------------------------------------------------
        # REDUÇÃO DE PICO (NOVA MÉTRICA CRÍTICA)
        # --------------------------------------------------

        if p_peak_grid_ref_kw > 0:
            peak_reduction_percent = (1.0 - (p_peak_grid_opt_kw / p_peak_grid_ref_kw)) * 100.0
        else:
            peak_reduction_percent = 0.0

        # --------------------------------------------------
        # OBJETIVO COMPOSTO
        # --------------------------------------------------

        grid_dependency_objective = (
                peak_grid_dependency_ratio
                + self.total_grid_dependency_weight * total_grid_dependency_ratio
        )

        # --------------------------------------------------
        # SEGUNDO OBJETIVO
        # --------------------------------------------------

        if self.second_objective_name == "P_peak_grid_opt_kw":
            second_obj = p_peak_grid_opt_kw
        elif self.second_objective_name == "grid_dependency_objective":
            second_obj = grid_dependency_objective
        else:
            second_obj = peak_grid_dependency_ratio

        objectives = (lcoe, second_obj)

        # --------------------------------------------------
        # MÉTRICAS COMPLETAS (CBPE READY)
        # --------------------------------------------------

        metrics = {
            "milp_status": result.solver_status,
            "solve_time_sec": float(result.solve_time_sec),

            # ECONOMICS
            "lcoe_usd_kwh": lcoe,
            "capex_total_usd": float(economics["capex_total_usd"]),
            "annualized_capex_usd": float(economics["annualized_capex_usd"]),
            "fixed_opex_annual_usd": float(economics["fixed_opex_annual_usd"]),
            "variable_h2_opex_annual_usd": float(
                economics.get("variable_h2_opex_annual_usd", 0.0)
            ),
            "biomethane_fuel_opex_annual_usd": float(
                economics.get("biomethane_fuel_opex_annual_usd", 0.0)
            ),
            "chp_variable_opex_annual_usd": float(
                economics.get("chp_variable_opex_annual_usd", 0.0)
            ),
            "variable_biomethane_opex_annual_usd": float(
                economics.get("variable_biomethane_opex_annual_usd", 0.0)
            ),
            "variable_dispatchable_opex_annual_usd": float(
                economics.get("variable_dispatchable_opex_annual_usd", 0.0)
            ),
            "degradation_opex_annual_usd": float(
                economics.get("degradation_opex_annual_usd", 0.0)
            ),

            # GRID COST
            "grid_opex_annual_usd": grid_opex,
            "grid_peak_opex_annual_usd": grid_peak_opex,

            # ENERGIA
            "E_grid_total_kwh": e_grid_total,
            "E_load_total_kwh": e_load_total,

            # PONTA (ENERGIA)
            "E_grid_peak_opt_kwh": e_grid_peak_opt,
            "E_grid_peak_ref_kwh": e_grid_peak_ref,
            "R_peak_percent": float(peak_metrics["R_peak_percent"]),

            # PONTA (POTÊNCIA)
            "P_peak_grid_opt_kw": p_peak_grid_opt_kw,
            "P_peak_grid_ref_kw": p_peak_grid_ref_kw,
            "Peak_reduction_percent": peak_reduction_percent,

            # DEPENDÊNCIA
            "peak_grid_dependency_ratio": peak_grid_dependency_ratio,
            "peak_grid_dependency_percent": peak_grid_dependency_percent,
            "total_grid_dependency_ratio": total_grid_dependency_ratio,
            "grid_dependency_objective": grid_dependency_objective,
        }

        # --------------------------------------------------
        # FINALIZAÇÃO
        # --------------------------------------------------

        individual.objectives = objectives
        individual.metrics = metrics
        self._evaluation_cache[key] = (objectives, metrics)

    # -----------------------------------------------------------------
    # Núcleo do NSGA-II
    # -----------------------------------------------------------------
    @staticmethod
    def dominates(a: Individual, b: Individual) -> bool:
        assert a.objectives is not None and b.objectives is not None
        return (
            all(x <= y for x, y in zip(a.objectives, b.objectives))
            and any(x < y for x, y in zip(a.objectives, b.objectives))
        )

    def fast_non_dominated_sort(self, population: List[Individual]) -> List[List[Individual]]:
        fronts: List[List[Individual]] = []
        domination_count: Dict[int, int] = {}
        dominated_solutions: Dict[int, List[Individual]] = {}

        first_front: List[Individual] = []

        for p in population:
            dominated_solutions[id(p)] = []
            domination_count[id(p)] = 0

            for q in population:
                if p is q:
                    continue

                if self.dominates(p, q):
                    dominated_solutions[id(p)].append(q)
                elif self.dominates(q, p):
                    domination_count[id(p)] += 1

            if domination_count[id(p)] == 0:
                p.rank = 0
                first_front.append(p)

        fronts.append(first_front)

        i = 0
        while i < len(fronts) and fronts[i]:
            next_front: List[Individual] = []

            for p in fronts[i]:
                for q in dominated_solutions[id(p)]:
                    domination_count[id(q)] -= 1
                    if domination_count[id(q)] == 0:
                        q.rank = i + 1
                        next_front.append(q)

            i += 1
            if next_front:
                fronts.append(next_front)

        return fronts

    @staticmethod
    def compute_crowding_distance(front: List[Individual]) -> None:
        if not front:
            return

        for ind in front:
            ind.crowding_distance = 0.0

        num_objectives = len(front[0].objectives)

        for m in range(num_objectives):
            front.sort(key=lambda ind: ind.objectives[m])

            front[0].crowding_distance = float("inf")
            front[-1].crowding_distance = float("inf")

            f_min = front[0].objectives[m]
            f_max = front[-1].objectives[m]

            if f_max == f_min:
                continue

            for i in range(1, len(front) - 1):
                prev_val = front[i - 1].objectives[m]
                next_val = front[i + 1].objectives[m]
                front[i].crowding_distance += (next_val - prev_val) / (f_max - f_min)

    def tournament_select(self, population: List[Individual]) -> Individual:
        contenders = random.sample(population, self.tournament_size)
        contenders.sort(
            key=lambda ind: (
                ind.rank if ind.rank is not None else 1e9,
                -ind.crowding_distance,
            )
        )
        return copy.deepcopy(contenders[0])

    def crossover(self, p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
        if random.random() > self.crossover_prob:
            return copy.deepcopy(p1), copy.deepcopy(p2)

        c1: Dict[str, float] = {}
        c2: Dict[str, float] = {}

        for var in self.decision_variables:
            alpha = random.random()
            v1 = p1.capacities[var]
            v2 = p2.capacities[var]

            c1[var] = alpha * v1 + (1.0 - alpha) * v2
            c2[var] = alpha * v2 + (1.0 - alpha) * v1

        return Individual(self._repair(c1)), Individual(self._repair(c2))

    def mutate(self, ind: Individual) -> Individual:
        child = copy.deepcopy(ind)

        for var in self.decision_variables:
            if random.random() < self.mutation_prob:
                low, high = self.bounds[var]
                span = high - low
                mutated = child.capacities[var] + random.gauss(0.0, 0.1 * span)
                child.capacities[var] = mutated

        child.capacities = self._repair(child.capacities)
        child.objectives = None
        child.metrics = None
        child.rank = None
        child.crowding_distance = 0.0
        return child

    def make_offspring(self, population: List[Individual]) -> List[Individual]:
        offspring: List[Individual] = []

        while len(offspring) < self.population_size:
            p1 = self.tournament_select(population)
            p2 = self.tournament_select(population)

            c1, c2 = self.crossover(p1, p2)
            c1 = self.mutate(c1)
            c2 = self.mutate(c2)

            offspring.append(c1)
            if len(offspring) < self.population_size:
                offspring.append(c2)

        return offspring

    def environmental_selection(self, combined: List[Individual]) -> List[Individual]:
        new_population: List[Individual] = []
        fronts = self.fast_non_dominated_sort(combined)

        for front in fronts:
            self.compute_crowding_distance(front)

            if len(new_population) + len(front) <= self.population_size:
                new_population.extend(front)
            else:
                front.sort(key=lambda ind: -ind.crowding_distance)
                remaining = self.population_size - len(new_population)
                new_population.extend(front[:remaining])
                break

        return new_population

    # -----------------------------------------------------------------
    # Loop principal
    # -----------------------------------------------------------------
    def run(self, verbose: bool = True) -> pd.DataFrame:
        population = self._initialize_population()
        for ind in population:
            self.evaluate(ind)

        fronts = self.fast_non_dominated_sort(population)
        for front in fronts:
            self.compute_crowding_distance(front)

        for gen in range(self.generations):
            offspring = self.make_offspring(population)
            for ind in offspring:
                self.evaluate(ind)

            combined = population + offspring
            population = self.environmental_selection(combined)

            if verbose:
                best_front = self.fast_non_dominated_sort(population)[0]
                lcoes = [ind.objectives[0] for ind in best_front]

                if self.second_objective_name == "P_peak_grid_opt_kw":
                    vals = [ind.metrics["P_peak_grid_opt_kw"] for ind in best_front if ind.metrics]
                    label = "Grid peak kW"
                    val_str = f"{min(vals):.2f}" if vals else "nan"
                elif self.second_objective_name == "grid_dependency_objective":
                    vals = [ind.metrics["grid_dependency_objective"] for ind in best_front if ind.metrics]
                    label = "Grid dependency objective"
                    val_str = f"{min(vals):.4f}" if vals else "nan"
                else:
                    vals = [ind.metrics["peak_grid_dependency_ratio"] for ind in best_front if ind.metrics]
                    label = "Peak dependency ratio"
                    val_str = f"{min(vals):.4f}" if vals else "nan"

                print(
                    f"Generation {gen + 1}/{self.generations} | "
                    f"Front0 size={len(best_front)} | "
                    f"LCOE min={min(lcoes):.6f} | "
                    f"{label} min={val_str}"
                )

        final_front = self.fast_non_dominated_sort(population)[0]
        self.compute_crowding_distance(final_front)

        rows: List[Dict[str, Any]] = []
        for i, ind in enumerate(final_front):
            row: Dict[str, Any] = {
                "solution_id": i,
                **ind.capacities,
                "lcoe_usd_kwh": float(ind.objectives[0]),
                "rank": ind.rank,
                "crowding_distance": ind.crowding_distance,
            }

            if ind.metrics:
                row.update(ind.metrics)

            if "grid_dependency_objective" not in row:
                row["grid_dependency_objective"] = float(ind.objectives[1])

            rows.append(row)

        pareto_df = pd.DataFrame(rows)

        if not pareto_df.empty:
            second_col = self.second_objective_name

            if second_col not in pareto_df.columns:
                if second_col == "P_peak_grid_opt_kw":
                    second_col = "P_peak_grid_opt_kw"
                elif second_col == "grid_dependency_objective":
                    second_col = "grid_dependency_objective"
                else:
                    second_col = "peak_grid_dependency_ratio"

            dedup_cols = self.decision_variables + [
                "lcoe_usd_kwh",
                second_col,
            ]
            dedup_cols = [c for c in dedup_cols if c in pareto_df.columns]

            # --------------------------------------------------
            # LIMPEZA E ORDENAÇÃO ORIGINAL
            # --------------------------------------------------

            pareto_df = pareto_df.drop_duplicates(subset=dedup_cols)
            pareto_df = (pareto_df.sort_values(
                by=[second_col, "lcoe_usd_kwh"]
            ).reset_index(drop=True))
            pareto_df["solution_id"] = range(len(pareto_df))

            # --------------------------------------------------
            # ENRIQUECIMENTO DE MÉTRICAS (CBPE READY)
            # --------------------------------------------------

            # --------------------------------------------------
            # ENRIQUECIMENTO DE MÉTRICAS (CBPE READY)
            # --------------------------------------------------

            from src.models.milp_dispatch import MILPDispatchOptimizer
            from src.economics.lcoe import build_economics_summary
            from src.metrics.peak_metrics import build_peak_metrics_summary

            enriched_rows = []

            print("\n[INFO] Enriquecendo Pareto com métricas completas...")

            for _, row in pareto_df.iterrows():

                capacities = {
                    var: float(row[var])
                    for var in self.decision_variables
                }

                # -----------------------------------
                # MILP (usar dados da classe, não df solto)
                # -----------------------------------

                optimizer = MILPDispatchOptimizer(
                    config=self.config,
                    capacities=capacities,
                    degradation_model=None,
                )

                result = optimizer.run_annual_simulation(
                    df=self.df_h,  # <<< CORREÇÃO CRÍTICA
                    period_hours=self.pareto_period_hours,  # <<< CONSISTÊNCIA COM GA
                )

                dispatch = result.dispatch_df

                # -----------------------------------
                # FAIL SAFE (evita quebrar Pareto)
                # -----------------------------------

                if dispatch.empty or result.solver_status not in ("OPTIMAL", "FEASIBLE"):
                    continue  # <<< não inclui solução inválida no Pareto enriquecido

                # -----------------------------------
                # REFERÊNCIA (consistente com resto do código)
                # -----------------------------------

                df_ref = pd.DataFrame({
                    "hour": dispatch["hour"],
                    "demand_kw": dispatch["demand_kw"],
                    "p_grid_kw": dispatch["demand_kw"],
                })

                # -----------------------------------
                # ECONOMICS
                # -----------------------------------

                economics = build_economics_summary(
                    config=self.config,
                    capacities=capacities,
                    dispatch_df=dispatch,
                )

                # -----------------------------------
                # PEAK METRICS
                # -----------------------------------

                peak_metrics = build_peak_metrics_summary(
                    df_dispatch_opt=dispatch,
                    df_dispatch_ref=df_ref,
                    tariff_cfg=self.config["tariff"],
                    economics_opt=economics,
                    dt_hours=self.dt_hours,
                )

                # -----------------------------------
                # POTÊNCIA
                # -----------------------------------

                P_opt = float(dispatch["p_grid_kw"].max())
                P_ref = float(df_ref["p_grid_kw"].max())

                if P_ref > 0:
                    peak_reduction = (1.0 - (P_opt / P_ref)) * 100.0
                else:
                    peak_reduction = 0.0

                # -----------------------------------
                # ENRIQUECER LINHA
                # -----------------------------------

                enriched = row.to_dict()

                enriched.update({
                    "P_peak_grid_ref_kw": P_ref,
                    "Peak_reduction_percent": peak_reduction,
                    "R_peak_percent": float(peak_metrics["R_peak_percent"]),
                    "peak_grid_dependency_percent": float(peak_metrics["peak_grid_dependency_percent"]),
                })

                enriched_rows.append(enriched)

            # -----------------------------------
            # RECONSTRUIR DATAFRAME
            # -----------------------------------

            if len(enriched_rows) == 0:
                raise RuntimeError("Nenhuma solução válida no Pareto após enriquecimento.")

            pareto_df = pd.DataFrame(enriched_rows)

            # --------------------------------------------------
            # RETORNO FINAL
            # --------------------------------------------------

            return pareto_df


# ---------------------------------------------------------------------
# Helpers executáveis
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run NSGA-II optimization for the CBPE energy planning pipeline."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file. If omitted, uses config/base.yaml",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"Arquivo YAML vazio: {path}")

    if not isinstance(data, dict):
        raise ValueError(f"Arquivo YAML inválido (esperado dicionário): {path}")

    return data


def ensure_hour_column(df: pd.DataFrame) -> pd.DataFrame:
    if "hour" not in df.columns:
        df = df.copy()
        df["hour"] = range(len(df))
    return df


def validate_input_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes no CSV de entrada: {missing}")


def resolve_runs_root(project_root: Path) -> Path:
    candidates = [
        project_root / "results" / "runs",
        project_root / "src" / "run" / "results" / "runs",
    ]
    for c in candidates:
        if c.exists():
            return c

    candidates[0].mkdir(parents=True, exist_ok=True)
    return candidates[0]


def infer_run_name_from_config(
    config_path: Path,
    route: str = "hydrogen",
) -> str:
    config_name = config_path.stem

    route = str(route).strip().lower()

    if route == "hydrogen":
        suffix = "with_h2"
    elif route == "biomethane":
        suffix = "with_biomethane"
    else:
        raise ValueError(
            f"Unsupported system.route: {route!r}"
        )

    if config_name.endswith("_no_penalty"):
        return config_name.replace(
            "_no_penalty",
            f"_{suffix}_no_penalty",
        )

    return f"{config_name}_{suffix}"


def build_output_dirs(
    project_root: Path,
    config_path: Path,
    route: str = "hydrogen",
) -> tuple[Path, Path]:
    runs_root = resolve_runs_root(project_root)

    scenario_name = infer_run_name_from_config(
        config_path,
        route=route,
    )

    scenario_dir = runs_root / scenario_name

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = scenario_dir / f"pareto_{timestamp}"
    latest_dir = scenario_dir / "pareto_latest"

    run_dir.mkdir(parents=True, exist_ok=True)
    latest_dir.mkdir(parents=True, exist_ok=True)

    return run_dir, latest_dir


def save_outputs(
    pareto_df: pd.DataFrame,
    cfg: Dict[str, Any],
    run_dir: Path,
    latest_dir: Path,
) -> None:
    run_csv = run_dir / "pareto.csv"
    latest_csv = latest_dir / "pareto.csv"

    pareto_df.to_csv(run_csv, index=False)
    pareto_df.to_csv(latest_csv, index=False)

    if cfg.get("reproducibility", {}).get("save_config_snapshot", False):
        snapshot_path = run_dir / "config_snapshot.yaml"
        with open(snapshot_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def main() -> None:
    args = parse_args()

    print("=" * 72)
    print("NSGA-II OPTIMIZATION - CBPE 2026")
    print("=" * 72)

    if args.config is not None:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path
    else:
        config_path = PROJECT_ROOT / "config" / "base.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {config_path}")

    cfg = load_yaml(config_path)

    data_path = Path(cfg["data"]["demand_profile_csv"])
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path

    if not data_path.exists():
        raise FileNotFoundError(f"Arquivo de dados não encontrado: {data_path}")

    print(f"Config: {config_path}")
    print(f"Data:   {data_path}")

    df_h = pd.read_csv(data_path)
    df_h = ensure_hour_column(df_h)
    validate_input_columns(df_h, cfg["data"]["required_columns"])

    df_h = df_h.sort_values("hour").reset_index(drop=True)
    df_h["t_global"] = range(len(df_h))

    print(f"Linhas: {len(df_h)}")

    optimizer = NSGA2Optimizer(cfg, df_h)

    print("\nExecutando GA...")
    pareto_df = optimizer.run(verbose=True)

    if pareto_df.empty:
        raise RuntimeError("Pareto vazio — todas as soluções ficaram inviáveis.")

    run_dir, latest_dir = build_output_dirs(
        PROJECT_ROOT,
        config_path,
        route=optimizer.route,
    )
    save_outputs(pareto_df, cfg, run_dir, latest_dir)

    print("\nArquivos gerados:")
    print(f"  {run_dir / 'pareto.csv'}")
    print(f"  {latest_dir / 'pareto.csv'}")

    print("\nResumo Pareto:")
    preview_cols = [c for c in ["solution_id", "lcoe_usd_kwh", "P_peak_grid_opt_kw", "peak_grid_dependency_ratio"] if c in pareto_df.columns]
    print(pareto_df[preview_cols].head())

    print("\n" + "=" * 72)
    print("OTIMIZAÇÃO FINALIZADA")
    print("=" * 72)


if __name__ == "__main__":
    main()