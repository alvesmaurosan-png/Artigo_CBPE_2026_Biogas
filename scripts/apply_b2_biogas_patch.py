from pathlib import Path

path = Path("src/models/milp_dispatch.py")
text = path.read_text(encoding="utf-8")


def replace_once(old: str, new: str, label: str) -> None:
    global text
    assert old in text, f"{label}: anchor not found"
    text = text.replace(old, new, 1)


# ============================================================
# PATCH 1 — DispatchResult: estado final específico de biogás
# ============================================================

replace_once(
'''    milp_gap: Optional[float] = None
    final_biomethane_nm3: float = 0.0
''',
'''    milp_gap: Optional[float] = None
    final_biomethane_nm3: float = 0.0
    final_biogas_nm3: float = 0.0
''',
"PATCH 1",
)


# ============================================================
# PATCH 2 — aceitar a nova rota "biogas"
# ============================================================

replace_once(
'''        if self.route not in {"hydrogen", "biomethane"}:
            raise ValueError(
                f"Unsupported system.route: {self.route!r}. "
                "Expected 'hydrogen' or 'biomethane'."
            )
''',
'''        if self.route not in {"hydrogen", "biomethane", "biogas"}:
            raise ValueError(
                f"Unsupported system.route: {self.route!r}. "
                "Expected 'hydrogen', 'biomethane' or 'biogas'."
            )
''',
"PATCH 2",
)


# ============================================================
# PATCH 3 — capacidades B2
# ============================================================

replace_once(
'''        elif self.route == "biomethane":
            self.biomethane_storage_nm3 = float(
                capacities["biomethane_storage_nm3"]
            )
            self.chp_kw = float(capacities["chp_kw"])

        # ---------------------------------------------------------
        # BATERIA
''',
'''        elif self.route == "biomethane":
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
''',
"PATCH 3",
)


# ============================================================
# PATCH 4 — parâmetros tecnológicos B2
# ============================================================

anchor = '''            if self.biomethane_delivery_max_nm3_day < 0:
                raise ValueError(
                    "delivery.max_nm3_per_day must be >= 0"
                )

        # ---------------------------------------------------------
        # TARIFA
'''

new = '''            if self.biomethane_delivery_max_nm3_day < 0:
                raise ValueError(
                    "delivery.max_nm3_per_day must be >= 0"
                )

        elif self.route == "biogas":
            bg_cfg = self.config["technology"]["biogas"]
            bg_storage_cfg = self.config["technology"]["biogas_storage"]
            chp_bg_cfg = self.config["technology"]["chp_biogas"]

            # -----------------------------------------------------
            # Produção agregada de biogás
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
            # Gasômetro
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

            # Valor consolidado, mas sua tradução temporal ainda
            # NÃO é aplicada como derating contínuo.
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
'''

replace_once(anchor, new, "PATCH 4")


# ============================================================
# PATCH 5 — OPEX variável CHP B2 para decisão operacional
# ============================================================

replace_once(
'''            if self.chp_variable_cost_usd_kwh < 0:
                raise ValueError(
                    "economics.opex_variable_biomethane."
                    "chp_usd_per_kwh must be >= 0"
                )

        # penalidade leve de throughput da bateria para evitar cycling artificial
''',
'''            if self.chp_variable_cost_usd_kwh < 0:
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
''',
"PATCH 5",
)


# ============================================================
# PATCH 6 — variáveis específicas B2
#
# Anchor deliberately uses executable Python rather than
# comments to avoid encoding-dependent matching.
# ============================================================

replace_once(
'''        if self.demand_charge_in_milp_objective:
            variables["P_peak_month"] = solver.NumVar(
''',
'''        # -----------------------------------------------------
        # Rota biogas — B2
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
                    # Potência elétrica líquida entregue pela
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

                    # Consumo horário de biogás.
                    "biogas_use": [
                        solver.NumVar(
                            0,
                            solver.infinity(),
                            f"bg_use_{t}",
                        )
                        for t in range(T)
                    ],

                    # Estoque no gasômetro limitado diretamente
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
''',
"PATCH 6",
)
# ============================================================
# PATCH 7 — terminal target no método de restrições
# ============================================================

replace_once(
'''            prev_peak: float,
            prev_monthly_peak: float = 0.0,
    ) -> None:
''',
'''            prev_peak: float,
            prev_monthly_peak: float = 0.0,
            terminal_dispatchable_target: float | None = None,
    ) -> None:
''',
"PATCH 7",
)


# ============================================================
# PATCH 8 — balanço elétrico B2
# ============================================================

replace_once(
'''            elif self.route == "biomethane":
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
''',
'''            elif self.route == "biomethane":
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
''',
"PATCH 8",
)


# ============================================================
# PATCH 9 — dinâmica física B2 dentro do loop horário
# ============================================================

anchor = '''                solver.Add(
                    v["biomethane_level"][t]
                    >= (
                        self.biomethane_storage_nm3
                        * self.biomethane_soc_min_fraction
                    )
                )

        # =====================================================
        # RESTRICOES GLOBAIS POR ROTA
'''

new = '''                solver.Add(
                    v["biomethane_level"][t]
                    >= (
                        self.biomethane_storage_nm3
                        * self.biomethane_soc_min_fraction
                    )
                )

            # =================================================
            # ROTA BIOGAS — B2
            # =================================================
            elif self.route == "biogas":

                # ---------------------------------------------
                # CHP ligado/desligado + carga mínima
                #
                # chp_kw representa potência elétrica líquida
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
                # Biogás -> eletricidade líquida
                #
                # E_net =
                # V_BG * PCI_BG * eta_el * (1 - parasitic)
                #
                # Esta equação reproduz a convenção consolidada
                # da planilha B2-2026:
                # 196,812 kWh/t bruto -> 177,1308 kWh/t líquido.
                # ---------------------------------------------
                solver.Add(
                    v["biogas_use"][t]
                    * self.biogas_lhv_kwh_per_nm3
                    * self.biogas_chp_efficiency_el
                    * (1.0 - self.biogas_parasitic_fraction)
                    == v["p_chp_net"][t] * dt
                )

                # ---------------------------------------------
                # Produção contínua agregada de biogás
                # ---------------------------------------------
                biogas_prod = (
                    self.biogas_production_nm3_hour
                    * dt
                )

                # ---------------------------------------------
                # Dinâmica do gasômetro
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
'''

replace_once(anchor, new, "PATCH 9")


# ============================================================
# PATCH 10 — condição terminal anual B2
# ============================================================

replace_once(
'''            # A condicao terminal ciclica nao e aplicada aqui.
            # Ela deve ser imposta somente no fim do horizonte
            # anual, e nao em cada bloco do rolling horizon.
    # ---------------------------------------------------------
''',
'''            # A condicao terminal ciclica nao e aplicada aqui.
            # Ela deve ser imposta somente no fim do horizonte
            # anual, e nao em cada bloco do rolling horizon.

        elif self.route == "biogas":
            # Não existe entrega externa em B2.
            # O recurso entra exclusivamente pela produção
            # contínua derivada do substrato.

            if terminal_dispatchable_target is not None:
                solver.Add(
                    v["biogas_level"][T - 1]
                    == terminal_dispatchable_target
                )

    # ---------------------------------------------------------
''',
"PATCH 10",
)


# ============================================================
# PATCH 11 — OPEX variável do CHP B2 no objetivo
# ============================================================

replace_once(
'''            elif self.route == "biomethane":
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
''',
'''            elif self.route == "biomethane":
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
                # Como p_chp_net é líquido:
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
''',
"PATCH 11",
)


# ============================================================
# PATCH 12 — solve_period recebe terminal target
# ============================================================

replace_once(
'''        prev_peak: float = 0,
        prev_monthly_peak: float = 0.0,
    ) -> DispatchResult:
''',
'''        prev_peak: float = 0,
        prev_monthly_peak: float = 0.0,
        terminal_dispatchable_target: float | None = None,
    ) -> DispatchResult:
''',
"PATCH 12a",
)

replace_once(
'''            init_dispatchable_state,
            prev_peak,
            prev_monthly_peak,
        )
''',
'''            init_dispatchable_state,
            prev_peak,
            prev_monthly_peak,
            terminal_dispatchable_target,
        )
''',
"PATCH 12b",
)


# ============================================================
# PATCH 13 — erro de solver: estado final por rota
# ============================================================

replace_once(
'''            if self.route == "hydrogen":
                final_h2_kg = init_dispatchable_state
                final_biomethane_nm3 = 0.0
            else:
                final_h2_kg = 0.0
                final_biomethane_nm3 = init_dispatchable_state

            return DispatchResult(
''',
'''            if self.route == "hydrogen":
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
''',
"PATCH 13a",
)

replace_once(
'''                milp_gap=None,
                final_biomethane_nm3=final_biomethane_nm3,
            )
''',
'''                milp_gap=None,
                final_biomethane_nm3=final_biomethane_nm3,
                final_biogas_nm3=final_biogas_nm3,
            )
''',
"PATCH 13b",
)


# ============================================================
# PATCH 14 — colunas horárias B2 no dispatch
# ============================================================

replace_once(
'''            elif self.route == "biomethane":
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
''',
'''            elif self.route == "biomethane":
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
''',
"PATCH 14",
)


# ============================================================
# PATCH 15 — estado final solve_period por rota
# ============================================================

replace_once(
'''        if self.route == "hydrogen":
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
''',
'''        if self.route == "hydrogen":
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
''',
"PATCH 15a",
)

replace_once(
'''            milp_gap=None,
            final_biomethane_nm3=final_biomethane_nm3,
        )
''',
'''            milp_gap=None,
            final_biomethane_nm3=final_biomethane_nm3,
            final_biogas_nm3=final_biogas_nm3,
        )
''',
"PATCH 15b",
)


# ============================================================
# PATCH 16 — estado inicial anual B2
# ============================================================

replace_once(
'''        elif self.route == "biomethane":
            dispatchable_state = (
                self.biomethane_storage_nm3
                * self.biomethane_soc_init_fraction
            )

        peak = 0.0
''',
'''        elif self.route == "biomethane":
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
''',
"PATCH 16",
)


# ============================================================
# PATCH 17 — target terminal somente no último bloco anual
# ============================================================

replace_once(
'''            result = self.solve_period(
                part,
                bat,
                dispatchable_state,
                peak,
                monthly_peak,
            )
''',
'''            terminal_target = None

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
''',
"PATCH 17",
)


# ============================================================
# PATCH 18 — retorno inviável anual por rota
# ============================================================

replace_once(
'''                if self.route == "hydrogen":
                    final_h2_kg = dispatchable_state
                    final_biomethane_nm3 = 0.0
                else:
                    final_h2_kg = 0.0
                    final_biomethane_nm3 = dispatchable_state

                return DispatchResult(
''',
'''                if self.route == "hydrogen":
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
''',
"PATCH 18a",
)

# This occurrence belongs to run_annual_simulation.
replace_once(
'''                    milp_gap=None,
                    final_biomethane_nm3=final_biomethane_nm3,
                )
''',
'''                    milp_gap=None,
                    final_biomethane_nm3=final_biomethane_nm3,
                    final_biogas_nm3=final_biogas_nm3,
                )
''',
"PATCH 18b",
)


# ============================================================
# PATCH 19 — propagação do gasômetro entre blocos
# ============================================================

replace_once(
'''            elif self.route == "biomethane":
                dispatchable_state = (
                    result.final_biomethane_nm3
                )

            total_objective += result.objective_value
''',
'''            elif self.route == "biomethane":
                dispatchable_state = (
                    result.final_biomethane_nm3
                )

            elif self.route == "biogas":
                dispatchable_state = (
                    result.final_biogas_nm3
                )

            total_objective += result.objective_value
''',
"PATCH 19",
)


# ============================================================
# PATCH 20 — estado final anual por rota + DispatchResult
#
# Não usa replace_once(), porque existem vários DispatchResult
# estruturalmente semelhantes no arquivo. O bloco anual é
# localizado pela última ocorrência do estado final da rota e
# pelo solver_status="OPTIMAL".
# ============================================================

annual_state_marker = (
    '        if self.route == "hydrogen":\n'
    '            final_h2_kg = dispatchable_state\n'
)

annual_start = text.rfind(annual_state_marker)

assert annual_start >= 0, (
    "PATCH 20: annual final-state block not found"
)

annual_return_start = text.find(
    '        return DispatchResult(',
    annual_start,
)

assert annual_return_start >= 0, (
    "PATCH 20: annual DispatchResult start not found"
)

optimal_marker = (
    '            solver_status="OPTIMAL",'
)

optimal_pos = text.find(
    optimal_marker,
    annual_return_start,
)

assert optimal_pos >= 0, (
    "PATCH 20: annual OPTIMAL return not found"
)

annual_end = text.find(
    '\n        )',
    optimal_pos,
)

assert annual_end >= 0, (
    "PATCH 20: annual DispatchResult end not found"
)

annual_end += len('\n        )')


new_annual_block = '''        if self.route == "hydrogen":
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
        )'''


text = (
    text[:annual_start]
    + new_annual_block
    + text[annual_end:]
)


# ============================================================
# WRITE PATCHED FILE
# ============================================================

path.write_text(text, encoding="utf-8")

print("B2 biogas physical-route patch applied successfully.")