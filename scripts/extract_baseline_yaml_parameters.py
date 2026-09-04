from pathlib import Path

import pandas as pd
import yaml


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

CONFIG_DIR = ROOT / "configs" / "paper"

H2_CONFIG = (
    CONFIG_DIR
    / "pv_bsv_h2_1500.yaml"
)

B1_CONFIG = (
    CONFIG_DIR
    / "pv_bsv_biomethane_1500.yaml"
)

OUT_DIR = (
    ROOT
    / "results"
    / "paper"
    / "source_data"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUT_CSV = (
    OUT_DIR
    / "baseline_2026_yaml_parameters_extracted.csv"
)

OUT_TXT = (
    OUT_DIR
    / "baseline_2026_yaml_parameters_extracted.txt"
)


# ============================================================
# YAML LOAD
# ============================================================

def load_yaml(path: Path):

    if not path.exists():
        raise FileNotFoundError(
            f"Config not found: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(
            f"Invalid YAML root in {path}"
        )

    return data


h2 = load_yaml(H2_CONFIG)
b1 = load_yaml(B1_CONFIG)


# ============================================================
# HELPERS
# ============================================================

def get_nested(
    data,
    path,
    default=None,
):

    current = data

    for key in path.split("."):

        if not isinstance(
            current,
            dict,
        ):
            return default

        if key not in current:
            return default

        current = current[key]

    return current


def get_first_existing(
    data,
    paths,
):

    """
    Try several YAML paths and return:

        value,
        resolved_path

    The first existing path wins.
    """

    for path in paths:

        value = get_nested(
            data,
            path,
            default=None,
        )

        if value is not None:
            return value, path

    return None, None


def add(
    rows,
    id_,
    group,
    route,
    parameter,
    symbol,
    unit,
    yaml_paths,
    source_dict,
    classification="Parametro de entrada",
    status_if_found="A - Consolidado 2026",
    prospect_status="",
    notes="",
):

    if isinstance(
        yaml_paths,
        str,
    ):
        yaml_paths = [
            yaml_paths
        ]

    value, resolved_path = (
        get_first_existing(
            source_dict,
            yaml_paths,
        )
    )

    if value is not None:
        status = status_if_found
    else:
        status = (
            "B - Nao encontrado "
            "nos caminhos candidatos"
        )

    rows.append({

        "id":
            id_,

        "group":
            group,

        "route":
            route,

        "parameter":
            parameter,

        "symbol":
            symbol,

        "unit":
            unit,

        "value_2026":
            value,

        "classification":
            classification,

        "status":
            status,

        "prospective_status":
            prospect_status,

        "resolved_yaml_path":
            resolved_path,

        "candidate_yaml_paths":
            " | ".join(
                yaml_paths
            ),

        "notes":
            notes,
    })


def add_derived(
    rows,
    id_,
    group,
    route,
    parameter,
    symbol,
    unit,
    value,
    classification="Parametro derivado",
    status="A - Derivado do YAML 2026",
    prospect_status="",
    notes="",
):

    rows.append({

        "id":
            id_,

        "group":
            group,

        "route":
            route,

        "parameter":
            parameter,

        "symbol":
            symbol,

        "unit":
            unit,

        "value_2026":
            value,

        "classification":
            classification,

        "status":
            status,

        "prospective_status":
            prospect_status,

        "resolved_yaml_path":
            None,

        "candidate_yaml_paths":
            None,

        "notes":
            notes,
    })


rows = []


# ============================================================
# COMMON ECONOMICS / PV / BSV
# ============================================================

for route, cfg in [
    ("H2", h2),
    ("B1", b1),
]:

    prefix = (
        "H"
        if route == "H2"
        else "M"
    )


    # --------------------------------------------------------
    # ECONOMICS
    # --------------------------------------------------------

    add(
        rows,
        f"{prefix}-EC01",
        "Economia",
        route,
        "Taxa de desconto / WACC real",
        "r",
        "p.u.",
        [
            "economics.wacc_real",
            "economics.discount_rate",
        ],
        cfg,
        prospect_status=(
            "D - Driver prospectivo "
            "2030-2050"
        ),
    )

    add(
        rows,
        f"{prefix}-EC02",
        "Economia",
        route,
        "Horizonte economico",
        "n",
        "anos",
        [
            "economics.analysis_horizon_years",
            "economics.project_lifetime_years",
        ],
        cfg,
        prospect_status=(
            "Avaliar manutencao "
            "ou harmonizacao"
        ),
    )


    # --------------------------------------------------------
    # PV
    # --------------------------------------------------------

    add(
        rows,
        f"{prefix}-PV01",
        "PV",
        route,
        "CAPEX PV",
        "",
        "USD/kW",
        [
            "economics.capex.pv_usd_per_kw",
        ],
        cfg,
        prospect_status=(
            "D - Driver prospectivo central"
        ),
    )

    add(
        rows,
        f"{prefix}-PV02",
        "PV",
        route,
        "OPEX fixo PV",
        "",
        "fracao CAPEX/ano",
        [
            (
                "economics.opex_fixed."
                "pv_fraction_of_capex_per_year"
            ),
        ],
        cfg,
        prospect_status=(
            "D - Prospectivo se houver "
            "fonte diferenciada"
        ),
    )


    # --------------------------------------------------------
    # BSV - CAPEX DECOMPOSED
    # --------------------------------------------------------

    add(
        rows,
        f"{prefix}-B01A",
        "BSV",
        route,
        "CAPEX modulo BSV",
        "",
        "USD/kWh",
        [
            (
                "economics.capex."
                "bsv_module_usd_per_kwh"
            ),
        ],
        cfg,
        prospect_status=(
            "D - Driver prospectivo central"
        ),
    )

    add(
        rows,
        f"{prefix}-B01B",
        "BSV",
        route,
        "CAPEX repurpose BSV",
        "",
        "USD/kWh",
        [
            (
                "economics.capex."
                "bsv_repurpose_usd_per_kwh"
            ),
        ],
        cfg,
        prospect_status=(
            "D - Driver prospectivo / "
            "circularidade"
        ),
    )

    add(
        rows,
        f"{prefix}-B01C",
        "BSV",
        route,
        "CAPEX integracao BSV",
        "",
        "USD/kWh",
        [
            (
                "economics.capex."
                "bsv_integration_usd_per_kwh"
            ),
        ],
        cfg,
        prospect_status=(
            "D - Driver prospectivo"
        ),
    )


    module = get_nested(
        cfg,
        (
            "economics.capex."
            "bsv_module_usd_per_kwh"
        ),
    )

    repurpose = get_nested(
        cfg,
        (
            "economics.capex."
            "bsv_repurpose_usd_per_kwh"
        ),
    )

    integration = get_nested(
        cfg,
        (
            "economics.capex."
            "bsv_integration_usd_per_kwh"
        ),
    )


    if all(
        v is not None
        for v in [
            module,
            repurpose,
            integration,
        ]
    ):

        total_bsv_capex = (
            float(module)
            + float(repurpose)
            + float(integration)
        )

    else:
        total_bsv_capex = None


    add_derived(
        rows,
        f"{prefix}-B01D",
        "BSV",
        route,
        "CAPEX BSV total",
        "",
        "USD/kWh",
        total_bsv_capex,
        prospect_status=(
            "D - Resultado da soma dos "
            "componentes prospectivos"
        ),
        notes=(
            "Modulo + repurpose + integracao."
        ),
    )


    # --------------------------------------------------------
    # BSV OPEX / PHYSICAL
    # --------------------------------------------------------

    add(
        rows,
        f"{prefix}-B02",
        "BSV",
        route,
        "OPEX fixo BSV",
        "",
        "fracao CAPEX/ano",
        [
            (
                "economics.opex_fixed."
                "bsv_fraction_of_capex_per_year"
            ),
        ],
        cfg,
        prospect_status=(
            "D - Prospectivo se necessario"
        ),
    )

    add(
        rows,
        f"{prefix}-B03",
        "BSV",
        route,
        "SOC inicial BSV",
        "SOC0",
        "p.u.",
        [
            "technology.battery.soc_init_fraction",
            "technology.bsv.soc_init_fraction",
        ],
        cfg,
        prospect_status=(
            "Manter salvo sensibilidade"
        ),
    )

    add(
        rows,
        f"{prefix}-B04",
        "BSV",
        route,
        "SOC minimo BSV",
        "SOCmin",
        "p.u.",
        [
            "technology.battery.min_soc_fraction",
            "technology.bsv.min_soc_fraction",
        ],
        cfg,
        prospect_status=(
            "Manter salvo sensibilidade"
        ),
    )

    add(
        rows,
        f"{prefix}-B05",
        "BSV",
        route,
        "SOH inicial BSV",
        "SOH0",
        "p.u.",
        [
            "technology.battery.soh_initial",
            "technology.bsv.soh_initial",
        ],
        cfg,
        prospect_status=(
            "D - Pode evoluir "
            "tecnologicamente"
        ),
    )

    add(
        rows,
        f"{prefix}-B06",
        "BSV",
        route,
        "Janela SOC util",
        "DeltaSOC",
        "p.u.",
        [
            (
                "technology.battery."
                "usable_soc_window"
            ),
            (
                "technology.bsv."
                "usable_soc_window"
            ),
        ],
        cfg,
        prospect_status=(
            "D - Sensibilidade tecnologica"
        ),
    )

    add(
        rows,
        f"{prefix}-B07",
        "BSV",
        route,
        "RTE BSV",
        "eta_RT",
        "p.u.",
        [
            (
                "technology.battery."
                "battery_roundtrip"
            ),
            (
                "technology.bsv."
                "battery_roundtrip"
            ),
        ],
        cfg,
        prospect_status=(
            "D - Driver tecnologico"
        ),
    )

    add(
        rows,
        f"{prefix}-B08",
        "BSV",
        route,
        "C-rate maximo",
        "Cmax",
        "h^-1",
        [
            "technology.battery.c_rate_max",
            "technology.bsv.c_rate_max",
        ],
        cfg,
        prospect_status=(
            "D - Driver tecnologico"
        ),
    )


# ============================================================
# GRID / TARIFF
# ============================================================

for route, cfg in [
    ("H2", h2),
    ("B1", b1),
]:

    prefix = (
        "H"
        if route == "H2"
        else "M"
    )


    add(
        rows,
        f"{prefix}-G01",
        "Rede",
        route,
        "Modelo tarifario",
        "",
        "",
        [
            "tariff.pricing_model",
        ],
        cfg,
        prospect_status=(
            "D - Driver de politica "
            "tarifaria"
        ),
    )

    add(
        rows,
        f"{prefix}-G02",
        "Rede",
        route,
        "Tarifa fora de ponta",
        "c_off",
        "USD/kWh",
        [
            "tariff.offpeak_price_usd_kwh",
            "tariff.offpeak_usd_per_kwh",
        ],
        cfg,
        prospect_status=(
            "D - Driver prospectivo central"
        ),
    )

    add(
        rows,
        f"{prefix}-G03",
        "Rede",
        route,
        "Tarifa ponta",
        "c_peak",
        "USD/kWh",
        [
            "tariff.peak_price_usd_kwh",
            "tariff.peak_usd_per_kwh",
        ],
        cfg,
        prospect_status=(
            "D - Driver prospectivo central"
        ),
    )

    add(
        rows,
        f"{prefix}-G04",
        "Rede",
        route,
        "Inicio da ponta",
        "",
        "hora",
        [
            "tariff.peak_window.start_hour",
            "tariff.peak_start_hour",
        ],
        cfg,
        prospect_status=(
            "D - Politica tarifaria"
        ),
    )

    add(
        rows,
        f"{prefix}-G05",
        "Rede",
        route,
        "Fim da ponta",
        "",
        "hora",
        [
            "tariff.peak_window.end_hour",
            "tariff.peak_end_hour",
        ],
        cfg,
        prospect_status=(
            "D - Politica tarifaria"
        ),
    )

    add(
        rows,
        f"{prefix}-G06",
        "Rede",
        route,
        "Inclui demanda contratada",
        "",
        "boolean",
        [
            "tariff.include_demand_charge",
        ],
        cfg,
        prospect_status=(
            "D - Politica tarifaria"
        ),
    )

    add(
        rows,
        f"{prefix}-G07",
        "Rede",
        route,
        "Tarifa de demanda",
        "",
        "USD/kW.mes",
        [
            "tariff.demand_charge_usd_kw_month",
        ],
        cfg,
        prospect_status=(
            "D - Driver prospectivo"
        ),
    )


# ============================================================
# H2-SPECIFIC
# ============================================================

add(
    rows,
    "H01",
    "H2",
    "H2",
    "Eficiencia eletrolisador",
    "eta_ELZ",
    "p.u.",
    [
        "technology.efficiencies.electrolyzer",
        "technology.hydrogen.electrolyzer_efficiency",
        "technology.electrolyzer.efficiency",
        "technology.electrolyzer.eta",
        "technology.hydrogen.eta_elz",
    ],
    h2,
    prospect_status=(
        "D - Driver tecnologico central"
    ),
)

add(
    rows,
    "H02",
    "H2",
    "H2",
    "PCI H2",
    "PCI_H2",
    "kWh/kg",
    [
        "technology.hydrogen.lhv_kwh_per_kg",
        "technology.h2.lhv_kwh_per_kg",
        "technology.hydrogen.pci_kwh_per_kg",
    ],
    h2,
    prospect_status=(
        "Parametro fisico - manter"
    ),
)

add(
    rows,
    "H03",
    "H2",
    "H2",
    "Eficiencia fuel cell",
    "eta_FC",
    "p.u.",
    [
        "technology.efficiencies.fuelcell",
        "technology.hydrogen.fuelcell_efficiency",
        "technology.fuelcell.efficiency",
        "technology.fuel_cell.efficiency",
        "technology.hydrogen.eta_fc",
    ],
    h2,
    prospect_status=(
        "D - Driver tecnologico central"
    ),
)


# ------------------------------------------------------------
# H2 ECONOMICS
# ------------------------------------------------------------

add(
    rows,
    "H04",
    "H2",
    "H2",
    "CAPEX eletrolisador",
    "",
    "USD/kW",
    [
        (
            "economics.capex."
            "electrolyzer_usd_per_kw"
        ),
    ],
    h2,
    prospect_status=(
        "D - Driver prospectivo central"
    ),
)

add(
    rows,
    "H05",
    "H2",
    "H2",
    "CAPEX tanque H2",
    "",
    "USD/kg",
    [
        (
            "economics.capex."
            "h2_tank_usd_per_kg"
        ),
    ],
    h2,
    prospect_status=(
        "D - Driver prospectivo"
    ),
)

add(
    rows,
    "H06",
    "H2",
    "H2",
    "CAPEX fuel cell",
    "",
    "USD/kW",
    [
        (
            "economics.capex."
            "fuelcell_usd_per_kw"
        ),
    ],
    h2,
    prospect_status=(
        "D - Driver prospectivo central"
    ),
)

add(
    rows,
    "H07",
    "H2",
    "H2",
    "OPEX fixo eletrolisador",
    "",
    "fracao CAPEX/ano",
    [
        (
            "economics.opex_fixed."
            "electrolyzer_fraction_of_capex_per_year"
        ),
    ],
    h2,
    prospect_status=(
        "D - Prospectivo"
    ),
)

add(
    rows,
    "H08",
    "H2",
    "H2",
    "OPEX fixo fuel cell",
    "",
    "fracao CAPEX/ano",
    [
        (
            "economics.opex_fixed."
            "fuelcell_fraction_of_capex_per_year"
        ),
    ],
    h2,
    prospect_status=(
        "D - Prospectivo"
    ),
)

add(
    rows,
    "H09",
    "H2",
    "H2",
    "OPEX fixo tanque H2",
    "",
    "fracao CAPEX/ano",
    [
        (
            "economics.opex_fixed."
            "h2_tank_fraction_of_capex_per_year"
        ),
    ],
    h2,
    prospect_status=(
        "D - Prospectivo"
    ),
)


# ------------------------------------------------------------
# H2 OPERATION / CONSTRAINTS
# ------------------------------------------------------------

add(
    rows,
    "H10",
    "H2",
    "H2",
    "SOC inicial tanque H2",
    "",
    "p.u.",
    [
        (
            "technology.h2_storage."
            "soc_init_fraction"
        ),
    ],
    h2,
    prospect_status=(
        "Premissa operacional"
    ),
)

add(
    rows,
    "H11",
    "H2",
    "H2",
    "Limite eletrolisador",
    "",
    "kW",
    [
        (
            "optimization.constraints."
            "electrolyzer_kw.max"
        ),
    ],
    h2,
    classification="Restricao / espaco de decisao",
    prospect_status=(
        "D - Pode variar "
        "por cenario"
    ),
)

add(
    rows,
    "H12",
    "H2",
    "H2",
    "Limite tanque H2",
    "",
    "kg",
    [
        (
            "optimization.constraints."
            "h2_tank_kg.max"
        ),
    ],
    h2,
    classification="Restricao / espaco de decisao",
    prospect_status=(
        "D - Pode variar "
        "por cenario"
    ),
)

add(
    rows,
    "H13",
    "H2",
    "H2",
    "Limite fuel cell",
    "",
    "kW",
    [
        (
            "optimization.constraints."
            "fuelcell_kw.max"
        ),
    ],
    h2,
    classification="Restricao / espaco de decisao",
    prospect_status=(
        "D - Pode variar "
        "por cenario"
    ),
)


# ============================================================
# B1 - BIOMETHANE
# ============================================================

add(
    rows,
    "M01",
    "Biometano",
    "B1",
    "PCI biometano",
    "PCI_BM",
    "kWh/Nm3",
    [
        (
            "technology.biomethane."
            "lhv_kwh_per_nm3"
        ),
    ],
    b1,
    prospect_status=(
        "Parametro fisico - manter"
    ),
)

add(
    rows,
    "M02",
    "Biometano",
    "B1",
    "Fracao CH4",
    "x_CH4",
    "p.u.",
    [
        (
            "technology.biomethane."
            "methane_fraction"
        ),
    ],
    b1,
    prospect_status=(
        "Auditar qualidade / "
        "faixa futura"
    ),
)

add(
    rows,
    "M03",
    "Biometano",
    "B1",
    "Oferta maxima biometano",
    "V_sup_max",
    "Nm3/dia",
    [
        (
            "technology.biomethane."
            "max_supply_nm3_day"
        ),
    ],
    b1,
    classification="Restricao logistica",
    prospect_status=(
        "D - Driver prospectivo "
        "importante"
    ),
)

add(
    rows,
    "M04",
    "Biometano",
    "B1",
    "SOC inicial armazenamento BM",
    "SOC_BM0",
    "p.u.",
    [
        (
            "technology.biomethane_storage."
            "soc_init_fraction"
        ),
    ],
    b1,
    prospect_status=(
        "Premissa operacional"
    ),
)

add(
    rows,
    "M05",
    "Biometano",
    "B1",
    "SOC minimo armazenamento BM",
    "",
    "p.u.",
    [
        (
            "technology.biomethane_storage."
            "soc_min_fraction"
        ),
    ],
    b1,
    prospect_status=(
        "Premissa operacional"
    ),
)

add(
    rows,
    "M06",
    "Biometano",
    "B1",
    "Vida util armazenamento BM",
    "",
    "anos",
    [
        (
            "technology.biomethane_storage."
            "lifetime_years"
        ),
    ],
    b1,
    prospect_status=(
        "D - Revisar se houver "
        "fonte futura"
    ),
)


# ------------------------------------------------------------
# B1 ECONOMICS
# ------------------------------------------------------------

add(
    rows,
    "M07",
    "Biometano",
    "B1",
    "CAPEX armazenamento BM",
    "",
    "USD/Nm3",
    [
        (
            "economics.capex."
            "biomethane_storage_usd_per_nm3"
        ),
    ],
    b1,
    prospect_status=(
        "D - Driver prospectivo"
    ),
)

add(
    rows,
    "M08",
    "Biometano",
    "B1",
    "Preco biometano",
    "c_BM",
    "USD/Nm3",
    [
        (
            "economics."
            "opex_variable_biomethane."
            "biomethane_usd_per_nm3"
        ),
    ],
    b1,
    prospect_status=(
        "D - Driver prospectivo central"
    ),
)

add(
    rows,
    "M09",
    "Biometano",
    "B1",
    "Preco BM - baixo",
    "",
    "USD/Nm3",
    [
        (
            "economics."
            "price_sensitivity_biomethane."
            "low_usd_per_nm3"
        ),
    ],
    b1,
    classification="Sensibilidade baseline",
    prospect_status=(
        "Pode auxiliar S1-S3"
    ),
)

add(
    rows,
    "M10",
    "Biometano",
    "B1",
    "Preco BM - base",
    "",
    "USD/Nm3",
    [
        (
            "economics."
            "price_sensitivity_biomethane."
            "base_usd_per_nm3"
        ),
    ],
    b1,
    classification="Sensibilidade baseline",
    prospect_status=(
        "Referencia"
    ),
)

add(
    rows,
    "M11",
    "Biometano",
    "B1",
    "Preco BM - alto",
    "",
    "USD/Nm3",
    [
        (
            "economics."
            "price_sensitivity_biomethane."
            "high_usd_per_nm3"
        ),
    ],
    b1,
    classification="Sensibilidade baseline",
    prospect_status=(
        "Pode auxiliar S1-S3"
    ),
)


# ============================================================
# CHP
# ============================================================

add(
    rows,
    "C01-BM",
    "CHP",
    "B1",
    "Eficiencia eletrica CHP",
    "eta_el_CHP",
    "p.u.",
    [
        "technology.chp.eta_el",
        "technology.chp.electrical_efficiency",
    ],
    b1,
    prospect_status=(
        "D - Driver tecnologico"
    ),
)

add(
    rows,
    "C02-BM",
    "CHP",
    "B1",
    "CAPEX CHP",
    "",
    "USD/kW",
    [
        (
            "economics.capex."
            "chp_usd_per_kw"
        ),
    ],
    b1,
    prospect_status=(
        "D - Driver prospectivo central"
    ),
)

add(
    rows,
    "C03-BM",
    "CHP",
    "B1",
    "O&M variavel CHP",
    "",
    "USD/kWh",
    [
        (
            "economics."
            "opex_variable_biomethane."
            "chp_usd_per_kwh"
        ),
    ],
    b1,
    prospect_status=(
        "D - Prospectivo"
    ),
)

add(
    rows,
    "C04-BM",
    "CHP",
    "B1",
    "OPEX fixo CHP",
    "",
    "fracao CAPEX/ano",
    [
        (
            "economics.opex_fixed."
            "chp_fraction_of_capex_per_year"
        ),
    ],
    b1,
    prospect_status=(
        "C - Auditar fonte / "
        "intencionalidade"
    ),
    notes=(
        "Se valor 0.0 estiver no YAML, "
        "registrar como valor usado no baseline, "
        "mas auditar antes da prospectiva."
    ),
)

add(
    rows,
    "C05-BM",
    "CHP",
    "B1",
    "Limite CHP",
    "",
    "kW",
    [
        (
            "optimization.constraints."
            "chp_kw.max"
        ),
    ],
    b1,
    classification="Restricao / espaco de decisao",
    prospect_status=(
        "D - Pode variar "
        "por cenario"
    ),
)

add(
    rows,
    "C06-BM",
    "CHP",
    "B1",
    "Limite armazenamento BM",
    "",
    "Nm3",
    [
        (
            "optimization.constraints."
            "biomethane_storage_nm3.max"
        ),
    ],
    b1,
    classification="Restricao / espaco de decisao",
    prospect_status=(
        "D - Infraestrutura / "
        "logistica"
    ),
)


# ============================================================
# OPTIMIZATION / REPRODUCIBILITY
# ============================================================

for route, cfg in [
    ("H2", h2),
    ("B1", b1),
]:

    prefix = (
        "H"
        if route == "H2"
        else "M"
    )


    add(
        rows,
        f"{prefix}-O01",
        "Otimizacao",
        route,
        "Population",
        "",
        "individuos",
        [
            "optimization.ga.population",
            "optimization.population",
            "ga.population",
            "population",
        ],
        cfg,
        classification="Configuracao computacional",
        prospect_status=(
            "Nao e driver prospectivo"
        ),
    )

    add(
        rows,
        f"{prefix}-O02",
        "Otimizacao",
        route,
        "Generations",
        "",
        "geracoes",
        [
            "optimization.ga.generations",
            "optimization.generations",
            "ga.generations",
            "generations",
        ],
        cfg,
        classification="Configuracao computacional",
        prospect_status=(
            "Nao e driver prospectivo"
        ),
    )

    add(
        rows,
        f"{prefix}-O03",
        "Otimizacao",
        route,
        "Pareto period",
        "",
        "h",
        [
            "optimization.pareto_period_hours",
        ],
        cfg,
        classification="Configuracao computacional",
        prospect_status=(
            "Nao e driver prospectivo"
        ),
    )

    add(
        rows,
        f"{prefix}-O04",
        "Otimizacao",
        route,
        "Single-case period",
        "",
        "h",
        [
            "optimization.single_case_period_hours",
        ],
        cfg,
        classification="Configuracao computacional",
        prospect_status=(
            "Nao e driver prospectivo"
        ),
    )

    add(
        rows,
        f"{prefix}-O05",
        "Otimizacao",
        route,
        "Seed",
        "",
        "",
        [
            "reproducibility.seed",
            "optimization.seed",
            "seed",
        ],
        cfg,
        classification="Reprodutibilidade",
        prospect_status=(
            "Nao e driver prospectivo"
        ),
    )


# ============================================================
# BUILD DATAFRAME
# ============================================================

df = pd.DataFrame(rows)


# ============================================================
# CONSISTENCY CHECKS
# ============================================================

# Route field in YAML
route_h2 = get_nested(
    h2,
    "system.route",
)

route_b1 = get_nested(
    b1,
    "system.route",
)


if route_h2 is not None:

    normalized = str(
        route_h2
    ).strip().lower()

    if normalized not in [
        "hydrogen",
        "h2",
    ]:
        raise ValueError(
            "Unexpected H2 system.route: "
            f"{route_h2!r}"
        )


if route_b1 is not None:

    normalized = str(
        route_b1
    ).strip().lower()

    if normalized != "biomethane":
        raise ValueError(
            "Unexpected B1 system.route: "
            f"{route_b1!r}"
        )


# Detect duplicate IDs
duplicate_ids = (
    df[
        df["id"].duplicated(
            keep=False
        )
    ]
)

if not duplicate_ids.empty:
    raise RuntimeError(
        "Duplicate parameter IDs found:\n"
        + duplicate_ids[
            [
                "id",
                "route",
                "parameter",
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# SAVE CSV
# ============================================================

df.to_csv(
    OUT_CSV,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# SAVE HUMAN-READABLE TXT
# ============================================================

with open(
    OUT_TXT,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "BASELINE 2026 - PARAMETROS EXTRAIDOS DOS YAMLs\n"
    )

    f.write(
        "=" * 100
        + "\n"
    )

    f.write(
        f"H2_CONFIG: {H2_CONFIG}\n"
    )

    f.write(
        f"B1_CONFIG: {B1_CONFIG}\n"
    )

    f.write(
        "=" * 100
        + "\n\n"
    )


    for route in [
        "H2",
        "B1",
    ]:

        f.write(
            f"\n=== {route} ===\n\n"
        )

        subset = (
            df[
                df["route"] == route
            ]
            .copy()
        )

        display_cols = [
            "id",
            "group",
            "parameter",
            "unit",
            "value_2026",
            "status",
            "resolved_yaml_path",
            "prospective_status",
        ]

        f.write(
            subset[
                display_cols
            ].to_string(
                index=False
            )
        )

        f.write(
            "\n\n"
        )


    missing = df[
        df["value_2026"].isna()
    ]

    f.write(
        "\n=== NOT FOUND ===\n\n"
    )

    if missing.empty:

        f.write(
            "NONE\n"
        )

    else:

        f.write(
            missing[
                [
                    "route",
                    "parameter",
                    "candidate_yaml_paths",
                ]
            ].to_string(
                index=False
            )
        )

        f.write(
            "\n"
        )


# ============================================================
# CONSOLE REPORT
# ============================================================

print()

print(
    "=== YAML EXTRACTION COMPLETE ==="
)

print(
    "H2 config =",
    H2_CONFIG,
)

print(
    "B1 config =",
    B1_CONFIG,
)

print()

print(
    "rows =",
    len(df),
)


# ------------------------------------------------------------
# Resolved counts
# ------------------------------------------------------------

print()

print(
    "=== RESOLUTION SUMMARY ==="
)

for route in [
    "H2",
    "B1",
]:

    s = df[
        df["route"] == route
    ]

    found = int(
        s[
            "value_2026"
        ].notna().sum()
    )

    missing_count = int(
        s[
            "value_2026"
        ].isna().sum()
    )

    print(
        f"{route}: "
        f"found={found} "
        f"missing={missing_count}"
    )


# ------------------------------------------------------------
# Missing
# ------------------------------------------------------------

print()

print(
    "=== NOT FOUND ==="
)

missing = df[
    df["value_2026"].isna()
]


if missing.empty:

    print(
        "NONE"
    )

else:

    print(
        missing[
            [
                "route",
                "parameter",
                "candidate_yaml_paths",
            ]
        ].to_string(
            index=False
        )
    )


# ------------------------------------------------------------
# Important common parameters side by side
# ------------------------------------------------------------

print()

print(
    "=== COMMON PARAMETERS H2 x B1 ==="
)

common_names = [
    "Taxa de desconto / WACC real",
    "Horizonte economico",
    "CAPEX PV",
    "CAPEX modulo BSV",
    "CAPEX repurpose BSV",
    "CAPEX integracao BSV",
    "CAPEX BSV total",
    "OPEX fixo BSV",
    "SOH inicial BSV",
    "Janela SOC util",
    "RTE BSV",
    "C-rate maximo",
    "Tarifa fora de ponta",
    "Tarifa ponta",
    "Inicio da ponta",
    "Fim da ponta",
]


common = df[
    df["parameter"].isin(
        common_names
    )
][
    [
        "route",
        "parameter",
        "value_2026",
    ]
]


pivot = common.pivot_table(
    index="parameter",
    columns="route",
    values="value_2026",
    aggfunc="first",
)


print(
    pivot.to_string()
)


# ------------------------------------------------------------
# Files
# ------------------------------------------------------------

print()

print(
    "=== FILES ==="
)

print(
    OUT_CSV.relative_to(ROOT)
)

print(
    OUT_TXT.relative_to(ROOT)
)