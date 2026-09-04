from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    ROOT
    / "results"
    / "paper"
    / "source_data"
)

INPUT_XLSX = (
    DATA_DIR
    / "tabela_mestra_parametros_2026_2050.xlsx"
)

OUTPUT_CSV = (
    DATA_DIR
    / "tabela_mestra_parametros_2026_2050_v2.csv"
)

OUTPUT_XLSX = (
    DATA_DIR
    / "tabela_mestra_parametros_2026_2050_v2.xlsx"
)


# ============================================================
# LOAD V1
# ============================================================

if not INPUT_XLSX.exists():
    raise FileNotFoundError(INPUT_XLSX)

df = pd.read_excel(
    INPUT_XLSX,
    sheet_name="Tabela_Mestra",
)

print("=== INPUT V1 ===")
print("rows =", len(df))
print("columns =", len(df.columns))

if len(df) != 84:
    raise ValueError(
        f"Esperados 84 registros; encontrados {len(df)}"
    )


# ============================================================
# NORMALIZATION
# ============================================================

def text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def lower(value):
    return text(value).lower()


# ============================================================
# COMMON PARAMETERS
#
# Estes aparecem nos dois YAMLs, mas representam a mesma
# condição sistêmica e não devem divergir entre H2 e B1
# nos cenários prospectivos.
# ============================================================

COMMON_PARAMETERS = {
    "Taxa de desconto / WACC real",
    "Horizonte economico",

    "CAPEX PV",
    "OPEX fixo PV",

    "CAPEX modulo BSV",
    "CAPEX repurpose BSV",
    "CAPEX integracao BSV",
    "CAPEX BSV total",
    "OPEX fixo BSV",

    "SOC inicial BSV",
    "SOC minimo BSV",
    "SOH inicial BSV",
    "Janela SOC util",
    "RTE BSV",
    "C-rate maximo",

    "Modelo tarifario",
    "Tarifa fora de ponta",
    "Tarifa ponta",
    "Inicio da ponta",
    "Fim da ponta",
    "Inclui demanda contratada",
    "Tarifa de demanda",
}


# ============================================================
# EXPLICIT CLASSIFICATION
# ============================================================

F_PARAMETERS = {
    "PCI H2",
    "PCI biometano",
}

O_PARAMETERS = {
    "Horizonte economico",
    "SOC inicial BSV",
    "SOC minimo BSV",
    "SOC inicial tanque H2",
    "SOC inicial armazenamento BM",
    "SOC minimo armazenamento BM",
}

T_PARAMETERS = {
    "CAPEX PV",

    "CAPEX modulo BSV",
    "CAPEX repurpose BSV",
    "CAPEX integracao BSV",
    "SOH inicial BSV",
    "Janela SOC util",
    "RTE BSV",
    "C-rate maximo",

    "Eficiencia eletrolisador",
    "Eficiencia fuel cell",
    "CAPEX eletrolisador",
    "CAPEX tanque H2",
    "CAPEX fuel cell",

    "CAPEX armazenamento BM",
    "Vida util armazenamento BM",

    "Eficiencia eletrica CHP",
    "CAPEX CHP",
}

E_PARAMETERS = {
    "Taxa de desconto / WACC real",

    "OPEX fixo PV",
    "OPEX fixo BSV",

    "OPEX fixo eletrolisador",
    "OPEX fixo fuel cell",
    "OPEX fixo tanque H2",

    "Preco biometano",
    "Preco BM - baixo",
    "Preco BM - base",
    "Preco BM - alto",

    "O&M variavel CHP",
    "OPEX fixo CHP",

    "Tarifa fora de ponta",
    "Tarifa ponta",
    "Tarifa de demanda",
}

R_PARAMETERS = {
    "Modelo tarifario",
    "Inicio da ponta",
    "Fim da ponta",
    "Inclui demanda contratada",

    "Oferta maxima biometano",
    "Fracao CH4",
}

D_PARAMETERS = {
    "CAPEX BSV total",
}

L_PARAMETERS = {
    "Limite eletrolisador",
    "Limite tanque H2",
    "Limite fuel cell",
    "Limite CHP",
    "Limite armazenamento BM",
}

C_PARAMETERS = {
    "Population",
    "Generations",
    "Pareto period",
    "Single-case period",
    "Seed",
}


# ============================================================
# CLASSIFICATION FUNCTION
# ============================================================

def classify_model(row):

    p = text(row["Parametro"])

    if p in D_PARAMETERS:
        return "D"

    if p in C_PARAMETERS:
        return "C"

    if p in L_PARAMETERS:
        return "L"

    if p in F_PARAMETERS:
        return "F"

    if p in O_PARAMETERS:
        return "O"

    if p in T_PARAMETERS:
        return "T"

    if p in E_PARAMETERS:
        return "E"

    if p in R_PARAMETERS:
        return "R"

    # Fallback based on original classification
    original = lower(row["Classificacao"])

    if "computacional" in original:
        return "C"

    if "reprodutibilidade" in original:
        return "C"

    if "restricao" in original:
        return "L"

    if "derivado" in original:
        return "D"

    return "REVISAR"


# ============================================================
# SCOPE
# ============================================================

def classify_scope(row):

    p = text(row["Parametro"])

    if p in COMMON_PARAMETERS:
        return "COMMON"

    route = text(row["Rota"])

    if route == "H2":
        return "H2"

    if route == "B1":
        return "B1"

    return "REVISAR"


# ============================================================
# PROJECTION RULES
# ============================================================

# Core prospective drivers
PROJECT_PARAMETERS = {

    # Common
    "Taxa de desconto / WACC real",
    "CAPEX PV",
    "OPEX fixo PV",

    "CAPEX modulo BSV",
    "CAPEX repurpose BSV",
    "CAPEX integracao BSV",
    "OPEX fixo BSV",
    "SOH inicial BSV",
    "RTE BSV",

    "Tarifa fora de ponta",
    "Tarifa ponta",
    "Tarifa de demanda",

    # H2
    "Eficiencia eletrolisador",
    "Eficiencia fuel cell",
    "CAPEX eletrolisador",
    "CAPEX tanque H2",
    "CAPEX fuel cell",
    "OPEX fixo eletrolisador",
    "OPEX fixo fuel cell",
    "OPEX fixo tanque H2",

    # B1
    "Oferta maxima biometano",
    "CAPEX armazenamento BM",
    "Preco biometano",
    "Vida util armazenamento BM",

    "Eficiencia eletrica CHP",
    "CAPEX CHP",
    "O&M variavel CHP",
    "OPEX fixo CHP",
}


# Parameters projected through components rather than directly
DERIVED_PROJECT_PARAMETERS = {
    "CAPEX BSV total",
}


# Core S1/S2/S3 differentiators
SCENARIO_PARAMETERS = {

    "Taxa de desconto / WACC real",
    "CAPEX PV",

    "CAPEX modulo BSV",
    "CAPEX repurpose BSV",
    "CAPEX integracao BSV",
    "SOH inicial BSV",
    "RTE BSV",

    "Tarifa fora de ponta",
    "Tarifa ponta",
    "Tarifa de demanda",

    "Eficiencia eletrolisador",
    "Eficiencia fuel cell",
    "CAPEX eletrolisador",
    "CAPEX tanque H2",
    "CAPEX fuel cell",

    "Oferta maxima biometano",
    "CAPEX armazenamento BM",
    "Preco biometano",

    "Eficiencia eletrica CHP",
    "CAPEX CHP",
    "O&M variavel CHP",
}


# ============================================================
# SENSITIVITY
# ============================================================

SENSITIVITY_PARAMETERS = {

    "SOC inicial BSV",
    "SOC minimo BSV",
    "SOH inicial BSV",
    "Janela SOC util",
    "RTE BSV",
    "C-rate maximo",

    "Modelo tarifario",
    "Inicio da ponta",
    "Fim da ponta",
    "Inclui demanda contratada",

    "Fracao CH4",

    "Preco BM - baixo",
    "Preco BM - base",
    "Preco BM - alto",

    "Limite eletrolisador",
    "Limite tanque H2",
    "Limite fuel cell",
    "Limite CHP",
    "Limite armazenamento BM",
}


# ============================================================
# MODELING DECISION FUNCTIONS
# ============================================================

def project_flag(row):

    p = text(row["Parametro"])

    if p in DERIVED_PROJECT_PARAMETERS:
        return "Derivado"

    if p in PROJECT_PARAMETERS:
        return "Sim"

    return "Nao"


def scenario_flag(row):

    p = text(row["Parametro"])

    if p in SCENARIO_PARAMETERS:
        return "Sim"

    return "Nao"


def sensitivity_flag(row):

    p = text(row["Parametro"])

    return (
        "Sim"
        if p in SENSITIVITY_PARAMETERS
        else "Nao"
    )


def source_required(row):

    project = project_flag(row)

    if project == "Sim":
        return "Sim"

    if sensitivity_flag(row) == "Sim":
        return "Sim"

    return "Nao"


# ============================================================
# RESEARCH PRIORITY
# ============================================================

HIGH_PRIORITY = {
    "CAPEX PV",
    "CAPEX modulo BSV",
    "CAPEX repurpose BSV",
    "CAPEX integracao BSV",
    "SOH inicial BSV",
    "RTE BSV",

    "Eficiencia eletrolisador",
    "Eficiencia fuel cell",
    "CAPEX eletrolisador",
    "CAPEX tanque H2",
    "CAPEX fuel cell",

    "Oferta maxima biometano",
    "Preco biometano",
    "CAPEX armazenamento BM",
    "Eficiencia eletrica CHP",
    "CAPEX CHP",

    "Tarifa fora de ponta",
    "Tarifa ponta",
    "Tarifa de demanda",
    "Taxa de desconto / WACC real",
}


def research_priority(row):

    p = text(row["Parametro"])

    if p in HIGH_PRIORITY:
        return "Alta"

    if (
        project_flag(row) == "Sim"
        or sensitivity_flag(row) == "Sim"
    ):
        return "Media"

    return "Baixa"


# ============================================================
# JUSTIFICATION
# ============================================================

CLASS_DESCRIPTIONS = {
    "F": (
        "Constante fisica; nao projetar temporalmente "
        "salvo mudanca de especificacao do combustivel."
    ),
    "O": (
        "Premissa operacional mantida para comparabilidade; "
        "pode ser explorada em sensibilidade quando pertinente."
    ),
    "T": (
        "Driver tecnologico; pode evoluir com maturidade, "
        "learning e desempenho da tecnologia."
    ),
    "E": (
        "Driver economico ou de mercado; sujeito a "
        "trajetorias temporais e cenarios."
    ),
    "R": (
        "Parametro regulatorio, tarifario, logistico ou "
        "de infraestrutura."
    ),
    "D": (
        "Parametro derivado; nao deve ser projetado "
        "independentemente dos seus componentes."
    ),
    "C": (
        "Configuracao computacional ou de reprodutibilidade; "
        "nao representa evolucao prospectiva."
    ),
    "L": (
        "Limite do espaco de decisao; altera o dominio "
        "de busca, mas nao e driver prospectivo por si so."
    ),
    "REVISAR": (
        "Classificacao ainda nao determinada."
    ),
}


def justification(row):

    cls = classify_model(row)

    return CLASS_DESCRIPTIONS.get(
        cls,
        "Revisar classificacao."
    )


# ============================================================
# DERIVED FROM
# ============================================================

def derived_from(row):

    p = text(row["Parametro"])

    if p == "CAPEX BSV total":
        return (
            "CAPEX modulo BSV + "
            "CAPEX repurpose BSV + "
            "CAPEX integracao BSV"
        )

    return ""


# ============================================================
# APPLY V2 CURATION
# ============================================================

df["Classe_Modelagem"] = df.apply(
    classify_model,
    axis=1,
)

df["Escopo_Parametro"] = df.apply(
    classify_scope,
    axis=1,
)

df["Projetar_2030_2050"] = df.apply(
    project_flag,
    axis=1,
)

df["Variar_S1_S2_S3"] = df.apply(
    scenario_flag,
    axis=1,
)

df["Sensibilidade"] = df.apply(
    sensitivity_flag,
    axis=1,
)

df["Derivado_De"] = df.apply(
    derived_from,
    axis=1,
)

df["Fonte_Prospectiva_Requerida"] = df.apply(
    source_required,
    axis=1,
)

df["Prioridade_Pesquisa"] = df.apply(
    research_priority,
    axis=1,
)

df["Justificativa_Classificacao"] = df.apply(
    justification,
    axis=1,
)


# ============================================================
# SPECIAL NOTES
# ============================================================

df["Nota_Metodologica_V2"] = ""


def set_note(parameter, note):

    mask = (
        df["Parametro"]
        == parameter
    )

    df.loc[
        mask,
        "Nota_Metodologica_V2"
    ] = note


set_note(
    "CAPEX BSV total",
    (
        "Nao preencher diretamente nos cenarios; "
        "recalcular a partir dos tres componentes."
    ),
)

set_note(
    "Preco BM - baixo",
    (
        "Sensibilidade do baseline 2026; "
        "nao equivale automaticamente ao cenario S1."
    ),
)

set_note(
    "Preco BM - base",
    (
        "Referencia da sensibilidade 2026; "
        "nao equivale automaticamente ao cenario S2."
    ),
)

set_note(
    "Preco BM - alto",
    (
        "Sensibilidade do baseline 2026; "
        "nao equivale automaticamente ao cenario S3."
    ),
)

set_note(
    "OPEX fixo CHP",
    (
        "Valor 0.0 usado no baseline deve ser auditado "
        "antes da modelagem prospectiva."
    ),
)

for p in [
    "Modelo tarifario",
    "Inicio da ponta",
    "Fim da ponta",
    "Inclui demanda contratada",
]:
    set_note(
        p,
        (
            "Manter no cenario prospectivo principal; "
            "eventual mudanca estrutural deve ser tratada "
            "como analise especifica de politica."
        ),
    )

for p in [
    "Limite eletrolisador",
    "Limite tanque H2",
    "Limite fuel cell",
    "Limite CHP",
    "Limite armazenamento BM",
]:
    set_note(
        p,
        (
            "Limite do espaco de busca. "
            "Nao projetar automaticamente; ampliar apenas "
            "se houver justificativa fisica/metodologica."
        ),
    )


# ============================================================
# CHECK UNCLASSIFIED
# ============================================================

unclassified = df[
    df["Classe_Modelagem"]
    == "REVISAR"
]

if not unclassified.empty:

    print()
    print("=== ATENCAO: PARAMETROS A REVISAR ===")

    print(
        unclassified[
            [
                "ID",
                "Grupo",
                "Rota",
                "Parametro",
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# BUILD PROSPECTIVE MATRIX
# ============================================================

# Scenario columns already existing in V1
scenario_columns = [
    c
    for c in df.columns
    if (
        c.startswith("S1_")
        or c.startswith("S2_")
        or c.startswith("S3_")
    )
]


# ------------------------------------------------------------
# Validate common values across H2 and B1
# ------------------------------------------------------------

common_source = df[
    df["Escopo_Parametro"]
    == "COMMON"
].copy()


common_conflicts = []


for parameter, group in common_source.groupby(
    "Parametro"
):

    values = (
        group["Valor_2026"]
        .dropna()
        .astype(str)
        .unique()
    )

    if len(values) > 1:
        common_conflicts.append(
            {
                "Parametro": parameter,
                "Valores": " | ".join(values),
            }
        )


if common_conflicts:

    conflict_df = pd.DataFrame(
        common_conflicts
    )

    raise RuntimeError(
        "Parametros COMMON divergentes entre H2 e B1:\n"
        + conflict_df.to_string(
            index=False
        )
    )


# ------------------------------------------------------------
# Deduplicate COMMON
# ------------------------------------------------------------

common = (
    common_source
    .sort_values(
        ["Parametro", "Rota"]
    )
    .drop_duplicates(
        subset=["Parametro"],
        keep="first",
    )
    .copy()
)

common["ID_Prospectivo"] = (
    "COM-"
    + (
        common.reset_index(drop=True)
        .index
        + 1
    ).astype(str).str.zfill(3)
)

common["Rota_Origem_2026"] = "H2 + B1"


# ------------------------------------------------------------
# Specific H2 and B1
# ------------------------------------------------------------

h2_specific = df[
    df["Escopo_Parametro"] == "H2"
].copy()

h2_specific["ID_Prospectivo"] = (
    "H2-"
    + (
        h2_specific.reset_index(drop=True)
        .index
        + 1
    ).astype(str).str.zfill(3)
)

h2_specific["Rota_Origem_2026"] = "H2"


b1_specific = df[
    df["Escopo_Parametro"] == "B1"
].copy()

b1_specific["ID_Prospectivo"] = (
    "B1-"
    + (
        b1_specific.reset_index(drop=True)
        .index
        + 1
    ).astype(str).str.zfill(3)
)

b1_specific["Rota_Origem_2026"] = "B1"


prospective = pd.concat(
    [
        common,
        h2_specific,
        b1_specific,
    ],
    ignore_index=True,
    sort=False,
)


# ============================================================
# PROSPECTIVE MATRIX COLUMN ORDER
# ============================================================

prospective_base_columns = [
    "ID_Prospectivo",
    "Escopo_Parametro",
    "Grupo",
    "Parametro",
    "Simbolo",
    "Unidade",
    "Valor_2026",

    "Classe_Modelagem",
    "Projetar_2030_2050",
    "Variar_S1_S2_S3",
    "Sensibilidade",

    "Prioridade_Pesquisa",
    "Fonte_Prospectiva_Requerida",

    "Derivado_De",
    "Justificativa_Classificacao",
    "Nota_Metodologica_V2",

    "Rota_Origem_2026",
    "Arquivo_Fonte_2026",
    "Caminho_YAML_Resolvido",
]


prospective_columns = (
    prospective_base_columns
    + scenario_columns
)


prospective = prospective[
    [
        c
        for c in prospective_columns
        if c in prospective.columns
    ]
].copy()


# ============================================================
# RESEARCH AGENDA
# ============================================================

research_agenda = prospective[
    prospective[
        "Fonte_Prospectiva_Requerida"
    ] == "Sim"
].copy()

research_agenda = research_agenda[
    [
        "ID_Prospectivo",
        "Escopo_Parametro",
        "Grupo",
        "Parametro",
        "Unidade",
        "Valor_2026",
        "Classe_Modelagem",
        "Projetar_2030_2050",
        "Variar_S1_S2_S3",
        "Sensibilidade",
        "Prioridade_Pesquisa",
        "Nota_Metodologica_V2",
    ]
].copy()


priority_order = {
    "Alta": 0,
    "Media": 1,
    "Baixa": 2,
}

research_agenda["_priority"] = (
    research_agenda[
        "Prioridade_Pesquisa"
    ]
    .map(priority_order)
    .fillna(9)
)

research_agenda = (
    research_agenda
    .sort_values(
        [
            "_priority",
            "Escopo_Parametro",
            "Grupo",
            "Parametro",
        ]
    )
    .drop(
        columns="_priority"
    )
    .reset_index(drop=True)
)


# ============================================================
# CLASS LEGEND
# ============================================================

legend = pd.DataFrame(
    [
        [
            "F",
            "Constante fisica",
            (
                "Nao projetar temporalmente; "
                "manter salvo alteracao de especificacao."
            ),
        ],
        [
            "O",
            "Premissa operacional",
            (
                "Manter para comparabilidade; "
                "eventualmente testar em sensibilidade."
            ),
        ],
        [
            "T",
            "Driver tecnologico",
            (
                "Pode evoluir com learning, maturidade "
                "e desempenho da tecnologia."
            ),
        ],
        [
            "E",
            "Driver economico/mercado",
            (
                "Pode variar temporalmente e entre cenarios."
            ),
        ],
        [
            "R",
            "Regulacao/infraestrutura",
            (
                "Representa tarifa, politica, logistica "
                "ou disponibilidade de infraestrutura."
            ),
        ],
        [
            "D",
            "Parametro derivado",
            (
                "Nao projetar diretamente; recalcular "
                "a partir dos componentes."
            ),
        ],
        [
            "C",
            "Configuracao computacional",
            (
                "Serve a reproducibilidade; "
                "nao representa futuro tecnologico."
            ),
        ],
        [
            "L",
            "Limite do espaco de decisao",
            (
                "Define dominio da otimizacao; "
                "nao e driver prospectivo por si so."
            ),
        ],
    ],
    columns=[
        "Codigo",
        "Classe",
        "Definicao",
    ],
)


# ============================================================
# SCENARIO DEFINITION
# ============================================================

scenario_definition = pd.DataFrame(
    [
        [
            "S1",
            "Evolucao conservadora",
            (
                "Reducao de custos e ganhos tecnologicos "
                "mais lentos; infraestrutura e condicoes "
                "economicas menos favoraveis."
            ),
        ],
        [
            "S2",
            "Evolucao tendencial",
            (
                "Trajetoria central baseada nas fontes "
                "prospectivas selecionadas."
            ),
        ],
        [
            "S3",
            "Evolucao acelerada",
            (
                "Maior learning tecnologico, maturidade "
                "de infraestrutura e condicoes economicas "
                "mais favoraveis."
            ),
        ],
    ],
    columns=[
        "Cenario",
        "Nome",
        "Definicao",
    ],
)


# ============================================================
# THERMAL NOTE
# ============================================================

thermal_note = pd.DataFrame(
    [
        [
            "Baseline B1",
            "T0",
            (
                "Sem monetizacao do calor da CHP."
            ),
        ],
        [
            "Sensibilidade",
            "T1",
            (
                "Demanda termica parcial; "
                "nao e cenario temporal S1/S2/S3."
            ),
        ],
        [
            "Sensibilidade",
            "T2",
            (
                "Demanda termica elevada; "
                "nao e cenario temporal S1/S2/S3."
            ),
        ],
        [
            "Limite teorico",
            "TMAX",
            (
                "100% do calor disponivel valorizado; "
                "nao pertence ao baseline."
            ),
        ],
    ],
    columns=[
        "Categoria",
        "Caso",
        "Definicao",
    ],
)


# ============================================================
# SUMMARY
# ============================================================

summary_rows = []

for cls in [
    "F",
    "O",
    "T",
    "E",
    "R",
    "D",
    "C",
    "L",
    "REVISAR",
]:

    count = int(
        (
            df[
                "Classe_Modelagem"
            ] == cls
        ).sum()
    )

    if count > 0:
        summary_rows.append(
            [
                "Classe",
                cls,
                count,
            ]
        )


for scope in [
    "COMMON",
    "H2",
    "B1",
    "REVISAR",
]:

    count = int(
        (
            df[
                "Escopo_Parametro"
            ] == scope
        ).sum()
    )

    if count > 0:
        summary_rows.append(
            [
                "Escopo",
                scope,
                count,
            ]
        )


for flag in [
    "Sim",
    "Nao",
    "Derivado",
]:

    count = int(
        (
            df[
                "Projetar_2030_2050"
            ] == flag
        ).sum()
    )

    if count > 0:
        summary_rows.append(
            [
                "Projetar",
                flag,
                count,
            ]
        )


summary = pd.DataFrame(
    summary_rows,
    columns=[
        "Dimensao",
        "Categoria",
        "Quantidade",
    ],
)


# ============================================================
# SAVE CSV
# ============================================================

df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# SAVE XLSX
# ============================================================

with pd.ExcelWriter(
    OUTPUT_XLSX,
    engine="xlsxwriter",
) as writer:

    df.to_excel(
        writer,
        sheet_name="Tabela_Mestra_V2",
        index=False,
    )

    prospective.to_excel(
        writer,
        sheet_name="Matriz_Prospectiva",
        index=False,
    )

    research_agenda.to_excel(
        writer,
        sheet_name="Agenda_Pesquisa",
        index=False,
    )

    summary.to_excel(
        writer,
        sheet_name="Resumo",
        index=False,
    )

    legend.to_excel(
        writer,
        sheet_name="Legenda_Classes",
        index=False,
    )

    scenario_definition.to_excel(
        writer,
        sheet_name="Cenarios",
        index=False,
    )

    thermal_note.to_excel(
        writer,
        sheet_name="Tratamento_Termico",
        index=False,
    )


    workbook = writer.book


    # ========================================================
    # FORMATS
    # ========================================================

    header_fmt = workbook.add_format(
        {
            "bold": True,
            "font_color": "white",
            "bg_color": "#1F4E78",
            "border": 1,
            "align": "center",
            "valign": "vcenter",
            "text_wrap": True,
        }
    )

    wrap_fmt = workbook.add_format(
        {
            "text_wrap": True,
            "valign": "top",
        }
    )

    common_fmt = workbook.add_format(
        {
            "bg_color": "#D9EAF7",
        }
    )

    h2_fmt = workbook.add_format(
        {
            "bg_color": "#E2F0D9",
        }
    )

    b1_fmt = workbook.add_format(
        {
            "bg_color": "#FCE4D6",
        }
    )

    future_fmt = workbook.add_format(
        {
            "bg_color": "#FFF2CC",
        }
    )

    alert_fmt = workbook.add_format(
        {
            "bg_color": "#F4CCCC",
        }
    )


    # ========================================================
    # FORMAT SHEETS
    # ========================================================

    for sheet_name, data in [
        ("Tabela_Mestra_V2", df),
        ("Matriz_Prospectiva", prospective),
        ("Agenda_Pesquisa", research_agenda),
        ("Resumo", summary),
        ("Legenda_Classes", legend),
        ("Cenarios", scenario_definition),
        ("Tratamento_Termico", thermal_note),
    ]:

        ws = writer.sheets[sheet_name]

        for col_num, value in enumerate(
            data.columns
        ):
            ws.write(
                0,
                col_num,
                value,
                header_fmt,
            )

        ws.freeze_panes(
            1,
            0,
        )

        ws.autofilter(
            0,
            0,
            len(data),
            len(data.columns) - 1,
        )


    # ========================================================
    # MASTER WIDTHS
    # ========================================================

    ws = writer.sheets[
        "Tabela_Mestra_V2"
    ]

    ws.set_column("A:A", 13)
    ws.set_column("B:B", 18)
    ws.set_column("C:C", 9)
    ws.set_column("D:D", 31)
    ws.set_column("E:F", 16)
    ws.set_column("G:G", 15)
    ws.set_column("H:L", 22)
    ws.set_column("M:V", 28)
    ws.set_column("W:AZ", 22)
    ws.set_column("BA:ZZ", 38, wrap_fmt)

    ws.freeze_panes(
        1,
        7,
    )


    # ========================================================
    # PROSPECTIVE WIDTHS
    # ========================================================

    ws = writer.sheets[
        "Matriz_Prospectiva"
    ]

    ws.set_column("A:A", 15)
    ws.set_column("B:B", 13)
    ws.set_column("C:C", 18)
    ws.set_column("D:D", 31)
    ws.set_column("E:F", 16)
    ws.set_column("G:G", 15)
    ws.set_column("H:M", 20)
    ws.set_column("N:S", 36, wrap_fmt)
    ws.set_column("T:ZZ", 22)

    ws.freeze_panes(
        1,
        7,
    )


    # Scope conditional formatting
    scope_col = prospective.columns.get_loc(
        "Escopo_Parametro"
    )

    ws.conditional_format(
        1,
        scope_col,
        len(prospective),
        scope_col,
        {
            "type": "text",
            "criteria": "containing",
            "value": "COMMON",
            "format": common_fmt,
        },
    )

    ws.conditional_format(
        1,
        scope_col,
        len(prospective),
        scope_col,
        {
            "type": "text",
            "criteria": "containing",
            "value": "H2",
            "format": h2_fmt,
        },
    )

    ws.conditional_format(
        1,
        scope_col,
        len(prospective),
        scope_col,
        {
            "type": "text",
            "criteria": "containing",
            "value": "B1",
            "format": b1_fmt,
        },
    )


    # Research priority
    ws_agenda = writer.sheets[
        "Agenda_Pesquisa"
    ]

    ws_agenda.set_column(
        "A:L",
        23,
        wrap_fmt,
    )

    priority_col = (
        research_agenda.columns.get_loc(
            "Prioridade_Pesquisa"
        )
    )

    ws_agenda.conditional_format(
        1,
        priority_col,
        len(research_agenda),
        priority_col,
        {
            "type": "text",
            "criteria": "containing",
            "value": "Alta",
            "format": alert_fmt,
        },
    )


    # Other sheets
    writer.sheets["Resumo"].set_column(
        "A:C",
        25,
    )

    writer.sheets[
        "Legenda_Classes"
    ].set_column(
        "A:A",
        12,
    )

    writer.sheets[
        "Legenda_Classes"
    ].set_column(
        "B:B",
        30,
    )

    writer.sheets[
        "Legenda_Classes"
    ].set_column(
        "C:C",
        90,
        wrap_fmt,
    )

    writer.sheets[
        "Cenarios"
    ].set_column(
        "A:A",
        12,
    )

    writer.sheets[
        "Cenarios"
    ].set_column(
        "B:B",
        28,
    )

    writer.sheets[
        "Cenarios"
    ].set_column(
        "C:C",
        90,
        wrap_fmt,
    )

    writer.sheets[
        "Tratamento_Termico"
    ].set_column(
        "A:B",
        20,
    )

    writer.sheets[
        "Tratamento_Termico"
    ].set_column(
        "C:C",
        90,
        wrap_fmt,
    )


# ============================================================
# REPORT
# ============================================================

print()
print("=" * 72)
print("TABELA-MESTRA V2 GERADA")
print("=" * 72)

print()
print("Registros de auditoria 2026 =", len(df))
print("Linhas da matriz prospectiva =", len(prospective))
print("Linhas da agenda de pesquisa =", len(research_agenda))

print()
print("=== CLASSES ===")
print(
    df[
        "Classe_Modelagem"
    ].value_counts().to_string()
)

print()
print("=== ESCOPO ===")
print(
    df[
        "Escopo_Parametro"
    ].value_counts().to_string()
)

print()
print("=== PROJETAR 2030-2050 ===")
print(
    df[
        "Projetar_2030_2050"
    ].value_counts().to_string()
)

print()
print("=== VARIAR S1-S2-S3 ===")
print(
    df[
        "Variar_S1_S2_S3"
    ].value_counts().to_string()
)

print()
print("=== SENSIBILIDADE ===")
print(
    df[
        "Sensibilidade"
    ].value_counts().to_string()
)

print()
print("=== PRIORIDADE DE PESQUISA ===")
print(
    research_agenda[
        "Prioridade_Pesquisa"
    ].value_counts().to_string()
)

print()
print("=== FILES ===")
print(OUTPUT_CSV.relative_to(ROOT))
print(OUTPUT_XLSX.relative_to(ROOT))