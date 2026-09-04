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

INPUT_CSV = (
    DATA_DIR
    / "baseline_2026_yaml_parameters_extracted.csv"
)

OUTPUT_CSV = (
    DATA_DIR
    / "tabela_mestra_parametros_2026_2050.csv"
)

OUTPUT_XLSX = (
    DATA_DIR
    / "tabela_mestra_parametros_2026_2050.xlsx"
)


# ============================================================
# LOAD
# ============================================================

if not INPUT_CSV.exists():
    raise FileNotFoundError(
        f"Arquivo de entrada nao encontrado: {INPUT_CSV}"
    )

df = pd.read_csv(INPUT_CSV)

print("=== INPUT ===")
print("rows =", len(df))
print("columns =", len(df.columns))


# ============================================================
# VALIDATION
# ============================================================

required = [
    "id",
    "group",
    "route",
    "parameter",
    "symbol",
    "unit",
    "value_2026",
    "classification",
    "status",
    "prospective_status",
    "resolved_yaml_path",
    "candidate_yaml_paths",
    "notes",
]

missing = [
    col
    for col in required
    if col not in df.columns
]

if missing:
    raise KeyError(
        "Colunas obrigatorias ausentes: "
        + ", ".join(missing)
    )


if df["id"].duplicated().any():
    duplicates = df.loc[
        df["id"].duplicated(keep=False),
        ["id", "route", "parameter"],
    ]

    raise RuntimeError(
        "IDs duplicados encontrados:\n"
        + duplicates.to_string(index=False)
    )


if df["value_2026"].isna().any():
    unresolved = df.loc[
        df["value_2026"].isna(),
        [
            "id",
            "route",
            "parameter",
            "candidate_yaml_paths",
        ],
    ]

    raise RuntimeError(
        "Ainda existem valores 2026 nao resolvidos:\n"
        + unresolved.to_string(index=False)
    )


# ============================================================
# NORMALIZE NAMES
# ============================================================

df = df.rename(
    columns={
        "id": "ID",
        "group": "Grupo",
        "route": "Rota",
        "parameter": "Parametro",
        "symbol": "Simbolo",
        "unit": "Unidade",
        "value_2026": "Valor_2026",
        "classification": "Classificacao",
        "status": "Status_2026",
        "prospective_status": "Tratamento_Prospectivo",
        "resolved_yaml_path": "Caminho_YAML_Resolvido",
        "candidate_yaml_paths": "Caminhos_YAML_Candidatos",
        "notes": "Notas_Extracao",
    }
)


# ============================================================
# IDENTIFY SOURCE YAML
# ============================================================

df["Arquivo_Fonte_2026"] = df["Rota"].map(
    {
        "H2": "configs/paper/pv_bsv_h2_1500.yaml",
        "B1": "configs/paper/pv_bsv_biomethane_1500.yaml",
    }
)

df["Tipo_Fonte_2026"] = "Configuracao YAML vigente"


# ============================================================
# AUDIT / PROSPECTIVE CLASSIFICATION
# ============================================================

def classify_audit(row):

    classification = str(
        row["Classificacao"]
    ).lower()

    treatment = str(
        row["Tratamento_Prospectivo"]
    ).lower()

    if (
        "configuracao computacional" in classification
        or "reprodutibilidade" in classification
    ):
        return "Nao prioritario"

    if "auditar" in treatment:
        return "Prioritario"

    return "Documentar fonte"


def classify_driver(row):

    classification = str(
        row["Classificacao"]
    ).lower()

    treatment = str(
        row["Tratamento_Prospectivo"]
    ).lower()

    if (
        "configuracao computacional" in classification
        or "reprodutibilidade" in classification
    ):
        return "Nao"

    if (
        "driver prospectivo" in treatment
        or "prospectivo" in treatment
        or "pode variar" in treatment
        or "politica" in treatment
        or "tecnologico" in treatment
    ):
        return "Sim"

    return "Avaliar"


df["Auditoria_Fonte_2026"] = df.apply(
    classify_audit,
    axis=1,
)

df["Driver_Prospectivo"] = df.apply(
    classify_driver,
    axis=1,
)


# ============================================================
# SCENARIO COLUMNS
# ============================================================

scenario_columns = []

for year in [2030, 2040, 2050]:
    for scenario in ["S1", "S2", "S3"]:

        value_col = f"{scenario}_{year}_Valor"
        source_col = f"{scenario}_{year}_Fonte"
        justification_col = (
            f"{scenario}_{year}_Justificativa"
        )

        df[value_col] = pd.NA
        df[source_col] = pd.NA
        df[justification_col] = pd.NA

        scenario_columns.extend(
            [
                value_col,
                source_col,
                justification_col,
            ]
        )


# ============================================================
# UNCERTAINTY / GENERAL DOCUMENTATION
# ============================================================

df["Faixa_Incerteza_Baixa"] = pd.NA
df["Faixa_Incerteza_Alta"] = pd.NA

df["Ano_Base_Fonte"] = 2026

df["Moeda_Base"] = df["Unidade"].apply(
    lambda x: (
        "USD"
        if "USD" in str(x)
        else ""
    )
)

df["Regra_Conversao"] = pd.NA

df["Fonte_Primaria_2026"] = pd.NA

df["Justificativa_2026"] = (
    "Valor efetivamente utilizado no baseline 2026; "
    "extraido automaticamente do YAML vigente."
)

df["Observacoes"] = pd.NA


# ============================================================
# COLUMN ORDER
# ============================================================

base_columns = [
    "ID",
    "Grupo",
    "Rota",
    "Parametro",
    "Simbolo",
    "Unidade",
    "Valor_2026",
    "Classificacao",
    "Status_2026",
    "Auditoria_Fonte_2026",
    "Driver_Prospectivo",
    "Tratamento_Prospectivo",
    "Arquivo_Fonte_2026",
    "Caminho_YAML_Resolvido",
    "Tipo_Fonte_2026",
    "Fonte_Primaria_2026",
    "Ano_Base_Fonte",
    "Moeda_Base",
    "Regra_Conversao",
    "Justificativa_2026",
    "Faixa_Incerteza_Baixa",
    "Faixa_Incerteza_Alta",
]

final_columns = (
    base_columns
    + scenario_columns
    + [
        "Caminhos_YAML_Candidatos",
        "Notas_Extracao",
        "Observacoes",
    ]
)

master = df[final_columns].copy()


# ============================================================
# SORT
# ============================================================

route_order = {
    "H2": 0,
    "B1": 1,
}

master["_route_order"] = (
    master["Rota"]
    .map(route_order)
    .fillna(9)
)

master = (
    master
    .sort_values(
        [
            "_route_order",
            "Grupo",
            "ID",
        ]
    )
    .drop(
        columns="_route_order"
    )
    .reset_index(drop=True)
)


# ============================================================
# SCENARIO DEFINITION SHEET
# ============================================================

scenario_definition = pd.DataFrame(
    [
        [
            "S1",
            "BAU / conservador",
            (
                "Evolucao tecnologica e de mercado "
                "conservadora; infraestrutura e politicas "
                "avancam de forma incremental."
            ),
        ],
        [
            "S2",
            "Transicao coordenada",
            (
                "Evolucao tecnologica intermediaria, "
                "planejamento coordenado de infraestrutura "
                "e instrumentos de politica consistentes."
            ),
        ],
        [
            "S3",
            "Transicao acelerada",
            (
                "Maior learning tecnologico, escala, "
                "infraestrutura madura e sinais fortes "
                "de politica e flexibilidade."
            ),
        ],
    ],
    columns=[
        "Cenario",
        "Nome",
        "Definicao",
    ],
)

horizons = pd.DataFrame(
    [
        [2030, "Curto/medio prazo prospectivo"],
        [2040, "Medio/longo prazo prospectivo"],
        [2050, "Horizonte de transicao de longo prazo"],
    ],
    columns=[
        "Horizonte",
        "Interpretacao",
    ],
)


# ============================================================
# LEGEND
# ============================================================

legend = pd.DataFrame(
    [
        [
            "Valor_2026",
            (
                "Valor efetivamente usado para gerar "
                "o caso-base e as fronteiras 2026."
            ),
        ],
        [
            "Parametro de entrada",
            (
                "Valor fornecido ao modelo e nao resultado "
                "da otimizacao."
            ),
        ],
        [
            "Restricao / espaco de decisao",
            (
                "Limite que define o dominio admissivel "
                "da variavel."
            ),
        ],
        [
            "Parametro derivado",
            (
                "Valor calculado a partir de outros "
                "parametros do YAML."
            ),
        ],
        [
            "Configuracao computacional",
            (
                "Parametro de reproducao numerica; "
                "nao deve ser confundido com driver futuro."
            ),
        ],
        [
            "Driver_Prospectivo = Sim",
            (
                "Parametro candidato a assumir valores "
                "distintos em S1/S2/S3 e/ou 2030-2050."
            ),
        ],
        [
            "Auditoria_Fonte_2026 = Prioritario",
            (
                "Valor usado no baseline que exige "
                "fortalecimento documental antes da "
                "versao final do estudo."
            ),
        ],
        [
            "S1 / S2 / S3",
            (
                "Cenarios prospectivos; nao sao "
                "novas funcoes-objetivo do NSGA-II."
            ),
        ],
        [
            "T0",
            (
                "Caso-base B1 sem monetizacao do calor "
                "da CHP. T1/T2/TMAX permanecem como "
                "sensibilidades separadas."
            ),
        ],
    ],
    columns=[
        "Termo",
        "Definicao",
    ],
)


# ============================================================
# EXPORT CSV
# ============================================================

master.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# EXPORT XLSX
# ============================================================

try:
    import xlsxwriter

    engine = "xlsxwriter"

except ImportError:
    engine = "openpyxl"


with pd.ExcelWriter(
    OUTPUT_XLSX,
    engine=engine,
) as writer:

    master.to_excel(
        writer,
        sheet_name="Tabela_Mestra",
        index=False,
    )

    scenario_definition.to_excel(
        writer,
        sheet_name="Definicao_Cenarios",
        index=False,
        startrow=0,
    )

    horizons.to_excel(
        writer,
        sheet_name="Definicao_Cenarios",
        index=False,
        startrow=7,
    )

    legend.to_excel(
        writer,
        sheet_name="Legenda",
        index=False,
    )


    # --------------------------------------------------------
    # XLSXWRITER FORMATTING
    # --------------------------------------------------------

    if engine == "xlsxwriter":

        workbook = writer.book

        ws = writer.sheets[
            "Tabela_Mestra"
        ]

        ws_scen = writer.sheets[
            "Definicao_Cenarios"
        ]

        ws_leg = writer.sheets[
            "Legenda"
        ]


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

        common_fmt = workbook.add_format(
            {
                "bg_color": "#D9EAF7",
            }
        )

        input_fmt = workbook.add_format(
            {
                "bg_color": "#E2F0D9",
            }
        )

        future_fmt = workbook.add_format(
            {
                "bg_color": "#FFF2CC",
            }
        )

        alert_fmt = workbook.add_format(
            {
                "bg_color": "#FCE4D6",
            }
        )

        wrap_fmt = workbook.add_format(
            {
                "text_wrap": True,
                "valign": "top",
            }
        )


        # Header
        for col_num, value in enumerate(
            master.columns
        ):
            ws.write(
                0,
                col_num,
                value,
                header_fmt,
            )


        ws.freeze_panes(1, 6)
        ws.autofilter(
            0,
            0,
            len(master),
            len(master.columns) - 1,
        )


        # Column widths
        widths = {
            "A": 12,
            "B": 17,
            "C": 8,
            "D": 30,
            "E": 14,
            "F": 17,
            "G": 15,
            "H": 27,
            "I": 22,
            "J": 23,
            "K": 19,
            "L": 32,
            "M": 44,
            "N": 48,
            "O": 23,
            "P": 35,
            "Q": 14,
            "R": 13,
            "S": 24,
            "T": 46,
            "U": 20,
            "V": 20,
        }

        for col, width in widths.items():
            ws.set_column(
                f"{col}:{col}",
                width,
            )


        # Scenario columns
        scenario_start = len(
            base_columns
        )

        for idx in range(
            scenario_start,
            scenario_start
            + len(scenario_columns),
        ):
            ws.set_column(
                idx,
                idx,
                20,
                future_fmt,
            )


        # Long text
        ws.set_column(
            len(final_columns) - 3,
            len(final_columns) - 1,
            45,
            wrap_fmt,
        )


        # Conditional formatting:
        # driver prospective
        driver_col = (
            master.columns
            .get_loc("Driver_Prospectivo")
        )

        audit_col = (
            master.columns
            .get_loc("Auditoria_Fonte_2026")
        )

        ws.conditional_format(
            1,
            driver_col,
            len(master),
            driver_col,
            {
                "type": "text",
                "criteria": "containing",
                "value": "Sim",
                "format": future_fmt,
            },
        )

        ws.conditional_format(
            1,
            audit_col,
            len(master),
            audit_col,
            {
                "type": "text",
                "criteria": "containing",
                "value": "Prioritario",
                "format": alert_fmt,
            },
        )


        # Scenario definition
        for col_num, value in enumerate(
            scenario_definition.columns
        ):
            ws_scen.write(
                0,
                col_num,
                value,
                header_fmt,
            )

        for col_num, value in enumerate(
            horizons.columns
        ):
            ws_scen.write(
                7,
                col_num,
                value,
                header_fmt,
            )

        ws_scen.set_column(
            "A:A",
            14,
        )
        ws_scen.set_column(
            "B:B",
            28,
        )
        ws_scen.set_column(
            "C:C",
            90,
            wrap_fmt,
        )


        # Legend
        for col_num, value in enumerate(
            legend.columns
        ):
            ws_leg.write(
                0,
                col_num,
                value,
                header_fmt,
            )

        ws_leg.set_column(
            "A:A",
            34,
        )

        ws_leg.set_column(
            "B:B",
            100,
            wrap_fmt,
        )

        ws_leg.freeze_panes(
            1,
            0,
        )


# ============================================================
# REPORT
# ============================================================

print()
print("=== TABELA-MESTRA GERADA ===")

print(
    "Parametros =",
    len(master),
)

print(
    "Drivers prospectivos =",
    int(
        (
            master[
                "Driver_Prospectivo"
            ] == "Sim"
        ).sum()
    ),
)

print(
    "Auditoria prioritaria =",
    int(
        (
            master[
                "Auditoria_Fonte_2026"
            ] == "Prioritario"
        ).sum()
    ),
)

print()
print("CSV:")
print(OUTPUT_CSV.relative_to(ROOT))

print()
print("XLSX:")
print(OUTPUT_XLSX.relative_to(ROOT))