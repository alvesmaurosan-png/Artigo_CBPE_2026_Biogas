from __future__ import annotations
from pathlib import Path
import hashlib
import json
import math
from datetime import datetime, timezone
from typing import Any
import pandas as pd
import yaml
# =====================================================================
# ROOT / BASELINE
# =====================================================================
ROOT = Path.cwd()
BASELINE_TAG = (
    "baseline-2026-dual-route-v3"
)
BASELINE_DEFINITION = (
    "H2 corrected-2026 frozen regression AND "
    "B2 Pareto smoke with M2 commit=24h / lookahead=6h"
)
SCHEMA_VERSION = "3.0"
# =====================================================================
# INPUTS
# =====================================================================
GA_SOURCE = (
    ROOT
    / "src"
    / "optimization"
    / "ga_nsga2.py"
)
B2_CONFIG = (
    ROOT
    / "configs"
    / "paper"
    / "pv_bsv_biogas_b2_pareto_smoke_2026.yaml"
)
H2_CONFIG = (
    ROOT
    / "configs"
    / "paper"
    / "pv_bsv_h2_1500_regression_2026.yaml"
)
B2_CSV = (
    ROOT
    / "results"
    / "runs"
    / "pv_bsv_biogas_b2_pareto_smoke_2026_with_biogas"
    / "pareto_latest"
    / "pareto_2026_enriched.csv"
)
H2_CSV = (
    ROOT
    / "results"
    / "runs"
    / "pv_bsv_h2_1500_regression_2026_with_h2"
    / "pareto_latest"
    / "pareto_2026_enriched.csv"
)
H2_REFERENCE = (
    ROOT
    / "results"
    / "validation"
    / "h2_eta_compression_regression"
    / "regression_summary.csv"
)
OUTPUT_DIR = (
    ROOT
    / "results"
    / "validation"
    / "pareto2026_dual_route_v3"
)
OUTPUT_JSON = (
    OUTPUT_DIR
    / "dual_route_regression_summary.json"
)
# =====================================================================
# FROZEN REFERENCES
# =====================================================================
H2_REFERENCE_CASE = (
    "H2_eta061_comp250"
)
H2_FIXED_CAPACITIES = {
    "pv_kw":
        990.0,
    "bsv_kwh":
        1241.0,
    "electrolyzer_kw":
        441.0,
    "h2_tank_kg":
        200.0,
    "fuelcell_kw":
        117.0,
}
# B2 causal + robustness qualification reference.
#
# This value was independently established by the LA6/LA24
# causal and robustness audits and is not inferred from the
# current Pareto CSV.
B2_M2_LA6_REFERENCE_PGRID_KW = (
    447.908125846
)
B2_COMMIT_HOURS = 24
B2_LOOKAHEAD_HOURS = 6
# =====================================================================
# NUMERICAL TOLERANCES
# =====================================================================
ABS_TOL_PGRID_KW = 1.0e-6
ABS_TOL_ENERGY_KWH = 1.0e-3
ABS_TOL_RATIO = 1.0e-9
ABS_TOL_LCOE = 1.0e-9
ABS_TOL_CAPACITY = 1.0e-9
ABS_TOL_PHYSICAL = 1.0e-9
ABS_TOL_IDENTITY = 1.0e-9
# =====================================================================
# HELPERS
# =====================================================================
def native(
    value: Any,
) -> Any:
    if hasattr(
        value,
        "item",
    ):
        try:
            return value.item()
        except Exception:
            pass
    return value
def finite(
    value: Any,
) -> bool:
    try:
        return math.isfinite(
            float(value)
        )
    except (
        TypeError,
        ValueError,
    ):
        return False
def close(
    actual: Any,
    expected: Any,
    atol: float,
) -> bool:
    if not (
        finite(actual)
        and
        finite(expected)
    ):
        return False
    return abs(
        float(actual)
        - float(expected)
    ) <= atol
def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()
    with path.open(
        "rb"
    ) as f:
        for chunk in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                chunk
            )
    return digest.hexdigest()
def artifact_record(
    path: Path,
) -> dict[str, Any]:
    return {
        "path":
            path.relative_to(
                ROOT
            ).as_posix(),
        "sha256":
            sha256_file(
                path
            ),
        "size_bytes":
            int(
                path.stat().st_size
            ),
    }
def load_yaml(
    path: Path,
) -> dict:
    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as f:
        value = yaml.safe_load(
            f
        )
    if not isinstance(
        value,
        dict,
    ):
        raise RuntimeError(
            f"Invalid YAML: {path}"
        )
    return value
def load_single_row(
    path: Path,
) -> tuple[
    pd.DataFrame,
    pd.Series,
]:
    df = pd.read_csv(
        path
    )
    if df.empty:
        raise RuntimeError(
            f"Empty Pareto CSV: {path}"
        )
    if len(df) != 1:
        raise RuntimeError(
            f"Regression smoke expected exactly "
            f"1 final Pareto row in {path}; "
            f"found {len(df)}."
        )
    return (
        df,
        df.iloc[0],
    )
def first_existing_column(
    df: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    for name in candidates:
        if name in df.columns:
            return name
    return None
def require_column(
    df: pd.DataFrame,
    candidates: list[str],
    label: str,
) -> str:
    result = first_existing_column(
        df,
        candidates,
    )
    if result is None:
        raise RuntimeError(
            f"Could not resolve {label}. "
            f"Tried {candidates}."
        )
    return result
def bool_dict(
    values: dict[str, Any],
) -> dict[str, bool]:
    return {
        key:
            bool(value)
        for key, value
        in values.items()
    }
# =====================================================================
# PRE-FLIGHT FILE GATES
# =====================================================================
required_paths = [
    GA_SOURCE,
    B2_CONFIG,
    H2_CONFIG,
    B2_CSV,
    H2_CSV,
    H2_REFERENCE,
]
missing = [
    p
    for p in required_paths
    if not p.exists()
]
if missing:
    raise FileNotFoundError(
        "Missing required regression artifacts:\n"
        + "\n".join(
            str(p)
            for p in missing
        )
    )
# =====================================================================
# LOAD INPUTS
# =====================================================================
b2_cfg = load_yaml(
    B2_CONFIG
)
h2_cfg = load_yaml(
    H2_CONFIG
)
b2_df, b2 = load_single_row(
    B2_CSV
)
h2_df, h2 = load_single_row(
    H2_CSV
)
h2_ref_df = pd.read_csv(
    H2_REFERENCE
)
# =====================================================================
# H2 FROZEN REFERENCE
# =====================================================================
if (
    "case"
    not in h2_ref_df.columns
):
    raise RuntimeError(
        "H2 reference CSV lacks 'case'."
    )
ref_matches = (
    h2_ref_df.loc[
        h2_ref_df[
            "case"
        ]
        == H2_REFERENCE_CASE
    ]
)
if len(
    ref_matches
) != 1:
    raise RuntimeError(
        f"Expected exactly one "
        f"{H2_REFERENCE_CASE} reference row, "
        f"found {len(ref_matches)}."
    )
h2_ref = (
    ref_matches.iloc[0]
)
# =====================================================================
# RESOLVE CURRENT H2 COLUMNS
# =====================================================================
h2_pgrid_col = require_column(
    h2_df,
    [
        "P_peak_grid_opt_kw",
        "P_peak_grid_kW",
    ],
    "H2 Pgrid",
)
h2_grid_dependency_col = require_column(
    h2_df,
    [
        "total_grid_dependency_ratio",
        "grid_dependency",
    ],
    "H2 grid dependency",
)
h2_legacy_lcoe_col = require_column(
    h2_df,
    [
        "lcoe_legacy_usd_kwh",
        "LCOE_USD_kWh",
        "LCOE_USD_kwh",
    ],
    "H2 legacy LCOE",
)
h2_egrid_col = first_existing_column(
    h2_df,
    [
        "E_grid_kWh",
        "E_grid_kwh",
        "grid_energy_kwh",
    ],
)
h2_eload_col = first_existing_column(
    h2_df,
    [
        "E_load_kWh",
        "E_load_kwh",
        "load_energy_kwh",
    ],
)
# =====================================================================
# H2 FIXED CAPACITY GATE
# =====================================================================
h2_capacity_checks = {}
for key, expected in (
    H2_FIXED_CAPACITIES.items()
):
    h2_capacity_checks[
        key
    ] = (
        key in h2_df.columns
        and close(
            h2[
                key
            ],
            expected,
            ABS_TOL_CAPACITY,
        )
    )
H2_FIXED_CAPACITY_GATE = all(
    h2_capacity_checks.values()
)
# =====================================================================
# H2 NUMERICAL BASELINE
# =====================================================================
delta_h2_pgrid = (
    float(
        h2[
            h2_pgrid_col
        ]
    )
    -
    float(
        h2_ref[
            "P_peak_grid_kW"
        ]
    )
)
delta_h2_grid_dependency = (
    float(
        h2[
            h2_grid_dependency_col
        ]
    )
    -
    float(
        h2_ref[
            "grid_dependency"
        ]
    )
)
delta_h2_legacy_lcoe = (
    float(
        h2[
            h2_legacy_lcoe_col
        ]
    )
    -
    float(
        h2_ref[
            "LCOE_USD_kWh"
        ]
    )
)
h2_numerical_checks = {
    "PGRID":
        abs(
            delta_h2_pgrid
        )
        <= ABS_TOL_PGRID_KW,
    "GRID_DEPENDENCY":
        abs(
            delta_h2_grid_dependency
        )
        <= ABS_TOL_RATIO,
    "LEGACY_LCOE":
        abs(
            delta_h2_legacy_lcoe
        )
        <= ABS_TOL_LCOE,
}
delta_h2_egrid = math.nan
if (
    h2_egrid_col is not None
    and
    "E_grid_kWh"
    in h2_ref.index
):
    delta_h2_egrid = (
        float(
            h2[
                h2_egrid_col
            ]
        )
        -
        float(
            h2_ref[
                "E_grid_kWh"
            ]
        )
    )
    h2_numerical_checks[
        "EGRID"
    ] = (
        abs(
            delta_h2_egrid
        )
        <= ABS_TOL_ENERGY_KWH
    )
H2_BASELINE_NUMERICAL_GATE = all(
    h2_numerical_checks.values()
)
# =====================================================================
# H2 GRID DEPENDENCY IDENTITY
# =====================================================================
h2_grid_identity_error = math.nan
if (
    h2_egrid_col is not None
    and
    h2_eload_col is not None
):
    expected_ratio = (
        float(
            h2[
                h2_egrid_col
            ]
        )
        /
        float(
            h2[
                h2_eload_col
            ]
        )
    )
    h2_grid_identity_error = (
        float(
            h2[
                h2_grid_dependency_col
            ]
        )
        -
        expected_ratio
    )
    H2_GRID_DEPENDENCY_IDENTITY_GATE = (
        abs(
            h2_grid_identity_error
        )
        <= ABS_TOL_IDENTITY
    )
else:
    # Current-pipeline numerical comparison already freezes
    # grid dependency if energy columns are unavailable.
    H2_GRID_DEPENDENCY_IDENTITY_GATE = True
# =====================================================================
# H2 ROUTE ISOLATION
# =====================================================================
B2_ONLY_FIELDS = [
    "chp_kw",
    "biogas_storage_nm3",
    "substrate_t_day",
    "substrate_t_year",
]
h2_b2_contamination = {
    field:
        field in h2_df.columns
    for field in B2_ONLY_FIELDS
}
H2_ROUTE_ISOLATION_GATE = not any(
    h2_b2_contamination.values()
)
# =====================================================================
# H2 SCHEMA
# =====================================================================
h2_required_columns = [
    "pv_kw",
    "bsv_kwh",
    "electrolyzer_kw",
    "h2_tank_kg",
    "fuelcell_kw",
    h2_pgrid_col,
    h2_grid_dependency_col,
    h2_legacy_lcoe_col,
]
H2_SCHEMA_GATE = all(
    col in h2_df.columns
    for col in h2_required_columns
)
# =====================================================================
# H2 ROUTE CONFIGURATION
# =====================================================================
H2_CONFIG_ROUTE_GATE = (
    h2_cfg.get(
        "system",
        {},
    ).get(
        "route"
    )
    == "hydrogen"
)
# =====================================================================
# GA SOURCE ? B2 TEMPORAL ARCHITECTURE CONTRACT
# =====================================================================
ga_source = GA_SOURCE.read_text(
    encoding="utf-8-sig"
)
helper_start = ga_source.find(
    "    def _run_pareto_dispatch("
)
helper_end = ga_source.find(
    "    def evaluate(",
    helper_start,
)
if (
    helper_start < 0
    or
    helper_end < 0
):
    helper_source = ""
else:
    helper_source = ga_source[
        helper_start:
        helper_end
    ]
b2_temporal_checks = {
    "HELPER_DEFINITION":
        ga_source.count(
            "def _run_pareto_dispatch("
        ) == 1,
    "HELPER_CALLS":
        ga_source.count(
            "self._run_pareto_dispatch("
        ) == 2,
    "BIOGAS_BRANCH":
        helper_source.count(
            'if self.route == "biogas":'
        ) == 1,
    "COMMIT_24":
        helper_source.count(
            "commit_hours=24,"
        ) == 1,
    "LOOKAHEAD_6":
        helper_source.count(
            "lookahead_hours=6,"
        ) == 1,
    "H2_NO_SPECIAL_BRANCH":
        (
            'self.route == "hydrogen"'
            not in helper_source
        ),
    "NON_B2_NO_COMMIT":
        helper_source.count(
            "commit_hours="
        ) == 1,
    "NON_B2_NO_LOOKAHEAD":
        helper_source.count(
            "lookahead_hours="
        ) == 1,
    "ROUTE_BINDING_MARKER":
        (
            "B2_M2_LA6_ROUTE_BINDING"
            in helper_source
        ),
}
B2_TEMPORAL_ARCHITECTURE_GATE = all(
    b2_temporal_checks.values()
)
# =====================================================================
# B2 CONFIG / OBJECTIVES
# =====================================================================
B2_CONFIG_ROUTE_GATE = (
    b2_cfg.get(
        "system",
        {},
    ).get(
        "route"
    )
    == "biogas"
)
expected_b2_objectives = [
    "lcoe_harmonized_usd_kwh",
    "P_peak_grid_opt_kw",
]
B2_OBJECTIVE_GATE = (
    b2_cfg.get(
        "optimization",
        {},
    ).get(
        "objective"
    )
    == expected_b2_objectives
)
# =====================================================================
# B2 SCHEMA
# =====================================================================
b2_required_columns = [
    "pv_kw",
    "bsv_kwh",
    "chp_kw",
    "biogas_storage_nm3",
    "substrate_t_day",
    "substrate_t_year",
    "lcoe_harmonized_usd_kwh",
    "lcoe_legacy_usd_kwh",
    "P_peak_grid_opt_kw",
    "total_grid_dependency_ratio",
]
B2_SCHEMA_CONTRACT_GATE = all(
    col in b2_df.columns
    for col in b2_required_columns
)
# =====================================================================
# B2 REFERENCE PEAK ? NEW V3 NUMERICAL ANCHOR
# =====================================================================
b2_pgrid = float(
    b2[
        "P_peak_grid_opt_kw"
    ]
)
delta_b2_pgrid_vs_la6_reference = (
    b2_pgrid
    -
    B2_M2_LA6_REFERENCE_PGRID_KW
)
B2_M2_LA6_REFERENCE_PEAK_GATE = (
    abs(
        delta_b2_pgrid_vs_la6_reference
    )
    <= ABS_TOL_PGRID_KW
)
# Explicitly prove that the old degenerate floor is gone.
B2_OLD_468_FLOOR_REMOVED_GATE = (
    b2_pgrid
    <
    468.247600114102
    - 1.0
)
# =====================================================================
# B2 PHYSICAL COUPLING
# =====================================================================
scaling = b2_cfg.get(
    "biogas_pareto_scaling",
    {},
)
storage_scale = safe_storage_scale = (
    scaling.get(
        "biogas_storage_nm3_per_kw_chp"
    )
)
substrate_scale = (
    scaling.get(
        "substrate_t_day_per_kw_chp"
    )
)
if (
    storage_scale is None
    or
    substrate_scale is None
):
    B2_PHYSICAL_COUPLING_GATE = False
    storage_error = math.nan
    substrate_error = math.nan
else:
    expected_storage = (
        float(
            b2[
                "chp_kw"
            ]
        )
        *
        float(
            storage_scale
        )
    )
    expected_substrate_day = (
        float(
            b2[
                "chp_kw"
            ]
        )
        *
        float(
            substrate_scale
        )
    )
    storage_error = (
        float(
            b2[
                "biogas_storage_nm3"
            ]
        )
        -
        expected_storage
    )
    substrate_error = (
        float(
            b2[
                "substrate_t_day"
            ]
        )
        -
        expected_substrate_day
    )
    B2_PHYSICAL_COUPLING_GATE = (
        abs(
            storage_error
        )
        <= ABS_TOL_PHYSICAL
        and
        abs(
            substrate_error
        )
        <= ABS_TOL_PHYSICAL
    )
# =====================================================================
# B2 ECONOMIC / OUTPUT SANITY
# =====================================================================
b2_economic_fields = [
    "lcoe_harmonized_usd_kwh",
    "lcoe_legacy_usd_kwh",
    "total_grid_dependency_ratio",
]
B2_ECONOMIC_FINITE_GATE = all(
    finite(
        b2[
            field
        ]
    )
    for field
    in b2_economic_fields
)
B2_HARMONIZED_ECONOMICS_GATE = (
    B2_ECONOMIC_FINITE_GATE
    and
    float(
        b2[
            "lcoe_harmonized_usd_kwh"
        ]
    )
    > 0.0
    and
    float(
        b2[
            "lcoe_legacy_usd_kwh"
        ]
    )
    > 0.0
)
# =====================================================================
# B2 GRID DEPENDENCY IDENTITY ? IF ENERGY FIELDS AVAILABLE
# =====================================================================
b2_egrid_col = first_existing_column(
    b2_df,
    [
        "E_grid_kWh",
        "E_grid_kwh",
        "grid_energy_kwh",
    ],
)
b2_eload_col = first_existing_column(
    b2_df,
    [
        "E_load_kWh",
        "E_load_kwh",
        "load_energy_kwh",
    ],
)
b2_grid_identity_error = math.nan
if (
    b2_egrid_col is not None
    and
    b2_eload_col is not None
):
    expected_grid_dependency = (
        float(
            b2[
                b2_egrid_col
            ]
        )
        /
        float(
            b2[
                b2_eload_col
            ]
        )
    )
    b2_grid_identity_error = (
        float(
            b2[
                "total_grid_dependency_ratio"
            ]
        )
        -
        expected_grid_dependency
    )
    B2_GRID_DEPENDENCY_IDENTITY_GATE = (
        abs(
            b2_grid_identity_error
        )
        <= ABS_TOL_IDENTITY
    )
else:
    B2_GRID_DEPENDENCY_IDENTITY_GATE = True
# =====================================================================
# CONSOLIDATED GATES
# =====================================================================
b2_gates = bool_dict(
    {
        "B2_SCHEMA_CONTRACT_GATE":
            B2_SCHEMA_CONTRACT_GATE,
        "B2_CONFIG_ROUTE_GATE":
            B2_CONFIG_ROUTE_GATE,
        "B2_OBJECTIVE_GATE":
            B2_OBJECTIVE_GATE,
        "B2_TEMPORAL_ARCHITECTURE_GATE":
            B2_TEMPORAL_ARCHITECTURE_GATE,
        "B2_M2_LA6_REFERENCE_PEAK_GATE":
            B2_M2_LA6_REFERENCE_PEAK_GATE,
        "B2_OLD_468_FLOOR_REMOVED_GATE":
            B2_OLD_468_FLOOR_REMOVED_GATE,
        "B2_PHYSICAL_COUPLING_GATE":
            B2_PHYSICAL_COUPLING_GATE,
        "B2_HARMONIZED_ECONOMICS_GATE":
            B2_HARMONIZED_ECONOMICS_GATE,
        "B2_GRID_DEPENDENCY_IDENTITY_GATE":
            B2_GRID_DEPENDENCY_IDENTITY_GATE,
    }
)
B2_M2_LA6_END_TO_END_GATE = all(
    b2_gates.values()
)
h2_gates = bool_dict(
    {
        "H2_FIXED_CAPACITY_GATE":
            H2_FIXED_CAPACITY_GATE,
        "H2_BASELINE_NUMERICAL_GATE":
            H2_BASELINE_NUMERICAL_GATE,
        "H2_GRID_DEPENDENCY_IDENTITY_GATE":
            H2_GRID_DEPENDENCY_IDENTITY_GATE,
        "H2_ROUTE_ISOLATION_GATE":
            H2_ROUTE_ISOLATION_GATE,
        "H2_SCHEMA_GATE":
            H2_SCHEMA_GATE,
        "H2_CONFIG_ROUTE_GATE":
            H2_CONFIG_ROUTE_GATE,
    }
)
H2_CURRENT_PIPELINE_REGRESSION = all(
    h2_gates.values()
)
BASELINE_2026_DUAL_ROUTE_V3_GATE = (
    B2_M2_LA6_END_TO_END_GATE
    and
    H2_CURRENT_PIPELINE_REGRESSION
)
# =====================================================================
# RESOLVE RUN DIRECTORIES
# =====================================================================
def resolve_latest_real_run(
    latest_csv: Path,
) -> str:
    run_root = (
        latest_csv
        .parent
        .parent
    )
    candidates = sorted(
        [
            p
            for p in run_root.iterdir()
            if (
                p.is_dir()
                and
                p.name.startswith(
                    "pareto_20"
                )
            )
        ],
        key=lambda p:
            p.name,
    )
    if not candidates:
        return ""
    return (
        candidates[-1]
        .relative_to(
            ROOT
        )
        .as_posix()
    )
b2_run_path = resolve_latest_real_run(
    B2_CSV
)
h2_run_path = resolve_latest_real_run(
    H2_CSV
)
# =====================================================================
# ARTIFACT HASHES
# =====================================================================
artifacts = {
    "b2_pareto_2026_enriched_csv":
        artifact_record(
            B2_CSV
        ),
    "h2_pareto_2026_enriched_csv":
        artifact_record(
            H2_CSV
        ),
    "h2_regression_summary_csv":
        artifact_record(
            H2_REFERENCE
        ),
    "ga_nsga2_py":
        artifact_record(
            GA_SOURCE
        ),
    "b2_smoke_config_yaml":
        artifact_record(
            B2_CONFIG
        ),
    "h2_regression_config_yaml":
        artifact_record(
            H2_CONFIG
        ),
}
# =====================================================================
# PERSIST V3 EVIDENCE
# =====================================================================
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)
timestamp = (
    datetime.now(
        timezone.utc
    )
    .isoformat()
)
summary = {
    "schema_version":
        SCHEMA_VERSION,
    "baseline_tag":
        BASELINE_TAG,
    "baseline_definition":
        BASELINE_DEFINITION,
    "timestamp":
        timestamp,
    "timestamp_timezone":
        "UTC",
    "baseline_status":
        (
            "PASS"
            if BASELINE_2026_DUAL_ROUTE_V3_GATE
            else "FAIL"
        ),
    "BASELINE_2026_DUAL_ROUTE_V3_GATE":
        bool(
            BASELINE_2026_DUAL_ROUTE_V3_GATE
        ),
    "b2_run_path":
        b2_run_path,
    "h2_run_path":
        h2_run_path,
    "h2_reference_path":
        H2_REFERENCE.relative_to(
            ROOT
        ).as_posix(),
    "B2": {
        "route":
            "biogas",
        "temporal_architecture": {
            "mode":
                "M2 / receding horizon",
            "commit_hours":
                B2_COMMIT_HOURS,
            "lookahead_hours":
                B2_LOOKAHEAD_HOURS,
            "reference_pgrid_kw":
                B2_M2_LA6_REFERENCE_PGRID_KW,
            "observed_pgrid_kw":
                b2_pgrid,
        },
        "gates":
            {
                **b2_gates,
                "B2_M2_LA6_END_TO_END_GATE":
                    bool(
                        B2_M2_LA6_END_TO_END_GATE
                    ),
            },
        "residuals": {
            "delta_pgrid_vs_la6_reference_kw":
                delta_b2_pgrid_vs_la6_reference,
            "storage_error_nm3":
                storage_error,
            "substrate_error_t_day":
                substrate_error,
            "grid_dependency_identity_error":
                b2_grid_identity_error,
        },
        "outputs": {
            "lcoe_harmonized_usd_kwh":
                float(
                    b2[
                        "lcoe_harmonized_usd_kwh"
                    ]
                ),
            "lcoe_legacy_usd_kwh":
                float(
                    b2[
                        "lcoe_legacy_usd_kwh"
                    ]
                ),
            "P_peak_grid_opt_kw":
                b2_pgrid,
            "total_grid_dependency_ratio":
                float(
                    b2[
                        "total_grid_dependency_ratio"
                    ]
                ),
        },
        "temporal_contract_checks":
            bool_dict(
                b2_temporal_checks
            ),
    },
    "H2": {
        "route":
            "hydrogen",
        "reference_case":
            H2_REFERENCE_CASE,
        "fixed_capacities":
            H2_FIXED_CAPACITIES,
        "gates":
            {
                **h2_gates,
                "H2_CURRENT_PIPELINE_REGRESSION":
                    bool(
                        H2_CURRENT_PIPELINE_REGRESSION
                    ),
            },
        "deltas": {
            "delta_pgrid_kw":
                delta_h2_pgrid,
            "delta_egrid_kwh":
                delta_h2_egrid,
            "delta_grid_dependency":
                delta_h2_grid_dependency,
            "delta_legacy_lcoe_usd_kwh":
                delta_h2_legacy_lcoe,
            "grid_dependency_identity_error":
                h2_grid_identity_error,
        },
        "b2_field_contamination":
            h2_b2_contamination,
        "outputs": {
            "P_peak_grid_opt_kw":
                float(
                    h2[
                        h2_pgrid_col
                    ]
                ),
            "total_grid_dependency_ratio":
                float(
                    h2[
                        h2_grid_dependency_col
                    ]
                ),
            "lcoe_legacy_usd_kwh":
                float(
                    h2[
                        h2_legacy_lcoe_col
                    ]
                ),
        },
    },
    "artifacts":
        artifacts,
}
with OUTPUT_JSON.open(
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        summary,
        f,
        indent=2,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=True,
    )
    f.write(
        "\n"
    )
# =====================================================================
# CONSOLE REPORT
# =====================================================================
print("=" * 120)
print(
    "PARETO 2026 ? DUAL-ROUTE REGRESSION V3"
)
print("=" * 120)
print()
print(
    "B2 ? M2 24+6"
)
for key, value in (
    b2_gates.items()
):
    print(
        f"{key:48s} = "
        f"{'PASS' if value else 'FAIL'}"
    )
print(
    f"{'B2_M2_LA6_END_TO_END_GATE':48s} = "
    f"{'PASS' if B2_M2_LA6_END_TO_END_GATE else 'FAIL'}"
)
print()
print(
    "B2 numerical residuals"
)
print(
    "  delta Pgrid vs LA6 ref [kW] = "
    f"{delta_b2_pgrid_vs_la6_reference:+.12e}"
)
print(
    "  storage error [Nm3]         = "
    f"{storage_error:+.12e}"
)
print(
    "  substrate error [t/day]     = "
    f"{substrate_error:+.12e}"
)
print(
    "  grid dependency identity    = "
    f"{b2_grid_identity_error:+.12e}"
)
print()
print("-" * 120)
print()
print(
    "H2 ? FROZEN CURRENT-PIPELINE REGRESSION"
)
for key, value in (
    h2_gates.items()
):
    print(
        f"{key:48s} = "
        f"{'PASS' if value else 'FAIL'}"
    )
print(
    f"{'H2_CURRENT_PIPELINE_REGRESSION':48s} = "
    f"{'PASS' if H2_CURRENT_PIPELINE_REGRESSION else 'FAIL'}"
)
print()
print(
    "H2 regression deltas"
)
print(
    "  delta Pgrid [kW]            = "
    f"{delta_h2_pgrid:+.12e}"
)
print(
    "  delta Egrid [kWh]           = "
    f"{delta_h2_egrid:+.12e}"
)
print(
    "  delta grid dependency       = "
    f"{delta_h2_grid_dependency:+.12e}"
)
print(
    "  delta legacy LCOE           = "
    f"{delta_h2_legacy_lcoe:+.12e}"
)
print()
print(
    "H2 B2-field contamination"
)
for key, value in (
    h2_b2_contamination.items()
):
    print(
        f"  {key:30s} = "
        f"{value}"
    )
print()
print("=" * 120)
print(
    f"{'B2_M2_LA6_END_TO_END_GATE':48s} = "
    f"{'PASS' if B2_M2_LA6_END_TO_END_GATE else 'FAIL'}"
)
print(
    f"{'H2_CURRENT_PIPELINE_REGRESSION':48s} = "
    f"{'PASS' if H2_CURRENT_PIPELINE_REGRESSION else 'FAIL'}"
)
print("-" * 120)
print(
    "BASELINE_2026_DUAL_ROUTE_V3_GATE"
    + " " * 14
    + "= "
    + (
        "PASS"
        if BASELINE_2026_DUAL_ROUTE_V3_GATE
        else "FAIL"
    )
)
print("=" * 120)
print()
print(
    "ARTIFACTS"
)
print(
    f"B2 CSV : {B2_CSV}"
)
print(
    f"B2 RUN : {b2_run_path}"
)
print(
    f"H2 CSV : {H2_CSV}"
)
print(
    f"H2 RUN : {h2_run_path}"
)
print(
    f"H2 REF : {H2_REFERENCE}"
)
print(
    f"V3 JSON: {OUTPUT_JSON}"
)
if not BASELINE_2026_DUAL_ROUTE_V3_GATE:
    raise SystemExit(2)
