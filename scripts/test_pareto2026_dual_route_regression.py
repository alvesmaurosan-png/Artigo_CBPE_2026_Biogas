from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sys
import pandas as pd
# =====================================================================
# PARETO 2026 — DUAL-ROUTE REGRESSION GATE
#
# Purpose
# -------
# Validate simultaneously:
#
#   1) B2 end-to-end Pareto smoke:
#      GA -> physical coupling -> MILP -> economics -> enrichment
#
#   2) H2 corrected 2026 baseline regression:
#      current shared pipeline must reproduce H2_eta061_comp250
#
# Final requirement:
#
#   B2 works
#       AND
#   H2 corrected baseline remains invariant
#
# The script validates existing artifacts. It does NOT rerun NSGA-II/MILP.
# =====================================================================
ROOT = Path.cwd()
# =====================================================================
# CONSTANTS — BASELINE 2026
# =====================================================================
VKM_ANNUAL = 3_327_000.0
PKM_ANNUAL = 76_521_000.0
B2_K_STORAGE = (
    2.258218220659535
)
B2_K_SUBSTRATE = (
    0.016936636654946513
)
H2_REFERENCE_CASE = (
    "H2_eta061_comp250"
)
H2_EXPECTED_CAPACITIES = {
    "pv_kw": 990.0,
    "bsv_kwh": 1241.0,
    "electrolyzer_kw": 441.0,
    "h2_tank_kg": 200.0,
    "fuelcell_kw": 117.0,
}
# =====================================================================
# TOLERANCES
# =====================================================================
ABS_TOL = {
    "capacity": 1e-12,
    "b2_storage": 1e-9,
    "b2_substrate_day": 1e-12,
    "b2_substrate_year": 1e-9,
    "dependency": 1e-12,
    "harmonized_lcoe": 1e-10,
    "service_cost": 1e-12,
    "h2_pgrid": 1e-6,
    "h2_egrid": 1e-4,
    "h2_dependency": 1e-10,
    "h2_legacy_lcoe": 1e-10,
}
# =====================================================================
# PATHS
# =====================================================================
B2_RUN_ROOT = (
    ROOT
    / "results"
    / "runs"
    / "pv_bsv_biogas_b2_pareto_smoke_2026_with_biogas"
)
H2_RUN_ROOT = (
    ROOT
    / "results"
    / "runs"
    / "pv_bsv_h2_1500_regression_2026_with_h2"
)
H2_REFERENCE_PATH = (
    ROOT
    / "results"
    / "validation"
    / "h2_eta_compression_regression"
    / "regression_summary.csv"
)
# =====================================================================
# HELPERS
# =====================================================================
def latest_timestamped_run(
    root: Path,
) -> Path:
    if not root.exists():
        raise FileNotFoundError(
            f"Run root not found: {root}"
        )
    candidates = sorted(
        [
            p
            for p in root.glob("pareto_*")
            if p.is_dir()
            and p.name != "pareto_latest"
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No timestamped Pareto run found in {root}"
        )
    return candidates[0]
def load_schema_gates(
    run_dir: Path,
) -> dict:
    p = (
        run_dir
        / "pareto_2026_schema_gates.json"
    )
    if not p.exists():
        raise FileNotFoundError(p)
    with p.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)
def load_latest_enriched(
    root: Path,
) -> tuple[pd.DataFrame, Path]:
    p = (
        root
        / "pareto_latest"
        / "pareto_2026_enriched.csv"
    )
    if not p.exists():
        raise FileNotFoundError(p)
    return (
        pd.read_csv(p),
        p,
    )
def schema_contract_pass(
    gates: dict,
) -> bool:
    required = [
        "PARETO_OUTPUT_SCHEMA_GATE",
        "PARETO_OBJECTIVES_GATE",
        "UF_VKM_GATE",
        "UF_PKM_GATE",
        "GRID_DEPENDENCY_IDENTITY_GATE",
    ]
    return all(
        bool(
            gates.get(
                key,
                False,
            )
        )
        for key in required
    )
def print_gate(
    name: str,
    value: bool,
    width: int = 44,
) -> None:
    print(
        f"{name:{width}s} = "
        f"{'PASS' if value else 'FAIL'}"
    )
# =====================================================================
# B2 END-TO-END VALIDATION
# =====================================================================
def validate_b2() -> dict:
    df, csv_path = (
        load_latest_enriched(
            B2_RUN_ROOT
        )
    )
    run_dir = (
        latest_timestamped_run(
            B2_RUN_ROOT
        )
    )
    schema_gates = (
        load_schema_gates(
            run_dir
        )
    )
    if df.empty:
        raise RuntimeError(
            "B2 enriched Pareto is empty."
        )
    required = {
        "solution_id",
        "route",
        "objective_1_name",
        "objective_2_name",
        "lcoe_legacy_usd_kwh",
        "lcoe_harmonized_usd_kwh",
        "annual_cost_harmonized_usd",
        "P_peak_grid_opt_kw",
        "E_grid_total_kwh",
        "E_load_total_kwh",
        "total_grid_dependency_ratio",
        "vehicle_km_annual",
        "passenger_km_annual",
        "energy_system_cost_usd_per_vehicle_km",
        "energy_system_cost_usd_per_passenger_km",
        "pv_kw",
        "bsv_kwh",
        "chp_kw",
        "biogas_storage_nm3",
        "substrate_t_day",
        "substrate_t_year",
    }
    columns_gate = (
        required
        <= set(df.columns)
    )
    route_gate = (
        set(
            df["route"].astype(str)
        )
        == {"biogas"}
    )
    objective_gate = (
        set(
            df[
                "objective_1_name"
            ].astype(str)
        )
        == {
            "lcoe_harmonized_usd_kwh"
        }
        and
        set(
            df[
                "objective_2_name"
            ].astype(str)
        )
        == {
            "P_peak_grid_opt_kw"
        }
    )
    service_gate = (
        (
            df["vehicle_km_annual"]
            == VKM_ANNUAL
        ).all()
        and
        (
            df["passenger_km_annual"]
            == PKM_ANNUAL
        ).all()
    )
    chp_domain_gate = (
        (
            df["chp_kw"]
            >= 50.0
        ).all()
        and
        (
            df["chp_kw"]
            <= 200.0
        ).all()
    )
    # -----------------------------------------------------------------
    # Physical B2 coupling
    # -----------------------------------------------------------------
    expected_storage = (
        df["chp_kw"]
        * B2_K_STORAGE
    )
    expected_substrate = (
        df["chp_kw"]
        * B2_K_SUBSTRATE
    )
    expected_substrate_year = (
        expected_substrate
        * 365.0
    )
    storage_error = (
        df["biogas_storage_nm3"]
        - expected_storage
    ).abs()
    substrate_error = (
        df["substrate_t_day"]
        - expected_substrate
    ).abs()
    substrate_year_error = (
        df["substrate_t_year"]
        - expected_substrate_year
    ).abs()
    storage_gate = (
        float(
            storage_error.max()
        )
        <= ABS_TOL["b2_storage"]
    )
    substrate_gate = (
        float(
            substrate_error.max()
        )
        <= ABS_TOL[
            "b2_substrate_day"
        ]
    )
    substrate_year_gate = (
        float(
            substrate_year_error.max()
        )
        <= ABS_TOL[
            "b2_substrate_year"
        ]
    )
    physical_coupling_gate = all([
        chp_domain_gate,
        storage_gate,
        substrate_gate,
        substrate_year_gate,
    ])
    # -----------------------------------------------------------------
    # Grid dependency
    # -----------------------------------------------------------------
    dependency_expected = (
        df["E_grid_total_kwh"]
        / df["E_load_total_kwh"]
    )
    dependency_error = (
        df[
            "total_grid_dependency_ratio"
        ]
        - dependency_expected
    ).abs()
    dependency_gate = (
        float(
            dependency_error.max()
        )
        <= ABS_TOL["dependency"]
    )
    # -----------------------------------------------------------------
    # Harmonized economics
    # -----------------------------------------------------------------
    lcoe_expected = (
        df[
            "annual_cost_harmonized_usd"
        ]
        / df[
            "E_load_total_kwh"
        ]
    )
    lcoe_error = (
        df[
            "lcoe_harmonized_usd_kwh"
        ]
        - lcoe_expected
    ).abs()
    lcoe_gate = (
        float(
            lcoe_error.max()
        )
        <= ABS_TOL[
            "harmonized_lcoe"
        ]
    )
    cost_vkm_expected = (
        df[
            "annual_cost_harmonized_usd"
        ]
        / VKM_ANNUAL
    )
    cost_pkm_expected = (
        df[
            "annual_cost_harmonized_usd"
        ]
        / PKM_ANNUAL
    )
    cost_vkm_error = (
        df[
            "energy_system_cost_usd_per_vehicle_km"
        ]
        - cost_vkm_expected
    ).abs()
    cost_pkm_error = (
        df[
            "energy_system_cost_usd_per_passenger_km"
        ]
        - cost_pkm_expected
    ).abs()
    service_cost_gate = (
        float(
            cost_vkm_error.max()
        )
        <= ABS_TOL["service_cost"]
        and
        float(
            cost_pkm_error.max()
        )
        <= ABS_TOL["service_cost"]
    )
    economics_gate = all([
        dependency_gate,
        lcoe_gate,
        service_cost_gate,
    ])
    schema_gate = (
        schema_contract_pass(
            schema_gates
        )
    )
    all_pass = all([
        columns_gate,
        route_gate,
        objective_gate,
        service_gate,
        physical_coupling_gate,
        economics_gate,
        schema_gate,
    ])
    return {
        "rows":
            len(df),
        "csv_path":
            csv_path,
        "run_dir":
            run_dir,
        "columns_gate":
            columns_gate,
        "route_gate":
            route_gate,
        "objective_gate":
            objective_gate,
        "service_gate":
            service_gate,
        "physical_coupling_gate":
            physical_coupling_gate,
        "economics_gate":
            economics_gate,
        "schema_gate":
            schema_gate,
        "storage_max_error":
            float(
                storage_error.max()
            ),
        "substrate_max_error":
            float(
                substrate_error.max()
            ),
        "dependency_max_error":
            float(
                dependency_error.max()
            ),
        "lcoe_max_error":
            float(
                lcoe_error.max()
            ),
        "all_pass":
            all_pass,
    }
# =====================================================================
# H2 CURRENT-PIPELINE REGRESSION
# =====================================================================
def validate_h2() -> dict:
    if not H2_REFERENCE_PATH.exists():
        raise FileNotFoundError(
            H2_REFERENCE_PATH
        )
    reference_df = pd.read_csv(
        H2_REFERENCE_PATH
    )
    ref_matches = (
        reference_df.loc[
            reference_df["case"]
            == H2_REFERENCE_CASE
        ]
    )
    if len(ref_matches) != 1:
        raise RuntimeError(
            "Expected exactly one "
            f"{H2_REFERENCE_CASE!r} row; "
            f"found {len(ref_matches)}."
        )
    ref = ref_matches.iloc[0]
    df, csv_path = (
        load_latest_enriched(
            H2_RUN_ROOT
        )
    )
    if len(df) != 1:
        raise RuntimeError(
            "H2 fixed-capacity regression "
            "must contain exactly one "
            f"Pareto row; found {len(df)}."
        )
    row = df.iloc[0]
    run_dir = (
        latest_timestamped_run(
            H2_RUN_ROOT
        )
    )
    schema_gates = (
        load_schema_gates(
            run_dir
        )
    )
    # -----------------------------------------------------------------
    # Fixed capacities
    # -----------------------------------------------------------------
    capacity_gate = all(
        abs(
            float(row[key])
            - expected
        )
        <= ABS_TOL["capacity"]
        for key, expected
        in H2_EXPECTED_CAPACITIES.items()
    )
    # -----------------------------------------------------------------
    # Numerical regression against corrected H2 baseline
    # -----------------------------------------------------------------
    calc = {
        "P_peak_grid_kW":
            float(
                row[
                    "P_peak_grid_opt_kw"
                ]
            ),
        "E_grid_kWh":
            float(
                row[
                    "E_grid_total_kwh"
                ]
            ),
        "grid_dependency":
            float(
                row[
                    "total_grid_dependency_ratio"
                ]
            ),
        "LCOE_USD_kWh":
            float(
                row[
                    "lcoe_legacy_usd_kwh"
                ]
            ),
    }
    ref_values = {
        key: float(ref[key])
        for key in calc
    }
    delta = {
        key:
            calc[key]
            - ref_values[key]
        for key in calc
    }
    numerical_gates = {
        "P_peak_grid_kW":
            abs(
                delta[
                    "P_peak_grid_kW"
                ]
            )
            <= ABS_TOL[
                "h2_pgrid"
            ],
        "E_grid_kWh":
            abs(
                delta[
                    "E_grid_kWh"
                ]
            )
            <= ABS_TOL[
                "h2_egrid"
            ],
        "grid_dependency":
            abs(
                delta[
                    "grid_dependency"
                ]
            )
            <= ABS_TOL[
                "h2_dependency"
            ],
        "LCOE_USD_kWh":
            abs(
                delta[
                    "LCOE_USD_kWh"
                ]
            )
            <= ABS_TOL[
                "h2_legacy_lcoe"
            ],
    }
    baseline_numerical_gate = all(
        numerical_gates.values()
    )
    # -----------------------------------------------------------------
    # Harmonized identities
    # -----------------------------------------------------------------
    annual_cost_harm = float(
        row[
            "annual_cost_harmonized_usd"
        ]
    )
    e_load = float(
        row[
            "E_load_total_kwh"
        ]
    )
    lcoe_harm_expected = (
        annual_cost_harm
        / e_load
    )
    lcoe_harm_error = (
        float(
            row[
                "lcoe_harmonized_usd_kwh"
            ]
        )
        - lcoe_harm_expected
    )
    lcoe_harm_gate = (
        abs(
            lcoe_harm_error
        )
        <= ABS_TOL[
            "harmonized_lcoe"
        ]
    )
    vkm = float(
        row[
            "vehicle_km_annual"
        ]
    )
    pkm = float(
        row[
            "passenger_km_annual"
        ]
    )
    vkm_cost_error = (
        float(
            row[
                "energy_system_cost_usd_per_vehicle_km"
            ]
        )
        - (
            annual_cost_harm
            / vkm
        )
    )
    pkm_cost_error = (
        float(
            row[
                "energy_system_cost_usd_per_passenger_km"
            ]
        )
        - (
            annual_cost_harm
            / pkm
        )
    )
    service_cost_gate = (
        abs(
            vkm_cost_error
        )
        <= ABS_TOL[
            "service_cost"
        ]
        and
        abs(
            pkm_cost_error
        )
        <= ABS_TOL[
            "service_cost"
        ]
    )
    harmonized_identities_gate = all([
        lcoe_harm_gate,
        service_cost_gate,
    ])
    # -----------------------------------------------------------------
    # Route isolation
    # -----------------------------------------------------------------
    route_gate = (
        str(
            row[
                "route"
            ]
        )
        == "hydrogen"
    )
    b2_fields = [
        "chp_kw",
        "biogas_storage_nm3",
        "substrate_t_day",
        "substrate_t_year",
    ]
    contamination = {}
    for col in b2_fields:
        if col not in df.columns:
            contamination[col] = False
        else:
            value = row[col]
            contamination[col] = (
                pd.notna(value)
                and
                abs(float(value))
                > 1e-12
            )
    no_b2_contamination_gate = (
        not any(
            contamination.values()
        )
    )
    route_isolation_gate = all([
        route_gate,
        no_b2_contamination_gate,
    ])
    schema_gate = (
        schema_contract_pass(
            schema_gates
        )
    )
    all_pass = all([
        capacity_gate,
        baseline_numerical_gate,
        harmonized_identities_gate,
        route_isolation_gate,
        schema_gate,
    ])
    return {
        "csv_path":
            csv_path,
        "run_dir":
            run_dir,
        "reference_case":
            H2_REFERENCE_CASE,
        "capacity_gate":
            capacity_gate,
        "baseline_numerical_gate":
            baseline_numerical_gate,
        "harmonized_identities_gate":
            harmonized_identities_gate,
        "route_isolation_gate":
            route_isolation_gate,
        "schema_gate":
            schema_gate,
        "pgrid_delta":
            delta[
                "P_peak_grid_kW"
            ],
        "egrid_delta":
            delta[
                "E_grid_kWh"
            ],
        "dependency_delta":
            delta[
                "grid_dependency"
            ],
        "legacy_lcoe_delta":
            delta[
                "LCOE_USD_kWh"
            ],
        "harmonized_lcoe_error":
            lcoe_harm_error,
        "vkm_cost_error":
            vkm_cost_error,
        "pkm_cost_error":
            pkm_cost_error,
        "contamination":
            contamination,
        "all_pass":
            all_pass,
    }
# =====================================================================
# VERSIONABLE EVIDENCE
# =====================================================================
EVIDENCE_DIR = (
    ROOT
    / "results"
    / "validation"
    / "pareto2026_dual_route"
)
EVIDENCE_PATH = (
    EVIDENCE_DIR
    / "dual_route_regression_summary.json"
)
def sha256_file(
    path: Path,
) -> str:
    """
    Calculate SHA-256 without loading the entire file into memory.
    """
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
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
            digest.update(chunk)
    return digest.hexdigest()
def project_relative_path(
    path: Path,
) -> str:
    """
    Return a repository-relative POSIX path whenever possible.
    """
    path = path.resolve()
    root = ROOT.resolve()
    try:
        return (
            path
            .relative_to(root)
            .as_posix()
        )
    except ValueError:
        return str(path)
def artifact_record(
    path: Path,
) -> dict:
    """
    Build immutable identity metadata for one evidence artifact.
    """
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return {
        "path":
            project_relative_path(
                path
            ),
        "sha256":
            sha256_file(
                path
            ),
        "size_bytes":
            int(
                path.stat().st_size
            ),
    }
def write_dual_route_evidence(
    b2: dict,
    h2: dict,
    dual_route_gate: bool,
) -> Path:
    """
    Persist the current baseline-2026 dual-route validation state.
    The evidence file is intentionally generated from the same
    in-memory gate results printed by this regression script.
    """
    timestamp = (
        datetime.now(
            timezone.utc
        )
        .isoformat(
            timespec="seconds"
        )
    )
    b2_csv = Path(
        b2["csv_path"]
    )
    h2_csv = Path(
        h2["csv_path"]
    )
    h2_reference = Path(
        H2_REFERENCE_PATH
    )
    summary = {
        "schema_version":
            "1.0",
        "timestamp":
            timestamp,
        "timestamp_timezone":
            "UTC",
        "baseline_status":
            (
                "PASS"
                if dual_route_gate
                else "FAIL"
            ),
        "baseline_definition":
            (
                "B2 end-to-end smoke AND "
                "H2 corrected-2026 current-pipeline regression"
            ),
        "b2_run_path":
            project_relative_path(
                Path(
                    b2["run_dir"]
                )
            ),
        "h2_run_path":
            project_relative_path(
                Path(
                    h2["run_dir"]
                )
            ),
        "h2_reference_path":
            project_relative_path(
                h2_reference
            ),
        "B2": {
            "gates": {
                "B2_SCHEMA_CONTRACT_GATE":
                    bool(
                        b2[
                            "schema_gate"
                        ]
                    ),
                "B2_ROUTE_GATE":
                    bool(
                        b2[
                            "route_gate"
                        ]
                    ),
                "B2_OBJECTIVE_GATE":
                    bool(
                        b2[
                            "objective_gate"
                        ]
                    ),
                "B2_SERVICE_GATE":
                    bool(
                        b2[
                            "service_gate"
                        ]
                    ),
                "B2_PHYSICAL_COUPLING_GATE":
                    bool(
                        b2[
                            "physical_coupling_gate"
                        ]
                    ),
                "B2_HARMONIZED_ECONOMICS_GATE":
                    bool(
                        b2[
                            "economics_gate"
                        ]
                    ),
                "B2_END_TO_END_SMOKE_GATE":
                    bool(
                        b2[
                            "all_pass"
                        ]
                    ),
            },
            "residuals": {
                "storage_max_error_nm3":
                    float(
                        b2[
                            "storage_max_error"
                        ]
                    ),
                "substrate_max_error_t_day":
                    float(
                        b2[
                            "substrate_max_error"
                        ]
                    ),
                "grid_dependency_max_error":
                    float(
                        b2[
                            "dependency_max_error"
                        ]
                    ),
                "harmonized_lcoe_max_error_usd_kwh":
                    float(
                        b2[
                            "lcoe_max_error"
                        ]
                    ),
            },
        },
        "H2": {
            "reference_case":
                str(
                    h2[
                        "reference_case"
                    ]
                ),
            "gates": {
                "H2_FIXED_CAPACITY_GATE":
                    bool(
                        h2[
                            "capacity_gate"
                        ]
                    ),
                "H2_BASELINE_NUMERICAL_GATE":
                    bool(
                        h2[
                            "baseline_numerical_gate"
                        ]
                    ),
                "H2_HARMONIZED_IDENTITIES_GATE":
                    bool(
                        h2[
                            "harmonized_identities_gate"
                        ]
                    ),
                "H2_ROUTE_ISOLATION_GATE":
                    bool(
                        h2[
                            "route_isolation_gate"
                        ]
                    ),
                "H2_SCHEMA_GATE":
                    bool(
                        h2[
                            "schema_gate"
                        ]
                    ),
                "H2_CURRENT_PIPELINE_REGRESSION":
                    bool(
                        h2[
                            "all_pass"
                        ]
                    ),
            },
            "deltas": {
                "delta_pgrid_kw":
                    float(
                        h2[
                            "pgrid_delta"
                        ]
                    ),
                "delta_egrid_kwh":
                    float(
                        h2[
                            "egrid_delta"
                        ]
                    ),
                "delta_grid_dependency":
                    float(
                        h2[
                            "dependency_delta"
                        ]
                    ),
                "delta_legacy_lcoe_usd_kwh":
                    float(
                        h2[
                            "legacy_lcoe_delta"
                        ]
                    ),
                "harmonized_lcoe_identity_error_usd_kwh":
                    float(
                        h2[
                            "harmonized_lcoe_error"
                        ]
                    ),
                "vehicle_km_cost_identity_error":
                    float(
                        h2[
                            "vkm_cost_error"
                        ]
                    ),
                "passenger_km_cost_identity_error":
                    float(
                        h2[
                            "pkm_cost_error"
                        ]
                    ),
            },
            "b2_field_contamination": {
                key:
                    bool(value)
                for key, value
                in h2[
                    "contamination"
                ].items()
            },
        },
        "artifacts": {
            "b2_pareto_2026_enriched_csv":
                artifact_record(
                    b2_csv
                ),
            "h2_pareto_2026_enriched_csv":
                artifact_record(
                    h2_csv
                ),
            "h2_regression_summary_csv":
                artifact_record(
                    h2_reference
                ),
        },
        "BASELINE_2026_DUAL_ROUTE_GATE":
            bool(
                dual_route_gate
            ),
    }
    EVIDENCE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary_path = (
        EVIDENCE_PATH
        .with_suffix(
            ".json.tmp"
        )
    )
    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        f.write(
            "\n"
        )
    # Atomic replacement prevents a partially-written
    # evidence JSON from being mistaken for a valid baseline.
    temporary_path.replace(
        EVIDENCE_PATH
    )
    return EVIDENCE_PATH
# =====================================================================
# MAIN
# =====================================================================
def main() -> None:
    print("=" * 112)
    print(
        "PARETO 2026 — DUAL-ROUTE REGRESSION"
    )
    print("=" * 112)
    try:
        b2 = validate_b2()
        print()
        print("B2 END-TO-END")
        print_gate(
            "B2_SCHEMA_CONTRACT_GATE",
            b2["schema_gate"],
        )
        print_gate(
            "B2_ROUTE_GATE",
            b2["route_gate"],
        )
        print_gate(
            "B2_OBJECTIVE_GATE",
            b2["objective_gate"],
        )
        print_gate(
            "B2_SERVICE_GATE",
            b2["service_gate"],
        )
        print_gate(
            "B2_PHYSICAL_COUPLING_GATE",
            b2[
                "physical_coupling_gate"
            ],
        )
        print_gate(
            "B2_HARMONIZED_ECONOMICS_GATE",
            b2[
                "economics_gate"
            ],
        )
        print_gate(
            "B2_END_TO_END_SMOKE_GATE",
            b2["all_pass"],
        )
        print()
        print(
            "B2 numerical residuals"
        )
        print(
            "  storage max error [Nm3]    = "
            f"{b2['storage_max_error']:.12e}"
        )
        print(
            "  substrate max error [t/d]  = "
            f"{b2['substrate_max_error']:.12e}"
        )
        print(
            "  grid dependency max error  = "
            f"{b2['dependency_max_error']:.12e}"
        )
        print(
            "  harmonized LCOE max error  = "
            f"{b2['lcoe_max_error']:.12e}"
        )
        h2 = validate_h2()
        print()
        print("-" * 112)
        print()
        print(
            "H2 CURRENT-PIPELINE REGRESSION"
        )
        print_gate(
            "H2_FIXED_CAPACITY_GATE",
            h2["capacity_gate"],
        )
        print_gate(
            "H2_BASELINE_NUMERICAL_GATE",
            h2[
                "baseline_numerical_gate"
            ],
        )
        print_gate(
            "H2_HARMONIZED_IDENTITIES_GATE",
            h2[
                "harmonized_identities_gate"
            ],
        )
        print_gate(
            "H2_ROUTE_ISOLATION_GATE",
            h2[
                "route_isolation_gate"
            ],
        )
        print_gate(
            "H2_SCHEMA_GATE",
            h2[
                "schema_gate"
            ],
        )
        print_gate(
            "H2_CURRENT_PIPELINE_REGRESSION",
            h2["all_pass"],
        )
        print()
        print(
            "H2 regression deltas"
        )
        print(
            "  delta Pgrid [kW]           = "
            f"{h2['pgrid_delta']:+.12e}"
        )
        print(
            "  delta Egrid [kWh]          = "
            f"{h2['egrid_delta']:+.12e}"
        )
        print(
            "  delta grid dependency      = "
            f"{h2['dependency_delta']:+.12e}"
        )
        print(
            "  delta legacy LCOE          = "
            f"{h2['legacy_lcoe_delta']:+.12e}"
        )
        print(
            "  harmonized LCOE identity   = "
            f"{h2['harmonized_lcoe_error']:+.12e}"
        )
        print()
        print(
            "H2 B2-field contamination"
        )
        for col, bad in (
            h2[
                "contamination"
            ].items()
        ):
            print(
                f"  {col:28s} = "
                f"{bad}"
            )
        # =============================================================
        # COMPOSITE BASELINE GATE
        # =============================================================
        dual_route_gate = (
            b2["all_pass"]
            and
            h2["all_pass"]
        )
        print()
        print("=" * 112)
        print_gate(
            "B2_END_TO_END_SMOKE_GATE",
            b2["all_pass"],
        )
        print_gate(
            "H2_CURRENT_PIPELINE_REGRESSION",
            h2["all_pass"],
        )
        print("-" * 112)
        print_gate(
            "BASELINE_2026_DUAL_ROUTE_GATE",
            dual_route_gate,
        )
        print("=" * 112)
        print()
        print("ARTIFACTS")
        print(
            f"B2 CSV : "
            f"{b2['csv_path']}"
        )
        print(
            f"B2 RUN : "
            f"{b2['run_dir']}"
        )
        print(
            f"H2 CSV : "
            f"{h2['csv_path']}"
        )
        print(
            f"H2 RUN : "
            f"{h2['run_dir']}"
        )
        print(
            f"H2 REF : "
            f"{H2_REFERENCE_PATH}"
        )
        evidence_path = (
            write_dual_route_evidence(
                b2=b2,
                h2=h2,
                dual_route_gate=dual_route_gate,
            )
        )
        print()
        print(
            "VERSIONABLE EVIDENCE"
        )
        print(
            f"JSON   : "
            f"{evidence_path}"
        )
        print(
            "STATUS : "
            + (
                "PASS"
                if dual_route_gate
                else "FAIL"
            )
        )
        if not dual_route_gate:
            raise SystemExit(2)
    except Exception as exc:
        print()
        print("=" * 112)
        print(
            "BASELINE_2026_DUAL_ROUTE_GATE = ERROR"
        )
        print(
            f"{type(exc).__name__}: {exc}"
        )
        print("=" * 112)
        raise
if __name__ == "__main__":
    main()
