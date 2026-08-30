from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.optimization.ga_nsga2 import NSGA2Optimizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "paper"
    / "pv_bsv_h2_1500.yaml"
)

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "fleet_demand_sp.csv"
)


SEEDS = [
    {
        "pv_kw": 1000,
        "bsv_kwh": 924,
        "electrolyzer_kw": 450,
        "h2_tank_kg": 205,
        "fuelcell_kw": 117,
    },
    {
        "pv_kw": 847,
        "bsv_kwh": 1289,
        "electrolyzer_kw": 485,
        "h2_tank_kg": 140,
        "fuelcell_kw": 111,
    },
    {
        "pv_kw": 940,
        "bsv_kwh": 1267,
        "electrolyzer_kw": 407,
        "h2_tank_kg": 128,
        "fuelcell_kw": 105,
    },
    {
        "pv_kw": 938,
        "bsv_kwh": 1319,
        "electrolyzer_kw": 399,
        "h2_tank_kg": 106,
        "fuelcell_kw": 100,
    },
]


def _load_config() -> dict:
    return yaml.safe_load(
        CONFIG_PATH.read_text(encoding="utf-8")
    )


def _build_optimizer(config: dict) -> NSGA2Optimizer:
    df = pd.read_csv(DATA_PATH).head(24)
    return NSGA2Optimizer(config, df)


def _population_keys(
    optimizer: NSGA2Optimizer,
    population,
):
    return [
        optimizer._capacities_key(ind.capacities)
        for ind in population
    ]


def test_legacy_population_initialization_is_preserved():
    config = _load_config()

    optimizer = _build_optimizer(config)
    population = optimizer._initialize_population()

    keys = _population_keys(
        optimizer,
        population,
    )

    assert len(population) == optimizer.population_size
    assert len(set(keys)) == optimizer.population_size
    assert not config.get(
        "warm_start",
        {},
    ).get("enabled", False)


def test_warm_start_seeds_are_inserted_first():
    config = _load_config()

    config["warm_start"] = {
        "enabled": True,
        "seeds": copy.deepcopy(SEEDS),
    }

    optimizer = _build_optimizer(config)
    population = optimizer._initialize_population()

    expected = [
        optimizer._capacities_key(seed)
        for seed in SEEDS
    ]

    got = _population_keys(
        optimizer,
        population[: len(SEEDS)],
    )

    assert len(population) == optimizer.population_size
    assert got == expected


def test_duplicate_warm_start_seed_is_removed():
    config = _load_config()

    seeds = copy.deepcopy(SEEDS)
    seeds.append(copy.deepcopy(SEEDS[0]))

    config["warm_start"] = {
        "enabled": True,
        "seeds": seeds,
    }

    optimizer = _build_optimizer(config)
    population = optimizer._initialize_population()

    keys = _population_keys(
        optimizer,
        population,
    )

    first_seed_key = optimizer._capacities_key(
        SEEDS[0]
    )

    assert len(population) == optimizer.population_size
    assert len(set(keys)) == optimizer.population_size
    assert keys.count(first_seed_key) == 1


def test_warm_start_seed_is_repaired_to_bounds():
    config = _load_config()

    config["warm_start"] = {
        "enabled": True,
        "seeds": [
            {
                "pv_kw": 9999,
                "bsv_kwh": 9999,
                "electrolyzer_kw": 9999,
                "h2_tank_kg": 9999,
                "fuelcell_kw": 9999,
            }
        ],
    }

    optimizer = _build_optimizer(config)
    population = optimizer._initialize_population()

    repaired = population[0].capacities

    assert repaired["pv_kw"] == 1200.0
    assert repaired["bsv_kwh"] == 1500.0
    assert repaired["electrolyzer_kw"] == 500.0
    assert repaired["h2_tank_kg"] == 300.0
    assert repaired["fuelcell_kw"] == 500.0


def test_incomplete_warm_start_seed_is_rejected():
    config = _load_config()

    config["warm_start"] = {
        "enabled": True,
        "seeds": [
            {
                "pv_kw": 1000,
                "bsv_kwh": 924,
            }
        ],
    }

    optimizer = _build_optimizer(config)

    with pytest.raises(
        ValueError,
        match="missing variables",
    ):
        optimizer._initialize_population()
