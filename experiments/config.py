from dataclasses import dataclass
from cbx.dynamics import ParticleDynamic
from cbx.utils.objective_handling import cbx_objective
from typing import Generator


@dataclass
class ConfigContainerDynamic:
    name_dynamic: str
    name_f: str
    dynamic: ParticleDynamic
    f: cbx_objective
    index_config: int
    config_dynamic: dict


@dataclass
class ExperimentConfig:
    experiment_name: str
    config_container_dynamic_gen: Generator[ConfigContainerDynamic, None, None]
    config_opt: dict
