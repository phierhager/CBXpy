import yaml
import itertools
from objective_dispatcher import dispatch_objective, get_available_objectives
from dynamics_dispatcher import dispatch_dynamics
from config import ExperimentConfig, ConfigContainerDynamic


def load_config(filename: str):
    with open(filename, "r") as file:
        return yaml.safe_load(file)


def range_generator(param_config):  #
    """Generate values out of range and step"""
    start, end = param_config["range"]
    step = param_config["step"]
    values = []
    for x in list(
        float(start) + step * i for i in range(int((end - start) / step) + 1)
    ):
        if isinstance(step, int):
            values.append(round(x))
        else:
            values.append(round(x, 4))
    return values


def _dict_product_generator(d: dict):
    """Make a cartesian product out of a dict."""
    keys = d.keys()
    for combination in itertools.product(
        *[v if isinstance(v, list) else [v] for v in d.values()]
    ):
        result = {}
        for key, value in zip(keys, combination):
            result[key] = value
        yield result


def _generate_configs_dynamics(config_dynamics: dict):
    """Create config of dynamic"""
    config = {}
    for cfg_name, value in config_dynamics.items():
        if isinstance(value, dict) and "range" in value and "step" in value:
            config[cfg_name] = range_generator(value)
        elif cfg_name == "name_f":  # dispatch and parse keyword all for function
            if value == "all":
                config[cfg_name] = [obj_name for obj_name in get_available_objectives()]
            else:
                config[cfg_name] = value
        else:
            config[cfg_name] = value
    config_dynamic_generator = _dict_product_generator(config)
    return config_dynamic_generator


def _get_name_and_config_dynamics(experiment_config: dict) -> list:
    selected_dynamics = experiment_config["selected_dynamics"]
    return [
        (name_dynamic, experiment_config["config_dynamics"][name_dynamic])
        for name_dynamic in selected_dynamics
    ]


def generate_dynamics_product(experiment_config: dict[str, dict]):
    """generate dynamics config product"""
    dynamics_name_and_cfg = _get_name_and_config_dynamics(experiment_config)
    for name_dynamic, _tmp_config_dynamic in dynamics_name_and_cfg:
        for i, configuration in enumerate(
            _generate_configs_dynamics(_tmp_config_dynamic)
        ):
            name_f = configuration["name_f"]
            print(configuration["name_f"])
            f = dispatch_objective(name_f)
            configuration.pop("name_f", None)
            dynamic = dispatch_dynamics(name_dynamic)
            yield ConfigContainerDynamic(
                name_dynamic=name_dynamic,
                name_f=name_f,
                dynamic=dynamic,
                f=f,
                index_config=i,
                config_dynamic=configuration,
            )


def create_experiment_config(file_path: str) -> ExperimentConfig:
    cfg = load_config(file_path)
    config_container_dynamic_gen = generate_dynamics_product(cfg)
    experiment_name = cfg["name"]
    experiment_config = ExperimentConfig(
        experiment_name=experiment_name,
        config_container_dynamic_gen=config_container_dynamic_gen,
        config_opt=cfg["config_opt"],
    )
    return experiment_config
