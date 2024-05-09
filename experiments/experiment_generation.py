import yaml
from tqdm.contrib.itertools import product as tqdm_product
from objective_dispatcher import dispatch_objective, get_available_objectives
from dynamics_dispatcher import dispatch_dynamics
from config import ExperimentConfig, ConfigContainerDynamic
from typing import Any, Generator


def load_config(filename: str) -> Any:
    """Load a configuration file.

    Args:
        filename (str): The path to the configuration file.

    Returns:
        Any: The configuration file.
    """
    with open(filename, "r") as file:
        return yaml.safe_load(file)


def range_generator(param_config: dict) -> list:  #
    """Generate values out of range and step.

    Args:
        param_config (dict): A dictionary containing the range and step.

    Returns:
        list: A list of values.
    """
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


def _dict_product_generator(d: dict) -> Generator[dict, Any, Any]:
    """Generate all possible combinations of values in a dictionary.

    Args:
        d (dict): A dictionary containing the values to combine.

    Yields:
        dict: A dictionary containing the values of the combinations.
    """
    keys = d.keys()
    for combination in tqdm_product(
        *[v if isinstance(v, list) else [v] for v in d.values()]
    ):
        result = {}
        for key, value in zip(keys, combination):
            result[key] = value
        yield result


def _generate_configs_dynamics(config_dynamics: dict) -> dict:
    """Generate configurations for dynamics.

    Args:
        config_dynamics (dict): A dictionary containing the dynamics configurations.

    Returns:
        dict: A dictionary containing the configurations.
    """
    config = {}
    for cfg_name, value in config_dynamics.items():
        if isinstance(value, dict) and "range" in value and "step" in value:
            config[cfg_name] = range_generator(value)
        elif cfg_name == "name_f":  # dispatch and parse keyword all for function
            if value == "all":
                config[cfg_name] = [obj_name for obj_name in get_available_objectives()]
            elif isinstance(value, list):
                config[cfg_name] = value
            else:
                config[cfg_name] = value
        else:
            config[cfg_name] = value
    config_dynamic_generator = _dict_product_generator(config)
    return config_dynamic_generator


def _get_name_and_config_dynamics(experiment_config: dict) -> list:
    """Get the name and configuration of dynamics.

    Args:
        experiment_config (dict): A dictionary containing the experiment configurations.

    Returns:
        list: A list of tuples containing the name and configuration of dynamics.
    """
    selected_dynamics = experiment_config["selected_dynamics"]
    return [
        (name_dynamic, experiment_config["config_dynamics"][name_dynamic])
        for name_dynamic in selected_dynamics
    ]


def generate_dynamics_product(
    experiment_config: dict[str, dict]
) -> Generator[ConfigContainerDynamic, Any, Any]:
    """Generate the product of dynamics configurations.

    Args:
        experiment_config (dict[str, dict]): A dictionary containing the experiment configurations.

    Yields:
        ConfigContainerDynamic: A container containing the dynamics configurations.
    """
    dynamics_name_and_cfg = _get_name_and_config_dynamics(experiment_config)
    for name_dynamic, _tmp_config_dynamic in dynamics_name_and_cfg:
        for i, configuration in enumerate(
            _generate_configs_dynamics(_tmp_config_dynamic)
        ):
            name_f = configuration["name_f"]
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
    """Create an experiment configuration.

    Args:
        file_path (str): The path to the configuration file.

    Returns:
        ExperimentConfig: The experiment configuration.
    """
    cfg = load_config(file_path)
    config_container_dynamic_gen = generate_dynamics_product(cfg)
    experiment_name = cfg["name"]
    experiment_config = ExperimentConfig(
        experiment_name=experiment_name,
        config_container_dynamic_gen=config_container_dynamic_gen,
        config_opt=cfg["config_opt"],
    )
    return experiment_config
