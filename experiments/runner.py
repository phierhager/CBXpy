import scipy.optimize
import cbx.utils
import cbx.utils.objective_handling
from config import ExperimentConfig, ConfigContainerDynamic
from result import ExperimentResult, ResultDynamicRun
import cbx
from cbx.scheduler import multiply
import time
import os
import numpy as np
from scipy import optimize
from tqdm import tqdm
from typing import Callable


def timing(method: Callable):
    """Time the execution of a method.

    Args:
        method (_type_): _description_
    """

    def wrapper(*args, **kw):
        startTime = int(round(time.time() * 1000))
        result = method(*args, **kw)
        endTime = int(round(time.time() * 1000))
        elapsed_time = endTime - startTime
        return elapsed_time, result

    return wrapper


@timing
def optimize_wrapper(dynamic: cbx.dynamics.ParticleDynamic, *args, **kwargs):
    # dynamic.verbosity = 0
    return dynamic.optimize(*args, **kwargs)


class Runner:
    def __init__(
        self,
        experiment_config: ExperimentConfig,
        result_dir: str = "results",
    ) -> None:
        self.experiment_config: ExperimentConfig = experiment_config
        self.experiment_result: ExperimentResult = ExperimentResult(
            experiment_config.experiment_name,
            experiment_config.config_opt,
            results_dynamic=[],
        )
        self.result_dir = result_dir
        self.starting_positions = None  # will be set in the first run to sth
        self.local_minimum_functions = (
            {}
        )  # will be filled for every funnction calculate nearest local minimum with scipy

    def run_experiment(self):
        results_dynamic = self.run_dynamic_configs()
        self.experiment_result.results_dynamic = results_dynamic
        if self.result_dir is not None:
            if not os.path.exists(self.result_dir):
                os.makedirs(self.result_dir)
            self.experiment_result.to_csv(
                self.result_dir
                + "/"
                + "_".join(str(self.experiment_result.experiment_name).lower().split())
                + ".csv"
            )
        return self.experiment_result

    def run_dynamic_configs(self):
        results_dynamic = []
        opt_dict = self.get_opt_dict()
        for config_container_dynamic in tqdm(
            self.experiment_config.config_container_dynamic_gen
        ):
            if self.starting_positions is None:
                self.set_starting_positions(
                    config_container_dynamic
                )  # only if not yet done
            if not config_container_dynamic.f in self.local_minimum_functions.keys():
                _prom_f = cbx.utils.objective_handling._promote_objective(
                    config_container_dynamic.f,
                    config_container_dynamic.config_dynamic["f_dim"],
                )
                local_min_x_values = np.squeeze(
                    np.apply_along_axis(
                        lambda x: optimize.fmin(_prom_f, x, disp=False),
                        axis=2,
                        arr=self.starting_positions,
                    )
                )
                self.local_minimum_functions[config_container_dynamic.f] = np.squeeze(
                    np.min(
                        np.apply_along_axis(_prom_f, axis=2, arr=local_min_x_values),
                        axis=1,
                    )
                )
            result = self.run_dynamic_config(config_container_dynamic, opt_dict)
            results_dynamic.append(result)
        return results_dynamic

    def run_dynamic_config(
        self,
        config_container_dynamic: ConfigContainerDynamic,
        opt_dict: dict,
    ):
        f = config_container_dynamic.f
        config_dynamic = config_container_dynamic.config_dynamic
        config_dynamic.pop("x")
        dynamic = config_container_dynamic.dynamic(
            f, x=self.starting_positions, **config_dynamic
        )
        time, best_x = optimize_wrapper(dynamic, **opt_dict)  # optimize
        best_f = f(best_x)
        success = np.where(
            best_f < self.local_minimum_functions[config_container_dynamic.f], 1, 0
        )
        return ResultDynamicRun(
            name_dynamic=config_container_dynamic.name_dynamic,
            name_f=config_container_dynamic.name_f,
            index_config=config_container_dynamic.index_config,
            config_dynamic=config_dynamic,
            time=time,
            best_f=best_f,
            best_x=best_x,
            success=success,
            local_min_f_values=self.local_minimum_functions[config_container_dynamic.f],
            starting_positions=self.starting_positions,
        )

    def set_starting_positions(self, config_container_dynamic: ConfigContainerDynamic):
        shape = (
            config_container_dynamic.config_dynamic["M"],
            config_container_dynamic.config_dynamic["N"],
            config_container_dynamic.config_dynamic["d"],
        )
        self.starting_positions = np.random.uniform(
            config_container_dynamic.config_dynamic["x_min"],
            config_container_dynamic.config_dynamic["x_max"],
            shape,
        )

    def get_opt_dict(self):
        _config_opt = self.experiment_config.config_opt
        return {
            "sched": multiply(
                factor=_config_opt["sched"]["factor"],
                maximum=_config_opt["sched"]["maximum"],
            ),
            "print_int": _config_opt["print_int"],
        }
