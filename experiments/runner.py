from config import ExperimentConfig
from result import ExperimentResult, ResultDynamicRun
import cbx
from cbx.scheduler import multiply
import time
import os


def timing(method):
    def wrapper(*args, **kw):
        startTime = int(round(time.time() * 1000))
        result = method(*args, **kw)
        endTime = int(round(time.time() * 1000))
        elapsed_time = endTime - startTime
        return elapsed_time, result

    return wrapper


@timing
def optimize_wrapper(dynamic, *args, **kwargs):
    return dynamic.optimize(*args, **kwargs)


class Runner:
    def __init__(
        self, experiment_config: ExperimentConfig, result_dir: str = "results"
    ) -> None:
        self.experiment_config: ExperimentConfig = experiment_config
        self.experiment_result: ExperimentResult = ExperimentResult(
            experiment_config.experiment_name,
            experiment_config.config_opt,
            results_dynamic=[],
        )
        self.result_dir = result_dir

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
        _config_opt = self.experiment_config.config_opt
        opt_dict = {
            "sched": multiply(
                factor=_config_opt["sched"]["factor"],
                maximum=_config_opt["sched"]["maximum"],
            ),
            "print_int": _config_opt["print_int"],
        }
        for (
            config_container_dynamic
        ) in self.experiment_config.config_container_dynamic_gen:
            f = config_container_dynamic.f
            config_dynamic = config_container_dynamic.config_dynamic
            dynamic = config_container_dynamic.dynamic(f, **config_dynamic)
            time, best_x = optimize_wrapper(dynamic, **opt_dict)
            best_f = f(best_x)
            results_dynamic.append(
                ResultDynamicRun(
                    name_dynamic=config_container_dynamic.name_dynamic,
                    name_f=config_container_dynamic.name_f,
                    index_config=config_container_dynamic.index_config,
                    config_dynamic=config_dynamic,
                    time=time,
                    best_f=best_f,
                    best_x=best_x,
                )
            )
        return results_dynamic

    # TODO: Implement starting positions for comparability
    def set_starting_positions(
        self, experiment: ExperimentConfig, x_min=-3.0, x_max=3.0
    ):
        return cbx.utils.init_particles(
            shape=(
                experiment.config_dynamic["M"],
                experiment.config_dynamic["N"],
                experiment.config_dynamic["d"],
            ),
            x_min=x_min,
            x_max=x_max,
        )
