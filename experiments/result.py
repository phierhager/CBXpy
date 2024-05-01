from dataclasses import dataclass
import numpy as np
import csv


@dataclass
class ResultDynamicRun:
    name_dynamic: str
    name_f: str
    index_config: int
    config_dynamic: dict
    time: float
    best_f: np.ndarray
    best_x: np.ndarray

    def to_list(self):
        return [
            self.name_dynamic,
            self.name_f,
            self.index_config,
            self.time,
            self.best_f.tolist(),
            self.best_x.tolist(),
            str(self.config_dynamic),
        ]


@dataclass
class ExperimentResult:
    experiment_name: str
    config_opt: dict
    results_dynamic: list[ResultDynamicRun]

    @classmethod
    def from_csv(cls, filename):
        with open(filename, "r", newline="") as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            experiment_name = None
            config_opt = None
            results_dynamic = []
            for row in reader:
                if experiment_name is None:
                    experiment_name = row[0]
                    config_opt = {}  # Assuming config_opt is not stored in CSV
                result_dynamic = ResultDynamicRun(
                    name_dynamic=row[1],
                    name_f=row[2],
                    index_config=int(row[3]),
                    config_dynamic=eval(
                        row[4]
                    ),  # Convert string representation of dict to dict
                    time=float(row[5]),
                    best_f=np.array(
                        eval(row[6])
                    ),  # Convert string representation of list to numpy array
                    best_x=np.array(
                        eval(row[7])
                    ),  # Convert string representation of list to numpy array
                )
                results_dynamic.append(result_dynamic)

            return cls(experiment_name, config_opt, results_dynamic)

    def to_csv(self, filename):
        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "experiment_name",
                    "name_dynamic",
                    "name_f",
                    "index_config",
                    "time",
                    "best_f",
                    "best_x",
                    "config_dynamic",
                ]
            )
            for result in self.results_dynamic:
                writer.writerow([self.experiment_name] + result.to_list())
