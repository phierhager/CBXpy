from dataclasses import dataclass
import numpy as np
import pandas as pd
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
    success: np.ndarray  # binary array
    # starting_positions: np.ndarray

    def to_list(self) -> list:
        """Convert the result to a list.

        Returns:
            list: A list containing the result.
        """
        return [
            self.name_dynamic,
            self.name_f,
            self.index_config,
            self.time,
            self.best_f.tolist(),
            self.best_x.tolist(),
            self.success.tolist(),
            # self.starting_positions.tolist(),
            str(self.config_dynamic),
        ]


@dataclass
class ExperimentResult:
    experiment_name: str
    config_opt: dict
    results_dynamic: list[ResultDynamicRun]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the experiment result to a pandas DataFrame.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the experiment result.
        """
        data = []

        # only temporary, as here should not be any logic
        # add relative function value
        maximal_f_values_reached = {}
        minimal_f_values_reached = {}
        for result in self.results_dynamic:
            min_i = np.argmin(result.best_f)
            if result.name_f not in maximal_f_values_reached:
                maximal_f_values_reached[result.name_f] = result.best_f[min_i]
            elif maximal_f_values_reached[result.name_f] < result.best_f[min_i]:
                maximal_f_values_reached[result.name_f] = result.best_f[min_i]

            if result.name_f not in minimal_f_values_reached:
                minimal_f_values_reached[result.name_f] = result.best_f[min_i]
            elif minimal_f_values_reached[result.name_f] > result.best_f[min_i]:
                minimal_f_values_reached[result.name_f] = result.best_f[min_i]

        for result in self.results_dynamic:
            min_i = np.argmin(result.best_f)
            rel_f_value = (
                result.best_f[min_i] - minimal_f_values_reached[result.name_f]
            ) / (
                maximal_f_values_reached[result.name_f]
                - minimal_f_values_reached[result.name_f]
            )
            config_data = {
                "Experiment Name": self.experiment_name,
                "Dynamic Name": result.name_dynamic,
                "Function Name": result.name_f,
                "Time": result.time,
                "Best Function Value": result.best_f[min_i],
                "Relative Function Value": rel_f_value,
                "Best Solution": result.best_x[min_i],
                "Success": result.success,
                # "Starting Positions": result.starting_positions,
            }
            config_data.update(
                result.config_dynamic
            )  # Include all config_dynamic keys and values
            data.append(config_data)
        return pd.DataFrame(data)

    @classmethod
    def from_csv(cls, filename: str) -> "ExperimentResult":
        """Create an ExperimentResult object from a CSV file.

        Args:
            filename (str): The filename of the CSV file.

        Returns:
            ExperimentResult: An ExperimentResult object.
        """
        csv.field_size_limit(1000000)
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
                    time=float(row[4]),
                    best_f=np.array(
                        eval(row[5])
                    ),  # Convert string representation of list to numpy array
                    best_x=np.array(
                        eval(row[6])
                    ),  # Convert string representation of list to numpy array
                    success=np.array(
                        eval(row[7])
                    ),  # Convert string representation of list to numpy array
                    # starting_positions=np.array(
                    #     eval(row[8])
                    # ),  # Convert string representation of list to numpy array
                    config_dynamic=eval(
                        row[8]
                    ),  # Convert string representation of dict to dict
                )
                results_dynamic.append(result_dynamic)

            return cls(experiment_name, config_opt, results_dynamic)

    def to_csv(self, filename: str):
        """Write the ExperimentResult object to a CSV file.

        Args:
            filename (str): The filename of the CSV file.
        """
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
                    "success",
                    "config_dynamic",
                ]
            )
            for result in self.results_dynamic:
                writer.writerow([self.experiment_name] + result.to_list())
