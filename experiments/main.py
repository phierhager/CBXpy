from runner import Runner
from experiment_generation import create_experiment_config
import sys

if __name__ == "__main__":
    if sys.argv[1]:
        file_path = sys.argv[1]
    else:
        file_path = r"experiments\configs\qcbo_cbo_3.yaml"  # Provide the path to your YAML configuration file
    experiment_config = create_experiment_config(file_path)
    runner = Runner(experiment_config)
    experiment_result = runner.run_experiment()
