from runner import Runner
from experiment_generation import create_experiment_config
import sys

if __name__ == "__main__":
    if len(sys.argv) == 2:
        file_path = sys.argv[1]
    else:
        file_path = r".\experiments\configs\33_test.yaml"  # Provide the path to your YAML configuration file
    experiment_config = create_experiment_config(file_path)
    runner = Runner(experiment_config)
    experiment_result = runner.run_experiment()
