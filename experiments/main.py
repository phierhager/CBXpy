from runner import Runner
from experiment_generation import create_experiment_config

if __name__ == "__main__":
    file_path = (
        r"experiments\easy.yaml"  # Provide the path to your YAML configuration file
    )
    experiment_config = create_experiment_config(file_path)
    runner = Runner(experiment_config)
    experiment_result = runner.run_experiment()
