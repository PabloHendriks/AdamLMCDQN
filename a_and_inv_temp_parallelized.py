import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed

from agents.adamLMCdqn import main as adamLMCDQN

def worker(base_config, a, inverse_temperature):
    # Copy base config to avoid shared-state mutations
    config = base_config.copy()
    config["a"] = a
    config["inverse_temperature"] = inverse_temperature
    # Run the training function
    return adamLMCDQN(config)

if __name__ == "__main__":
    # Load base configuration
    with open("configs/cartPole.yaml", "r") as f:
        base_config = yaml.safe_load(f)

    # Load hyperparameter sweep values
    with open("configs/sweeps/a_and_inverse_temperature_sweep.yaml", "r") as f:
        sweep = yaml.safe_load(f)

    # Prepare argument tuples for each combination
    tasks = [
        (base_config, a, inv_temp)
        for a in sweep["a"]
        for inv_temp in sweep["inverse_temperature"]
    ]

    # Execute in parallel across all CPU cores
    with ProcessPoolExecutor(max_workers=12) as executor:
        futures = [
            executor.submit(worker, cfg, a, inv)
            for cfg, a, inv in tasks
        ]
        for future in as_completed(futures):
            # Optionally handle return values or exceptions
            result = future.result()
