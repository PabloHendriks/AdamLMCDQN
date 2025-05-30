import yaml
import os
import jax
from concurrent.futures import ProcessPoolExecutor, as_completed

from agents.adamLMCdqn import main as adamLMCDQN
from agents.egreedy import main as egreedy


def worker(base_config, a, inverse_temperature, j, size):
    # Copy base config to avoid shared-state mutations
    config = base_config.copy()
    config["a"] = a
    config["inverse_temperature"] = inverse_temperature
    config["J"] = j
    config["size_deepSea"] = size

    return adamLMCDQN(config)

def parallelized(tasks):
    with ProcessPoolExecutor(max_workers=12) as executor:
        futures = [
            executor.submit(worker, cfg, a, inv, j)
            for cfg, a, inv, j in tasks
        ]
        for future in as_completed(futures):
            # Optionally handle return values or exceptions
            result = future.result()

def sequential(tasks):
    for cfg, a, inv, j, size in tasks:
        worker(cfg, a, inv, j, size)

if __name__ == "__main__":
    # Load base configuration
    with open("configs/defaultConfig.yaml", "r") as f:
        base_config = yaml.safe_load(f)

    # Load hyperparameter sweep values
    with open("configs/experiments/j_experiment.yaml", "r") as f:
        sweep = yaml.safe_load(f)

    tasks = [
        (base_config, a, inv_temp, j, size)
        for a in sweep["a"]
        for inv_temp in sweep["inverse_temperature"]
        for j in sweep["J"]
        for size in sweep["size"]
    ]

    if sweep["mode"] == "parallelized":
        parallelized(tasks)
    else:
        sequential(tasks)

    egreedy(config=base_config)

