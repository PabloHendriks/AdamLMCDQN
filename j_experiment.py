import yaml
import os
import jax
from concurrent.futures import ProcessPoolExecutor, as_completed

from agents.adamLMCdqn import main as adamLMCDQN
from agents.egreedy import main as egreedy


def worker(base_config, a, inverse_temperature, j, size=20):
    # Copy base config to avoid shared-state mutations
    config = base_config.copy()
    config["a"] = a
    config["inverse_temperature"] = inverse_temperature
    config["J"] = j
    if config["ENV_NAME"] == "DeepSea-bsuite":
        config["size_deepSea"] = size

    return adamLMCDQN(config)

def parallelized(tasks):
    if tasks[0][0]["ENV_NAME"] == "DeepSea-bsuite":
        with ProcessPoolExecutor(max_workers=12) as executor:
            futures = [
                executor.submit(worker, cfg, a, inv, j, size)
                for cfg, a, inv, j, size in tasks
            ]
            for future in as_completed(futures):
                # Optionally handle return values or exceptions
                result = future.result()
    else:
        with ProcessPoolExecutor(max_workers=12) as executor:
            futures = [
                executor.submit(worker, cfg, a, inv, j)
                for cfg, a, inv, j in tasks
            ]
            for future in as_completed(futures):
                # Optionally handle return values or exceptions
                result = future.result()

def sequential(tasks):
    if tasks[0][0]["ENV_NAME"] == "DeepSea-bsuite":
        for cfg, a, inv, j, size in tasks:
            worker(cfg, a, inv, j, size)
    else:
        for cfg, a, inv, j in tasks:
            worker(cfg, a, inv, j)

if __name__ == "__main__":
    # Load base configuration
    with open("configs/defaultConfig.yaml", "r") as f:
        base_config = yaml.safe_load(f)

    # Load hyperparameter sweep values
    with open("configs/experiments/j_experiment.yaml", "r") as f:
        sweep = yaml.safe_load(f)

    if base_config["ENV_NAME"] == "DeepSea-bsuite":
        tasks = [
            (base_config, a, inv_temp, j, size)
            for a in sweep["a"]
            for inv_temp in sweep["inverse_temperature"]
            for j in sweep["J"]
            for size in sweep["size"]
        ]
    else :
        tasks = [
            (base_config, a, inv_temp, j)
            for a in sweep["a"]
            for inv_temp in sweep["inverse_temperature"]
            for j in sweep["J"]
        ]


    if sweep["mode"] == "parallelized":
        parallelized(tasks)
    else:
        sequential(tasks)

    egreedy(config=base_config)

