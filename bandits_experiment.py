import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed

from agents.adamLMCdqn import main as adamLMCDQN
from agents.egreedy import main as egreedy

def worker(base_config, a, bandit, temp):
    # Copy base config to avoid shared-state mutations
    config = base_config.copy()
    config["a"] = a
    config["ENV_NAME"] = bandit
    config["inverse_temperature"] = temp
    # Run the training function
    return adamLMCDQN(config)

def egreedyWorker(base_config, bandit):
    config = base_config.copy()
    config["ENV_NAME"] = bandit
    config["EPSILON_ANNEAL_TIME"] = 1e+5

    return egreedy(config)

def sequential(tasks, egreedyTasks):
    for (base_config, a, bandit, temp) in tasks:
        worker(base_config, a, bandit, temp)

    for (base_config, bandit) in egreedyTasks:
        egreedyWorker(base_config, bandit)

def parallelized(tasks, egreedyTasks):
    with ProcessPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(worker, cfg, a, bandit, temp)
            for cfg, a, bandit, temp in tasks
        ]
        for future in as_completed(futures):
            # Optionally handle return values or exceptions
            result = future.result()

    with ProcessPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(egreedyWorker, cfg, bandit)
            for cfg, bandit in egreedyTasks
        ]
        for future in as_completed(futures):
            # Optionally handle return values or exceptions
            result = future.result()


if __name__ == "__main__":
    # Load base configuration
    with open("configs/defaultConfig.yaml", "r") as f:
        base_config = yaml.safe_load(f)

    # Load hyperparameter sweep values
    with open("configs/experiments/bandit_experiment.yaml", "r") as f:
        bandit_list = yaml.safe_load(f)

    base_config["NUM_SEEDS"] = 10

    tasks = [
        (base_config, a, bandit, temp)
        for a in bandit_list["a"]
        for bandit in bandit_list["bandits"]
        for temp in bandit_list["inverse_temperature"]
    ]

    egreedyTasks = [
        (base_config, bandit) for bandit in bandit_list["bandits"]
    ]

    if bandit_list.get("mode", "sequential") == "parallelized":
        parallelized(tasks, egreedyTasks)
    else:
        sequential(tasks, egreedyTasks)

