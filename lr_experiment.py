import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed

from agents.adamLMCdqn import main as adamLMCDQN
from agents.egreedy import main as egreedy

def worker(base_config, a, env, lr, size=20):
    # Copy base config to avoid shared-state mutations
    config = base_config.copy()
    config["a"] = a
    config["ENV_NAME"] = env
    config["LR"] = lr
    config["EXPERIMENT_NAME"] = "Freeway LR Experiment v1"
    config["inv_temperature"] = 100000

    if env == "DeepSea-bsuite":
        config["size_deepSea"] = size
        config["EXPERIMENT_NAME"] = "DeepSea LR Experiment v1"


    # Run the training function
    return adamLMCDQN(config)

# def egreedyWorker(base_config, env):
#     config = base_config.copy()
#     config["ENV_NAME"] = env
#     config["EPSILON_ANNEAL_TIME"] = 1e+5
#
#     return egreedy(config)

def sequential(cartPole_tasks, deepSea_tasks):
    for (base_config, a, env, lr) in cartPole_tasks:
        worker(base_config, a, env, lr)

    # for (base_config, a, env, lr, size) in deepSea_tasks:
    #     worker(base_config, a, env, lr, size)

if __name__ == "__main__":
    # Load base configuration
    with open("configs/defaultConfig.yaml", "r") as f:
        base_config = yaml.safe_load(f)

    # Load hyperparameter sweep values
    with open("configs/experiments/lr_experiment.yaml", "r") as f:
        env_list = yaml.safe_load(f)

    cartPole_tasks = [
        (base_config, a, "Freeway-MinAtar", lr)
        for a in env_list["a"]
        for lr in env_list["lr"]
    ]

    deepSea_tasks = [
        (base_config, a, "DeepSea-bsuite", lr, size)
        for a in env_list["a"]
        for size in env_list["size"]
        for lr in env_list["lr"]
    ]

    # egreedyTasks = [
    #     (base_config, env) for env in env_list["envs"]
    # ]

    if env_list.get("mode", "sequential") == "parallelized":
        raise ValueError("Parallel not implemented")
    else:
        sequential(cartPole_tasks, deepSea_tasks)

