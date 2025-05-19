import yaml

from agents.adamLMCdqn import main as adamLMCDQN
from agents.egreedy import main as egreedy

def run():
    with open("configs/cartPole.yaml", "r") as f:
        config = yaml.safe_load(f)

    with open("configs/sweeps/a_and_inverse_temperature_sweep.yaml", "r") as f:
        experiment_hyperparameters = yaml.safe_load(f)

    egreedy(config=config)

    for a_param in experiment_hyperparameters["a"]:
        for inverse_temperature in experiment_hyperparameters["inverse_temperature"]:
            config["a"] = a_param
            config["inverse_temperature"] = inverse_temperature
            adamLMCDQN(config)



if __name__ == "__main__":
    run()