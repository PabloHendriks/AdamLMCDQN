import yaml

from agents.adamLMCdqn import main as adamLMCDQN
from agents.egreedy import main as egreedy

def run():
    with open("configs/defaultConfig.yaml", "r") as f:
        config = yaml.safe_load(f)

    adamLMCDQN(config=config)
    egreedy(config=config)


if __name__ == "__main__":
    run()
