![tudelftLogo.png](tudelftLogo.png)

# AdamLMCDQN

This repository contains an implementation of a Deep Q-Network (DQN) reinforcement learning algorithm. We implement the LMC-LSVI and Adam LMCDQN algorithms, as proposed by [Provable and Practical: Efficient Exploration in Reinforcement Learning via Langevin Monte Carlo](https://arxiv.org/abs/2305.18246).
We implement the algorithms in JAX, a high-performance numerical computing library. For environment-specific implementations, we use the [gymnax](https://github.com/RobertTLange/gymnax?tab=readme-ov-file) as they offer fully vectorised environments. 

This repository is part of the thesis work of Pablo Hendriks Bardaji, supervised by Pascal van der Vaart and Neil Yorke-Smith, made for the [Research Project](https://github.com/TU-Delft-CSE/Research-Project) (CSE 3000) at the TU Delft (2025). 

## Installation

For installation, we recommend using a package manager like [uv](https://docs.astral.sh/uv/getting-started/installation/) or similar.

```bash
git clone https://github.com/PabloHendriks/AdamLMCDQN.git
cd AdamLMCDQN
uv sync
````

This will install all the required dependencies, including JAX and Gymnax. 

The project requires Python 3.11 or higher. For the complete list of dependencies, please refer to the `pyproject.toml` file.

We use [WandB](https://wandb.ai/site/) for logging and keeping track of the experiments; therefore, the first time you run the code, you will need to log in to an account. 


## Experiments

To run the agents, you can use the following command:

```bash
uv run main.py
````

This will use the default config to run LMC-LSVI or Adam LMCDQN, and also the E-Greedy agents. To run LMC-LSVI, use ``a: 0`` and for Adam LMCDQN use ``a: > 0`` in the config file, we recommend ``a: 1``. 

To run any of the experiments, you can use the following command:

```bash
uv run <experiment name>.py
````

Where experiment name can be one of the following:
- bandits_experiment
- j_experiment
- a_temp_lr_experiment
- lr_experiment
- robust_set_experiment

To see or change the hyperparameters, you can edit the default config file and the respective experiment config file, which you can find under `configs/experiments`. Keep in mind that the parameters of the default config will be overwritten by the ones defined in the specific experiment config. 

Most experiments have implemented a sequential and a parallelised version. We recommend using the sequential version for GPU usage, and the parallelised version for CPU. The parallelisation over several GPUs is not implemented.

## Citation

If you use LMC-LSVI or Adam LMCDQN in your research, please cite the following paper:

```bibtex
@inproceedings{ishfaq2024provable,
  title={Provable and Practical: Efficient Exploration in Reinforcement Learning via Langevin Monte Carlo},
  author={Ishfaq, Haque and Lan, Qingfeng and Xu, Pan and Mahmood, A Rupam and Precup, Doina and Anandkumar, Anima and Azizzadenesheli, Kamyar},
  booktitle={International Conference on Learning Representations},
  year={2024}
}
````

## Acknowledgments
We base our implementation on the [purejaxrl](https://github.com/luchris429/purejaxrl) implementation.
