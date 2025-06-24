import jax
import jax.numpy as jnp

import chex
import flax
import wandb
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
from gymnax.wrappers.purerl import FlattenObservationWrapper, LogWrapper
import gymnax
import flashbax as fbx

from optimizers.adamLMCDQN_optimiser import langevin_adam


class QNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(120)(x)
        x = nn.relu(x)
        x = nn.Dense(84)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x


@chex.dataclass(frozen=True)
class TimeStep:
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array


class CustomTrainState(TrainState):
    target_network_params: flax.core.FrozenDict
    timesteps: int
    n_updates: int

@jax.jit
def compute_entropy(q_vals):
    # Convert Q-values to probabilities using softmax
    probs = jax.nn.softmax(q_vals, axis=-1)

    # Compute entropy: H = -sum(p * log(p))
    # Add small epsilon to avoid log(0)
    eps = 1e-8
    entropy = -jnp.sum(probs * jnp.log(probs + eps), axis=-1)

    return entropy


def make_train(config):

    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_ENVS"]

    # Add extra parameter to specify size when running deep sea environment
    if config["ENV_NAME"] == "DeepSea-bsuite":
        if not config["size_deepSea"]:
            config["size_deepSea"] = 20
        basic_env, env_params = gymnax.make(config["ENV_NAME"], size=config["size_deepSea"])
    else:
        basic_env, env_params = gymnax.make(config["ENV_NAME"] )

    env = FlattenObservationWrapper(basic_env)
    env = LogWrapper(env)

    vmap_reset = lambda n_envs: lambda rng: jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(rng, n_envs), env_params
    )
    vmap_step = lambda n_envs: lambda rng, env_state, action: jax.vmap(
        env.step, in_axes=(0, 0, 0, None)
    )(jax.random.split(rng, n_envs), env_state, action, env_params)

    def train(rng):

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        init_obs, env_state = vmap_reset(config["NUM_ENVS"])(_rng)

        # INIT BUFFER
        buffer = fbx.make_flat_buffer(
            max_length=config["BUFFER_SIZE"],
            min_length=config["BUFFER_BATCH_SIZE"],
            sample_batch_size=config["BUFFER_BATCH_SIZE"],
            add_sequences=False,
            add_batch_size=config["NUM_ENVS"],
        )
        buffer = buffer.replace(
            init=jax.jit(buffer.init),
            add=jax.jit(buffer.add, donate_argnums=0),
            sample=jax.jit(buffer.sample),
            can_sample=jax.jit(buffer.can_sample),
        )
        dummy_rng = jax.random.PRNGKey(0)  # use a dummy rng here TODO change the rng
        _action = basic_env.action_space().sample(dummy_rng)
        _, _env_state = env.reset(dummy_rng, env_params)
        _obs, _, _reward, _done, _ = env.step(dummy_rng, _env_state, _action, env_params)
        _timestep = TimeStep(obs=_obs, action=_action, reward=_reward, done=_done)
        buffer_state = buffer.init(_timestep)

        # INIT NETWORK AND OPTIMIZER
        network = QNetwork(action_dim=env.action_space(env_params).n)
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        network_params = network.init(_rng, init_x)

        rng, _rng = jax.random.split(rng)

        tx = langevin_adam(
            base_rng=_rng,
            learning_rate=config['LR'],
            alpha1=config['alpha1'],
            alpha2=config['alpha2'],
            eps=config['eps'],
            inverse_temperature=config['inverse_temperature'],
            a=config['a'],
        )

        train_state = CustomTrainState.create(
            apply_fn=network.apply,
            params=network_params,
            target_network_params=jax.tree.map(lambda x: jnp.copy(x), network_params),
            tx=tx,
            timesteps=0,
            n_updates=0,
        )

        # TRAINING LOOP
        def _update_step(runner_state, unused):

            train_state, buffer_state, env_state, last_obs, rng = runner_state

            # STEP THE ENV
            rng, rng_a, rng_s = jax.random.split(rng, 3)
            q_vals = network.apply(train_state.params, last_obs)
            action = jnp.argmax(q_vals, axis=1)
            obs, env_state, reward, done, info = vmap_step(config["NUM_ENVS"])(
                rng_s, env_state, action
            )
            train_state = train_state.replace(
                timesteps=train_state.timesteps + config["NUM_ENVS"]
            )  # update timesteps count

            # BUFFER UPDATE
            timestep = TimeStep(obs=last_obs, action=action, reward=reward, done=done)
            buffer_state = buffer.add(buffer_state, timestep)

            # NETWORKS UPDATE
            def _learn_phase(train_state, rng):

                learn_batch = buffer.sample(buffer_state, rng).experience

                q_next_target = network.apply(
                    train_state.target_network_params, learn_batch.second.obs
                )  # (batch_size, num_actions)

                entropy = compute_entropy(q_next_target)

                q_next_target = jnp.max(q_next_target, axis=-1)  # (batch_size,)
                target = (
                        learn_batch.first.reward
                        + (1 - learn_batch.first.done) * config["GAMMA"] * q_next_target
                )

                def loss_fn(params):
                    lambda_reg = 1e-8
                    # predicted Q for the taken actions
                    q_vals = network.apply(
                        params, learn_batch.first.obs
                    )  # (batch_size, num_actions)
                    chosen_action_qvals = jnp.take_along_axis(
                        q_vals,
                        jnp.expand_dims(learn_batch.first.action, axis=-1),
                        axis=-1,
                    ).squeeze(axis=-1)

                    # per-sample squared TD error
                    td_errors = jnp.square(chosen_action_qvals - target)  # (batch,)

                    # average over the batch
                    td_loss = jnp.mean(td_errors)

                    # L2 norm of all parameters
                    squared = jax.tree_util.tree_map(lambda p: jnp.sum(jnp.square(p)), params)
                    l2_norm = jax.tree_util.tree_reduce(lambda x, y: x + y, squared, 0.0)

                    # Create dictionary for future plotting
                    loss_dictionary = {"td_loss": td_loss, "l2_norm": lambda_reg * l2_norm}

                    # total loss = TD loss + λ * ||w||²
                    total_loss = td_loss + lambda_reg * l2_norm
                    return total_loss, loss_dictionary

                def j_loop(state, _):
                    (loss, loss_dict), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
                    return state.apply_gradients(grads=grads), (loss, loss_dict)

                train_state, (loss, loss_dict) = jax.lax.scan(j_loop, train_state, None, config['J'])

                results_dict = {
                    "td_loss": loss_dict["td_loss"].mean(),
                    "l2_norm": loss_dict["l2_norm"].mean(),
                    "entropy": entropy.mean(),
                }

                train_state = train_state.replace(n_updates=train_state.n_updates + 1)
                return train_state, loss.mean(), results_dict

            rng, _rng = jax.random.split(rng)
            is_learn_time = (
                (buffer.can_sample(buffer_state))
                & (  # enough experience in buffer
                    train_state.timesteps > config["LEARNING_STARTS"]
                )
                & (  # pure exploration phase ended
                    train_state.timesteps % config["TRAINING_INTERVAL"] == 0
                )  # training interval
            )

            train_state, loss, results_dict = jax.lax.cond(
                is_learn_time,
                lambda train_state, rng: _learn_phase(train_state, rng),
                lambda train_state, rng: (train_state, jnp.array(0.0), {"td_loss": 0.0, "l2_norm": 0.0, "entropy": 1.0}),  # do nothing
                train_state,
                _rng,
            )

            # update target network
            train_state = jax.lax.cond(
                train_state.timesteps % config["TARGET_UPDATE_INTERVAL"] == 0,
                lambda train_state: train_state.replace(
                    target_network_params=optax.incremental_update(
                        train_state.params,
                        train_state.target_network_params,
                        config["TAU"],
                    )
                ),
                lambda train_state: train_state,
                operand=train_state,
            )

            metrics = {
                "timesteps": train_state.timesteps,
                "updates": train_state.n_updates,
                "loss": loss.mean(),
                "returns": info["returned_episode_returns"].mean(),
                "count": train_state.opt_state.count,
                "td_loss": results_dict["td_loss"],
                "l2_norm": results_dict["l2_norm"],
                "entropy": results_dict["entropy"],
            }

            # report on wandb if required
            if config.get("WANDB_MODE", "disabled") == "online":

                def callback(metrics):
                    if metrics["timesteps"] % 100 == 0:
                        wandb.log(metrics)

                jax.debug.callback(callback, metrics)

            runner_state = (train_state, buffer_state, env_state, obs, rng)

            return runner_state, metrics

        # train
        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, buffer_state, env_state, init_obs, _rng)

        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train


def main(config):
    experiment = config["EXPERIMENT_NAME"]
    if experiment != "":
        tags = config["TAGS"] + ['AdamLMCDQN', experiment]
    else:
        tags = config["TAGS"] + ['AdamLMCDQN']

    base_seed = jax.random.PRNGKey(config["SEED"])
    seeds = jax.random.split(base_seed, num=config["NUM_SEEDS"])

    train_jit = jax.jit(make_train(config))

    for seed in seeds:
        wandb.init(
            entity=config["ENTITY"],
            project=config["PROJECT"],
            tags=tags,
            name=f'rp_adamLMCDQN_{config["ENV_NAME"]}',
            config=config,
            mode=config["WANDB_MODE"],
        )

        outs = jax.block_until_ready(train_jit(seed))

        wandb.finish()
