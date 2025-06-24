import jax.numpy as jnp
import jax
import optax

from jax.scipy.special import logsumexp

from optimizers.adamLMCDQN_optimiser import langevin_adam

key = jax.random.PRNGKey(0)
dim = 2

x0 = jnp.zeros(dim)
optim = langevin_adam(key, a=0, inverse_temperature=1)
opt_state = optim.init(x0)

mus = jnp.array([
    [ 2.0,  0.0],
    [-1.0,  1.732],
    [-1.0, -1.732]
])

sigma = 1
cov = (sigma**2) * jnp.eye(2)
invcov = jnp.linalg.inv(cov)
log_norm = -0.5 * (
    2 * jnp.log(2 * jnp.pi) + jnp.linalg.slogdet(cov)[1]
)
weights = jnp.ones(3) / 3.0

def log_prob_mixture(x):
    # x: shape (2,)
    # compute [log N(x|mu_k, cov) for k in 0..2]
    def logpdf_k(mu):
        diff = x - mu
        return log_norm - 0.5 * (diff @ invcov @ diff)
    lps = jax.vmap(logpdf_k)(mus)           # shape (3,)
    return logsumexp(lps + jnp.log(weights))  # log ∑_k w_k exp(logpdf_k)

def loss_fn(x):
    return -log_prob_mixture(x)


@jax.jit
def step(x, state, rng):
    # compute gradient of loss
    grads = jax.grad(loss_fn)(x)
    updates, new_state = optim.update(grads, state, params=x)
    x_new = optax.apply_updates(x, updates)
    return x_new, new_state

def main():
    n_steps = 1000000
    burn_in = 0

    samples = []
    x, state = x0, opt_state

    for i in range(n_steps):
        x, state = step(x, state, None)
        if i >= burn_in:
            samples.append(x)

    samples = jnp.stack(samples)

    import numpy as np
    import matplotlib.pyplot as plt

    # Target parameters
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))

    # Grid for plotting
    n_grid = 200
    xs = np.linspace(-6, 6, n_grid)
    ys = np.linspace(-6, 6, n_grid)
    X, Y = np.meshgrid(xs, ys)
    pts = np.stack([X.ravel(), Y.ravel()], axis=-1)  # (n_grid**2, 2)

    # vectorize our JAX log_prob_mixture
    logp_fn = jax.jit(jax.vmap(log_prob_mixture))
    logps = np.array(logp_fn(pts))  # to NumPy
    Z = np.exp(logps).reshape(n_grid, n_grid)


    # Left: true density contourf
    cf = ax0.contourf(
        X, Y, Z,
        levels=50,
        cmap='viridis'
    )
    ax0.set_title("True Density")
    # fig.colorbar(cf, ax=ax0)

    # Right: empirical density via 2D histogram
    hb = ax1.hist2d(
        samples[:, 0], samples[:, 1],
        bins=100,
        range=[[-6, 6], [-6, 6]],
        cmap='viridis'
    )
    ax1.set_title("Empirical Density (Langevin MC Samples)")
    # fig.colorbar(hb[3], ax=ax1)

    plt.tight_layout()
    # plt.show()
    plt.savefig("langevin_samples.png", dpi=300)

if __name__ == '__main__':
    main()