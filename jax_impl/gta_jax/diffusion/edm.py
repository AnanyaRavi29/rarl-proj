import jax
import jax.numpy as jnp


SIGMA_MIN = 0.002
SIGMA_MAX = 80.0
SIGMA_DATA = 1.0
RHO = 7.0

P_MEAN = -1.2
P_STD = 1.2

S_CHURN = 80.0
S_TMIN = 0.05
S_TMAX = 50.0
S_NOISE = 1.003


def normalize(x, mean, std):
    return (x - mean) / std


def unnormalize(x, mean, std):
    return x * std + mean


def c_skip(sigma, sigma_data=SIGMA_DATA):
    return (
        sigma_data ** 2
        / (sigma ** 2 + sigma_data ** 2)
    )


def c_out(sigma, sigma_data=SIGMA_DATA):
    return (
        sigma
        * sigma_data
        / jnp.sqrt(
            sigma_data ** 2 + sigma ** 2
        )
    )


def c_in(sigma, sigma_data=SIGMA_DATA):
    return 1.0 / jnp.sqrt(
        sigma ** 2 + sigma_data ** 2
    )


def c_noise(sigma):
    return 0.25 * jnp.log(
        jnp.maximum(sigma, 1e-20)
    )


def loss_weight(
    sigma,
    sigma_data=SIGMA_DATA,
):
    return (
        sigma ** 2 + sigma_data ** 2
    ) / (
        sigma * sigma_data
    ) ** 2


def sample_training_sigmas(
    key,
    batch_size,
    p_mean=P_MEAN,
    p_std=P_STD,
):
    noise = jax.random.normal(
        key,
        shape=(batch_size,),
        dtype=jnp.float32,
    )

    return jnp.exp(
        p_mean + p_std * noise
    )


def preconditioned_forward(
    model,
    params,
    constants,
    x_noisy,
    sigma,
    cond,
    dropout_key,
    deterministic=False,
):
    """
    Equivalent to GTA ElucidatedDiffusion
    preconditioned_network_forward().
    """

    sigma_3d = sigma[:, None, None]

    network_input = (
        c_in(sigma_3d)
        * x_noisy
    )

    variables = {
        "params": params,
        "constants": constants,
    }

    net_out = model.apply(
        variables,
        network_input,
        c_noise(sigma),
        cond,
        deterministic=deterministic,
        rngs={
            "dropout": dropout_key,
        },
    )

    denoised = (
        c_skip(sigma_3d)
        * x_noisy
        +
        c_out(sigma_3d)
        * net_out
    )

    return denoised


def edm_loss(
    model,
    params,
    constants,
    inputs,
    cond,
    mean,
    std,
    rng,
    obs_dim=17,
):
    """
    GTA EDM training loss.

    inputs:
        [B, 32, 24]

    cond:
        [B, 1]
    """

    # Same Standard Normalizer as PyTorch GTA.
    inputs = normalize(
        inputs,
        mean,
        std,
    )

    batch_size = inputs.shape[0]

    (
        sigma_key,
        noise_key,
        dropout_key,
    ) = jax.random.split(
        rng,
        3,
    )

    # sigma ~ LogNormal(P_mean, P_std)
    sigma = sample_training_sigmas(
        sigma_key,
        batch_size,
    )

    sigma_3d = sigma[:, None, None]

    noise = jax.random.normal(
        noise_key,
        shape=inputs.shape,
        dtype=inputs.dtype,
    )

    noised_inputs = (
        inputs
        + sigma_3d * noise
    )

    denoised = preconditioned_forward(
        model=model,
        params=params,
        constants=constants,
        x_noisy=noised_inputs,
        sigma=sigma,
        cond=cond,
        dropout_key=dropout_key,
        deterministic=False,
    )

    # -------------------------------------
    # Match PyTorch GTA's terminal mask:
    #
    # mask[:, -1, obs_dim:] = 0
    #
    # Final row contains:
    # [next_state, zero_action, zero_reward]
    #
    # Do not train the fake action/reward.
    # -------------------------------------

    mask = jnp.ones_like(inputs)

    mask = mask.at[
        :,
        -1,
        obs_dim:
    ].set(0.0)

    squared_error = (
        denoised - inputs
    ) ** 2

    squared_error = (
        squared_error * mask
    )

    # PyTorch:
    # reduce(losses, 'b ... -> b', 'mean')
    per_sample_loss = jnp.mean(
        squared_error,
        axis=(1, 2),
    )

    weights = loss_weight(
        sigma
    )

    weighted_loss = (
        per_sample_loss
        * weights
    )

    loss = jnp.mean(
        weighted_loss
    )

    metrics = {
        "loss": loss,
        "sigma_mean": jnp.mean(sigma),
        "sigma_min": jnp.min(sigma),
        "sigma_max": jnp.max(sigma),
        "denoised_abs_mean":
            jnp.mean(jnp.abs(denoised)),
    }

    return loss, metrics
