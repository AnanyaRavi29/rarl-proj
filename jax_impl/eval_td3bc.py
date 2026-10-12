import argparse
import numpy as np
import gym
import d4rl


def relu(x):
    return np.maximum(x, 0.0)


class NumpyActor:
    def __init__(self, checkpoint_path):
        p = np.load(checkpoint_path)

        self.w0 = p["dense0_kernel"]
        self.b0 = p["dense0_bias"]

        self.w1 = p["dense1_kernel"]
        self.b1 = p["dense1_bias"]

        self.w2 = p["dense2_kernel"]
        self.b2 = p["dense2_bias"]

    def __call__(self, state):
        x = relu(state @ self.w0 + self.b0)
        x = relu(x @ self.w1 + self.b1)
        x = np.tanh(x @ self.w2 + self.b2)
        return x.astype(np.float32)


def reset_env(env, seed):
    # Compatible with old Gym/D4RL.
    try:
        env.seed(seed)
    except Exception:
        pass

    result = env.reset()

    if isinstance(result, tuple):
        return result[0]

    return result


def step_env(env, action):
    result = env.step(action)

    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        done = terminated or truncated
        return obs, reward, done, info

    obs, reward, done, info = result
    return obs, reward, done, info


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--env",
        default="halfcheetah-medium-v2",
    )

    parser.add_argument(
        "--actor",
        default=(
            "jax_impl/checkpoints/"
            "td3bc_halfcheetah_medium_seed0/"
            "actor_step_1000000.npz"
        ),
    )

    parser.add_argument(
        "--normalization",
        default=(
            "jax_impl/checkpoints/"
            "td3bc_halfcheetah_medium_seed0/"
            "normalization.npz"
        ),
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    env = gym.make(args.env)
    actor = NumpyActor(args.actor)

    norm = np.load(args.normalization)

    state_mean = norm["state_mean"]
    state_std = norm["state_std"]

    returns = []

    print("=" * 70)
    print("JAX TD3+BC EVALUATION")
    print("=" * 70)
    print("Environment :", args.env)
    print("Episodes    :", args.episodes)
    print()

    for episode in range(args.episodes):
        state = reset_env(
            env,
            args.seed + episode,
        )

        done = False
        episode_return = 0.0
        episode_steps = 0

        while not done:
            normalized_state = (
                state - state_mean
            ) / state_std

            action = actor(
                normalized_state
            )

            action = np.clip(
                action,
                env.action_space.low,
                env.action_space.high,
            )

            state, reward, done, _ = step_env(
                env,
                action,
            )

            episode_return += float(reward)
            episode_steps += 1

        returns.append(
            episode_return
        )

        print(
            f"Episode {episode + 1:02d}: "
            f"return={episode_return:9.2f} "
            f"steps={episode_steps}"
        )

    returns = np.asarray(
        returns,
        dtype=np.float64,
    )

    mean_return = float(
        returns.mean()
    )

    std_return = float(
        returns.std()
    )

    normalized_score = float(
        env.get_normalized_score(
            mean_return
        ) * 100.0
    )

    normalized_scores = np.asarray(
        [
            env.get_normalized_score(r) * 100.0
            for r in returns
        ]
    )

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)

    print(
        f"Mean raw return       : {mean_return:.2f}"
    )

    print(
        f"Std raw return        : {std_return:.2f}"
    )

    print(
        f"D4RL normalized score : {normalized_score:.2f}"
    )

    print(
        "Normalized score std  : "
        f"{normalized_scores.std():.2f}"
    )

    print("=" * 70)

    env.close()


if __name__ == "__main__":
    main()
