"""TD3+BC (CORL implementation, paper App. C.2/C.3 settings) on original data + one GTA condition."""
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def load_dataset(env_name, gta_path=None):
    """Original D4RL transitions, plus the GTA-generated transitions if gta_path is given."""
    with open(f"data/{env_name}.pkl", "rb") as f:
        episodes = pickle.load(f)
    keys = ["observations", "actions", "rewards", "next_observations", "terminals"]
    data = {k: np.concatenate([np.asarray(ep[k], np.float32).reshape(len(ep[k]), -1) for ep in episodes])
            for k in keys}
    data["rewards"] = data["rewards"].reshape(-1)
    data["terminals"] = data["terminals"].reshape(-1)
    if gta_path is not None:
        z = np.load(gta_path)
        for k in keys:
            data[k] = np.concatenate([data[k], z[k].astype(np.float32).reshape(len(z[k]), *data[k].shape[1:])])
    return data


def run_td3bc(env_name, condition, gta_path, seed, out_dir, steps=1_000_000, batch_size=1024,
              eval_freq=10_000, n_episodes=10, device="cuda"):
    os.environ["WANDB_MODE"] = "disabled"
    import gym
    import d4rl  # noqa: F401
    from corl.algorithms.td3_bc import TD3_BC, Actor, Critic
    from corl.shared.buffer import StateNormalizer, prepare_replay_buffer
    from corl.shared.utils import compute_mean_std, eval_actor, set_seed, wrap_env

    run_dir = Path(out_dir) / f"{condition}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "done.json").exists():
        print(f"{run_dir.name}: already finished")
        return

    data = load_dataset(env_name, gta_path)
    env = gym.make(env_name)
    state_dim, action_dim = env.observation_space.shape[0], env.action_space.shape[0]
    state_mean, state_std = compute_mean_std(data["observations"], eps=1e-3)
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    buffer = prepare_replay_buffer(state_dim, action_dim, len(data["observations"]), data, device=device,
                                   state_normalizer=StateNormalizer(state_mean, state_std))
    del data
    set_seed(seed, env)

    max_action = float(env.action_space.high[0])
    actor = Actor(state_dim, action_dim, max_action).to(device)
    critic_1, critic_2 = Critic(state_dim, action_dim).to(device), Critic(state_dim, action_dim).to(device)
    trainer = TD3_BC(
        max_action=max_action,
        actor=actor, actor_optimizer=torch.optim.Adam(actor.parameters(), lr=3e-4),
        critic_1=critic_1, critic_1_optimizer=torch.optim.Adam(critic_1.parameters(), lr=3e-4),
        critic_2=critic_2, critic_2_optimizer=torch.optim.Adam(critic_2.parameters(), lr=3e-4),
        discount=0.99, tau=0.005, device=device,
        policy_noise=0.2 * max_action, noise_clip=0.5 * max_action, policy_freq=2, alpha=2.5,
    )

    evals = []
    t0 = time.time()
    pbar = tqdm(range(steps), desc=f"{condition} seed{seed}", unit="it", mininterval=5)
    for t in pbar:
        trainer.train([b.to(device) for b in buffer.sample(batch_size)])
        if (t + 1) % eval_freq == 0 or t == steps - 1:
            score = env.get_normalized_score(eval_actor(env, actor, device, n_episodes, seed).mean()) * 100
            evals.append((t + 1, float(score)))
            pd.DataFrame(evals, columns=["step", "d4rl_score"]).to_csv(run_dir / "evals.csv", index=False)
            pbar.set_postfix(score=f"{score:.1f}")
    torch.save(trainer.state_dict(), run_dir / "checkpoint.pt")
    last = [s for _, s in evals[-5:]]
    (run_dir / "done.json").write_text(json.dumps(
        {"condition": condition, "seed": seed, "final_score_last5": float(np.mean(last)),
         "n_transitions": len(buffer), "minutes": (time.time() - t0) / 60}, indent=2))
    print(f"{run_dir.name}: final D4RL score (mean of last 5 evals) = {np.mean(last):.2f}")
