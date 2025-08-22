import os
import glob
import numpy as np
from stable_baselines3 import PPO
import matplotlib.pyplot as plt

from config.settings import (
    DAILY_CSV_DIR,
    LIVE_FOREX_PAIRS,
    LONG_OBS_WINDOW,
    SEED
)
from trainers.train_long_policy import LongBacktestEnv

ckpt_dir = "models/checkpoints_long/"
n_eval_episodes = 10

ckpt_files = glob.glob(os.path.join(ckpt_dir, "long_policy_ckpt_*_steps.zip"))
ckpt_files = sorted(
    ckpt_files,
    key=lambda x: int(os.path.basename(x).split("_ckpt_")[1].split("_steps")[0])
)

best_mean_reward = float('-inf')
best_ckpt = None

def make_eval_env():
    return LongBacktestEnv(
        csv_dir=DAILY_CSV_DIR,
        symbols=LIVE_FOREX_PAIRS,
        window=LONG_OBS_WINDOW,
        seed=SEED,
        max_steps=100000,
    )

for ckpt in ckpt_files:
    print(f"Evaluating {ckpt} ...")
    model = PPO.load(ckpt)
    env = make_eval_env()
    rewards = []

    for ep in range(n_eval_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
        rewards.append(ep_reward)
        print(f"Episode {ep+1}: Reward = {ep_reward:.2f}, Steps = {env.current_step}")

    mean_reward = np.mean(rewards)
    print(f" Mean reward over {n_eval_episodes} episodes: {mean_reward:.2f}")

    if mean_reward > best_mean_reward:
        best_mean_reward = mean_reward
        best_ckpt = ckpt

    # Plot for current checkpoint
    plt.plot(rewards, marker='o')
    plt.title(f"Rewards per Episode for {os.path.basename(ckpt)}")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.show()

print("="*60)
print(f"Best checkpoint is: {best_ckpt}")
print(f"With mean reward: {best_mean_reward:.2f}")
