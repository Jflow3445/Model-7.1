import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

from stable_baselines3 import PPO
from models.medium_policy import MediumTermOHLCPolicy
from trainers.train_medium_policy import MediumBacktestEnv
from config.settings import HOURLY_CSV_DIR, LIVE_FOREX_PAIRS, MEDIUM_OBS_WINDOW

LOGS_DIR = "D:/Model 7.1/models/logs"
CKPT_DIR = "D:/Model 7.1/models/checkpoints_medium"
WINDOW = MEDIUM_OBS_WINDOW

def aggregate_training_logs(logs_dir):
    rewards = []
    for fname in glob.glob(os.path.join(logs_dir, "medium_worker_*/monitor.csv")):
        if not os.path.isfile(fname):
            continue
        # Read and filter rows: only rows with 3 columns, numeric reward
        df = pd.read_csv(fname, skiprows=1, names=["r", "l", "t"])
        # Filter NaN, malformed, or non-numeric rows
        df = df[pd.to_numeric(df["r"], errors="coerce").notnull()]
        rewards.extend(df["r"].astype(float).values)
    rewards = np.array(rewards)
    return rewards

def plot_training_rewards(rewards):
    plt.figure(figsize=(12, 5))
    plt.plot(rewards, label="Episode reward", alpha=0.6)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training episode rewards (all workers)")
    plt.legend()
    plt.tight_layout()
    plt.show()

def evaluate_checkpoint(model_path, n_episodes=5):
    env = MediumBacktestEnv(
        csv_dir=HOURLY_CSV_DIR,
        symbols=LIVE_FOREX_PAIRS,
        window=WINDOW,
        max_steps=1000,
        seed=999,
    )
    model = PPO.load(model_path, env=env)
    episode_rewards = []
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
        episode_rewards.append(total_reward)
    avg_reward = np.mean(episode_rewards)
    return avg_reward

def evaluate_all_checkpoints(ckpt_dir):
    ckpt_paths = sorted([
        os.path.join(ckpt_dir, f)
        for f in os.listdir(ckpt_dir)
        if f.endswith('.zip') and "ckpt_" in f
    ])
    results = []
    steps = []
    for idx, ckpt in enumerate(ckpt_paths):
        avg_reward = evaluate_checkpoint(ckpt, n_episodes=3)
        # Extract steps from filename
        match = re.search(r'ckpt_(\d+)_steps', ckpt)
        if match:
            step = int(match.group(1))
        else:
            step = idx  # fallback: order index if no step in name
        steps.append(step)
        results.append((ckpt, avg_reward))
        print(f"{ckpt}: avg_reward={avg_reward:.2f}")
    return steps, results

def plot_eval_rewards(steps, results):
    eval_rewards = [score for _, score in results]
    plt.figure(figsize=(12, 5))
    plt.plot(steps, eval_rewards, marker='o', label='Eval avg reward')
    plt.xlabel("Training steps")
    plt.ylabel("Evaluation average reward")
    plt.title("Checkpoint Evaluation Rewards")
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    print("Aggregating training logs...")
    rewards = aggregate_training_logs(LOGS_DIR)
    print(f"Aggregated {len(rewards)} training episode rewards.")
    plot_training_rewards(rewards)

    print("\nEvaluating checkpoints...")
    steps, results = evaluate_all_checkpoints(CKPT_DIR)
    plot_eval_rewards(steps, results)

    best_ckpt, best_score = max(results, key=lambda x: x[1])
    print(f"\nBest checkpoint: {best_ckpt} with avg_reward {best_score:.2f}")

if __name__ == "__main__":
    main()
