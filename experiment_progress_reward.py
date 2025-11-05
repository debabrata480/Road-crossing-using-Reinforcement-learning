# === experiment_progress_reward.py ===
# Run two experiments: baseline (no shaping) and shaped (reward_progress_scale>0)
# Collect 1000-episode datasets and plot/save two graphs (before/after).
# Optimized hyperparameters for better success rates.

import math
import os
from collections import deque, Counter
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple




import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import DQN
from agent import DQNAgent
from game_crossy import CrossyEnv


@dataclass
class EpisodeStats:
    episode: int
    reward: float
    steps: int
    termination: str  # one of: success, collision, oob, None
    epsilon_end: float


def train_and_collect(env: CrossyEnv, episodes: int = 500, max_steps: int = 2000,
                      lr: float = 1e-3, batch_size: int = 64, target_update: int = 500,
                      eps_start: float = 1.0, eps_end: float = 0.05, eps_decay: int = 5000,
                      save_path: str = None, label: str = "run") -> List[EpisodeStats]:
    # Set device for GPU acceleration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if label == "baseline" or (label == "run" and episodes == 1):
        # Only print once at the start
        print("\n" + "="*60)
        print("DEVICE INFORMATION")
        print("="*60)
        if torch.cuda.is_available():
            print(f"✓ CUDA Available: Yes")
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            print(f"✗ CUDA Available: No")
            print(f"  Using CPU for training")
        print("="*60 + "\n")
    
    obs_shape = env.observation_shape
    n_actions = env.n_actions

    model = DQN(obs_shape, n_actions)
    agent = DQNAgent(model, lr=lr, batch_size=batch_size, target_update=target_update, device=device)

    all_stats: List[EpisodeStats] = []

    total_steps = 0
    for ep in range(1, episodes + 1):
        state = env.reset()
        episode_reward = 0.0
        steps = 0
        termination_reason = None
        last_eps = eps_end

        for t in range(max_steps):
            eps = eps_end + (eps_start - eps_end) * math.exp(-1.0 * total_steps / eps_decay)
            last_eps = eps
            action = agent.select_action(state, n_actions, eps=eps)
            next_state, reward, done, info = env.step(action)

            termination_reason = info.get("termination") if isinstance(info, dict) else None

            agent.push_transition(state, action, reward, next_state, float(done))
            agent.optimize()

            state = next_state
            episode_reward += float(reward)
            steps += 1
            total_steps += 1
            if done:
                break

        all_stats.append(EpisodeStats(
            episode=ep,
            reward=episode_reward,
            steps=steps,
            termination=termination_reason or "none",
            epsilon_end=last_eps,
        ))

        if ep % 10 == 0:
            print(f"[{label}] Episode {ep}/{episodes} reward={episode_reward:.2f} steps={steps}")
        if save_path and ep % 100 == 0:
            agent.save(save_path)

    if save_path:
        agent.save(save_path)

    return all_stats


def compute_metrics(stats: List[EpisodeStats], window: int = 20) -> Dict[str, np.ndarray]:
    rewards = np.array([s.reward for s in stats], dtype=np.float32)
    steps = np.array([s.steps for s in stats], dtype=np.int32)
    terminations = [s.termination for s in stats]
    success = np.array([1 if t == "success" else 0 for t in terminations], dtype=np.int32)
    collisions = np.array([1 if t == "collision" else 0 for t in terminations], dtype=np.int32)
    oobs = np.array([1 if t == "oob" else 0 for t in terminations], dtype=np.int32)

    def mov_avg(x: np.ndarray, k: int) -> np.ndarray:
        if len(x) < k:
            k = max(1, len(x))
        return np.convolve(x, np.ones(k) / k, mode="same")

    return {
        "reward": rewards,
        "reward_ma": mov_avg(rewards, window),
        "steps": steps,
        "steps_ma": mov_avg(steps, window),
        "success_rate_ma": mov_avg(success, window),
        "collision_rate_ma": mov_avg(collisions, window),
        "oob_rate_ma": mov_avg(oobs, window),
    }


def plot_before_after(before_metrics: Dict[str, np.ndarray], after_metrics: Dict[str, np.ndarray],
                      out_before: str, out_after: str):
    # BEFORE graph (no shaping)
    fig1, axes1 = plt.subplots(2, 1, figsize=(9, 7))
    x = np.arange(len(before_metrics["reward"]))
    axes1[0].set_title("Baseline (no reward shaping): Reward & Steps")
    axes1[0].plot(x, before_metrics["reward"], alpha=0.3, label="reward", color="tab:blue")
    axes1[0].plot(x, before_metrics["reward_ma"], label="reward MA", color="tab:green")
    ax2 = axes1[0].twinx()
    ax2.plot(x, before_metrics["steps_ma"], color="tab:orange", label="steps MA")
    axes1[0].set_xlabel("Episode")
    axes1[0].set_ylabel("Reward")
    ax2.set_ylabel("Steps")
    axes1[0].legend(loc="upper left")
    ax2.legend(loc="upper right")

    axes1[1].set_title("Baseline: Success/Collision/OOB rates (moving average)")
    axes1[1].plot(x, before_metrics["success_rate_ma"] , label="success", color="tab:green")
    axes1[1].plot(x, before_metrics["collision_rate_ma"], label="collision", color="tab:red")
    axes1[1].plot(x, before_metrics["oob_rate_ma"], label="oob", color="tab:purple")
    axes1[1].set_ylim(0, 1)
    axes1[1].set_xlabel("Episode")
    axes1[1].set_ylabel("Rate")
    axes1[1].legend(loc="upper right")

    fig1.tight_layout()
    fig1.savefig(out_before, dpi=150)

    # AFTER graph (with shaping)
    fig2, axes2 = plt.subplots(2, 1, figsize=(9, 7))
    x2 = np.arange(len(after_metrics["reward"]))
    axes2[0].set_title("After (progress shaping): Reward & Steps")
    axes2[0].plot(x2, after_metrics["reward"], alpha=0.3, label="reward", color="tab:blue")
    axes2[0].plot(x2, after_metrics["reward_ma"], label="reward MA", color="tab:green")
    ax2b = axes2[0].twinx()
    ax2b.plot(x2, after_metrics["steps_ma"], color="tab:orange", label="steps MA")
    axes2[0].set_xlabel("Episode")
    axes2[0].set_ylabel("Reward")
    ax2b.set_ylabel("Steps")
    axes2[0].legend(loc="upper left")
    ax2b.legend(loc="upper right")

    axes2[1].set_title("After: Success/Collision/OOB rates (moving average)")
    axes2[1].plot(x2, after_metrics["success_rate_ma"] , label="success", color="tab:green")
    axes2[1].plot(x2, after_metrics["collision_rate_ma"], label="collision", color="tab:red")
    axes2[1].plot(x2, after_metrics["oob_rate_ma"], label="oob", color="tab:purple")
    axes2[1].set_ylim(0, 1)
    axes2[1].set_xlabel("Episode")
    axes2[1].set_ylabel("Rate")
    axes2[1].legend(loc="upper right")

    fig2.tight_layout()
    fig2.savefig(out_after, dpi=150)


def run_experiment(episodes: int = 500, progress_scale: float = 0.5,
                   out_dir: str = "models_compressed") -> Tuple[List[EpisodeStats], List[EpisodeStats]]:
    os.makedirs(out_dir, exist_ok=True)
    # Ensure Ursina's first-run BAM conversion target exists
    os.makedirs(os.path.join(out_dir, "assets"), exist_ok=True)
    

    # Baseline: no shaping
    env_base = CrossyEnv(reward_progress_scale=0.0)
    print("Starting baseline run (no shaping)...")
    base_stats = train_and_collect(env_base, episodes=episodes,
                                   save_path=os.path.join(out_dir, "baseline_dqn.pth"), label="baseline")

    # Shaped: positive reward for forward progress
    env_shaped = CrossyEnv(reward_progress_scale=progress_scale)
    print(f"Starting shaped run (progress_scale={progress_scale})...")
    shaped_stats = train_and_collect(env_shaped, episodes=episodes,
                                     save_path=os.path.join(out_dir, "shaped_dqn.pth"), label="shaped")

    # Compute metrics and plot
    before_metrics = compute_metrics(base_stats)
    after_metrics = compute_metrics(shaped_stats)

    out_before = os.path.join(out_dir, "baseline_metrics.png")
    out_after = os.path.join(out_dir, "shaped_metrics.png")
    plot_before_after(before_metrics, after_metrics, out_before, out_after)
    plt.close('all')

    # Also persist raw data (simple CSV-like txt)
    def dump_stats(path: str, stats: List[EpisodeStats]):
        with open(path, "w", encoding="utf-8") as f:
            f.write("episode,reward,steps,termination,epsilon_end\n")
            for s in stats:
                f.write(f"{s.episode},{s.reward},{s.steps},{s.termination},{s.epsilon_end}\n")

    dump_stats(os.path.join(out_dir, "baseline_stats.csv"), base_stats)
    dump_stats(os.path.join(out_dir, "shaped_stats.csv"), shaped_stats)

    return base_stats, shaped_stats


if __name__ == "__main__":
    # Default run: 1000 episodes each, progress_scale=1.0 for better learning
    run_experiment(episodes=10000, progress_scale=1.0)
    print("Experiment complete. Plots and data saved in 'models_compressed'.")
