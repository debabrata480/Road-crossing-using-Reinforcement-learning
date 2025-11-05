"""
Enhanced training script with strategies to increase success rate over time.
This script implements multiple improvements to boost success rate.
"""

import math
import os
from collections import deque, Counter
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import random
import numpy as np
import torch
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import DQN
from agent import DQNAgent
import copy
from game_crossy import CrossyEnv


@dataclass
class EpisodeStats:
    episode: int
    reward: float
    steps: int
    termination: str
    epsilon_end: float


class EnhancedDQNAgent:
    """Extended DQNAgent with learning rate scheduling."""
    
    def __init__(self, model, lr=1e-3, gamma=0.99, batch_size=64, device=None, target_update=500,
                 lr_decay_episodes=2000, lr_decay_factor=0.5):
        # GPU device detection with explicit CUDA selection
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device
        
        # Initialize base agent functionality
        self.policy_net = model.to(self.device)
        self.target_net = copy.deepcopy(self.policy_net).to(self.device)
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        from agent import ReplayBuffer
        self.replay = ReplayBuffer()
        self.steps_done = 0
        self.target_update = target_update
        
        # LR scheduling attributes
        self.initial_lr = lr
        self.lr_decay_episodes = lr_decay_episodes
        self.lr_decay_factor = lr_decay_factor
        self.training_episodes = 0
    
    def update_learning_rate(self, episode):
        """Decay learning rate over time."""
        self.training_episodes = episode
        if episode > 0 and episode % self.lr_decay_episodes == 0:
            new_lr = self.initial_lr * (self.lr_decay_factor ** (episode // self.lr_decay_episodes))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            print(f"[Episode {episode}] Learning rate decayed to {new_lr:.6f}")
    
    # Delegate methods to base agent functionality
    def select_action(self, state, n_actions, eps=0.05):
        sample = random.random()
        self.steps_done += 1
        if sample < eps:
            return random.randrange(n_actions)
        else:
            self.policy_net.eval()
            with torch.no_grad():
                s = torch.as_tensor(state, device=self.device, dtype=torch.float32)
                if s.ndim == 1:
                    s = s.unsqueeze(0)
                q = self.policy_net(s)
                return int(q.argmax(1).item())
    
    def push_transition(self, state, action, reward, next_state, done):
        self.replay.push(state, action, reward, next_state, done)
    
    def optimize(self):
        if len(self.replay) < self.batch_size:
            return None
        trans = self.replay.sample(self.batch_size)
        
        def to_tensor(x, dtype=torch.float32):
            return torch.as_tensor(np.array(x), device=self.device, dtype=dtype)
        
        state = to_tensor(trans.state)
        action = torch.as_tensor(np.array(trans.action), device=self.device, dtype=torch.long).unsqueeze(1)
        reward = to_tensor(trans.reward).unsqueeze(1)
        next_state = to_tensor(trans.next_state)
        done = torch.as_tensor(np.array(trans.done), device=self.device, dtype=torch.float32).unsqueeze(1)
        
        q_values = self.policy_net(state).gather(1, action)
        with torch.no_grad():
            next_q = self.target_net(next_state).max(1)[0].unsqueeze(1)
            expected_q = reward + (1.0 - done) * self.gamma * next_q
        
        loss = torch.nn.MSELoss()(q_values, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()
        
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
    
    def save(self, path):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save(self.policy_net.state_dict(), path)
    
    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())


def train_with_improvements(
    env: CrossyEnv, 
    episodes: int = 10000, 
    max_steps: int = 2000,
    lr: float = 1e-3, 
    batch_size: int = 64, 
    target_update: int = 250,  # More frequent updates
    eps_start: float = 1.0, 
    eps_end: float = 0.01,  # Lower minimum epsilon for more exploitation
    eps_decay: int = 3000,  # Faster decay
    save_path: str = None, 
    label: str = "enhanced",
    use_lr_scheduling: bool = True,
    device: torch.device = None
) -> List[EpisodeStats]:
    """
    Train with improvements:
    - Learning rate scheduling
    - Better epsilon decay
    - More frequent target updates
    - Lower minimum epsilon
    """
    obs_shape = env.observation_shape
    n_actions = env.n_actions

    model = DQN(obs_shape, n_actions, hidden=512)  # Wider network
    # Set device if not provided
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if use_lr_scheduling:
        agent = EnhancedDQNAgent(model, lr=lr, batch_size=batch_size, target_update=target_update, device=device)
    else:
        agent = DQNAgent(model, lr=lr, batch_size=batch_size, target_update=target_update, device=device)

    all_stats: List[EpisodeStats] = []

    total_steps = 0
    best_success_rate = 0.0
    
    for ep in range(1, episodes + 1):
        state = env.reset()
        episode_reward = 0.0
        steps = 0
        termination_reason = None
        last_eps = eps_end

        # Update learning rate if using scheduling
        if use_lr_scheduling and isinstance(agent, EnhancedDQNAgent):
            agent.update_learning_rate(ep)

        for t in range(max_steps):
            # Exponential epsilon decay
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

        # Calculate recent success rate
        if ep >= 100:
            recent_success = sum(1 for s in all_stats[-100:] if s.termination == "success")
            current_success_rate = recent_success / 100.0
            if current_success_rate > best_success_rate:
                best_success_rate = current_success_rate
                if save_path:
                    # Save best model
                    best_path = save_path.replace('.pth', '_best.pth')
                    agent.save(best_path)

        if ep % 10 == 0:
            recent_stats = all_stats[-100:] if len(all_stats) >= 100 else all_stats
            recent_success = sum(1 for s in recent_stats if s.termination == "success")
            recent_success_rate = recent_success / len(recent_stats) * 100
            print(f"[{label}] Episode {ep}/{episodes} | "
                  f"Reward={episode_reward:.2f} | Steps={steps} | "
                  f"Termination={termination_reason} | "
                  f"Recent Success Rate={recent_success_rate:.1f}% | "
                  f"ε={last_eps:.3f}")
        
        if save_path and ep % 500 == 0:
            agent.save(save_path)

    if save_path:
        agent.save(save_path)

    return all_stats


def compute_metrics(stats: List[EpisodeStats], window: int = 20) -> Dict[str, np.ndarray]:
    """Compute metrics including success rate over time."""
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

    # Calculate cumulative success rate (overall success rate up to each episode)
    cumulative_success = np.cumsum(success) / np.arange(1, len(success) + 1)

    return {
        "reward": rewards,
        "reward_ma": mov_avg(rewards, window),
        "steps": steps,
        "steps_ma": mov_avg(steps, window),
        "success": success,
        "success_rate_ma": mov_avg(success, window),
        "cumulative_success_rate": cumulative_success,
        "collision_rate_ma": mov_avg(collisions, window),
        "oob_rate_ma": mov_avg(oobs, window),
    }


def plot_improvements(metrics: Dict[str, np.ndarray], out_path: str):
    """Plot metrics with success rate progression."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    x = np.arange(len(metrics["reward"]))

    # Plot 1: Reward and Steps
    axes[0].set_title("Reward & Steps Over Time")
    axes[0].plot(x, metrics["reward"], alpha=0.2, label="reward", color="tab:blue")
    axes[0].plot(x, metrics["reward_ma"], label="reward MA", color="tab:green", linewidth=2)
    ax2 = axes[0].twinx()
    ax2.plot(x, metrics["steps_ma"], color="tab:orange", label="steps MA", linewidth=2)
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward", color="tab:green")
    ax2.set_ylabel("Steps", color="tab:orange")
    axes[0].legend(loc="upper left")
    ax2.legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Success Rate (moving average and cumulative)
    axes[1].set_title("Success Rate Progression")
    axes[1].plot(x, metrics["success_rate_ma"], label="Success Rate (MA)", color="tab:green", linewidth=2)
    axes[1].plot(x, metrics["cumulative_success_rate"] * 100, label="Cumulative Success Rate", 
                color="tab:purple", linewidth=2, linestyle="--")
    axes[1].set_ylabel("Success Rate (%)")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylim(0, 100)
    axes[1].legend(loc="upper left")
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Termination Rates
    axes[2].set_title("Termination Rates (Moving Average)")
    axes[2].plot(x, metrics["success_rate_ma"] * 100, label="success", color="tab:green", linewidth=2)
    axes[2].plot(x, metrics["collision_rate_ma"] * 100, label="collision", color="tab:red", linewidth=2)
    axes[2].plot(x, metrics["oob_rate_ma"] * 100, label="oob", color="tab:purple", linewidth=2)
    axes[2].set_ylim(0, 100)
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Rate (%)")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {out_path}")


if __name__ == "__main__":
    # Configuration with improvements
    episodes = 15000  # More episodes
    progress_scale = 1.5  # Stronger reward shaping
    
    # Print GPU availability
    print("\n" + "="*60)
    print("DEVICE INFORMATION")
    print("="*60)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ CUDA Available: Yes")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        device = torch.device('cpu')
        print(f"✗ CUDA Available: No")
        print(f"  Using CPU for training (will be slower)")
    print("="*60 + "\n")
    
    out_dir = "models_improved"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "assets"), exist_ok=True)

    # Enhanced shaped environment
    env_shaped = CrossyEnv(reward_progress_scale=progress_scale)
    print(f"Starting enhanced training with {episodes} episodes...")
    print(f"Configuration:")
    print(f"  - Device: {device}")
    print(f"  - Progress scale: {progress_scale}")
    print(f"  - Episodes: {episodes}")
    print(f"  - Learning rate scheduling: Enabled")
    print(f"  - Wider network: 512 hidden units")
    print(f"  - More frequent target updates: 250")
    print(f"  - Lower minimum epsilon: 0.01")
    
    stats = train_with_improvements(
        env_shaped, 
        episodes=episodes,
        lr=1e-3,
        batch_size=64,
        target_update=250,
        eps_start=1.0,
        eps_end=0.01,
        eps_decay=3000,
        save_path=os.path.join(out_dir, "enhanced_dqn.pth"),
        label="enhanced",
        use_lr_scheduling=True,
        device=device
    )

    # Compute and plot metrics
    metrics = compute_metrics(stats)
    
    # Save plots
    plot_improvements(metrics, os.path.join(out_dir, "enhanced_metrics.png"))
    
    # Save statistics
    with open(os.path.join(out_dir, "enhanced_stats.csv"), "w", encoding="utf-8") as f:
        f.write("episode,reward,steps,termination,epsilon_end\n")
        for s in stats:
            f.write(f"{s.episode},{s.reward},{s.steps},{s.termination},{s.epsilon_end}\n")
    
    # Print final statistics
    final_stats = stats[-1000:]  # Last 1000 episodes
    final_success_rate = sum(1 for s in final_stats if s.termination == "success") / len(final_stats) * 100
    overall_success_rate = sum(1 for s in stats if s.termination == "success") / len(stats) * 100
    
    print("\n" + "="*60)
    print("FINAL STATISTICS")
    print("="*60)
    print(f"Total episodes: {len(stats)}")
    print(f"Overall success rate: {overall_success_rate:.2f}%")
    print(f"Final 1000 episodes success rate: {final_success_rate:.2f}%")
    print(f"Mean reward (last 1000): {np.mean([s.reward for s in final_stats]):.2f}")
    print(f"Mean steps (last 1000): {np.mean([s.steps for s in final_stats]):.2f}")
    print("="*60)

