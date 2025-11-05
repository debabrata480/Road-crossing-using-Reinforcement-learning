# === train_crossy.py ===
# Import torch FIRST to avoid DLL conflicts with Ursina/Panda3D
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

from model import DQN
from agent import DQNAgent
from game_crossy import CrossyEnv

def train(env, episodes=300, max_steps=2000, save_path="models/dqn_crossy.pth"):
    # Print GPU availability
    print("\n" + "="*60)
    print("DEVICE INFORMATION")
    print("="*60)
    if torch.cuda.is_available():
        print(f"✓ CUDA Available: Yes")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        device = torch.device('cuda')
    else:
        print(f"✗ CUDA Available: No")
        print(f"  Using CPU for training")
        device = torch.device('cpu')
    print("="*60 + "\n")
    
    obs_shape = env.observation_shape
    n_actions = env.n_actions

    model = DQN(obs_shape, n_actions)
    agent = DQNAgent(model, lr=1e-3, batch_size=64, target_update=500, device=device)

    eps_start, eps_end, eps_decay = 1.0, 0.05, 5000
    all_rewards, mean_rewards, losses = [], [], []
    smoothing = 20
    reward_deque, loss_deque = deque(maxlen=smoothing), deque(maxlen=smoothing)

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,6))

    total_steps = 0
    for ep in range(1, episodes+1):
        state = env.reset()
        episode_reward = 0.0

        for t in range(max_steps):
            eps = eps_end + (eps_start - eps_end) * math.exp(-1.0 * total_steps / eps_decay)
            action = agent.select_action(state, n_actions, eps=eps)

            next_state, reward, done, _ = env.step(action)
            agent.push_transition(state, action, reward, next_state, float(done))
            loss = agent.optimize()
            if loss is not None:
                losses.append(loss)
                loss_deque.append(loss)

            state = next_state
            episode_reward += reward
            total_steps += 1
            if done:
                break

        all_rewards.append(episode_reward)
        reward_deque.append(episode_reward)
        mean_rewards.append(np.mean(reward_deque))

        ax1.clear(); ax2.clear()
        ax1.set_title("Episode rewards")
        ax2.set_title("Loss")
        ax1.plot(all_rewards, label="reward")
        ax1.plot(mean_rewards, label=f"mean({smoothing})")
        ax1.legend()
        if losses:
            ax2.plot(np.convolve(losses, np.ones(10)/10, mode="valid"))
        plt.pause(0.001)

        print(f"Episode {ep}/{episodes} reward={episode_reward:.2f} mean={mean_rewards[-1]:.2f} eps={eps:.3f}")

        if ep % 50 == 0:
            agent.save(save_path)

    agent.save(save_path)
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    env = CrossyEnv()
    train(env, episodes=10000 )
