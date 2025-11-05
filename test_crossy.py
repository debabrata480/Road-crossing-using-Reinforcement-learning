# === test_crossy.py ===
# Import torch FIRST to avoid DLL conflicts with Ursina/Panda3D
import torch
import time
from game_crossy import CrossyEnv
from model import DQN
from agent import DQNAgent

def test(model_path="models/dqn_crossy.pth", episodes=15):
    # Set device for GPU acceleration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"[GPU] Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print(f"[CPU] Using CPU (CUDA not available)")
    
    env = CrossyEnv()
    obs_shape, n_actions = env.observation_shape, env.n_actions

    model = DQN(obs_shape, n_actions)
    agent = DQNAgent(model, device=device)
    agent.load(model_path)

    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        print(f"Episode {ep+1}/{episodes}")

        while not done:
            # greedy policy (eps=0 → no randomness)
            action = agent.select_action(state, n_actions, eps=0.0)
            state, reward, done, _ = env.step(action)
            env.render()     # step Ursina app
            total_reward += reward
            time.sleep(0.05) # slow down for visibility

        print(f"Episode {ep+1} finished with reward {total_reward:.2f}")

if __name__ == "__main__":
    test()
