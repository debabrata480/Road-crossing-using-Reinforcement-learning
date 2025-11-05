# === agent.py ===
# DQN agent with replay buffer and basic utilities
import random
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, model, lr=1e-3, gamma=0.99, batch_size=64, device=None, target_update=1000):
        # GPU device detection with explicit CUDA selection
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                # Print GPU information only once (if not already printed by caller)
                # This prevents duplicate messages
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device
        
        self.policy_net = model.to(self.device)
        # clone for target
        import copy
        self.target_net = copy.deepcopy(self.policy_net).to(self.device)
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay = ReplayBuffer()
        self.steps_done = 0
        self.target_update = target_update

    def select_action(self, state, n_actions, eps=0.05):
        # state: numpy array or torch tensor (single)
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
        # convert to tensors
        def to_tensor(x, dtype=torch.float32):
            return torch.as_tensor(np.array(x), device=self.device, dtype=dtype)

        state = to_tensor(trans.state)
        action = torch.as_tensor(np.array(trans.action), device=self.device, dtype=torch.long).unsqueeze(1)
        reward = to_tensor(trans.reward).unsqueeze(1)
        next_state = to_tensor(trans.next_state)
        done = torch.as_tensor(np.array(trans.done), device=self.device, dtype=torch.float32).unsqueeze(1)

        # current Q
        q_values = self.policy_net(state).gather(1, action)
        # next Q
        with torch.no_grad():
            next_q = self.target_net(next_state).max(1)[0].unsqueeze(1)
            expected_q = reward + (1.0 - done) * self.gamma * next_q

        loss = nn.MSELoss()(q_values, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()

        # soft/hard update
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
