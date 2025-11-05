# === model.py ===
# Simple, flexible DQN model that adapts to either image or vector observations.
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, obs_shape, n_actions, hidden=256):
        super().__init__()
        self.obs_shape = obs_shape
        self.n_actions = n_actions

        # obs_shape can be (channels, H, W) for images or (N,) for vector states
        if isinstance(obs_shape, (tuple, list)) and len(obs_shape) == 3:
            c, h, w = obs_shape
            # small conv net
            self.conv = nn.Sequential(
                nn.Conv2d(c, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU()
            )
            # compute conv output size dynamically with a dummy tensor
            # Note: This is computed before model is moved to device, so use CPU here
            # The actual forward pass will use the correct device
            with torch.no_grad():
                dummy = torch.zeros(1, c, h, w)
                conv_out = self.conv(dummy)
                conv_size = conv_out.view(1, -1).shape[1]
            self.head = nn.Sequential(
                nn.Linear(conv_size, hidden),
                nn.ReLU(),
                nn.Linear(hidden, n_actions)
            )
        else:
            # assume flat vector
            in_size = obs_shape[0] if isinstance(obs_shape, (tuple, list)) else int(obs_shape)
            self.conv = None
            self.head = nn.Sequential(
                nn.Linear(in_size, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, n_actions)
            )

    def forward(self, x):
        # expect torch tensor
        if self.conv is not None:
            # x shape (B, C, H, W)
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            return self.head(x)
        else:
            return self.head(x)
