import torch
import torch.nn as nn
from default_policy import PolicyNet



class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.actor = PolicyNet(obs_dim, act_dim)

        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        return self.actor(x), self.critic(x)
