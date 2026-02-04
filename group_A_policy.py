import torch
import torch.nn as nn

class PolicyNet(nn.Module):

    # Do NOT change this signature, otherwise the tournament loader will not work
    def __init__(self, obs_dim, act_dim): 
        super().__init__()

        # This network architecture is just a suggestion, feel free to change it as you like (but do not change the class name or __init__ signature)
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim),
            nn.Sigmoid()
        )

    # Do NOT change this signature, otherwise the tournament loader will not work
    def forward(self, x): 
        return self.net(x)