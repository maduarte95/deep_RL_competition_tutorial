import torch
import torch.nn as nn
from pettingzoo.mpe import simple_tag_v3
import numpy as np


#################################################
# Define environment
#################################################
env = simple_tag_v3.parallel_env(
    num_adversaries=3,
    num_good=1,
    max_cycles=300,
    continuous_actions=True,
)

obs = env.reset(seed=0)



#################################################
# Define a simple policy network
#################################################
class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim),
            nn.Tanh(),  # continuous actions ∈ [-1, 1]
        )

    def forward(self, obs):
        return torch.sigmoid(self.net(obs))


###############################################
# Initialize policies for each agent
###############################################
policies = {}
for agent in env.agents:
    obs_dim = env.observation_space(agent).shape[0]
    act_dim = env.action_space(agent).shape[0]
    policies[agent] = PolicyNet(obs_dim, act_dim)


#################################################
# Rollout loop
#################################################
scores = {agent: 0.0 for agent in env.agents}

for episode in range(5):
    obs, _ = env.reset()
    done = {agent: False for agent in env.agents}

    while not all(done.values()):
        actions = {}
        for agent, ob in obs.items():
            if agent == "agent_0":  # prey: simple scripted policy
                actions[agent] = np.random.uniform(0, 1, size=5).astype(np.float32)
            else:
                with torch.no_grad():
                    o = torch.tensor(ob, dtype=torch.float32)
                    actions[agent] = policies[agent](o).numpy()

        # print('Actions:', actions, type(actions['agent_0'][0]), type(actions['adversary_0'][0]))
        obs, rewards, terminations, truncations, infos = env.step(actions)

        for agent, r in rewards.items():
            scores[agent] += r

        done = {
            agent: terminations[agent] or truncations[agent]
            for agent in env.agents
        }


