import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torch.nn as nn
from pettingzoo.mpe import simple_tag_v3
import numpy as np


def get_agent_positions(env):
    world = env.unwrapped.world
    positions = {}
    for agent in world.agents:
        positions[agent.name] = agent.state.p_pos.copy()
    return positions

# def render_with_labels(env, scores, group_labels):
#     frame = env.render()  # RGB array
#     positions = get_agent_positions(env)

#     plt.imshow(frame)
#     plt.axis("off")

#     # Overlay agent labels
#     for agent, pos in positions.items():
#         x = (pos[0] + 1) * frame.shape[1] / 2
#         y = (1 - pos[1]) * frame.shape[0] / 2

#         plt.text(
#             x,
#             y - 10,
#             group_labels.get(agent, agent),
#             color="white",
#             fontsize=9,
#             ha="center",
#             bbox=dict(facecolor="black", alpha=0.5, pad=1),
#         )

#     # Scoreboard
#     score_text = "\n".join(
#         f"{group_labels.get(a, a)}: {scores[a]:.2f}"
#         for a in scores
#     )

#     plt.text(
#         10,
#         10,
#         score_text,
#         fontsize=10,
#         color="white",
#         va="top",
#         bbox=dict(facecolor="black", alpha=0.7),
#     )

#     plt.pause(0.01)
#     plt.clf()

# def render_with_labels(env, scores, group_labels):
#     frame = env.render()
#     world = env.unwrapped.world

#     plt.clf()
#     plt.imshow(frame, extent=[-1, 1, -1, 1])
#     plt.axis("off")

#     # Draw agent labels in WORLD coordinates
#     for agent in world.agents:
#         pos = agent.state.p_pos
#         name = agent.name
#         label = group_labels.get(name, name)

#         plt.text(
#             pos[0],
#             pos[1] + 0.05,
#             label,
#             color="white",
#             fontsize=9,
#             ha="center",
#             va="bottom",
#             bbox=dict(facecolor="black", alpha=0.6, pad=1),
#         )

#     # Scoreboard (screen-space)
#     score_text = "\n".join(
#         f"{group_labels.get(a, a)}: {scores[a]:.2f}"
#         for a in scores
#     )

#     plt.text(
#         -0.95, 0.95,
#         score_text,
#         fontsize=10,
#         color="white",
#         ha="left",
#         va="top",
#         bbox=dict(facecolor="black", alpha=0.7),
#     )

#     plt.pause(0.01)


def render_with_labels(env, scores, group_labels):
    world = env.unwrapped.world

    plt.clf()
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.gca().set_aspect("equal")
    plt.axis("off")

    # Draw landmarks
    for landmark in world.landmarks:
        pos = landmark.state.p_pos
        plt.scatter(pos[0], pos[1], c="gray", s=200, marker="s")

    # Draw agents
    for agent in world.agents:
        pos = agent.state.p_pos
        name = agent.name
        label = group_labels.get(name, name)

        color = "red" if "adversary" in name else "green"

        plt.scatter(pos[0], pos[1], c=color, s=200)
        plt.text(
            pos[0],
            pos[1] + 0.05,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            color="white",
            bbox=dict(facecolor="black", alpha=0.6, pad=1),
        )

    # Scoreboard
    score_text = "\n".join(
        f"{group_labels.get(a, a)}: {scores[a]:.2f}"
        for a in scores
    )

    plt.text(
        -0.95, 0.95,
        score_text,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(facecolor="black", alpha=0.7),
        color="white"
    )

    plt.pause(0.01)



plt.figure(figsize=(6, 6))


GROUP_LABELS = {
    "adversary_0": "Group A",
    "adversary_1": "Group B",
    "adversary_2": "Group C",
    "agent_0": "Prey",
}

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
    
#################################################
# Define environment
#################################################
env = simple_tag_v3.parallel_env(
    num_adversaries=3,
    num_good=1,
    max_cycles=300,
    continuous_actions=True,
    render_mode="rgb_array",
    # render_mode="human",
)

obs = env.reset(seed=0)
#################################################


###############################################
# Initialize policies for each agent
###############################################
policies = {}
for agent in env.agents:
    obs_dim = env.observation_space(agent).shape[0]
    act_dim = env.action_space(agent).shape[0]
    policies[agent] = PolicyNet(obs_dim, act_dim)
################################################

obs, _ = env.reset()
scores = {agent: 0.0 for agent in env.agents}

for t in range(300):
    actions = {}
    for agent, ob in obs.items():
        with torch.no_grad():
            actions[agent] = policies[agent](
                torch.tensor(ob, dtype=torch.float32)
            ).numpy()

    obs, rewards, terminations, truncations, infos = env.step(actions)

    for agent in rewards:
        scores[agent] += rewards[agent]

    render_with_labels(env, scores, GROUP_LABELS)

    if all(terminations.values()) or all(truncations.values()):
        break

plt.close()
