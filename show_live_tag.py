import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torch.nn as nn
from pettingzoo.mpe import simple_tag_v3
import numpy as np

from tournament_loader import load_all_policies, load_all_group_names
from pettingzoo_wrapper import AdversaryObsRewardWrapper



prey_groups = list(load_all_group_names('prey'))
predator_groups = list(load_all_group_names('predator'))


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
        plt.scatter(pos[0], pos[1], c="gray", s=2300)#, marker="s")

    # Draw agents
    for agent in world.agents:
        pos = agent.state.p_pos
        name = agent.name
        label = group_labels[name] #.get(name, name)

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

    plt.pause(0.05)


def render_tag(num_timesteps, env, scores, group_names, policies):

    plt.figure(figsize=(6, 6))

    obs = env.reset()
    scores = {agent: 0.0 for agent in env.agents}

    for t in range(num_timesteps):
        actions = {}
        for agent, ob in obs.items():
            with torch.no_grad():
                actions[agent] = policies[agent](
                    torch.tensor(ob, dtype=torch.float32))

        obs, reward, term, trunc, infos = env.step(actions)

        if all(term.values()) or all(trunc.values()):
            obs = env.reset()

        for agent in reward:
            scores[agent] += reward[agent]

        render_with_labels(env, scores, group_names)
    
    plt.pause(1.0)    

    plt.close()

    print('Final scores:')
    for agent, score in scores.items():
        print(f"{group_names[agent]} ({agent}): {score:.2f}")

    return scores


def plot_learning_curve(episode_rewards):
    plt.figure()
    plt.plot(np.convolve(episode_rewards, np.ones(50)/50, mode='valid'))
    plt.xlabel('Episode')
    plt.ylabel('Return (smoothed)')
    plt.title('Training Curve')
    plt.show()


if __name__ == "__main__":
    plt.figure(figsize=(6, 6))


    # GROUP_LABELS = {
    #     "adversary_0": "Group A",
    #     "adversary_1": "Group B",
    #     "adversary_2": "Group C",
    #     "agent_0": "Prey",
    # }

    # #################################################
    # # Define a simple policy network
    # #################################################
    # class PolicyNet(nn.Module):
    #     def __init__(self, obs_dim, act_dim):
    #         super().__init__()
    #         self.net = nn.Sequential(
    #             nn.Linear(obs_dim, 64),
    #             nn.ReLU(),
    #             nn.Linear(64, 64),
    #             nn.ReLU(),
    #             nn.Linear(64, act_dim),
    #             nn.Tanh(),  # continuous actions ∈ [-1, 1]
    #         )

    #     def forward(self, obs):
    #         return torch.sigmoid(self.net(obs))
        
    #################################################
    # Define environment
    #################################################

    # num_agents = number_of_submissions()  
    # print('Number of submissions:', num_agents)

    # env = simple_tag_v3.parallel_env(

        
    #     continuous_actions=True,
    #     render_mode="rgb_array",
    #     # render_mode="human",
    # )
    base_env = simple_tag_v3.parallel_env(
        num_adversaries=len(predator_groups),
        num_good=len(prey_groups),
        num_obstacles=2,
        max_cycles=300,
        continuous_actions=True
    )

    env = AdversaryObsRewardWrapper(base_env)

    # obs = env.reset(seed=0)
    #################################################


    # ###############################################
    # # Initialize policies for each agent
    # ###############################################
    # policies = {}
    # for agent in env.agents:
    #     obs_dim = env.observation_space(agent).shape[0]
    #     act_dim = env.action_space(agent).shape[0]
    #     policies[agent] = PolicyNet(obs_dim, act_dim)
    # ################################################


    ###############################################
    # Initialize policies for each agent
    ###############################################
    policies, group_names = load_all_policies(env, prey_groups, predator_groups)
    print('policies:', policies.keys())
    print('group_names:', group_names)

    # policies['agent_0'] = lambda obs: np.random.uniform(0, 1, size=env.action_space('agent_0').shape[0]).astype(np.float32)
    # group_names['agent_0'] = 'Prey'
    ################################################

    # obs = env.reset()
    # scores = {agent: 0.0 for agent in env.agents}

    # for t in range(200):
    #     actions = {}
    #     for agent, ob in obs.items():
    #         with torch.no_grad():
    #             actions[agent] = policies[agent](
    #                 torch.tensor(ob, dtype=torch.float32))

    #     obs, rewards, terminations, truncations, infos = env.step(actions)

    #     for agent in rewards:
    #         scores[agent] += rewards[agent]

    #     render_with_labels(env, scores, group_names)

    #     if all(terminations.values()) or all(truncations.values()):
    #         break

    # plt.close()
    render_tag(num_timesteps=500, env=env, scores={agent: 0.0 for agent in env.agents}, group_names=group_names, policies=policies)



