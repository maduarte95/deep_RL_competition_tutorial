import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torch.nn as nn
import numpy as np

from tournament_loader import load_all_policies, load_all_group_names
from pettingzoo_wrapper import AdversaryObsRewardWrapper, make_env





TIMESTEPS_PER_EPISODE = 300 # max timesteps per episode
prey_groups = list(load_all_group_names('prey'))
predator_groups = list(load_all_group_names('predator'))


torch.set_grad_enabled(False)

def get_agent_positions(env):
    world = env.unwrapped.world
    positions = {}
    for agent in world.agents:
        positions[agent.name] = agent.state.p_pos.copy()
    return positions



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
        
    #################################################
    # Define environment
    #################################################

    env = make_env(TIMESTEPS_PER_EPISODE, num_predators=len(predator_groups), num_preys=len(prey_groups))
    obs = env.reset()
    print('Environment created with', len(env.agents), 'agents.')


    ###############################################
    # Initialize policies for each agent
    ###############################################
    policies, group_names = load_all_policies(env, prey_groups, predator_groups)
    print('group_names:', group_names)


    render_tag(num_timesteps=500, env=env, scores={agent: 0.0 for agent in env.agents}, group_names=group_names, policies=policies)



