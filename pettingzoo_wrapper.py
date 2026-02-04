import numpy as np
from pettingzoo.mpe import simple_tag_v3

class AdversaryObsRewardWrapper:
    def __init__(self, env):
        self.env = env
        self.adversaries = [a for a in env.possible_agents if "adversary" in a]
        self.preys = [a for a in env.possible_agents if "agent" in a]

    def reset(self, seed=None):
        obs, _ = self.env.reset(seed=seed)
        return self._transform_obs(obs)

    def step(self, actions):
        obs, rewards, terminations, truncations, infos = self.env.step(actions)

        obs = self._transform_obs(obs)
        rewards = self._transform_rewards(rewards)

        return obs, rewards, terminations, truncations, infos

    def render(self):
        return self.env.render()

    @property
    def agents(self):
        return self.env.agents

    @property
    def possible_agents(self):
        return self.env.possible_agents
    
    @property
    def unwrapped(self):
        return self.env.unwrapped

    def observation_space(self, agent):
        if agent in self.adversaries:
            
            world = self.env.unwrapped.world
            agent_obj = next(a for a in world.agents if a.name == agent)

            # ego position
            obs_dim = len(agent_obj.state.p_pos)

            # relative prey positions
            obs_dim += len(self.preys) * len(agent_obj.state.p_pos)

            # relative obstacle positions
            obs_dim += len(world.landmarks) * len(agent_obj.state.p_pos)

            return obs_dim
        return self.env.observation_space(agent).shape[0]

    def action_space(self, agent):
        return self.env.action_space(agent).shape[0]

    # -----------------------
    # Custom observation
    # -----------------------
    def _transform_obs(self, obs):
        world = self.env.unwrapped.world
        new_obs = {}

        for agent in self.env.agents:
            if agent in self.adversaries:
                new_obs[agent] = self._adversary_obs(agent, world)
            else:
                new_obs[agent] = obs[agent]  # prey unchanged

        return new_obs

    def _adversary_obs(self, agent_name, world):
        agent = next(a for a in world.agents if a.name == agent_name)

        ego_pos = agent.state.p_pos

        # relative prey positions
        prey_rel = []
        for prey_name in self.preys:
            prey = next(a for a in world.agents if a.name == prey_name)
            prey_rel.append(prey.state.p_pos - ego_pos)

        # relative obstacle positions (landmarks)
        obs_rel = []
        for landmark in world.landmarks:
            obs_rel.append(landmark.state.p_pos - ego_pos)

        return np.concatenate([ego_pos] + prey_rel + obs_rel).astype(np.float32)

    # -----------------------
    # Custom reward
    # -----------------------
    def _transform_rewards(self, env_rewards):
        world = self.env.unwrapped.world
        rewards = {}

        # min_dist_to_prey = np.zeros(len(self.adversaries))
        for agent_name in self.env.agents:
            if agent_name in self.adversaries:
                agent = next(a for a in world.agents if a.name == agent_name)

                # distance to closest prey
                dists = []
                for prey_name in self.preys:
                    prey = next(a for a in world.agents if a.name == prey_name)
                    dists.append(np.linalg.norm(agent.state.p_pos - prey.state.p_pos))

                min_dist = min(dists)
                # if env_rewards[agent_name] != 0.0:
                #     min_dist_to_prey[self.adversaries.index(agent_name)] = min_dist
                    # print('Adversary:', agent_name, 'min_dist to prey:', min_dist)
                bonus = env_rewards[agent_name] * (min_dist < 0.2)
                rewards[agent_name] = bonus + 0.01 / (min_dist + 1e-6)
            else:
                rewards[agent_name] = env_rewards[agent_name] #0.0
        
        # if abs(min_dist_to_prey).sum() > 0:
        #     print('min dist to prey (adversaries):', min(min_dist_to_prey))

        return rewards


def make_env(timesteps_per_episode=300, num_predators=2, num_preys=1):
    base_env = simple_tag_v3.parallel_env(
        num_adversaries=num_predators,
        num_good=num_preys,
        num_obstacles=2,
        max_cycles=timesteps_per_episode,
        continuous_actions=True
    )
    env = AdversaryObsRewardWrapper(base_env)
    return env
