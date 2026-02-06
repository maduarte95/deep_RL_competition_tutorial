import torch
import torch.nn as nn

class PolicyNet(nn.Module):

    # Do NOT change this signature, otherwise the tournament loader will not work
    def __init__(self, obs_dim, act_dim): # DO NOT CHANGE THIS
        super().__init__()

        hidden_dim = 256

        # Input projection
        self.input_layer = nn.Linear(obs_dim, hidden_dim)

        # Residual blocks for deep learning capacity
        self.res_block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.res_block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.res_block3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Output head
        self.output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim),
            nn.Sigmoid()
        )

    # Do NOT change this signature, otherwise the tournament loader will not work
    def forward(self, x):
        x = self.input_layer(x)

        # Residual connections allow deep learning without gradient issues
        x = x + self.res_block1(x)
        x = x + self.res_block2(x)
        x = x + self.res_block3(x)

        return self.output_layer(x)

    # this is reinforce: policy gradient without critic


# ============================================================================
# Model-Based Actor-Critic Implementation
# ============================================================================

class ActorCriticNet(nn.Module):
    """
    Actor-Critic network with shared feature extractor.
    - Actor: outputs action probabilities
    - Critic: outputs state value V(s)
    """
    def __init__(self, obs_dim, act_dim):
        super().__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        # Actor head (policy) - outputs action probabilities
        self.actor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim),
            nn.Sigmoid()
        )

        # Critic head (value function) - outputs V(s)
        self.critic = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        """
        Returns both action probabilities and state value.
        For tournament compatibility, we return only actions by default.
        """
        features = self.shared(x)
        action_probs = self.actor(features)
        value = self.critic(features)
        return action_probs, value

    def get_action(self, x):
        """Get only action (for tournament/inference)"""
        features = self.shared(x)
        return self.actor(features)


class WorldModel(nn.Module):
    """
    World model that predicts:
    - Next observation given current observation and action
    - Reward given current observation and action

    This enables model-based learning by simulating trajectories.
    """
    def __init__(self, obs_dim, act_dim):
        super().__init__()

        # Encoder: process (obs, action) pair
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        # Next state prediction head
        self.next_obs_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, obs_dim)
        )

        # Reward prediction head
        self.reward_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, obs, action):
        """
        Predict next observation and reward.

        Args:
            obs: current observation [batch, obs_dim]
            action: action taken [batch, act_dim]

        Returns:
            next_obs_pred: predicted next observation [batch, obs_dim]
            reward_pred: predicted reward [batch, 1]
        """
        # Concatenate observation and action
        x = torch.cat([obs, action], dim=-1)

        # Encode
        features = self.encoder(x)

        # Predict next state and reward
        next_obs_pred = self.next_obs_head(features)
        reward_pred = self.reward_head(features)

        return next_obs_pred, reward_pred

    def rollout(self, initial_obs, policy, steps=10):
        """
        Generate imagined trajectory using the world model.

        Args:
            initial_obs: starting observation [obs_dim]
            policy: policy network to sample actions
            steps: number of steps to simulate

        Returns:
            observations: list of predicted observations
            actions: list of sampled actions
            rewards: list of predicted rewards
        """
        observations = [initial_obs]
        actions = []
        rewards = []

        obs = initial_obs
        for _ in range(steps):
            # Sample action from policy
            action_probs, _ = policy(obs)
            dist = torch.distributions.Bernoulli(action_probs)
            action = dist.sample()

            # Predict next state and reward
            next_obs, reward = self.forward(obs.unsqueeze(0), action.unsqueeze(0))
            next_obs = next_obs.squeeze(0)
            reward = reward.squeeze(0)

            observations.append(next_obs)
            actions.append(action)
            rewards.append(reward)

            obs = next_obs

        return observations, actions, rewards
    