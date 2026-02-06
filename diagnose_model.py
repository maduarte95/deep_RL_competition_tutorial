#!/usr/bin/env python3
"""
Diagnostic script to check what the actor-critic model is outputting
"""

import torch
import numpy as np
from pettingzoo_wrapper import make_env
from winners_prey_policy import ActorCriticNet

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create environment to get dimensions
env = make_env(300, num_predators=2, num_preys=1)
agent_id = "agent_0"  # prey
obs_dim = env.observation_space(agent_id)
act_dim = env.action_space(agent_id)

print(f"Observation dim: {obs_dim}, Action dim: {act_dim}")

# Load the actor-critic model
model_path = "winners_prey_prey_actor_critic.pt"
policy = ActorCriticNet(obs_dim, act_dim).to(device)
policy.load_state_dict(torch.load(model_path, map_location=device))
policy.eval()

print(f"\nLoaded model from: {model_path}")

# Get a few observations and see what actions the model outputs
obs_dict = env.reset()
print("\n" + "="*60)
print("Testing model outputs on random observations:")
print("="*60)

for i in range(5):
    obs_dict = env.reset()
    obs = torch.tensor(obs_dict[agent_id], dtype=torch.float32, device=device)

    with torch.no_grad():
        action_probs, value = policy(obs)

    print(f"\nTest {i+1}:")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Action probabilities: {action_probs.cpu().numpy()}")
    print(f"  Min: {action_probs.min().item():.4f}, Max: {action_probs.max().item():.4f}, Mean: {action_probs.mean().item():.4f}")
    print(f"  Value estimate: {value.item():.4f}")

    # Check if actions are all similar (would cause no movement)
    action_std = action_probs.std().item()
    if action_std < 0.01:
        print(f"  ⚠️  WARNING: Very low action variance (std={action_std:.6f}) - agent may not move much!")

print("\n" + "="*60)
print("Diagnosis complete!")
print("="*60)
