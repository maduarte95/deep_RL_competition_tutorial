# Visualization Guide

This guide explains how to visualize your trained agents in the Tag environment using the `visualize_rollout.py` script.

## Overview

The visualization script allows you to watch your trained agent play in the Tag environment in real-time using a pygame window. You can visualize either prey or predator agents, and all other agents will use default policies.

## Basic Usage

```bash
python visualize_rollout.py --model-path <path_to_model.pt> --role <prey|predator>
```

### Required Arguments

- `--model-path`: Path to your saved model weights file (`.pt` file)

### Optional Arguments

- `--role`: Which agent your model controls (default: `prey`)
  - `prey`: Your model controls a prey agent (agent_0)
  - `predator`: Your model controls a predator agent (adversary_0)

- `--policy-module`: Python module containing your policy network class (default: `winners_prey_policy`)

- `--num-prey`: Number of prey agents in the environment (default: 1)

- `--num-predators`: Number of predator agents in the environment (default: 2)

- `--timesteps`: Total timesteps to run the visualization (default: 500)

- `--episode-length`: Maximum timesteps per episode before reset (default: 300)

## Examples

### Visualize a Prey Agent

```bash
python visualize_rollout.py --model-path winners_prey_prey.pt --role prey
```

## Visualize both
```bash
python visualize_rollout.py --model-path my_predator.pt --role predator
```


This will:
- Load your model from `winners_prey_prey.pt`
- Control `agent_0` (the prey)
- Show 2 predators using default policies
- Run for 500 timesteps

### Visualize a Predator Agent

```bash
python visualize_rollout.py --model-path my_predator.pt --role predator
```

This will:
- Load your model from `my_predator.pt`
- Control `adversary_0` (a predator)
- Show prey agents using default policies

### Custom Environment Configuration

```bash
python visualize_rollout.py \
  --model-path my_model.pt \
  --role prey \
  --num-prey 2 \
  --num-predators 3 \
  --timesteps 1000 \
  --episode-length 500
```

This creates a larger environment with 2 prey and 3 predators, running for 1000 timesteps.

### Using a Custom Policy Module

```bash
python visualize_rollout.py \
  --model-path actor_critic_model.pt \
  --role prey \
  --policy-module actor_critic
```

This loads the policy network class from `actor_critic.py` instead of the default module.

## Understanding the Visualization

### Display Elements

- **Green flies**: Prey agents
- **Red flies**: Predator agents
- **Gray circles**: Obstacles/landmarks
- **"YOUR AGENT" label**: The agent controlled by your model
- **"CPU" label**: Agents using default policies
- **Scoreboard**: Current cumulative rewards for each agent
- **Timestep counter**: Current timestep in the simulation

### Controls

- Close the pygame window to stop the visualization
- The simulation runs at 20 FPS

## Model Requirements

Your model file (`.pt` file) should contain a PyTorch state dictionary that can be loaded into:

1. A `PolicyNet` class (for standard policies), or
2. An `ActorCriticNet` class (for actor-critic models)

The script will automatically detect actor-critic models if:
- Your policy module has an `ActorCriticNet` class
- Your model path contains the string `'actor_critic'`

## Output

After the visualization completes, you'll see final scores printed to the console:

```
============================================================
FINAL SCORES:
============================================================
YOUR AGENT      (agent_0):   125.45 █████████████████████████
CPU             (adversary_0):   -45.20 █████████
CPU             (adversary_1):   -52.15 ██████████
============================================================
```

## Troubleshooting

### Error: "Error loading policy"

Make sure:
- Your model path is correct
- Your policy module is in the same directory or on the Python path
- Your model's state dict matches the network architecture in the policy module

### Pygame window doesn't open

Make sure pygame is installed:
```bash
pip install pygame
```

### Model runs but behavior looks wrong

Check that:
- You're using the correct `--role` (prey vs predator)
- The `--policy-module` matches the module used during training
- The observation and action dimensions match your training setup
