# Stage 1.1: Environment Setup

## Introduction

This guide introduces the foundational concepts of creating PyBullet-based RL environments using the safe-control-gym framework. You'll learn how environments are structured, how to configure them, and how to run basic simulations.

## Learning Objectives

By the end of this guide, you will:
- Understand the BenchmarkEnv base class architecture
- Know how to create and configure environments
- Understand control vs simulation frequencies
- Be able to run basic simulations
- Understand state and action spaces

## The BenchmarkEnv Base Class

All environments in safe-control-gym inherit from `BenchmarkEnv`, which provides:

1. **Gymnasium Interface**: Standard `reset()`, `step()`, `render()` methods
2. **PyBullet Integration**: Physics simulation management
3. **Safety Features**: Constraint and disturbance handling
4. **Configuration Management**: YAML-based setup
5. **Symbolic Models**: CasADi integration for control

### Key Components

```python
from safe_control_gym.envs.benchmark_env import BenchmarkEnv, Task, Cost

class MyEnv(BenchmarkEnv):
    NAME = 'my_env'  # Environment identifier
    URDF_PATH = 'path/to/robot.urdf'  # Robot definition

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Custom initialization

    def reset(self):
        # Reset environment to initial state
        pass

    def step(self, action):
        # Execute action, return (obs, reward, done, info)
        pass

    def _get_observation(self):
        # Extract current state from PyBullet
        pass

    def _compute_reward(self):
        # Calculate reward based on current state
        pass
```

## Environment Configuration

Environments are configured through dictionaries or YAML files:

### Basic Configuration

```python
config = {
    'task': 'cartpole',           # Environment type
    'gui': False,                  # Show PyBullet GUI
    'verbose': True,               # Print debug info

    # Task parameters
    'task': Task.STABILIZATION,    # or TRAJ_TRACKING
    'cost': Cost.RL_REWARD,        # Reward function type

    # Simulation parameters
    'pyb_freq': 1000,              # PyBullet physics frequency (Hz)
    'ctrl_freq': 50,               # Control frequency (Hz)
    'episode_len_sec': 5,          # Episode duration (seconds)

    # Initial state
    'init_state': None,            # Use default
    'randomized_init': True,       # Randomize initial state
}
```

### YAML Configuration

```yaml
# config/basic_cartpole.yaml
task: cartpole
gui: False
verbose: True

task_config:
  task: stabilization
  cost: rl_reward
  pyb_freq: 1000
  ctrl_freq: 50
  episode_len_sec: 5
  randomized_init: True
```

## Control vs Simulation Frequency

Understanding the distinction between control and simulation frequencies is crucial:

### Simulation Frequency (`pyb_freq`)
- How often PyBullet updates physics
- Higher = more accurate but slower
- Typical range: 100-1000 Hz
- Must be divisible by control frequency

### Control Frequency (`ctrl_freq`)
- How often you send commands to the robot
- Your RL policy runs at this rate
- Typical range: 10-100 Hz
- Lower than simulation frequency

### Example Timing

```python
pyb_freq = 1000  # PyBullet steps 1000 times/second
ctrl_freq = 50   # Controller runs 50 times/second

# This means:
pyb_steps_per_ctrl = pyb_freq / ctrl_freq  # = 20
# For each controller action, PyBullet simulates 20 physics steps
```

**Why separate frequencies?**
- **Stability**: High physics frequency prevents tunneling, improves stability
- **Realism**: Real robots have control delays
- **Efficiency**: Don't need to run policy 1000 times/second

## Creating Your First Environment

Let's create a simple environment from scratch:

```python
from safe_control_gym.utils.registration import make

# Create environment with default settings
env = make('cartpole')

# Or with custom configuration
env = make('cartpole',
           gui=True,
           verbose=True,
           pyb_freq=1000,
           ctrl_freq=50,
           episode_len_sec=10)

# Check environment properties
print(f"State dimension: {env.state_dim}")
print(f"Action dimension: {env.action_dim}")
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")
```

## Running a Simulation

Basic simulation loop:

```python
import numpy as np

# Reset environment
obs, info = env.reset()
done = False
episode_reward = 0

# Run episode
while not done:
    # Random action (for now)
    action = env.action_space.sample()

    # Step environment
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    episode_reward += reward

    # Optional: render
    if env.GUI:
        env.render()

print(f"Episode reward: {episode_reward}")
env.close()
```

## State and Action Spaces

### Observation Space
What the agent sees (may differ from full state):

```python
# Cartpole observation: [x, x_dot, theta, theta_dot]
obs_space = env.observation_space
print(obs_space)  # Box(4,)
print(obs_space.low)   # Lower bounds
print(obs_space.high)  # Upper bounds
```

### Action Space
What the agent can control:

```python
# Cartpole action: force applied to cart
action_space = env.action_space
print(action_space)  # Box(1,)

# Normalized action space for RL
env_normalized = make('cartpole',
                      normalized_rl_action_space=True)
# Now actions are in [-1, 1], scaled internally
```

### State vs Observation

**State**: Complete description of system (may not be fully observable)
**Observation**: What agent actually sees

```python
# Full state (internal)
state = env.state  # [x, x_dot, theta, theta_dot, ...]

# Observation (what agent receives)
obs = env._get_observation()  # May be noisy, partial

# In safe-control-gym, typically obs == state for simplicity
```

## Environment Tasks

### Stabilization

Goal: Reach and maintain a target state

```python
env = make('cartpole',
           task=Task.STABILIZATION,
           task_info={'stabilization_goal': [0, 0, 0, 0]})

# Reward is based on distance to goal
# env.X_GOAL contains the target state
```

### Trajectory Tracking

Goal: Follow a reference trajectory

```python
env = make('cartpole',
           task=Task.TRAJ_TRACKING,
           task_info={
               'trajectory_type': 'circle',
               'trajectory_radius': 0.5,
               'trajectory_period': 10.0
           })

# env.X_GOAL is a time-indexed trajectory
# env.U_GOAL is the reference control sequence
```

## Episode Management

Understanding episode lifecycle:

```python
# Create environment
env = make('cartpole', episode_len_sec=5, ctrl_freq=50)

# Episode length in steps
max_steps = env.CTRL_STEPS  # = 5 * 50 = 250 steps

# Reset starts new episode
obs, info = env.reset()

# Step counter
for step in range(max_steps):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    # terminated: task-specific termination (e.g., fell over)
    # truncated: time limit reached
    if terminated or truncated:
        break

print(f"Episode finished at step {step}")
```

## Practical Exercise 1: Environment Exploration

Create a script that explores a cartpole environment:

```python
from safe_control_gym.utils.registration import make
import numpy as np

def explore_environment():
    # Create environment
    env = make('cartpole', gui=True, verbose=True)

    # Print environment info
    print("=== Environment Information ===")
    print(f"State dim: {env.state_dim}")
    print(f"Action dim: {env.action_dim}")
    print(f"Control freq: {env.CTRL_FREQ} Hz")
    print(f"PyBullet freq: {env.PYB_FREQ} Hz")
    print(f"Episode length: {env.EPISODE_LEN_SEC} sec")
    print(f"Max steps: {env.CTRL_STEPS}")
    print(f"\nObservation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Run random episode
    obs, info = env.reset()
    print(f"\nInitial state: {obs}")

    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if step % 10 == 0:
            print(f"Step {step}: state={obs}, reward={reward:.3f}")

        if terminated or truncated:
            print(f"Episode ended at step {step}")
            break

    env.close()

if __name__ == "__main__":
    explore_environment()
```

## Practical Exercise 2: Frequency Comparison

Compare different simulation frequencies:

```python
import time
import numpy as np

def benchmark_frequencies():
    configs = [
        {'pyb_freq': 100, 'ctrl_freq': 50},
        {'pyb_freq': 500, 'ctrl_freq': 50},
        {'pyb_freq': 1000, 'ctrl_freq': 50},
    ]

    for config in configs:
        env = make('cartpole', gui=False, **config)

        # Time 100 steps
        start_time = time.time()
        obs, _ = env.reset()

        for _ in range(100):
            action = env.action_space.sample()
            env.step(action)

        elapsed = time.time() - start_time

        print(f"PyBullet freq: {config['pyb_freq']} Hz")
        print(f"  Time for 100 steps: {elapsed:.3f} sec")
        print(f"  Speed-up: {100 * env.CTRL_TIMESTEP / elapsed:.1f}x")
        print()

        env.close()

if __name__ == "__main__":
    benchmark_frequencies()
```

## Key Takeaways

1. **BenchmarkEnv** is the base class for all environments
2. **Configuration** can be done via Python dicts or YAML files
3. **Frequencies** matter: use high `pyb_freq` for accuracy, moderate `ctrl_freq` for realism
4. **State != Observation** (though often equal in safe-control-gym)
5. **Tasks** can be stabilization or trajectory tracking
6. **Episodes** have time limits and task-specific termination conditions

## Common Pitfalls

1. **Frequency mismatch**: Ensure `pyb_freq % ctrl_freq == 0`
2. **GUI in training**: Always use `gui=False` for training (much faster)
3. **Action scaling**: Use `normalized_rl_action_space=True` for RL
4. **Episode length**: Too short = insufficient learning; too long = slow training

## Next Steps

Now that you understand basic environment setup, proceed to:
- [02_pybullet_basics.md](02_pybullet_basics.md) - Deep dive into PyBullet
- [03_custom_environment.md](03_custom_environment.md) - Build your own environment

## Additional Resources

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [PyBullet Quickstart](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA)
- [safe_control_gym.envs.benchmark_env source](../../safe_control_gym/envs/benchmark_env.py)
