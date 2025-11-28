# Quick Start Guide: PyBullet Safe RL Tutorial

A fast-track guide to get you up and running with safe reinforcement learning in PyBullet.

## Installation (5 minutes)

```bash
# Navigate to safe-control-gym directory
cd /home/egmc/safe-control-gym

# Create conda environment (optional but recommended)
conda create -n safe_rl python=3.10
conda activate safe_rl

# Install safe-control-gym
pip install -e .

# Install optional dependencies
conda install -c anaconda gmp  # For constraint handling
```

## 30-Minute Crash Course

### 1. Basic Environment (10 minutes)

Create and run a basic PyBullet environment:

```python
from safe_control_gym.utils.registration import make

# Create cartpole environment
env = make('cartpole', gui=True, ctrl_freq=50)

# Run episode
obs, _ = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)

env.close()
```

Or run the example:
```bash
cd pybullet_safe_rl_tutorial/examples
python stage_1_basic_env.py
```

### 2. Add Safety Constraints (10 minutes)

Add constraints to your environment:

```python
from safe_control_gym.utils.registration import make

# Define constraints
constraints = [{
    'constraint_type': 'BoundedConstraint',
    'constrained_variable': 'state',
    'active_dims': [0, 2],  # x position, theta angle
    'upper_bounds': [0.8, 0.3],
    'lower_bounds': [-0.8, -0.3]
}]

# Create constrained environment
env = make(
    'cartpole',
    gui=True,
    constraints=constraints,
    use_constraint_penalty=True,
    constraint_penalty=10.0
)

# Check for violations
obs, _ = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)

    if env.constraints.is_violated(env):
        print("Constraint violated!")

env.close()
```

Or run:
```bash
python stage_2_constrained_env.py
```

### 3. Train Safe RL Policy (10 minutes to start, 30+ minutes to train)

Train a PPO policy that respects constraints:

```python
from functools import partial
from safe_control_gym.utils.registration import make
from safe_control_gym.controllers.ppo import PPO

# Create environment factory
def create_env():
    return make(
        'cartpole',
        constraints=constraints,
        use_constraint_penalty=True,
        constraint_penalty=10.0
    )

# Create and train PPO
ppo = PPO(
    env_func=create_env,
    training=True,
    num_epochs=500,  # Increase to 5000+ for full training
    output_dir='./my_safe_policy'
)

ppo.reset()
ppo.learn()  # This takes time!
ppo.close()
```

Or run:
```bash
python stage_3_safe_ppo_training.py --train
```

Then evaluate:
```bash
python stage_3_safe_ppo_training.py --eval
```

## Core Concepts (5 minutes read)

### 1. Environment Structure

```
BenchmarkEnv (base class)
    ├── PyBullet physics simulation
    ├── Constraints (safety boundaries)
    ├── Symbolic model (for control)
    └── Task (stabilization/tracking)
```

### 2. Safety Constraints

Constraints define safe regions:

```
g(x, u) ≤ 0  → Safe
g(x, u) > 0  → Unsafe (violation)
```

Types:
- **State constraints**: g(x) ≤ 0 (position, velocity limits)
- **Input constraints**: g(u) ≤ 0 (actuation limits)
- **Combined**: g(x, u) ≤ 0 (state-dependent input limits)

### 3. Safe RL Approaches

| Approach | When to Use | Safety Guarantee |
|----------|-------------|------------------|
| **Reward shaping** | Fast training, soft safety | Soft (no guarantee) |
| **Safety filters (CBF)** | Critical safety, known dynamics | Strong (mathematical) |
| **Constrained RL** | Training-time safety | Medium |
| **Safe exploration** | Physical systems | Strong (during training) |

### 4. Workflow

```
1. Create environment with constraints
2. Train RL policy (PPO, SAC) with constraint penalties
3. (Optional) Add safety filter (CBF) for deployment
4. Evaluate: measure performance + safety
```

## Common Patterns

### Pattern 1: Simple Safe Environment

```python
from safe_control_gym.utils.registration import make

env = make(
    'cartpole',
    constraints=[{
        'constraint_type': 'BoundedConstraint',
        'constrained_variable': 'state',
        'active_dims': [0],
        'upper_bounds': [1.0],
        'lower_bounds': [-1.0]
    }],
    use_constraint_penalty=True,
    constraint_penalty=10.0
)
```

### Pattern 2: Train Safe PPO

```python
from safe_control_gym.controllers.ppo import PPO

ppo = PPO(
    env_func=lambda: env,
    training=True,
    num_epochs=5000,
    hidden_dim=256,
    actor_lr=3e-4,
    output_dir='./results'
)

ppo.reset()
ppo.learn()
```

### Pattern 3: Evaluate with Safety Filter

```python
from safe_control_gym.safety_filters.cbf import CBF

# Load trained policy
ppo = PPO(env_func=lambda: env, training=False)
ppo.load('model.pt')

# Create safety filter
cbf = CBF(env_func=lambda: env, slope=0.1)

# Run with safety
obs, _ = env.reset()
for _ in range(500):
    action_desired = ppo.select_action(obs)
    action_safe = cbf.compute_action(obs, action_desired)
    obs, _, _, _, _ = env.step(action_safe)
```

## Key Files Reference

### Documentation
- `README.md` - Main tutorial overview
- `stage_1_basics/` - Environment basics, PyBullet, dynamics
- `stage_2_constraints/` - Constraint theory and implementation
- `stage_3_rl_safety/` - Safe RL methods, CBF, training
- `stage_4_advanced/` - Advanced topics (not yet covered)

### Examples
- `examples/stage_1_basic_env.py` - Basic environment demo
- `examples/stage_2_constrained_env.py` - Constraints demo
- `examples/stage_3_safe_ppo_training.py` - Full training pipeline

### Source Code
- `safe_control_gym/envs/benchmark_env.py` - Base environment
- `safe_control_gym/envs/constraints.py` - Constraint classes
- `safe_control_gym/controllers/ppo/ppo.py` - PPO implementation
- `safe_control_gym/safety_filters/cbf/cbf.py` - CBF safety filter

## Troubleshooting

### "ModuleNotFoundError: No module named 'safe_control_gym'"
```bash
cd /home/egmc/safe-control-gym
pip install -e .
```

### PyBullet GUI won't open
```python
# Use headless mode
env = make('cartpole', gui=False)
```

### Training is too slow
```python
# Reduce epochs for testing
ppo = PPO(env_func=..., num_epochs=100)  # Instead of 5000

# Or use more workers
ppo = PPO(env_func=..., num_workers=8)  # Parallel environments
```

### Constraint violations during training
This is expected! Solutions:
- Increase `constraint_penalty` (10.0 → 20.0)
- Use safety filter for deployment (CBF)
- Train longer (more epochs)

## Next Steps

1. **Run all examples** in order (Stages 1-3)
2. **Experiment**: Modify constraints, tune hyperparameters
3. **Compare**: Train with/without constraints
4. **Advanced**: Try quadrotor, SAC algorithm, domain randomization
5. **Read docs**: Deep dive into specific topics in stage folders

## Cheat Sheet

```python
# Create environment
from safe_control_gym.utils.registration import make
env = make('cartpole', gui=True, constraints=[...])

# Add constraints
constraints = [{
    'constraint_type': 'BoundedConstraint',
    'constrained_variable': 'state',  # or 'input'
    'active_dims': [0, 2],
    'upper_bounds': [1.0, 0.3],
    'lower_bounds': [-1.0, -0.3]
}]

# Train PPO
from safe_control_gym.controllers.ppo import PPO
ppo = PPO(env_func=lambda: env, training=True, num_epochs=5000)
ppo.reset()
ppo.learn()

# Evaluate
ppo.load('model.pt')
obs, _ = env.reset()
action = ppo.select_action(obs)

# Safety filter
from safe_control_gym.safety_filters.cbf import CBF
cbf = CBF(env_func=lambda: env)
action_safe = cbf.compute_action(obs, action_desired)

# Check violations
violated = env.constraints.is_violated(env)
g_values = env.constraints.get_value(env)
```

## Resources

- **Tutorial**: `pybullet_safe_rl_tutorial/README.md`
- **Examples**: `pybullet_safe_rl_tutorial/examples/`
- **GitHub**: https://github.com/utiasDSL/safe-control-gym
- **Paper**: https://arxiv.org/abs/2109.06325
- **Review**: https://arxiv.org/abs/2108.06266

## Getting Help

1. Check documentation in `stage_N_*/` folders
2. Review example code in `examples/`
3. Search GitHub issues
4. Open new issue with minimal reproducible example

Happy safe learning!
