# Tutorial Examples

This directory contains runnable examples for each stage of the PyBullet Safe RL Tutorial.

## Quick Start

### Prerequisites

Ensure you have installed safe-control-gym:

```bash
cd /home/egmc/safe-control-gym
pip install -e .
```

### Running Examples

Navigate to the examples directory:

```bash
cd pybullet_safe_rl_tutorial/examples
```

## Stage 1: Basic Environment

**File**: `stage_1_basic_env.py`

Learn environment basics:
- Creating and configuring environments
- Running simulation loops
- Understanding control vs simulation frequencies
- Visualizing trajectories

**Run**:
```bash
python stage_1_basic_env.py
```

**Expected output**:
- PyBullet GUI showing cartpole
- Terminal output with environment info
- Trajectory plots saved to `stage_1_trajectory.png`
- Frequency comparison benchmarks

**Duration**: ~30 seconds

---

## Stage 2: Safety Constraints

**File**: `stage_2_constrained_env.py`

Add safety constraints:
- Defining state and input constraints
- Monitoring constraint violations
- Using constraint penalties
- Visualizing safe regions

**Run**:
```bash
python stage_2_constrained_env.py
```

**Expected output**:
- Constrained cartpole simulation
- Violation detection and counting
- Plots showing constraint violations
- Safe region visualization

**Duration**: ~30 seconds

**Key observations**:
- Random policy violates constraints frequently
- Constraint penalties encourage (but don't guarantee) safety

---

## Stage 3: Safe PPO Training

**File**: `stage_3_safe_ppo_training.py`

Train RL policy with safety:
- PPO training with constraint penalties
- Policy evaluation
- Optional CBF safety filtering
- Training progress visualization

### Training

Train a new safe PPO policy:

```bash
python stage_3_safe_ppo_training.py --train
```

**Duration**: ~10-30 minutes (500 epochs demo, use 5000+ for full training)

**Output**: Trained model saved to `./results_safe_ppo/model_latest.pt`

### Evaluation

Evaluate trained policy:

```bash
# Without safety filter
python stage_3_safe_ppo_training.py --eval

# With CBF safety filter
python stage_3_safe_ppo_training.py --eval --use_cbf
```

**Expected output**:
- 5 evaluation episodes with GUI
- Performance metrics (reward, violations)
- CBF intervention statistics (if --use_cbf)

### Plotting

Visualize training progress:

```bash
python stage_3_safe_ppo_training.py --plot
```

**Output**: `training_progress.png` with learning curves

---

## Advanced Usage

### Custom Configuration

Modify hyperparameters in the scripts:

```python
# In stage_3_safe_ppo_training.py

# Change constraint penalty
env_func = partial(create_safe_env, constraint_penalty=20.0)  # More conservative

# Change PPO hyperparameters
ppo = PPO(
    env_func=env_func,
    hidden_dim=512,  # Larger network
    actor_lr=1e-4,   # Slower learning
    num_epochs=10000,  # More training
)
```

### Using Different Environments

Replace `'cartpole'` with `'quadrotor'`:

```python
env = make(
    'quadrotor',  # 3D quadrotor
    constraints=quadrotor_constraints,
    # ... other config
)
```

### Adding More Constraints

```python
constraints = [
    # Position constraint
    {
        'constraint_type': 'BoundedConstraint',
        'constrained_variable': 'state',
        'active_dims': [0],
        'upper_bounds': [0.8],
        'lower_bounds': [-0.8],
    },
    # Velocity constraint
    {
        'constraint_type': 'BoundedConstraint',
        'constrained_variable': 'state',
        'active_dims': [1],  # x_dot
        'upper_bounds': [2.0],
        'lower_bounds': [-2.0],
    },
    # Add more as needed...
]
```

## Troubleshooting

### Import Errors

```
ModuleNotFoundError: No module named 'safe_control_gym'
```

**Solution**: Install safe-control-gym:
```bash
cd /home/egmc/safe-control-gym
pip install -e .
```

### PyBullet GUI Won't Open

**Solution**: Set `gui=False` for headless mode, or check X server on Linux.

### Training is Slow

**Solutions**:
- Reduce `num_epochs` for faster demo
- Increase `num_workers` for parallel environments
- Use `gui=False` (GUI is much slower)
- Reduce `rollout_batch_size`

### CBF Import Error

```
ImportError: cannot import name 'CBF'
```

**Solution**: CBF requires additional dependencies. Run evaluation without `--use_cbf` flag.

### Low Performance After Training

**Possible causes**:
- Not enough training epochs (use 5000+)
- Constraint penalty too high (reduces performance for safety)
- Bad hyperparameters

**Solutions**:
- Train longer
- Tune constraint_penalty (try 5.0, 10.0, 20.0)
- Adjust PPO learning rates

## Example Workflow

Complete workflow from scratch:

```bash
# 1. Understand basics
python stage_1_basic_env.py

# 2. Add constraints
python stage_2_constrained_env.py

# 3. Train safe policy (grab coffee, this takes a while)
python stage_3_safe_ppo_training.py --train --output_dir ./my_model

# 4. Evaluate learned policy
python stage_3_safe_ppo_training.py --eval --model_path ./my_model/model_latest.pt

# 5. Evaluate with CBF safety filter
python stage_3_safe_ppo_training.py --eval --use_cbf --model_path ./my_model/model_latest.pt

# 6. Visualize training
python stage_3_safe_ppo_training.py --plot --output_dir ./my_model
```

## Expected Results

### Stage 1
- Random policy fails quickly
- Cartpole falls over
- High variance in trajectories

### Stage 2
- Constraints clearly defined
- Violations detected and counted
- ~30-50% violation rate with random policy

### Stage 3 (After Training)
- Learned policy balances cartpole
- Significant violation reduction (~5-10% vs ~40% random)
- CBF further reduces violations to near 0%
- Performance trade-off: safety reduces task reward

## Next Steps

After completing these examples:

1. **Experiment**: Modify constraints, hyperparameters
2. **Compare**: Train without constraints, compare performance
3. **Advanced**: Try quadrotor, different RL algorithms (SAC, DDPG)
4. **Stage 4**: Explore advanced topics (domain randomization, RARL, GP-MPC)

## Additional Files

Create your own examples:

```python
# my_custom_example.py
from safe_control_gym.utils.registration import make

env = make('cartpole', gui=True, constraints=[...])
# Your code here...
```

## Getting Help

- Review stage documentation in `../stage_N_*/`
- Check [safe-control-gym docs](https://github.com/utiasDSL/safe-control-gym)
- Open issue if you find bugs

Happy learning!
