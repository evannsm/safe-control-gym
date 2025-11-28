# PyBullet Safe Reinforcement Learning Tutorial

A comprehensive, staged guide to building your own PyBullet-based RL methods with integrated safety constraints using the safe-control-gym framework.

## Overview

This tutorial walks you through creating safe reinforcement learning systems from the ground up, based on the safe-control-gym codebase. You'll learn how to:

- Set up PyBullet physics environments
- Implement safety constraints (state, input, and combined)
- Integrate RL algorithms (PPO, SAC) with safety mechanisms
- Use safety filters like Control Barrier Functions (CBF) and Model Predictive Safety Certification (MPSC)
- Train and evaluate safe RL policies

## Tutorial Structure

Each stage builds upon the previous one, with complete working examples:

### [Stage 1: Basic Environment Setup](stage_1_basics/)
**Duration**: ~30-45 minutes

Learn the fundamentals of creating PyBullet environments:
- PyBullet physics simulation basics
- Creating custom Gym environments
- State/action space definitions
- Rendering and visualization
- Basic dynamics modeling

**Topics covered**:
- BenchmarkEnv base class architecture
- PyBullet integration (URDF, physics parameters)
- Control and simulation frequencies
- Episode management
- Observation and action spaces

---

### [Stage 2: Safety Constraints](stage_2_constraints/)
**Duration**: ~45-60 minutes

Add safety to your environments:
- Understanding constraint formulations (g(x) <= 0)
- Implementing state constraints (position/velocity limits)
- Implementing input constraints (actuation limits)
- Combined state-input constraints
- Constraint violation detection and handling

**Topics covered**:
- Constraint class architecture
- Symbolic constraint modeling with CasADi
- Active dimensions and filtering
- Strict vs soft constraints
- Constraint evaluation and monitoring

---

### [Stage 3: RL Integration with Safety](stage_3_rl_safety/)
**Duration**: ~60-90 minutes

Combine RL with safety mechanisms:
- Implementing PPO/SAC controllers
- Safety filters (CBF, MPSC)
- Reward shaping with constraint penalties
- Safe exploration strategies
- Training with safety guarantees

**Topics covered**:
- PPO/SAC architecture for safe RL
- Control Barrier Functions (CBF)
- CBF-QP safety filtering
- Integration of learned policies with safety filters
- Observation/reward normalization
- Vectorized environments for parallel training

---

### [Stage 4: Advanced Topics](stage_4_advanced/)
**Duration**: ~60-90 minutes

Advanced techniques and best practices:
- Hyperparameter optimization
- Domain randomization
- Disturbance injection
- Robust policy training (RARL, RAP)
- Model-based safety (GP-MPC)
- Evaluation and metrics

**Topics covered**:
- Hyperparameter tuning with Optuna
- Inertial property randomization
- Adversarial disturbances
- Gaussian Process MPC
- Safety metrics and benchmarking
- Transferring policies to real systems

---

## Quick Start

### Prerequisites

```bash
# Python 3.10+ recommended
conda create -n safe_rl python=3.10
conda activate safe_rl

# Install safe-control-gym
cd /home/egmc/safe-control-gym
pip install -e .

# Optional: Install additional dependencies
conda install -c anaconda gmp  # For pycddlib
```

### Running Your First Example

```bash
# Navigate to tutorial directory
cd pybullet_safe_rl_tutorial/examples

# Run a basic environment example
python stage_1_basic_env.py

# Run a constrained environment example
python stage_2_constrained_env.py

# Run a safe RL training example
python stage_3_safe_ppo_training.py
```

## Learning Path

### Beginner Path (Start here if new to RL or PyBullet)
1. Start with Stage 1, complete all examples
2. Read Stage 2 documentation thoroughly
3. Complete Stage 2 examples with modifications
4. Move to Stage 3 when comfortable with constraints
5. Stage 4 is optional but recommended

### Intermediate Path (Familiar with RL, new to safety)
1. Skim Stage 1, run examples to understand architecture
2. Focus on Stage 2 - this is the core of safety
3. Deep dive into Stage 3 safety filters
4. Experiment with Stage 4 advanced techniques

### Advanced Path (Experienced with safe RL)
1. Review code architecture in Stage 1
2. Study constraint formulations in Stage 2
3. Implement custom safety filters in Stage 3
4. Extend Stage 4 with your own methods

## Key Concepts

### Safety in RL

Traditional RL maximizes cumulative reward, but safety-critical applications require additional guarantees:

- **Hard Constraints**: Never violate (e.g., collision avoidance)
- **Soft Constraints**: Minimize violations (e.g., energy consumption)
- **Safety Filters**: Post-process learned actions to ensure safety
- **Safe Exploration**: Maintain safety during training

### Framework Architecture

```
Environment (PyBullet)
    |
    v
Constraints Module -----> Constraint Violations
    |
    v
RL Controller (PPO/SAC)
    |
    v
Safety Filter (CBF/MPSC) -----> Safe Actions
    |
    v
Environment.step()
```

### Design Philosophy

1. **Modularity**: Separate components (env, constraints, controller, safety filter)
2. **Configurability**: YAML-based configuration for reproducibility
3. **Symbolic + Numeric**: CasADi for symbolic math, PyTorch for learning
4. **Safety First**: Multiple layers of safety (constraints, filters, penalties)

## File Structure

```
pybullet_safe_rl_tutorial/
├── README.md                           # This file
├── stage_1_basics/
│   ├── 01_environment_setup.md         # Detailed guide
│   ├── 02_pybullet_basics.md           # PyBullet fundamentals
│   ├── 03_custom_environment.md        # Build your own env
│   └── 04_dynamics_modeling.md         # Symbolic dynamics
├── stage_2_constraints/
│   ├── 01_constraint_theory.md         # Mathematical foundations
│   ├── 02_state_constraints.md         # Position/velocity limits
│   ├── 03_input_constraints.md         # Actuation limits
│   └── 04_constraint_evaluation.md     # Runtime checking
├── stage_3_rl_safety/
│   ├── 01_ppo_basics.md                # PPO implementation
│   ├── 02_safety_filters.md            # CBF and MPSC
│   ├── 03_safe_training.md             # Training with safety
│   └── 04_evaluation.md                # Testing and metrics
├── stage_4_advanced/
│   ├── 01_hyperparameter_tuning.md     # HPO guide
│   ├── 02_domain_randomization.md      # Robustness
│   ├── 03_disturbances.md              # Adversarial training
│   └── 04_real_world_transfer.md       # Sim-to-real
└── examples/
    ├── stage_1_basic_env.py
    ├── stage_2_constrained_env.py
    ├── stage_3_safe_ppo_training.py
    ├── stage_3_cbf_filtering.py
    ├── stage_4_domain_randomization.py
    └── configs/                         # YAML configurations
        ├── basic_cartpole.yaml
        ├── constrained_cartpole.yaml
        └── safe_ppo_cartpole.yaml
```

## Common Patterns

### Configuration-Driven Development

All experiments use YAML configuration files:

```yaml
# Example configuration
task: cartpole
task_config:
  task: stabilization
  cost: rl_reward
  constraints:
    - constraint: BoundedConstraint
      constrained_variable: state
      active_dims: [0, 2]  # x position, theta
      upper_bounds: [1.0, 0.3]
      lower_bounds: [-1.0, -0.3]
```

### Modular Controller Design

```python
# Standard pattern
env = make('cartpole', **config.task_config)
ctrl = make('ppo', env, **config.algo_config)
ctrl.reset()

# Training loop
for episode in range(n_episodes):
    obs, info = env.reset()
    while not done:
        action = ctrl.select_action(obs)
        obs, reward, done, info = env.step(action)
```

## Troubleshooting

### Common Issues

**Q: PyBullet GUI won't open**
- Ensure X server is running (Linux)
- Try `gui=False` for headless mode

**Q: CasADi import errors**
- Install: `pip install casadi`
- For Apple Silicon: may need to build from source

**Q: Constraint violations during training**
- This is expected! Safety filters help
- Increase constraint penalties
- Reduce exploration noise

**Q: Training is slow**
- Use vectorized environments (`num_workers > 1`)
- Reduce `pyb_freq` if high-fidelity physics not needed
- Use GPU: `use_gpu=True`

## Additional Resources

### Papers
- [Safe Learning in Robotics: From Learning-Based Control to Safe RL](https://arxiv.org/abs/2108.06266)
- [Safe-Control-Gym Benchmark Suite](https://arxiv.org/abs/2109.06325)
- [Control Barrier Functions: Theory and Applications](https://arxiv.org/abs/1903.11199)

### Repositories
- [safe-control-gym](https://github.com/utiasDSL/safe-control-gym)
- [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones)
- [stable-baselines3](https://github.com/DLR-RM/stable-baselines3)

### Documentation
- [PyBullet Quickstart](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA)
- [CasADi Documentation](https://web.casadi.org/docs/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)

## Contributing

Found an error or want to improve this tutorial?
- Issues: [GitHub Issues](https://github.com/utiasDSL/safe-control-gym/issues)
- Pull requests welcome!

## Next Steps

Ready to start? Head to [Stage 1: Basic Environment Setup](stage_1_basics/) to begin your journey into safe reinforcement learning!
