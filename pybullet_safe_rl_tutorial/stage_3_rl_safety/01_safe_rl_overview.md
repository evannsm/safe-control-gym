# Stage 3.1: Safe Reinforcement Learning Overview

## Introduction

This guide introduces the integration of reinforcement learning with safety constraints. You'll learn different approaches to safe RL and how to implement them in safe-control-gym.

## Learning Objectives

- Understand different safe RL paradigms
- Learn when to use each approach
- Implement safe PPO training
- Integrate safety filters with learned policies

## The Safe RL Problem

Traditional RL maximizes expected cumulative reward:

```
max E[Σ r_t]
```

Safe RL adds safety constraints:

```
max E[Σ r_t]
subject to: g(x_t, u_t) ≤ 0 ∀t (or with high probability)
```

### Challenges

1. **Exploration vs Safety**: Learning requires exploration, but exploration can be unsafe
2. **Unknown Dynamics**: Agent doesn't know what's safe initially
3. **Constraint Violations**: How many violations are acceptable during training?
4. **Policy Performance**: Safety shouldn't completely sacrifice performance

## Safe RL Approaches

### 1. Reward Shaping (Soft Constraints)

Add constraint penalties to reward:

```python
r_safe(x, u) = r_task(x, u) - λ * penalty(g(x, u))

# Example penalty functions
penalty_quadratic = max(0, g(x, u))^2
penalty_exponential = exp(λ * max(0, g(x, u))) - 1
penalty_barrier = -log(ε - g(x, u))  # Barrier method
```

**Pros**:
- Easy to implement
- Works with any RL algorithm
- Smooth learning signal

**Cons**:
- No hard guarantees
- Tuning λ is critical
- May still violate during training

**When to use**: Soft safety preferences, not safety-critical

### 2. Safety Filters (Post-Processing)

Learn unconstrained policy, then filter actions through safety layer:

```python
# Learn policy π(x)
action_desired = π(x)

# Safety filter ensures safety
action_safe = safety_filter(action_desired, x)

# Execute safe action
env.step(action_safe)
```

**Safety filters**:
- **CBF (Control Barrier Functions)**: QP to minimize distance from desired action
- **MPSC (Model Predictive Safety Certification)**: MPC with safety constraints
- **Simplex Architecture**: Switch to safe controller when needed

**Pros**:
- Strong safety guarantees
- Separates learning from safety
- Can provide formal certificates

**Cons**:
- Requires accurate model
- Computational overhead
- May conflict with learning

**When to use**: Safety-critical applications, known dynamics

### 3. Constrained RL Algorithms

Modify RL algorithm to enforce constraints during training:

**Algorithms**:
- **CPO (Constrained Policy Optimization)**: Trust region with constraints
- **PPO-Lagrangian**: Dual optimization with Lagrange multipliers
- **TRPO-CPO**: Constraint-aware trust region
- **Safe Actor-Critic**: Augment value function with safety

**Pros**:
- Principled constraint handling
- Theoretical guarantees
- Balances performance and safety

**Cons**:
- More complex than standard RL
- Slower convergence
- Requires careful tuning

**When to use**: Training-time safety important, moderate safety requirements

### 4. Safe Exploration

Ensure safety during exploration:

**Methods**:
- **Conservative initialization**: Start from safe policy
- **Safe action spaces**: Constrain exploration
- **Reachability analysis**: Only explore provably safe states
- **Shield functions**: Safety wrapper around policy

**Pros**:
- Prevents unsafe exploration
- Good for physical systems
- Can learn faster with safety

**Cons**:
- May limit exploration
- Requires prior knowledge
- Can be conservative

**When to use**: Physical robots, expensive failures

## Safe RL in safe-control-gym

The framework supports multiple safe RL approaches:

### Architecture

```
┌─────────────────┐
│   Environment   │
│  (PyBullet +    │
│  Constraints)   │
└────────┬────────┘
         │
         │ state, constraint info
         ▼
┌─────────────────┐
│  RL Controller  │
│  (PPO, SAC)     │
│  + Safety       │
└────────┬────────┘
         │
         │ action_desired
         ▼
┌─────────────────┐
│ Safety Filter   │ (optional)
│  (CBF, MPSC)    │
└────────┬────────┘
         │
         │ action_safe
         ▼
┌─────────────────┐
│ Environment     │
│    .step()      │
└─────────────────┘
```

### Available Methods

1. **PPO with Constraint Penalties**
   ```python
   env = make('cartpole',
              constraints=constraints,
              use_constraint_penalty=True,
              constraint_penalty=10.0)

   ctrl = PPO(env, ...)
   ```

2. **PPO with Safety Filters**
   ```python
   env = make('cartpole', constraints=constraints)
   ctrl = PPO(env, ...)
   safety_filter = CBF(env, ...)

   # During execution
   action_desired = ctrl.select_action(obs)
   action_safe = safety_filter.compute_action(obs, action_desired)
   ```

3. **Safe PPO (Safety Layer)**
   ```python
   from safe_control_gym.controllers.safe_explorer import SafePPO

   ctrl = SafePPO(
       env,
       safety_budget=25,  # Allowed violations per episode
       use_safety_layer=True
   )
   ```

4. **Robust RL (Domain Randomization)**
   ```python
   env = make('cartpole',
              randomized_inertial_prop=True,
              disturbances={'white_noise': {...}})

   ctrl = RARL(env, ...)  # Robust Adversarial RL
   ```

## Comparison of Methods

| Method | Safety Guarantee | Training Speed | Performance | Model Required |
|--------|-----------------|----------------|-------------|----------------|
| Reward Shaping | Soft | Fast | High | No |
| Safety Filters | Strong | Fast | Medium | Yes |
| Constrained RL | Medium | Slow | Medium-High | No |
| Safe Exploration | Strong | Medium | Medium | Yes |

## Choosing the Right Approach

### Decision Tree

1. **Is safety critical (lives at stake)?**
   - Yes → Use safety filters (CBF/MPSC)
   - No → Continue

2. **Do you have accurate dynamics model?**
   - Yes → Consider safety filters or MPC
   - No → Use reward shaping or constrained RL

3. **Can you tolerate training violations?**
   - Yes → Reward shaping or constrained RL
   - No → Safe exploration or sim-to-real transfer

4. **Is real-time performance critical?**
   - Yes → Avoid heavy safety filters
   - No → All methods viable

### Recommended Combinations

**Scenario 1: Learning in Simulation, Deploy on Robot**
- Training: Reward shaping for speed
- Deployment: CBF safety filter for guarantees

**Scenario 2: Physical System, No Failures Allowed**
- Training: Safe exploration + conservative init
- Deployment: Safety filter + learned policy

**Scenario 3: Soft Safety, Fast Learning**
- Training: Reward shaping
- Deployment: Learned policy only

**Scenario 4: Unknown Dynamics, Safety Important**
- Training: Constrained RL (CPO/PPO-Lagrangian)
- Deployment: Learned policy with monitoring

## Implementation Pattern

General pattern for safe RL in safe-control-gym:

```python
from safe_control_gym.utils.registration import make
from safe_control_gym.controllers.ppo import PPO
from safe_control_gym.safety_filters.cbf import CBF

# 1. Create environment with constraints
env = make(
    'cartpole',
    constraints=[{
        'constraint_type': 'BoundedConstraint',
        'constrained_variable': 'state',
        'active_dims': [0, 2],
        'upper_bounds': [1.0, 0.3],
        'lower_bounds': [-1.0, -0.3]
    }],
    use_constraint_penalty=True,
    constraint_penalty=10.0
)

# 2. Create RL controller
ctrl = PPO(
    lambda: env,
    training=True,
    checkpoint_path='model_latest.pt',
    output_dir='./results',

    # PPO hyperparameters
    hidden_dim=256,
    actor_lr=3e-4,
    critic_lr=1e-3,
    gamma=0.99,

    # Training config
    num_epochs=10000,
    rollout_batch_size=2048,
)

# 3. Train
ctrl.reset()
ctrl.learn()

# 4. Evaluate with safety filter (optional)
ctrl.training = False
safety_filter = CBF(lambda: env)

test_env = make('cartpole', gui=True, constraints=constraints)
obs, _ = test_env.reset()

for step in range(500):
    # Get action from learned policy
    action = ctrl.select_action(obs)

    # Filter for safety
    action_safe = safety_filter.compute_action(obs, action)

    # Execute
    obs, reward, done, truncated, info = test_env.step(action_safe)

    if done or truncated:
        break
```

## Metrics for Safe RL

Track both performance and safety:

```python
class SafeRLMetrics:
    def __init__(self):
        self.episode_rewards = []
        self.constraint_violations = []
        self.violation_magnitudes = []

    def record_episode(self, env, rewards):
        # Performance
        self.episode_rewards.append(sum(rewards))

        # Safety
        violations = [env.constraints.is_violated(env)]
        self.constraint_violations.append(sum(violations))

        # Severity
        magnitudes = [max(0, env.constraints.get_value(env))]
        self.violation_magnitudes.append(sum(magnitudes))

    def get_stats(self):
        return {
            'mean_reward': np.mean(self.episode_rewards),
            'mean_violations': np.mean(self.constraint_violations),
            'violation_rate': np.mean([v > 0 for v in self.constraint_violations]),
            'mean_magnitude': np.mean(self.violation_magnitudes)
        }
```

## Key Takeaways

1. **Multiple approaches** to safe RL: reward shaping, filters, constrained algorithms, safe exploration
2. **Trade-offs** between safety guarantees, performance, and complexity
3. **Hybrid approaches** work best: soft constraints for learning, hard filters for deployment
4. **Metrics matter**: track both performance and safety
5. **Choose based on application**: simulation vs real, soft vs critical safety

## Next Steps

- [02_safe_ppo_training.md](02_safe_ppo_training.md) - Implement safe PPO training
- [03_safety_filters.md](03_safety_filters.md) - Deep dive into CBF and MPSC
- [04_evaluation.md](04_evaluation.md) - Evaluate safe RL policies

## Additional Resources

- [Safe Learning in Robotics (Review)](https://arxiv.org/abs/2108.06266)
- [Constrained Policy Optimization](https://arxiv.org/abs/1705.10528)
- [Control Barrier Functions for Safe Learning](https://arxiv.org/abs/1903.11199)
