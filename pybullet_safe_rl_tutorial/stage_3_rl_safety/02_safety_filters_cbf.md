# Stage 3.2: Safety Filters - Control Barrier Functions (CBF)

## Introduction

Control Barrier Functions (CBFs) provide mathematical guarantees that a system stays within a safe set. This guide explains CBF theory and implementation in safe-control-gym.

## Learning Objectives

- Understand CBF mathematical foundation
- Implement CBF-based safety filters
- Integrate CBF with learned policies
- Tune CBF parameters for performance

## What is a Control Barrier Function?

A **barrier function** is a scalar function h(x) that defines a safe set:

```
Safe set: C = {x | h(x) ≥ 0}
Unsafe set: {x | h(x) < 0}
```

The **boundary** is where h(x) = 0.

### Example: Position Constraint

For a 1D position constraint |x| ≤ x_max:

```python
# Barrier function
h(x) = x_max² - x²

# Safe when h(x) ≥ 0
# h(x) = 0 → x = ±x_max (boundary)
# h(x) > 0 → |x| < x_max (interior, safe)
# h(x) < 0 → |x| > x_max (outside, unsafe)
```

## CBF Condition

For control-affine dynamics:

```
ẋ = f(x) + g(x)u
```

A function h(x) is a **Control Barrier Function** if there exists α > 0 such that:

```
sup_u [L_f h(x) + L_g h(x) · u + α h(x)] ≥ 0
```

Where:
- L_f h = ∇h · f(x): Lie derivative w.r.t. drift dynamics
- L_g h = ∇h · g(x): Lie derivative w.r.t. control
- α: Class-K function (often just a constant α > 0)

### Intuition

The CBF condition ensures that there *exists* a control input u that keeps the system safe:

```
ḣ(x, u) = ∇h · ẋ
        = ∇h · (f(x) + g(x)u)
        = L_f h + L_g h · u

CBF condition: ḣ(x, u) + α h(x) ≥ 0
```

This means h(x) doesn't decrease too fast → system stays in safe set.

## CBF-QP Safety Filter

Given a desired control u_des (e.g., from RL policy), find the *closest* safe control:

```
minimize    ||u - u_des||²
subject to  L_f h(x) + L_g h(x) · u + α h(x) ≥ 0
            u_min ≤ u ≤ u_max
```

This is a **Quadratic Program (QP)** that can be solved efficiently.

### Properties

1. **If u_des is safe**, QP returns u = u_des (no modification)
2. **If u_des is unsafe**, QP finds minimal modification
3. **Guarantees** system stays in safe set (if feasible)

## Example: Cartpole CBF

For cartpole with state constraint on position and angle:

### State Constraints

```python
# State: [x, x_dot, theta, theta_dot]
# Constraints:
#   |x| ≤ x_max
#   |theta| ≤ theta_max

# Barrier function (ellipse in x-theta space)
h(x, theta) = 1 - (x/x_max)² - (theta/theta_max)²

# Safe when h ≥ 0
```

### Dynamics

Cartpole dynamics (simplified):

```python
ẋ = x_dot
ẍ = (F + m*l*theta_dot²*sin(theta)) / M
θ̇ = theta_dot
θ̈ = (g*sin(theta) - F*cos(theta)/M) / l
```

In control-affine form: ẋ = f(x) + g(x)u

### Lie Derivatives

```python
# Gradient of h
∇h = [-2x/x_max², 0, -2θ/θ_max², 0]

# L_f h = ∇h · f(x) (drift term)
L_f_h = ∇h[0] * x_dot + ∇h[2] * theta_dot

# L_g h = ∇h · g(x) (control term)
L_g_h = ∇h[1] * (1/M) + ∇h[3] * (-cos(theta)/(M*l))
```

### CBF-QP

```python
# Find safe control
minimize    (u - u_des)²
subject to  L_f_h + L_g_h * u + α * h ≥ 0
            -F_max ≤ u ≤ F_max
```

## Implementation in safe-control-gym

### Using CBF Safety Filter

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
        'active_dims': [0, 2],  # x and theta
        'upper_bounds': [1.0, 0.3],
        'lower_bounds': [-1.0, -0.3]
    }]
)

# 2. Train RL policy (PPO)
ctrl = PPO(lambda: env, training=True)
ctrl.reset()
ctrl.learn()

# 3. Create CBF safety filter
cbf_filter = CBF(
    env_func=lambda: env,
    slope=0.1,  # α parameter
    soft_constrained=True,  # Use slack variable
    slack_weight=10000.0,  # Penalty on constraint violation
    slack_tolerance=1e-3  # Max allowed slack
)

# 4. Test with safety filter
env_test = make('cartpole', gui=True, constraints=env.constraints)
obs, _ = env_test.reset()

for step in range(500):
    # Get desired action from policy
    action_des = ctrl.select_action(obs)

    # Filter through CBF
    action_safe = cbf_filter.compute_action(obs, action_des)

    # Execute safe action
    obs, reward, done, truncated, info = env_test.step(action_safe)

    # Check if CBF modified action
    if np.linalg.norm(action_safe - action_des) > 1e-6:
        print(f"Step {step}: CBF intervention! "
              f"Desired: {action_des}, Safe: {action_safe}")

    if done or truncated:
        break
```

### CBF Parameters

**slope (α)**: Controls conservativeness
- **Large α**: More conservative, stays farther from boundary
- **Small α**: Less conservative, allows closer to boundary
- Typical range: 0.01 to 1.0

**soft_constrained**: Use slack variable
- **True**: Allow small violations with penalty (more robust)
- **False**: Strict constraint (may be infeasible)

**slack_weight**: Penalty on slack variable
- **Large**: Strictly enforce (10000+)
- **Small**: Allow more violations (100-)

**slack_tolerance**: Maximum allowed slack
- Typical: 1e-3 to 1e-1

## Advanced: Custom CBF

Implement a custom barrier function:

```python
import casadi as cs
from safe_control_gym.safety_filters.cbf import CBF

class CustomCBF(CBF):
    def __init__(self, env_func, **kwargs):
        super().__init__(env_func, **kwargs)

        # Define custom barrier function
        self.setup_custom_barrier()

    def setup_custom_barrier(self):
        """Define custom h(x) function."""

        # Get state variables
        x = self.X  # CasADi symbolic state

        # Custom barrier: exponential barrier for smoothness
        x_pos = x[0]
        theta = x[2]

        # Smooth barrier
        h_x = cs.exp(-(x_pos/0.9)**2)  # Decays near boundaries
        h_theta = cs.exp(-(theta/0.25)**2)

        # Combined barrier (product)
        h_combined = h_x * h_theta

        # Override CBF function
        self.cbf = cs.Function('h', [self.X], [h_combined])

        # Recompute Lie derivative
        self.lie_derivative = self.get_lie_derivative()
        self.setup_optimizer()
```

## CBF Limitations

### 1. Model Accuracy

CBF requires accurate dynamics model (f, g). Model mismatch can lead to:
- False sense of safety
- Overly conservative behavior
- Constraint violations

**Solution**: Add robustness margins, use adaptive CBF

### 2. Control Authority

CBF assumes sufficient control authority. May fail if:
- Actuator saturation
- Insufficient control power
- State already too close to boundary

**Solution**: Use exponential CBF, larger α, earlier intervention

### 3. Multiple Constraints

With many constraints, QP may become infeasible:

```python
# Multiple constraints
h1(x) ≥ 0  # Position
h2(x) ≥ 0  # Angle
h3(x) ≥ 0  # Velocity

# All must be satisfied simultaneously
# May be infeasible if constraints conflict
```

**Solution**: Prioritize constraints, use soft constraints

### 4. Computational Cost

QP must be solved at each control step (e.g., 50-100 Hz):
- Can be expensive for high-dimensional systems
- Real-time requirements

**Solution**: Use efficient QP solvers (OSQP, qpOASES), simplify barrier

## Comparison with Other Methods

| Method | Guarantees | Model Required | Computational Cost | Ease of Use |
|--------|-----------|----------------|-------------------|-------------|
| CBF | Strong | Yes | Medium (QP) | Medium |
| MPSC | Strong | Yes | High (MPC) | Hard |
| Reward Shaping | Soft | No | Low | Easy |
| Constrained RL | Medium | No | Low | Medium |

## Practical Tips

### 1. Start Simple

Begin with basic barrier functions:
```python
# Simple box constraint
h(x) = x_max - x  # One-sided
h(x) = x_max² - x²  # Two-sided (smoother)
```

### 2. Tune α Carefully

```python
# Test different α values
alphas = [0.01, 0.1, 0.5, 1.0]

for alpha in alphas:
    cbf = CBF(env_func, slope=alpha)
    # Evaluate performance and safety
    metrics = evaluate(cbf)
    print(f"α={alpha}: violations={metrics['violations']}, "
          f"performance={metrics['reward']}")
```

### 3. Visualize Barrier

```python
import matplotlib.pyplot as plt

def plot_barrier(cbf, x_range, theta_range):
    X, Theta = np.meshgrid(x_range, theta_range)
    H = np.zeros_like(X)

    for i in range(len(x_range)):
        for j in range(len(theta_range)):
            state = np.array([X[i,j], 0, Theta[i,j], 0])
            H[i,j] = cbf.cbf(state)

    plt.contourf(X, Theta, H, levels=20)
    plt.contour(X, Theta, H, levels=[0], colors='r', linewidths=2)
    plt.xlabel('x')
    plt.ylabel('theta')
    plt.title('Barrier Function h(x, theta)')
    plt.colorbar(label='h (safe when h≥0)')
    plt.show()
```

### 4. Monitor Interventions

Track how often CBF modifies actions:

```python
intervention_count = 0
total_steps = 0

for episode in range(n_episodes):
    obs, _ = env.reset()
    done = False

    while not done:
        action_des = policy(obs)
        action_safe = cbf.compute_action(obs, action_des)

        if np.linalg.norm(action_safe - action_des) > 1e-4:
            intervention_count += 1

        obs, _, done, _, _ = env.step(action_safe)
        total_steps += 1

print(f"CBF intervention rate: {intervention_count/total_steps:.2%}")
```

## Key Takeaways

1. **CBF** provides mathematical safety guarantees through barrier functions
2. **CBF-QP** filters actions to ensure safety with minimal modification
3. **Requires** accurate dynamics model and sufficient control authority
4. **Parameters** (α, slack weight) trade off conservativeness and performance
5. **Works well** with learned policies as a post-processing safety layer

## Next Steps

- [03_training_with_safety.md](03_training_with_safety.md) - Complete safe RL training pipeline
- [../stage_4_advanced/](../stage_4_advanced/) - Advanced topics: MPSC, domain randomization

## Additional Resources

- [CBF Tutorial Paper](https://arxiv.org/abs/1903.11199)
- [CBF source code](../../safe_control_gym/safety_filters/cbf/cbf.py)
- [CBF Neural Networks](../../safe_control_gym/safety_filters/cbf/cbf_nn.py)
