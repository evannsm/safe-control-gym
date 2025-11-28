# Stage 2.1: Constraint Theory and Formulation

## Introduction

Safety constraints are the foundation of safe reinforcement learning. This guide introduces constraint theory, formulations, and how to implement them in safe-control-gym.

## Learning Objectives

- Understand constraint formulations and types
- Learn the difference between hard and soft constraints
- Master constraint violation detection
- Implement custom constraint classes

## What Are Safety Constraints?

Safety constraints define **forbidden regions** of the state or action space. They ensure the system operates within safe bounds.

### Mathematical Formulation

A constraint is typically expressed as:

```
g(x, u) ≤ 0
```

Where:
- `g`: constraint function
- `x`: state
- `u`: control input
- **Safe region**: g(x, u) ≤ 0
- **Unsafe region**: g(x, u) > 0

**Examples**:

1. **Position limit**: g(x) = |x| - x_max ≤ 0
2. **Velocity limit**: g(x) = |ẋ| - v_max ≤ 0
3. **Input limit**: g(u) = |u| - u_max ≤ 0
4. **Combined**: g(x, u) = x² + u² - r² ≤ 0

## Types of Constraints

### 1. State Constraints

Constraints on system state only:

```python
g(x) ≤ 0
```

**Examples**:
- Position bounds: `-1 ≤ x ≤ 1` → `g(x) = |x| - 1 ≤ 0`
- Angle limits: `-π/4 ≤ θ ≤ π/4`
- Velocity limits: `|v| ≤ v_max`
- Safe regions: `x² + y² ≤ r²` (circular boundary)

**Use cases**:
- Workspace boundaries
- Joint limits
- Obstacle avoidance
- Collision avoidance

### 2. Input Constraints

Constraints on control inputs:

```python
g(u) ≤ 0
```

**Examples**:
- Actuation limits: `-F_max ≤ F ≤ F_max`
- Torque bounds: `|τ| ≤ τ_max`
- Thrust limits: `0 ≤ thrust ≤ thrust_max`

**Use cases**:
- Physical actuator limits
- Power constraints
- Safety margins on control authority

### 3. Combined Constraints

Constraints involving both state and input:

```python
g(x, u) ≤ 0
```

**Examples**:
- State-dependent input limits: `|u| ≤ f(x)`
- Energy constraints: `½mv² + mgh + ½ku² ≤ E_max`
- Control Barrier Functions (CBF): `Ḣ(x, u) + α H(x) ≥ 0`

**Use cases**:
- State-dependent safety margins
- Energy-based safety
- Formal safety guarantees

## Hard vs Soft Constraints

### Hard Constraints

**Never** allowed to be violated:

```python
g(x, u) ≤ 0  (must always hold)
```

**Enforcement methods**:
1. **Action clipping**: Project actions onto safe set
2. **Safety filters**: Post-process learned actions (CBF, MPSC)
3. **Safe exploration**: Constrain exploration during training
4. **Barrier methods**: Prevent reaching boundary

**Example**:
```python
# Hard position constraint: clip action
if x + u*dt > x_max:
    u = (x_max - x) / dt  # Clip to boundary
```

### Soft Constraints

**Prefer** not to violate, but allowed occasionally:

```python
g(x, u) ≤ 0  (minimize violations)
```

**Enforcement methods**:
1. **Penalty in reward**: `r = r_task - λ * max(0, g(x, u))²`
2. **Lagrangian relaxation**: Add constraint violation cost
3. **Chance constraints**: Allow violations with low probability

**Example**:
```python
# Soft energy constraint: penalize in reward
energy = 0.5 * m * v**2
energy_violation = max(0, energy - E_max)
reward = task_reward - 10.0 * energy_violation**2
```

### When to Use Each?

| Constraint Type | Hard | Soft |
|----------------|------|------|
| Safety-critical (collision) | ✓ | |
| Comfort (smooth motion) | | ✓ |
| Legal requirements | ✓ | |
| Performance goals | | ✓ |
| Physical limits (joint bounds) | ✓ | |
| Efficiency (energy) | | ✓ |

## Constraint Formulations

### Box Constraints

Simplest form: element-wise bounds:

```python
x_min ≤ x ≤ x_max

# Reformulated as g(x) ≤ 0:
g1(x) = x - x_max ≤ 0
g2(x) = x_min - x ≤ 0
```

**Implementation**:
```python
class BoundedConstraint(Constraint):
    def __init__(self, env, upper_bounds, lower_bounds):
        self.upper_bounds = np.array(upper_bounds)
        self.lower_bounds = np.array(lower_bounds)

    def get_value(self, env):
        x = env.state

        # Two constraints: upper and lower
        g_upper = x - self.upper_bounds
        g_lower = self.lower_bounds - x

        # Stack into single vector
        g = np.concatenate([g_upper, g_lower])

        return g  # All elements should be ≤ 0
```

### Norm Constraints

Constrain magnitude (L1, L2, L∞):

```python
# L2 norm: ||x|| ≤ r
g(x) = x² + y² - r² ≤ 0

# L∞ norm: max|x_i| ≤ r
g(x) = max(|x|) - r ≤ 0
```

**Use cases**:
- Circular workspace: `x² + y² ≤ r²`
- Velocity magnitude: `vₓ² + vᵧ² ≤ v_max²`

### Linear Constraints

Constraints defined by hyperplanes:

```python
# Half-space: aᵀx ≤ b
g(x) = aᵀx - b ≤ 0

# Polytope: Ax ≤ b (multiple half-spaces)
g(x) = Ax - b ≤ 0  (element-wise)
```

**Use cases**:
- Polyhedral safe regions
- Linear obstacle avoidance

### Nonlinear Constraints

Arbitrary smooth functions:

```python
# Ellipse: (x/a)² + (y/b)² ≤ 1
g(x) = (x/a)² + (y/b)² - 1 ≤ 0

# Trig functions: θ ≤ sin⁻¹(0.5)
g(θ) = sin(θ) - 0.5 ≤ 0
```

## Constraint Violation

### Detecting Violations

A constraint is **violated** when:

```python
g(x, u) > 0
```

With tolerance for numerical errors:

```python
violation = g(x, u) > tolerance  # e.g., tolerance = 1e-6
```

### Strict vs Non-Strict

```python
# Strict: g(x) < 0 required (boundary forbidden)
violation = g(x) >= 0

# Non-strict: g(x) ≤ 0 allowed (boundary OK)
violation = g(x) > 0
```

### Measuring Severity

How badly is the constraint violated?

```python
# Absolute violation
violation_amount = max(0, g(x, u))

# Relative violation
violation_ratio = max(0, g(x, u)) / |g_safe|

# Cumulative violations over episode
total_violation = sum(max(0, g(x_t, u_t)) for t in episode)
```

## Practical Considerations

### 1. Constraint Scaling

Different constraints have different magnitudes:

```python
# Before scaling
g1(x) = x - 1000  # Large scale
g2(x) = θ - 0.1   # Small scale

# After normalization
g1_norm(x) = (x - 1000) / 1000
g2_norm(x) = (θ - 0.1) / 0.1
```

**Benefits**:
- Easier to tune penalties
- Numerical stability
- Fair weighting in optimization

### 2. Active Dimensions

Apply constraints to specific state dimensions only:

```python
# Full state: [x, y, z, vx, vy, vz]
# Constrain only x and z position

active_dims = [0, 2]  # x and z indices
x_active = x[active_dims]  # [x, z]

g(x) = x_active - bounds
```

### 3. Safety Margins

Add buffer to constraints for robustness:

```python
# Physical limit
x_physical_max = 1.0

# Add safety margin (10%)
x_safe_max = 0.9

# Constraint
g(x) = x - x_safe_max ≤ 0
```

## Example: Cartpole Constraints

Comprehensive constraint set for cartpole:

```python
# State: [x, x_dot, theta, theta_dot]
# Control: [force]

# 1. Position constraint (cart on track)
g1(x) = |x| - 1.0 ≤ 0

# 2. Angle constraint (pole upright)
g2(x) = |theta| - 0.3 ≤ 0

# 3. Velocity constraint (safety)
g3(x) = |x_dot| - 2.0 ≤ 0

# 4. Input constraint (motor limits)
g4(u) = |force| - 20.0 ≤ 0

# 5. Combined constraint (state-dependent input limit)
# Reduce max force when near boundary
margin = 1.0 - |x|
force_limit = 20.0 * margin
g5(x, u) = |force| - force_limit ≤ 0
```

## Constraint Design Guidelines

### Do's

1. **Keep it simple**: Start with box constraints
2. **Add margins**: Don't use exact physical limits
3. **Scale properly**: Normalize different constraint types
4. **Test thoroughly**: Verify constraint logic
5. **Document clearly**: Explain constraint purpose

### Don'ts

1. **Over-constrain**: Too many constraints make learning impossible
2. **Conflicting constraints**: Ensure feasible region exists
3. **Ignore dynamics**: Consider reachability
4. **Forget tolerances**: Use numerical tolerances for floating point
5. **Hard-code values**: Make constraints configurable

## Practical Exercise: Design Constraints

For a quadrotor (state: [x, y, z, vx, vy, vz, roll, pitch, yaw]):

**Task**: Design a constraint set that ensures:
1. Stays within workspace: -5 ≤ x, y ≤ 5, 0 ≤ z ≤ 10
2. Speed limits: ||v|| ≤ 5 m/s
3. Attitude limits: |roll|, |pitch| ≤ 30°
4. Input limits: 0 ≤ thrust ≤ 20 N

**Solution**:
```python
# 1. Workspace bounds (6 constraints)
g1 = [x - 5, -5 - x, y - 5, -5 - y, z - 10, 0 - z]

# 2. Speed limit (1 constraint)
g2 = vx**2 + vy**2 + vz**2 - 25

# 3. Attitude limits (2 constraints)
g3 = [|roll| - π/6, |pitch| - π/6]

# 4. Thrust limits (2 constraints)
g4 = [thrust - 20, 0 - thrust]
```

## Key Takeaways

1. **Constraints** define safe regions: g(x, u) ≤ 0
2. **Three types**: state, input, or combined
3. **Hard constraints** must never be violated
4. **Soft constraints** are penalized but allowed
5. **Formulation matters**: choose appropriate form
6. **Design carefully**: margins, scaling, feasibility

## Next Steps

- [02_state_constraints.md](02_state_constraints.md) - Implement state constraints
- [03_input_constraints.md](03_input_constraints.md) - Implement input constraints
- [04_constraint_evaluation.md](04_constraint_evaluation.md) - Runtime checking

## Additional Resources

- [constraints.py source](../../safe_control_gym/envs/constraints.py)
- [Control Barrier Functions: Theory and Applications](https://arxiv.org/abs/1903.11199)
- [Safe RL Tutorial](https://arxiv.org/abs/2108.06266)
