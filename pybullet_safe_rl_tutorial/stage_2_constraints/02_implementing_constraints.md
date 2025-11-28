# Stage 2.2: Implementing Constraints in safe-control-gym

## Introduction

This guide shows how to implement and use constraints in your environments using the safe-control-gym framework.

## The Constraint Class

All constraints inherit from the `Constraint` base class:

```python
from safe_control_gym.envs.constraints import Constraint, ConstrainedVariableType

class MyConstraint(Constraint):
    def __init__(self, env, **kwargs):
        # Specify what this constraint acts on
        super().__init__(
            env,
            constrained_variable=ConstrainedVariableType.STATE,  # or INPUT, INPUT_AND_STATE
            **kwargs
        )

        # Define constraint-specific parameters
        self.num_constraints = 2  # Number of scalar constraints

    def get_value(self, env):
        """Evaluate constraint: returns g(x) where g(x) ≤ 0 is safe."""
        pass

    def get_symbolic_model(self):
        """Return CasADi symbolic form of constraint."""
        pass
```

## Example 1: Bounded State Constraint

Simplest constraint: box bounds on state variables.

```python
import numpy as np
import casadi as cs
from safe_control_gym.envs.constraints import Constraint, ConstrainedVariableType

class BoundedConstraint(Constraint):
    """Box constraints on state: lower ≤ state ≤ upper."""

    def __init__(self,
                 env,
                 constrained_variable=ConstrainedVariableType.STATE,
                 active_dims=None,
                 upper_bounds=None,
                 lower_bounds=None,
                 **kwargs):
        """Initialize bounded constraint.

        Args:
            env: Environment instance.
            active_dims: Which dimensions to constrain (None = all).
            upper_bounds: Upper limits for each dimension.
            lower_bounds: Lower limits for each dimension.
        """

        super().__init__(
            env,
            constrained_variable=constrained_variable,
            active_dims=active_dims,
            **kwargs
        )

        # Store bounds
        if upper_bounds is not None:
            self.upper_bounds = np.array(upper_bounds, ndmin=1)
        else:
            self.upper_bounds = np.inf * np.ones(self.dim)

        if lower_bounds is not None:
            self.lower_bounds = np.array(lower_bounds, ndmin=1)
        else:
            self.lower_bounds = -np.inf * np.ones(self.dim)

        # Total number of constraints (upper + lower for each dim)
        self.num_constraints = 2 * self.dim

    def get_value(self, env):
        """Evaluate constraint.

        Returns:
            g (np.array): Constraint values (≤ 0 is safe).
        """

        # Get current state
        if self.constrained_variable == ConstrainedVariableType.STATE:
            state = env.state
        elif self.constrained_variable == ConstrainedVariableType.INPUT:
            state = env.action  # Last action
        else:
            raise NotImplementedError

        # Apply filter to select active dimensions
        state_filtered = state @ self.constraint_filter.T

        # Upper bound constraints: x - x_max ≤ 0
        g_upper = state_filtered - self.upper_bounds

        # Lower bound constraints: x_min - x ≤ 0
        g_lower = self.lower_bounds - state_filtered

        # Stack into single vector
        g = np.concatenate([g_upper, g_lower])

        return g

    def is_violated(self, env, c_value=None):
        """Check if constraint is currently violated.

        Args:
            env: Environment instance.
            c_value: Pre-computed constraint value (optional).

        Returns:
            violated (bool): True if any constraint violated.
        """

        if c_value is None:
            c_value = self.get_value(env)

        # Check if any constraint is positive (violated)
        if self.strict:
            violated = np.any(c_value >= 0)
        else:
            violated = np.any(c_value > 0)

        return violated

    def get_symbolic_model(self):
        """Return symbolic constraint model for CBF/MPC.

        Returns:
            sym_dict (dict): Dictionary with symbolic constraint info.
        """

        # Create symbolic variable
        if self.constrained_variable == ConstrainedVariableType.STATE:
            x_sym = cs.SX.sym('x', self.dim)
        else:
            x_sym = cs.SX.sym('u', self.dim)

        # Symbolic constraints
        g_upper = x_sym - self.upper_bounds
        g_lower = self.lower_bounds - x_sym
        g_sym = cs.vertcat(g_upper, g_lower)

        # Create CasADi function
        g_func = cs.Function('g', [x_sym], [g_sym], ['x'], ['g'])

        return {
            'symbolic_model': g_func,
            'num_constraints': self.num_constraints
        }
```

### Usage Example

```python
from safe_control_gym.utils.registration import make

# Create environment with constraints
env = make(
    'cartpole',
    constraints=[{
        'constraint_type': 'BoundedConstraint',
        'constrained_variable': 'state',
        'active_dims': [0, 2],  # x position and theta
        'upper_bounds': [1.0, 0.3],  # Max 1m position, 0.3 rad angle
        'lower_bounds': [-1.0, -0.3]
    }]
)

# Check constraints during episode
obs, info = env.reset()

for step in range(100):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)

    # Check constraint violations
    if env.constraints.is_violated(env):
        print(f"Constraint violated at step {step}")
        constraint_values = env.constraints.get_value(env)
        print(f"Constraint values: {constraint_values}")
```

## Example 2: Velocity Constraint

Constrain velocity magnitude:

```python
class VelocityConstraint(Constraint):
    """Constrains velocity magnitude: ||v|| ≤ v_max."""

    def __init__(self,
                 env,
                 velocity_dims,  # Which dimensions are velocities
                 max_velocity,
                 **kwargs):
        """Initialize velocity constraint.

        Args:
            env: Environment.
            velocity_dims: Indices of velocity dimensions.
            max_velocity: Maximum velocity magnitude.
        """

        super().__init__(
            env,
            constrained_variable=ConstrainedVariableType.STATE,
            **kwargs
        )

        self.velocity_dims = np.array(velocity_dims)
        self.max_velocity = max_velocity
        self.num_constraints = 1  # Single constraint on magnitude

    def get_value(self, env):
        """Evaluate constraint."""

        # Extract velocities
        velocities = env.state[self.velocity_dims]

        # Velocity magnitude
        v_mag = np.linalg.norm(velocities)

        # Constraint: v_mag - v_max ≤ 0
        g = np.array([v_mag - self.max_velocity])

        return g

    def get_symbolic_model(self):
        """Symbolic model."""

        x_sym = cs.SX.sym('x', len(self.velocity_dims))

        # Symbolic L2 norm
        v_mag = cs.norm_2(x_sym)

        # Constraint
        g_sym = v_mag - self.max_velocity

        g_func = cs.Function('g', [x_sym], [g_sym], ['x'], ['g'])

        return {
            'symbolic_model': g_func,
            'num_constraints': self.num_constraints
        }
```

## Adding Constraints to Environments

### Method 1: Configuration File

```yaml
# config/safe_cartpole.yaml
task: cartpole
constraints:
  # Position constraint
  - constraint_type: BoundedConstraint
    constrained_variable: state
    active_dims: [0]  # x position
    upper_bounds: [1.0]
    lower_bounds: [-1.0]

  # Angle constraint
  - constraint_type: BoundedConstraint
    constrained_variable: state
    active_dims: [2]  # theta
    upper_bounds: [0.3]
    lower_bounds: [-0.3]

  # Input constraint
  - constraint_type: BoundedConstraint
    constrained_variable: input
    upper_bounds: [20.0]
    lower_bounds: [-20.0]
```

### Method 2: Python Dictionary

```python
constraints = [
    {
        'constraint_type': 'BoundedConstraint',
        'constrained_variable': 'state',
        'active_dims': [0, 2],
        'upper_bounds': [1.0, 0.3],
        'lower_bounds': [-1.0, -0.3]
    },
    {
        'constraint_type': 'BoundedConstraint',
        'constrained_variable': 'input',
        'upper_bounds': [20.0],
        'lower_bounds': [-20.0]
    }
]

env = make('cartpole', constraints=constraints)
```

### Method 3: Programmatic

```python
from safe_control_gym.envs.constraints import create_constraint_list

# Create environment
env = make('cartpole')

# Add constraints
constraint_list = [
    BoundedConstraint(
        env,
        constrained_variable=ConstrainedVariableType.STATE,
        active_dims=[0],
        upper_bounds=[1.0],
        lower_bounds=[-1.0]
    )
]

env.constraints = create_constraint_list(env, constraint_list)
```

## Constraint Penalties in Rewards

Penalize constraint violations in the reward:

```python
class CartpoleWithPenalty(Cartpole):
    def __init__(self,
                 constraint_penalty=10.0,
                 **kwargs):
        super().__init__(
            use_constraint_penalty=True,
            constraint_penalty=constraint_penalty,
            **kwargs
        )

    def _compute_reward(self):
        """Compute reward with constraint penalty."""

        # Base task reward
        task_reward = self._compute_task_reward()

        # Constraint violation penalty
        if self.use_constraint_penalty:
            constraint_values = self.constraints.get_value(self)

            # Penalize positive (violated) constraints
            violations = np.maximum(0, constraint_values)
            penalty = self.constraint_penalty * np.sum(violations**2)

            reward = task_reward - penalty
        else:
            reward = task_reward

        return reward
```

## Termination on Violation

End episode when constraint violated:

```python
env = make(
    'cartpole',
    constraints=constraints,
    done_on_violation=True  # Episode ends on violation
)

# During episode
obs, reward, terminated, truncated, info = env.step(action)

# terminated=True if constraint violated
if terminated and info.get('constraint_violation', False):
    print("Episode terminated due to constraint violation!")
```

## Monitoring Constraints

Track constraint violations during training:

```python
class ConstraintMonitor:
    def __init__(self):
        self.violations = []
        self.total_violations = 0

    def record_step(self, env):
        """Record constraint state at each step."""

        violated = env.constraints.is_violated(env)
        self.violations.append(violated)

        if violated:
            self.total_violations += 1

    def get_stats(self):
        """Get constraint statistics."""

        return {
            'total_steps': len(self.violations),
            'total_violations': self.total_violations,
            'violation_rate': self.total_violations / len(self.violations)
        }

# Usage
monitor = ConstraintMonitor()

for episode in range(num_episodes):
    obs, _ = env.reset()
    done = False

    while not done:
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        monitor.record_step(env)

print(f"Constraint stats: {monitor.get_stats()}")
```

## Advanced: Custom Constraint

Create a circular safe region constraint:

```python
class CircularConstraint(Constraint):
    """Keep state within circular region."""

    def __init__(self,
                 env,
                 center,
                 radius,
                 state_dims,  # Which dims define the circle (e.g., [0, 1] for x, y)
                 **kwargs):
        super().__init__(
            env,
            constrained_variable=ConstrainedVariableType.STATE,
            **kwargs
        )

        self.center = np.array(center)
        self.radius = radius
        self.state_dims = state_dims
        self.num_constraints = 1

    def get_value(self, env):
        """Evaluate: distance² - radius² ≤ 0."""

        # Extract relevant state dimensions
        state = env.state[self.state_dims]

        # Distance from center
        dist_sq = np.sum((state - self.center)**2)

        # Constraint: dist² - r² ≤ 0
        g = np.array([dist_sq - self.radius**2])

        return g

    def get_symbolic_model(self):
        """Symbolic model."""

        x_sym = cs.SX.sym('x', len(self.state_dims))

        # Distance squared
        dist_sq = cs.dot(x_sym - self.center, x_sym - self.center)

        # Constraint
        g_sym = dist_sq - self.radius**2

        g_func = cs.Function('g', [x_sym], [g_sym])

        return {
            'symbolic_model': g_func,
            'num_constraints': 1
        }

# Usage
env = make(
    'quadrotor',
    constraints=[{
        'constraint_type': 'CircularConstraint',
        'center': [0, 0],
        'radius': 5.0,
        'state_dims': [0, 1]  # x, y position
    }]
)
```

## Key Takeaways

1. **Inherit from Constraint** base class
2. **Implement `get_value()`** to evaluate constraints
3. **Implement `get_symbolic_model()`** for MPC/CBF use
4. **Specify constrained_variable**: STATE, INPUT, or INPUT_AND_STATE
5. **Use active_dims** to select specific state/input dimensions
6. **Multiple constraints** can be combined in a list

## Next Steps

- [Stage 3: RL Integration with Safety](../stage_3_rl_safety/) - Use constraints with RL algorithms

## Additional Resources

- [constraints.py source](../../safe_control_gym/envs/constraints.py)
- [Cartpole constraint examples](../../safe_control_gym/envs/gym_control/cartpole.py)
