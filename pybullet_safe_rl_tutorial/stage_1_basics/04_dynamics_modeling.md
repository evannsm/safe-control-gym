# Stage 1.4: Symbolic Dynamics Modeling with CasADi

## Introduction

While PyBullet handles numeric simulation, many control algorithms (MPC, iLQR, safety filters) require symbolic representations of system dynamics. This guide shows how to add symbolic dynamics to your environment using CasADi.

## Learning Objectives

- Understand the role of symbolic dynamics in safe control
- Learn CasADi basics for symbolic mathematics
- Implement symbolic models for environments
- Use symbolic models in controllers and safety filters

## Why Symbolic Dynamics?

**Numeric Simulation (PyBullet)**:
- High-fidelity physics
- Complex contact dynamics
- No analytical gradients
- Black-box to controllers

**Symbolic Dynamics (CasADi)**:
- Simplified analytical model
- Fast gradient computation
- Required for MPC, CBF, iLQR
- White-box for optimization

**Best Practice**: Use both!
- PyBullet for realistic simulation
- CasADi for control and safety

## CasADi Basics

### Installation

```bash
pip install casadi
```

### Symbolic Variables

```python
import casadi as cs

# Scalar variable
x = cs.SX.sym('x')

# Vector variable
x = cs.SX.sym('x', 4)  # 4D vector

# Matrix variable
A = cs.SX.sym('A', 3, 3)  # 3x3 matrix
```

### Symbolic Expressions

```python
import casadi as cs

# Define variables
x = cs.SX.sym('x')
v = cs.SX.sym('v')
a = cs.SX.sym('a')

# Build expressions
kinetic_energy = 0.5 * m * v**2
position_update = x + v * dt + 0.5 * a * dt**2

# Vector operations
state = cs.vertcat(x, v)  # Stack into vector
dot_product = cs.dot(state, state)  # Dot product
```

### Functions

```python
# Define symbolic function
x = cs.SX.sym('x', 2)
u = cs.SX.sym('u', 1)

# f(x, u) = [x[1], u[0] - 9.81]
f = cs.vertcat(x[1], u[0] - 9.81)

# Create callable function
f_func = cs.Function('f', [x, u], [f], ['x', 'u'], ['x_dot'])

# Evaluate
x_val = np.array([1.0, 0.5])
u_val = np.array([2.0])
x_dot_val = f_func(x_val, u_val)
print(x_dot_val)  # Output: [0.5, -7.81]
```

### Jacobians

```python
# Automatic differentiation
x = cs.SX.sym('x', 2)
u = cs.SX.sym('u', 1)

f = cs.vertcat(x[1], u[0] - 9.81 - 0.1 * x[1])

# Compute Jacobians
df_dx = cs.jacobian(f, x)  # ∂f/∂x
df_du = cs.jacobian(f, u)  # ∂f/∂u

# Create function that returns both f and Jacobians
f_jac = cs.Function('f_jac',
                    [x, u],
                    [f, df_dx, df_du],
                    ['x', 'u'],
                    ['f', 'df_dx', 'df_du'])
```

## Symbolic Dynamics in safe-control-gym

### The SymbolicModel Class

All environments provide a symbolic model through `get_symbolic_model()`:

```python
from safe_control_gym.math_and_models.symbolic_systems import SymbolicModel

class MySymbolicModel(SymbolicModel):
    def __init__(self):
        super().__init__()

        # Define symbolic variables
        self.nx = 4  # State dimension
        self.nu = 1  # Control dimension

        self.x_sym = cs.SX.sym('x', self.nx)
        self.u_sym = cs.SX.sym('u', self.nu)

        # Define dynamics: x_dot = f(x, u)
        self.x_dot = self._define_dynamics()

        # Create callable function
        self.fc_func = cs.Function(
            'fc',
            [self.x_sym, self.u_sym],
            [self.x_dot],
            ['x', 'u'],
            ['x_dot']
        )

    def _define_dynamics(self):
        """Define continuous-time dynamics."""
        # Implement system-specific dynamics
        pass
```

### Example: Pendulum Symbolic Model

```python
import casadi as cs
from safe_control_gym.math_and_models.symbolic_systems import SymbolicModel

class PendulumModel(SymbolicModel):
    """Symbolic model for simple pendulum.

    State: [theta, theta_dot]
    Control: [torque]

    Dynamics:
        theta_dot = omega
        omega_dot = (tau - m*g*L*sin(theta) - b*omega) / (m*L^2)
    """

    def __init__(self,
                 mass=0.15,
                 length=0.5,
                 gravity=9.81,
                 damping=0.01):
        self.mass = mass
        self.length = length
        self.gravity = gravity
        self.damping = damping

        # Dimensions
        self.nx = 2  # [theta, theta_dot]
        self.nu = 1  # [torque]

        # Create symbolic variables
        self.x_sym = cs.SX.sym('x', self.nx)
        self.u_sym = cs.SX.sym('u', self.nu)

        # Define dynamics
        self.x_dot = self._define_dynamics()

        # Create functions
        self._create_functions()

        super().__init__()

    def _define_dynamics(self):
        """Define continuous-time dynamics."""

        theta = self.x_sym[0]
        theta_dot = self.x_sym[1]
        tau = self.u_sym[0]

        # Inertia
        I = self.mass * self.length**2

        # Angular acceleration
        theta_ddot = (
            tau
            - self.mass * self.gravity * self.length * cs.sin(theta)
            - self.damping * theta_dot
        ) / I

        # State derivative
        x_dot = cs.vertcat(theta_dot, theta_ddot)

        return x_dot

    def _create_functions(self):
        """Create CasADi functions for dynamics and derivatives."""

        # Continuous dynamics: x_dot = f(x, u)
        self.fc_func = cs.Function(
            'fc',
            [self.x_sym, self.u_sym],
            [self.x_dot],
            ['x', 'u'],
            ['x_dot']
        )

        # Jacobians for linearization
        self.A = cs.jacobian(self.x_dot, self.x_sym)
        self.B = cs.jacobian(self.x_dot, self.u_sym)

        self.df_func = cs.Function(
            'df',
            [self.x_sym, self.u_sym],
            [self.A, self.B],
            ['x', 'u'],
            ['A', 'B']
        )

    def linearize(self, x0, u0):
        """Linearize dynamics around operating point.

        Args:
            x0: State operating point.
            u0: Control operating point.

        Returns:
            A: State Jacobian (nx x nx).
            B: Control Jacobian (nx x nu).
        """

        A, B = self.df_func(x0, u0)
        return np.array(A), np.array(B)
```

## Integrating Symbolic Models with Environments

Add symbolic model to your environment:

```python
class SimplePendulum(BenchmarkEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # ... environment setup ...

        # Create symbolic model
        self.symbolic = PendulumModel(
            mass=self.MASS,
            length=self.LENGTH,
            gravity=self.GRAVITY_ACC,
            damping=0.01
        )

    def get_symbolic_model(self):
        """Return symbolic dynamics model."""
        return self.symbolic

    def _symbolic_step(self, x, u, dt):
        """Simulate one step using symbolic model (for verification).

        Args:
            x: Current state.
            u: Control input.
            dt: Time step.

        Returns:
            x_next: Next state (Euler integration).
        """

        # Get dynamics
        x_dot = self.symbolic.fc_func(x, u)
        x_dot = np.array(x_dot).flatten()

        # Euler integration
        x_next = x + dt * x_dot

        return x_next
```

## Example: Cartpole Symbolic Dynamics

More complex example with coupled dynamics:

```python
class CartpoleModel(SymbolicModel):
    """Symbolic model for cartpole.

    State: [x, x_dot, theta, theta_dot]
    Control: [force]

    Nonlinear dynamics with coupling between cart and pole.
    """

    def __init__(self,
                 m_cart=1.0,
                 m_pole=0.1,
                 l_pole=0.5,
                 gravity=9.81):
        self.m_cart = m_cart
        self.m_pole = m_pole
        self.l_pole = l_pole
        self.gravity = gravity

        self.nx = 4
        self.nu = 1

        self.x_sym = cs.SX.sym('x', self.nx)
        self.u_sym = cs.SX.sym('u', self.nu)

        self.x_dot = self._define_dynamics()
        self._create_functions()

        super().__init__()

    def _define_dynamics(self):
        """Cartpole dynamics from Lagrangian mechanics."""

        # Unpack state
        x = self.x_sym[0]
        x_dot = self.x_sym[1]
        theta = self.x_sym[2]
        theta_dot = self.x_sym[3]

        # Control
        F = self.u_sym[0]

        # Masses and length
        mc = self.m_cart
        mp = self.m_pole
        l = self.l_pole
        g = self.gravity

        # Total mass
        m_total = mc + mp

        # Cartpole equations of motion
        sin_theta = cs.sin(theta)
        cos_theta = cs.cos(theta)

        # Denominator in dynamics
        denom = m_total - mp * cos_theta**2

        # Cart acceleration
        x_ddot = (
            F + mp * l * theta_dot**2 * sin_theta
            - mp * g * sin_theta * cos_theta
        ) / denom

        # Pole angular acceleration
        theta_ddot = (
            -F * cos_theta
            - mp * l * theta_dot**2 * sin_theta * cos_theta
            + m_total * g * sin_theta
        ) / (l * denom)

        # State derivative
        x_dot_vec = cs.vertcat(x_dot, x_ddot, theta_dot, theta_ddot)

        return x_dot_vec

    def _create_functions(self):
        """Create CasADi functions."""

        # Dynamics
        self.fc_func = cs.Function(
            'fc',
            [self.x_sym, self.u_sym],
            [self.x_dot],
            ['x', 'u'],
            ['x_dot']
        )

        # Jacobians
        self.A = cs.jacobian(self.x_dot, self.x_sym)
        self.B = cs.jacobian(self.x_dot, self.u_sym)

        self.df_func = cs.Function(
            'df',
            [self.x_sym, self.u_sym],
            [self.A, self.B],
            ['x', 'u'],
            ['A', 'B']
        )
```

## Using Symbolic Models

### 1. Model Predictive Control (MPC)

```python
from safe_control_gym.controllers.mpc import MPC

# Create environment with symbolic model
env = make('cartpole')

# Create MPC controller (uses symbolic model internally)
mpc = MPC(
    env_func=lambda: env,
    horizon=20,
    q_mpc=[1, 1, 10, 10],  # State weights
    r_mpc=[0.1]  # Control weight
)

# MPC uses env.symbolic to predict future states
```

### 2. Control Barrier Functions (CBF)

```python
# CBF needs symbolic dynamics for Lie derivatives
# See Stage 3 for details

def cbf_constraint(x, u, model):
    """CBF constraint: Lf_h + Lg_h * u + alpha * h >= 0"""

    # Barrier function (e.g., state constraint)
    h = cs.SX.sym('h')  # h(x) >= 0 defines safe set

    # Lie derivative with respect to dynamics
    dh_dx = cs.gradient(h, x)
    Lf_h = cs.dot(dh_dx, model.fc_func(x, 0))  # Drift term
    Lg_h = cs.dot(dh_dx, model.B)  # Control term

    # CBF constraint
    constraint = Lf_h + Lg_h @ u + alpha * h

    return constraint
```

### 3. Linearization for LQR

```python
from safe_control_gym.controllers.lqr import LQR

# Create environment
env = make('cartpole')

# Linearize around equilibrium
x_eq = np.array([0, 0, 0, 0])  # Upright position
u_eq = np.array([0])

A, B = env.symbolic.linearize(x_eq, u_eq)

# Create LQR controller
lqr = LQR(
    env_func=lambda: env,
    Q=np.diag([1, 1, 10, 10]),
    R=np.array([[0.1]])
)

# LQR uses linearized model
```

## Verifying Symbolic vs Numeric Dynamics

Always verify your symbolic model matches PyBullet:

```python
def verify_dynamics(env, x0, u, dt=0.01, steps=100):
    """Compare symbolic and PyBullet dynamics."""

    # PyBullet trajectory
    env.reset()
    env.state = x0
    pyb_states = [x0.copy()]

    for _ in range(steps):
        obs, _, _, _, _ = env.step(u)
        pyb_states.append(obs.copy())

    pyb_states = np.array(pyb_states)

    # Symbolic trajectory (Euler integration)
    sym_states = [x0.copy()]
    x = x0.copy()

    for _ in range(steps):
        x_dot = env.symbolic.fc_func(x, u)
        x_dot = np.array(x_dot).flatten()
        x = x + dt * x_dot
        sym_states.append(x.copy())

    sym_states = np.array(sym_states)

    # Compare
    error = np.linalg.norm(pyb_states - sym_states, axis=1)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    plt.plot(error)
    plt.xlabel('Step')
    plt.ylabel('State Error')
    plt.title('PyBullet vs Symbolic Model Error')
    plt.grid()
    plt.show()

    print(f"Max error: {error.max():.6f}")
    print(f"Mean error: {error.mean():.6f}")
```

## Key Takeaways

1. **CasADi** provides symbolic mathematics for control
2. **SymbolicModel** class defines continuous dynamics x_dot = f(x, u)
3. **Automatic differentiation** computes Jacobians for linearization
4. **Both models** (PyBullet + symbolic) are used together
5. **Verification** ensures symbolic model matches physics

## Common Pitfalls

1. **Sign errors** in dynamics equations
2. **Unit mismatches** (radians vs degrees, etc.)
3. **Linearization points** must be equilibria
4. **Integration methods** (Euler vs RK4) affect accuracy
5. **Model mismatch** between symbolic and PyBullet

## Next Steps

Congratulations! You've completed Stage 1. You now understand:
- Environment structure
- PyBullet physics simulation
- Custom environment creation
- Symbolic dynamics modeling

Ready for safety? Proceed to:
- [Stage 2: Safety Constraints](../stage_2_constraints/) - Add constraints to your environments

## Additional Resources

- [CasADi Documentation](https://web.casadi.org/docs/)
- [symbolic_systems.py source](../../safe_control_gym/math_and_models/symbolic_systems.py)
- [cartpole.py symbolic model](../../safe_control_gym/envs/gym_control/cartpole.py) (search for `_setup_symbolic`)
