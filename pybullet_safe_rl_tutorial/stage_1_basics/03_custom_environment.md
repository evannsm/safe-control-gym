# Stage 1.3: Creating a Custom Environment

## Introduction

Now that you understand the basics of environments and PyBullet, let's build a custom environment from scratch. We'll create a simple "Pendulum" environment to demonstrate all key concepts.

## Learning Objectives

- Implement a complete BenchmarkEnv subclass
- Define custom state/action spaces
- Implement reward functions
- Handle episode termination
- Add URDF loading and state management

## The Pendulum Environment

We'll create a simple inverted pendulum:
- **State**: [theta, theta_dot] (angle and angular velocity)
- **Action**: torque applied to joint
- **Goal**: Balance upright (theta = 0)

### Step 1: Create the Class

```python
import numpy as np
import pybullet as p
from gymnasium import spaces

from safe_control_gym.envs.benchmark_env import BenchmarkEnv, Task, Cost

class SimplePendulum(BenchmarkEnv):
    """Simple inverted pendulum environment."""

    NAME = 'simple_pendulum'
    URDF_PATH = None  # We'll use a built-in one

    # Define available constraints (we'll add these in Stage 2)
    AVAILABLE_CONSTRAINTS = {}

    def __init__(self,
                 **kwargs):
        """Initialize the pendulum environment."""

        # Call parent constructor
        super().__init__(**kwargs)

        # Physical parameters
        self.MASS = 0.15  # kg
        self.LENGTH = 0.5  # m
        self.GRAVITY_ACC = 9.81  # m/s^2
        self.MAX_TORQUE = 2.0  # N*m

        # Dimensions
        self.state_dim = 2  # [theta, theta_dot]
        self.action_dim = 1  # [torque]

        # Define spaces
        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -8.0]),
            high=np.array([np.pi, 8.0]),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-self.MAX_TORQUE,
            high=self.MAX_TORQUE,
            shape=(1,),
            dtype=np.float32
        )

        # Setup PyBullet simulation
        self._setup_simulation()

    def _setup_simulation(self):
        """Initialize PyBullet physics."""

        # Connect to PyBullet
        if self.GUI:
            self.PYB_CLIENT = p.connect(p.GUI)
        else:
            self.PYB_CLIENT = p.connect(p.DIRECT)

        # Physics parameters
        p.setGravity(0, 0, -self.GRAVITY_ACC, physicsClientId=self.PYB_CLIENT)
        p.setTimeStep(self.PYB_TIMESTEP, physicsClientId=self.PYB_CLIENT)

        # Load ground plane
        import pybullet_data
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.PLANE_ID = p.loadURDF("plane.urdf", physicsClientId=self.PYB_CLIENT)

    def reset(self):
        """Reset the environment to initial state.

        Returns:
            obs (np.array): Initial observation.
            info (dict): Additional information.
        """

        # Reset simulation
        p.resetSimulation(physicsClientId=self.PYB_CLIENT)
        p.setGravity(0, 0, -self.GRAVITY_ACC, physicsClientId=self.PYB_CLIENT)
        p.setTimeStep(self.PYB_TIMESTEP, physicsClientId=self.PYB_CLIENT)

        # Reload ground
        import pybullet_data
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.PLANE_ID = p.loadURDF("plane.urdf", physicsClientId=self.PYB_CLIENT)

        # Create pendulum using URDF
        # For simplicity, we'll use PyBullet's built-in pendulum
        # In practice, you'd load your own URDF file
        self.ROBOT_ID = p.loadURDF(
            "pendulum.urdf",  # Simple pendulum URDF
            basePosition=[0, 0, 1],
            useFixedBase=True,
            physicsClientId=self.PYB_CLIENT
        )

        # Initialize state
        if self.INIT_STATE is not None:
            init_theta = self.INIT_STATE[0]
            init_theta_dot = self.INIT_STATE[1]
        else:
            # Random initialization around upright
            init_theta = np.random.uniform(-0.1, 0.1)
            init_theta_dot = np.random.uniform(-0.05, 0.05)

        # Set initial joint state
        p.resetJointState(
            self.ROBOT_ID, 0,
            targetValue=init_theta,
            targetVelocity=init_theta_dot,
            physicsClientId=self.PYB_CLIENT
        )

        # Disable default motor
        p.setJointMotorControl2(
            self.ROBOT_ID, 0,
            controlMode=p.VELOCITY_CONTROL,
            force=0,
            physicsClientId=self.PYB_CLIENT
        )

        # Reset counters
        self.pyb_step_counter = 0
        self.ctrl_step_counter = 0

        # Get initial observation
        obs = self._get_observation()
        info = {}

        return obs, info

    def step(self, action):
        """Advance the environment by one control step.

        Args:
            action (np.array): Control action (torque).

        Returns:
            obs (np.array): Current observation.
            reward (float): Reward for this step.
            terminated (bool): Whether episode ended due to task completion/failure.
            truncated (bool): Whether episode ended due to time limit.
            info (dict): Additional information.
        """

        # Clip action to valid range
        action = np.clip(action, -self.MAX_TORQUE, self.MAX_TORQUE)

        # Apply torque
        p.setJointMotorControl2(
            self.ROBOT_ID, 0,
            controlMode=p.TORQUE_CONTROL,
            force=float(action[0]),
            physicsClientId=self.PYB_CLIENT
        )

        # Step physics simulation (multiple substeps per control step)
        for _ in range(self.PYB_STEPS_PER_CTRL):
            p.stepSimulation(physicsClientId=self.PYB_CLIENT)
            self.pyb_step_counter += 1

        self.ctrl_step_counter += 1

        # Get observation
        obs = self._get_observation()

        # Compute reward
        reward = self._compute_reward(obs, action)

        # Check termination conditions
        terminated = self._check_termination(obs)
        truncated = self.ctrl_step_counter >= self.CTRL_STEPS

        # Info dict
        info = {
            'ctrl_step': self.ctrl_step_counter,
            'pyb_step': self.pyb_step_counter,
        }

        return obs, reward, terminated, truncated, info

    def _get_observation(self):
        """Extract current state from PyBullet.

        Returns:
            obs (np.array): Current observation [theta, theta_dot].
        """

        # Get joint state
        joint_state = p.getJointState(
            self.ROBOT_ID, 0,
            physicsClientId=self.PYB_CLIENT
        )

        theta = joint_state[0]  # Position
        theta_dot = joint_state[1]  # Velocity

        # Normalize angle to [-pi, pi]
        theta = self._normalize_angle(theta)

        obs = np.array([theta, theta_dot], dtype=np.float32)

        # Update internal state
        self.state = obs.copy()

        return obs

    def _compute_reward(self, obs, action):
        """Compute reward for current state and action.

        Args:
            obs (np.array): Current observation.
            action (np.array): Applied action.

        Returns:
            reward (float): Reward value.
        """

        theta, theta_dot = obs

        if self.TASK == Task.STABILIZATION:
            # Reward for staying upright with minimal velocity
            # Negative cost (will be minimized)
            angle_cost = theta**2
            velocity_cost = 0.1 * theta_dot**2
            action_cost = 0.01 * action[0]**2

            reward = -(angle_cost + velocity_cost + action_cost)

        else:
            # For trajectory tracking, compare to reference
            # (not implemented in this simple example)
            reward = 0.0

        return reward

    def _check_termination(self, obs):
        """Check if episode should terminate.

        Args:
            obs (np.array): Current observation.

        Returns:
            done (bool): Whether episode is done.
        """

        theta, theta_dot = obs

        # Terminate if pendulum falls too far
        if abs(theta) > np.pi / 2:  # More than 90 degrees
            return True

        # Terminate if spinning too fast
        if abs(theta_dot) > 10.0:
            return True

        return False

    def _normalize_angle(self, angle):
        """Normalize angle to [-pi, pi].

        Args:
            angle (float): Angle in radians.

        Returns:
            normalized (float): Angle in [-pi, pi].
        """
        return ((angle + np.pi) % (2 * np.pi)) - np.pi

    def render(self, mode='human'):
        """Render the environment.

        Args:
            mode (str): Rendering mode.
        """

        if mode == 'human' and self.GUI:
            # PyBullet GUI handles rendering automatically
            pass

    def close(self):
        """Clean up resources."""

        if hasattr(self, 'PYB_CLIENT'):
            p.disconnect(physicsClientId=self.PYB_CLIENT)
```

### Step 2: Register the Environment

To use your environment with the `make()` function:

```python
# In a registration file (e.g., __init__.py)
from safe_control_gym.utils.registration import register

register(
    id='simple_pendulum',
    entry_point='path.to.module:SimplePendulum',
)
```

### Step 3: Test Your Environment

```python
from safe_control_gym.utils.registration import make

# Create environment
env = make('simple_pendulum',
           gui=True,
           ctrl_freq=50,
           episode_len_sec=10)

# Test episode
obs, info = env.reset()
print(f"Initial state: {obs}")

for step in range(100):
    # Random action
    action = env.action_space.sample()

    # Step
    obs, reward, terminated, truncated, info = env.step(action)

    print(f"Step {step}: state={obs}, reward={reward:.3f}")

    if terminated or truncated:
        print("Episode ended")
        break

env.close()
```

## Adding Configuration Support

Make your environment configurable:

```python
class SimplePendulum(BenchmarkEnv):
    def __init__(self,
                 mass=0.15,
                 length=0.5,
                 max_torque=2.0,
                 **kwargs):
        """Initialize with configurable parameters.

        Args:
            mass (float): Pendulum mass in kg.
            length (float): Pendulum length in m.
            max_torque (float): Maximum torque in N*m.
        """

        # Store parameters
        self.MASS = mass
        self.LENGTH = length
        self.MAX_TORQUE = max_torque

        # Call parent constructor
        super().__init__(**kwargs)

        # ... rest of initialization
```

With YAML config:

```yaml
# config/my_pendulum.yaml
task: simple_pendulum
mass: 0.2
length: 0.6
max_torque: 3.0
ctrl_freq: 100
episode_len_sec: 15
gui: True
```

## Adding Trajectory Tracking

Extend for trajectory tracking tasks:

```python
def _compute_reward(self, obs, action):
    """Compute reward."""

    theta, theta_dot = obs

    if self.TASK == Task.STABILIZATION:
        # Stabilization reward (as before)
        angle_cost = theta**2
        velocity_cost = 0.1 * theta_dot**2
        action_cost = 0.01 * action[0]**2
        reward = -(angle_cost + velocity_cost + action_cost)

    elif self.TASK == Task.TRAJ_TRACKING:
        # Get reference from trajectory
        ref_theta = self.X_GOAL[self.ctrl_step_counter, 0]
        ref_theta_dot = self.X_GOAL[self.ctrl_step_counter, 1]

        # Tracking error
        theta_error = theta - ref_theta
        theta_dot_error = theta_dot - ref_theta_dot

        # Tracking reward
        tracking_cost = theta_error**2 + 0.1 * theta_dot_error**2
        action_cost = 0.01 * action[0]**2
        reward = -(tracking_cost + action_cost)

    return reward

def _generate_trajectory(self):
    """Generate reference trajectory for tracking tasks."""

    if self.TASK == Task.TRAJ_TRACKING:
        # Sinusoidal trajectory
        time = np.linspace(0, self.EPISODE_LEN_SEC, self.CTRL_STEPS)
        amplitude = 0.5  # radians
        frequency = 0.5  # Hz

        ref_theta = amplitude * np.sin(2 * np.pi * frequency * time)
        ref_theta_dot = amplitude * 2 * np.pi * frequency * np.cos(2 * np.pi * frequency * time)

        self.X_GOAL = np.column_stack([ref_theta, ref_theta_dot])
```

## Practical Exercise: Bouncing Ball Environment

Create a bouncing ball environment:

**Requirements**:
- **State**: [x, y, vx, vy] (position and velocity)
- **Action**: [fx, fy] (force applied to ball)
- **Goal**: Reach target position while bouncing
- **Constraints**: Stay within bounds
- **URDF**: Sphere with appropriate mass

**Hints**:
```python
class BouncingBall(BenchmarkEnv):
    NAME = 'bouncing_ball'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.state_dim = 4  # [x, y, vx, vy]
        self.action_dim = 2  # [fx, fy]
        self.GRAVITY_ACC = 9.81
        self.BALL_RADIUS = 0.1
        self.BALL_MASS = 0.5
        self.MAX_FORCE = 10.0

        # Define spaces
        self.observation_space = spaces.Box(
            low=np.array([-5, 0, -10, -10]),
            high=np.array([5, 5, 10, 10]),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-self.MAX_FORCE,
            high=self.MAX_FORCE,
            shape=(2,),
            dtype=np.float32
        )

    def reset(self):
        # Reset simulation
        p.resetSimulation(physicsClientId=self.PYB_CLIENT)
        p.setGravity(0, 0, -self.GRAVITY_ACC)

        # Load sphere
        collision_shape = p.createCollisionShape(
            p.GEOM_SPHERE,
            radius=self.BALL_RADIUS
        )
        visual_shape = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=self.BALL_RADIUS,
            rgbaColor=[1, 0, 0, 1]
        )

        self.BALL_ID = p.createMultiBody(
            baseMass=self.BALL_MASS,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[0, 0, 2],
            physicsClientId=self.PYB_CLIENT
        )

        # ... implementation

    def step(self, action):
        # Apply force
        p.applyExternalForce(
            self.BALL_ID,
            -1,  # Base link
            forceObj=[action[0], 0, action[1]],
            posObj=[0, 0, 0],
            flags=p.LINK_FRAME,
            physicsClientId=self.PYB_CLIENT
        )

        # ... rest of step
```

## Key Takeaways

1. **Inherit from BenchmarkEnv** for all custom environments
2. **Define state/action spaces** in `observation_space` and `action_space`
3. **Implement reset()** to initialize PyBullet and set initial state
4. **Implement step()** to apply actions and advance simulation
5. **Extract state** using PyBullet query functions
6. **Compute rewards** based on task (stabilization vs tracking)
7. **Handle termination** with both task-specific and time-based conditions

## Common Mistakes

1. **Forgetting to disable default motors** - Always use `VELOCITY_CONTROL` with `force=0`
2. **Not clipping actions** - Can cause instability
3. **Incorrect state extraction** - Match your state definition exactly
4. **Missing normalization** - Normalize angles to [-π, π]
5. **No error handling** - Check for NaN/Inf in states

## Next Steps

- [04_dynamics_modeling.md](04_dynamics_modeling.md) - Add symbolic dynamics with CasADi
- [Stage 2: Safety Constraints](../stage_2_constraints/) - Add safety to your environment

## Additional Resources

- [safe_control_gym.envs.benchmark_env](../../safe_control_gym/envs/benchmark_env.py) - Base class source
- [safe_control_gym.envs.gym_control.cartpole](../../safe_control_gym/envs/gym_control/cartpole.py) - Reference implementation
