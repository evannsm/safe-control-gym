# Stage 1.2: PyBullet Basics

## Introduction

PyBullet is a physics simulation engine that powers the safe-control-gym environments. Understanding how PyBullet works under the hood will help you create custom environments and debug physics issues.

## Learning Objectives

- Understand PyBullet's client-server architecture
- Learn to load and manipulate URDF models
- Master physics simulation parameters
- Know how to extract state information
- Understand collision detection and contacts

## PyBullet Architecture

### Client-Server Model

PyBullet uses a client-server architecture:

```python
import pybullet as p
import pybullet_data

# Start physics server
# GUI mode
client_id = p.connect(p.GUI)

# Or headless mode (faster)
# client_id = p.connect(p.DIRECT)

# Set gravity
p.setGravity(0, 0, -9.8)

# Load ground plane
p.setAdditionalSearchPath(pybullet_data.getDataPath())
plane_id = p.loadURDF("plane.urdf")

# When done
p.disconnect()
```

**Key concepts**:
- **Client ID**: Handle to physics server (important for multi-env)
- **Body ID**: Unique identifier for each loaded object
- **Link Index**: Index for multi-link robots (joints)

### In safe-control-gym

```python
# Environment handles PyBullet connection
class BenchmarkEnv:
    def _setup_simulation(self):
        # Connect to PyBullet
        if self.GUI:
            self.PYB_CLIENT = p.connect(p.GUI)
        else:
            self.PYB_CLIENT = p.connect(p.DIRECT)

        # Configure physics
        p.setGravity(0, 0, -self.GRAVITY_ACC, physicsClientId=self.PYB_CLIENT)
        p.setTimeStep(self.PYB_TIMESTEP, physicsClientId=self.PYB_CLIENT)
```

## URDF Models

### What is URDF?

URDF (Unified Robot Description Format) defines robot structure:
- Links: rigid bodies
- Joints: connections between links
- Visual: how it looks
- Collision: for physics
- Inertial: mass, center of mass, inertia tensor

### Example URDF (Cartpole)

```xml
<?xml version="1.0"?>
<robot name="cartpole">
    <!-- Base link (cart) -->
    <link name="cart">
        <inertial>
            <mass value="1.0"/>
            <inertia ixx="0.01" ixy="0" ixz="0"
                     iyy="0.01" iyz="0" izz="0.01"/>
        </inertial>
        <visual>
            <geometry>
                <box size="0.3 0.2 0.1"/>
            </geometry>
            <material name="blue">
                <color rgba="0 0 1 1"/>
            </material>
        </visual>
        <collision>
            <geometry>
                <box size="0.3 0.2 0.1"/>
            </geometry>
        </collision>
    </link>

    <!-- Pole link -->
    <link name="pole">
        <inertial>
            <origin xyz="0 0 0.5"/>
            <mass value="0.1"/>
            <inertia ixx="0.001" ixy="0" ixz="0"
                     iyy="0.001" iyz="0" izz="0.001"/>
        </inertial>
        <visual>
            <origin xyz="0 0 0.5"/>
            <geometry>
                <cylinder radius="0.02" length="1.0"/>
            </geometry>
            <material name="red">
                <color rgba="1 0 0 1"/>
            </material>
        </visual>
        <collision>
            <origin xyz="0 0 0.5"/>
            <geometry>
                <cylinder radius="0.02" length="1.0"/>
            </geometry>
        </collision>
    </link>

    <!-- Joint connecting cart and pole -->
    <joint name="cart_to_pole" type="continuous">
        <parent link="cart"/>
        <child link="pole"/>
        <origin xyz="0 0 0"/>
        <axis xyz="0 1 0"/>  <!-- Rotation axis -->
    </joint>

    <!-- Prismatic joint for cart movement -->
    <joint name="slider" type="prismatic">
        <parent link="world"/>
        <child link="cart"/>
        <axis xyz="1 0 0"/>  <!-- Translation axis -->
        <limit effort="100" lower="-10" upper="10" velocity="10"/>
    </joint>
</robot>
```

### Loading URDF in PyBullet

```python
import pybullet as p

# Load URDF
robot_id = p.loadURDF(
    "cartpole.urdf",
    basePosition=[0, 0, 0.1],
    baseOrientation=[0, 0, 0, 1],  # Quaternion
    useFixedBase=False,  # Can move
    flags=p.URDF_USE_INERTIA_FROM_FILE
)

# Get information
num_joints = p.getNumJoints(robot_id)
print(f"Robot has {num_joints} joints")

# Inspect joints
for i in range(num_joints):
    info = p.getJointInfo(robot_id, i)
    print(f"Joint {i}: {info[1].decode('utf-8')}")  # Joint name
    print(f"  Type: {info[2]}")  # Joint type
    print(f"  Limits: {info[8]} to {info[9]}")  # Position limits
```

## Physics Simulation

### Time Stepping

PyBullet advances simulation in discrete time steps:

```python
# Set time step (typically 1/240 to 1/1000 seconds)
p.setTimeStep(1./1000)

# Step simulation
p.stepSimulation()  # Advances by one time step

# For safe-control-gym environments
# Each env.step() calls p.stepSimulation() multiple times:
for _ in range(self.PYB_STEPS_PER_CTRL):
    p.stepSimulation()
```

### Physics Parameters

```python
# Gravity
p.setGravity(0, 0, -9.81)

# Simulation parameters
p.setPhysicsEngineParameter(
    fixedTimeStep=1./1000,
    numSubSteps=1,           # Subdivide timestep
    numSolverIterations=50,  # Constraint solver iterations
    enableConeFriction=True,
    solverResidualThreshold=1e-7
)

# Real-time simulation (optional, for visualization)
p.setRealTimeSimulation(0)  # 0=disabled (manual stepping)
```

### Damping and Friction

```python
# Joint damping (energy dissipation)
p.changeDynamics(
    robot_id,
    joint_index,
    linearDamping=0.04,
    angularDamping=0.04,
    jointDamping=0.0
)

# Friction
p.changeDynamics(
    robot_id,
    link_index,
    lateralFriction=1.0,  # Coulomb friction
    spinningFriction=0.1,
    rollingFriction=0.01
)
```

## State Extraction

### Joint States

```python
# Get joint state (position, velocity, forces, torque)
joint_state = p.getJointState(robot_id, joint_index)

position = joint_state[0]
velocity = joint_state[1]
reaction_forces = joint_state[2]  # 6D: force + torque
applied_torque = joint_state[3]

# Get multiple joints at once
joint_states = p.getJointStates(robot_id, range(num_joints))
```

### Base State (for floating bodies)

```python
# Get base position and orientation
pos, orn = p.getBasePositionAndOrientation(robot_id)
# pos: [x, y, z]
# orn: [qx, qy, qz, qw] (quaternion)

# Get base velocity
lin_vel, ang_vel = p.getBaseVelocity(robot_id)
# lin_vel: [vx, vy, vz]
# ang_vel: [wx, wy, wz]

# Convert quaternion to Euler angles
euler = p.getEulerFromQuaternion(orn)
# euler: [roll, pitch, yaw]
```

### Example: Cartpole State

```python
def get_cartpole_state(robot_id):
    # Cart position and velocity (prismatic joint)
    cart_state = p.getJointState(robot_id, joint_index=0)
    x = cart_state[0]
    x_dot = cart_state[1]

    # Pole angle and angular velocity (revolute joint)
    pole_state = p.getJointState(robot_id, joint_index=1)
    theta = pole_state[0]
    theta_dot = pole_state[1]

    return np.array([x, x_dot, theta, theta_dot])
```

## Applying Forces and Torques

### Direct Force Control

```python
# Apply force to a joint
p.setJointMotorControl2(
    robot_id,
    joint_index,
    controlMode=p.TORQUE_CONTROL,
    force=10.0  # Newton-meters for revolute, Newtons for prismatic
)

# Disable default motor (important!)
p.setJointMotorControl2(
    robot_id,
    joint_index,
    controlMode=p.VELOCITY_CONTROL,
    force=0  # Disable built-in motor
)
```

### Position Control

```python
# PD position control (built-in)
p.setJointMotorControl2(
    robot_id,
    joint_index,
    controlMode=p.POSITION_CONTROL,
    targetPosition=0.5,
    force=100,  # Max force
    positionGain=0.1,  # Kp
    velocityGain=1.0   # Kd
)
```

### Velocity Control

```python
p.setJointMotorControl2(
    robot_id,
    joint_index,
    controlMode=p.VELOCITY_CONTROL,
    targetVelocity=2.0,
    force=100
)
```

## Visualization and Debugging

### Camera Control

```python
# Set camera view
p.resetDebugVisualizerCamera(
    cameraDistance=2.0,
    cameraYaw=45,
    cameraPitch=-30,
    cameraTargetPosition=[0, 0, 0]
)

# Disable GUI elements
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
```

### Debug Drawing

```python
# Draw lines
p.addUserDebugLine(
    lineFromXYZ=[0, 0, 0],
    lineToXYZ=[1, 0, 0],
    lineColorRGB=[1, 0, 0],
    lineWidth=2,
    lifeTime=0  # 0 = permanent
)

# Draw text
p.addUserDebugText(
    text="Target",
    textPosition=[0, 0, 1],
    textColorRGB=[0, 1, 0],
    textSize=1.5
)
```

## Collision Detection

```python
# Check contacts
contact_points = p.getContactPoints(
    bodyA=robot_id,
    bodyB=ground_id
)

for contact in contact_points:
    contact_force = contact[9]  # Normal force
    position = contact[5]  # Contact position on A
    normal = contact[7]  # Contact normal

    if contact_force > 0:
        print(f"Collision at {position} with force {contact_force}N")
```

## Practical Example: Manual Cartpole Simulation

```python
import pybullet as p
import pybullet_data
import time
import numpy as np

def manual_cartpole_sim():
    # Connect to PyBullet
    client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Setup physics
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1./240)

    # Load ground and robot
    plane_id = p.loadURDF("plane.urdf")
    robot_id = p.loadURDF(
        "../../../safe_control_gym/envs/gym_control/assets/cartpole.urdf",
        basePosition=[0, 0, 0.05]
    )

    # Disable default motors
    for joint_idx in range(p.getNumJoints(robot_id)):
        p.setJointMotorControl2(
            robot_id, joint_idx,
            controlMode=p.VELOCITY_CONTROL,
            force=0
        )

    # Camera
    p.resetDebugVisualizerCamera(
        cameraDistance=2.5,
        cameraYaw=0,
        cameraPitch=-10,
        cameraTargetPosition=[0, 0, 0.5]
    )

    # Simulation loop
    for step in range(1000):
        # Get current state
        cart_state = p.getJointState(robot_id, 0)
        pole_state = p.getJointState(robot_id, 1)

        x = cart_state[0]
        x_dot = cart_state[1]
        theta = pole_state[0]
        theta_dot = pole_state[1]

        # Simple PD controller to balance
        force = -10.0 * theta - 5.0 * theta_dot - 1.0 * x - 2.0 * x_dot

        # Apply force
        p.setJointMotorControl2(
            robot_id, 0,
            controlMode=p.TORQUE_CONTROL,
            force=force
        )

        # Step simulation
        p.stepSimulation()
        time.sleep(1./240)

        # Print state every 60 steps
        if step % 60 == 0:
            print(f"Step {step}: x={x:.3f}, θ={theta:.3f}, "
                  f"ẋ={x_dot:.3f}, θ̇={theta_dot:.3f}")

    p.disconnect()

if __name__ == "__main__":
    manual_cartpole_sim()
```

## PyBullet in safe-control-gym

### How Environments Use PyBullet

```python
class Cartpole(BenchmarkEnv):
    def reset(self):
        # Reset PyBullet simulation
        p.resetSimulation(physicsClientId=self.PYB_CLIENT)
        p.setGravity(0, 0, -self.GRAVITY_ACC)

        # Reload URDF
        self.ROBOT_ID = p.loadURDF(
            self.URDF_PATH,
            basePosition=[0, 0, 0.05],
            physicsClientId=self.PYB_CLIENT
        )

        # Set initial state
        p.resetJointState(self.ROBOT_ID, 0, self.INIT_X, self.INIT_X_DOT)
        p.resetJointState(self.ROBOT_ID, 1, self.INIT_THETA, self.INIT_THETA_DOT)

    def step(self, action):
        # Apply action
        p.setJointMotorControl2(
            self.ROBOT_ID, 0,
            controlMode=p.TORQUE_CONTROL,
            force=action[0]
        )

        # Step physics (multiple substeps)
        for _ in range(self.PYB_STEPS_PER_CTRL):
            p.stepSimulation(physicsClientId=self.PYB_CLIENT)

        # Get observation
        obs = self._get_observation()
        reward = self._compute_reward()
        done = self._check_done()

        return obs, reward, done, {}
```

## Key Takeaways

1. **PyBullet** uses a client-server model with unique IDs for everything
2. **URDF files** define robot structure, mass, inertia, and appearance
3. **Time stepping** must match your control requirements
4. **State extraction** uses `getJointState()` and `getBasePositionAndOrientation()`
5. **Control modes**: TORQUE_CONTROL, POSITION_CONTROL, VELOCITY_CONTROL
6. **Physics parameters** (damping, friction) significantly affect behavior

## Common Issues

1. **Robot falls through floor**: Check collision geometries in URDF
2. **Unstable simulation**: Reduce time step or increase solver iterations
3. **Sluggish response**: Disable default motors with `force=0`
4. **Memory leaks**: Always call `p.disconnect()` when done

## Next Steps

- [03_custom_environment.md](03_custom_environment.md) - Build a custom environment
- [04_dynamics_modeling.md](04_dynamics_modeling.md) - Add symbolic dynamics

## Additional Resources

- [PyBullet Quickstart Guide](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA)
- [PyBullet GitHub Examples](https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet/examples)
- [URDF Tutorial](http://wiki.ros.org/urdf/Tutorials)
