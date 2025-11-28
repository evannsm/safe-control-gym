"""
Stage 2 Example: Environment with Safety Constraints

This example demonstrates:
1. Adding state and input constraints
2. Monitoring constraint violations
3. Using constraint penalties in rewards
4. Visualizing safe regions

Run with:
    python stage_2_constrained_env.py
"""

import numpy as np
import matplotlib.pyplot as plt
from safe_control_gym.utils.registration import make


def constrained_environment_demo():
    """Demonstrate environment with safety constraints."""

    print("=" * 60)
    print("Stage 2: Constrained Environment Demo")
    print("=" * 60)

    # Define constraints
    constraints = [
        {
            'constraint_type': 'BoundedConstraint',
            'constrained_variable': 'state',
            'active_dims': [0],  # Cart position
            'upper_bounds': [0.8],
            'lower_bounds': [-0.8],
            'strict': False
        },
        {
            'constraint_type': 'BoundedConstraint',
            'constrained_variable': 'state',
            'active_dims': [2],  # Pole angle
            'upper_bounds': [0.25],  # ~14 degrees
            'lower_bounds': [-0.25],
            'strict': False
        },
        {
            'constraint_type': 'BoundedConstraint',
            'constrained_variable': 'input',
            'upper_bounds': [15.0],  # Force limit
            'lower_bounds': [-15.0],
            'strict': False
        }
    ]

    # Create environment with constraints
    env = make(
        'cartpole',
        gui=True,
        constraints=constraints,
        done_on_violation=False,  # Don't terminate on violation
        use_constraint_penalty=True,  # Penalize violations in reward
        constraint_penalty=10.0,  # Penalty weight
        verbose=True
    )

    print(f"\nEnvironment created with {len(env.constraints.constraints)} constraints:")
    for i, constraint in enumerate(env.constraints.constraints):
        print(f"  Constraint {i+1}: {constraint.constrained_variable.value}")

    # Storage
    states = []
    actions = []
    rewards = []
    violations = []
    constraint_values_list = []

    # Run episode
    print(f"\nRunning episode with random policy...")
    obs, info = env.reset()

    done = False
    step = 0
    violation_count = 0

    while not done and step < 300:
        # Random action
        action = env.action_space.sample()

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Check constraint violations
        constraint_values = env.constraints.get_value(env)
        violated = env.constraints.is_violated(env)

        if violated:
            violation_count += 1

        # Store data
        states.append(obs.copy())
        actions.append(action.copy())
        rewards.append(reward)
        violations.append(violated)
        constraint_values_list.append(constraint_values.copy())

        # Print violations
        if violated and step % 10 == 0:
            print(f"Step {step:3d}: VIOLATION! "
                  f"x={obs[0]:6.3f}, θ={obs[2]:6.3f}")

        step += 1

    print(f"\nEpisode finished at step {step}")
    print(f"Total violations: {violation_count} ({violation_count/step*100:.1f}%)")
    print(f"Total reward: {sum(rewards):.2f}")

    env.close()

    # Visualize
    visualize_constrained_trajectory(
        states, actions, rewards, violations, constraint_values_list
    )


def visualize_constrained_trajectory(states, actions, rewards, violations,
                                     constraint_values_list):
    """Plot trajectory with constraint violations highlighted."""

    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    violations = np.array(violations)
    constraint_values = np.array(constraint_values_list)

    time = np.arange(len(states)) * 0.02

    fig, axes = plt.subplots(4, 1, figsize=(12, 10))

    # Position plot with constraints
    ax = axes[0]
    ax.plot(time, states[:, 0], label='x (cart position)', linewidth=2)
    ax.axhline(0.8, color='r', linestyle='--', label='Position limits', linewidth=1.5)
    ax.axhline(-0.8, color='r', linestyle='--', linewidth=1.5)

    # Highlight violations
    violation_times = time[violations]
    if len(violation_times) > 0:
        ax.scatter(violation_times, states[violations, 0],
                  color='red', s=50, zorder=5, alpha=0.6, label='Violations')

    ax.set_ylabel('Position (m)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_title('Cartpole with Safety Constraints')

    # Angle plot with constraints
    ax = axes[1]
    ax.plot(time, states[:, 2], label='θ (pole angle)', linewidth=2, color='orange')
    ax.axhline(0.25, color='r', linestyle='--', label='Angle limits', linewidth=1.5)
    ax.axhline(-0.25, color='r', linestyle='--', linewidth=1.5)

    # Highlight violations
    if len(violation_times) > 0:
        ax.scatter(violation_times, states[violations, 2],
                  color='red', s=50, zorder=5, alpha=0.6, label='Violations')

    ax.set_ylabel('Angle (rad)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Action plot with constraints
    ax = axes[2]
    ax.plot(time, actions, label='Force', linewidth=2, color='green')
    ax.axhline(15.0, color='r', linestyle='--', label='Force limits', linewidth=1.5)
    ax.axhline(-15.0, color='r', linestyle='--', linewidth=1.5)
    ax.set_ylabel('Action (N)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Constraint values
    ax = axes[3]

    # Plot each constraint
    for i in range(constraint_values.shape[1]):
        ax.plot(time, constraint_values[:, i], alpha=0.6, linewidth=1)

    ax.axhline(0, color='red', linestyle='-', linewidth=2, label='Safety threshold')
    ax.fill_between(time, -1, 0, alpha=0.2, color='green', label='Safe region')
    ax.fill_between(time, 0, 10, alpha=0.2, color='red', label='Unsafe region')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Constraint Values')
    ax.set_ylim([-1, 1])
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_title('Constraint Values (g ≤ 0 is safe)')

    plt.tight_layout()
    plt.savefig('stage_2_constrained_trajectory.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: stage_2_constrained_trajectory.png")
    plt.show()


def visualize_safe_region():
    """Visualize the safe region in state space."""

    print("\n" + "=" * 60)
    print("Visualizing Safe Region")
    print("=" * 60)

    # Create grid
    x = np.linspace(-1.5, 1.5, 100)
    theta = np.linspace(-0.5, 0.5, 100)
    X, Theta = np.meshgrid(x, theta)

    # Constraints
    x_max = 0.8
    theta_max = 0.25

    # Safe region indicator
    safe = (np.abs(X) <= x_max) & (np.abs(Theta) <= theta_max)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Safe region
    ax.contourf(X, Theta, safe.astype(float), levels=[0, 0.5, 1],
                colors=['lightcoral', 'lightgreen'], alpha=0.5)

    # Constraint boundaries
    ax.axvline(x_max, color='red', linestyle='--', linewidth=2, label='Position limits')
    ax.axvline(-x_max, color='red', linestyle='--', linewidth=2)
    ax.axhline(theta_max, color='blue', linestyle='--', linewidth=2, label='Angle limits')
    ax.axhline(-theta_max, color='blue', linestyle='--', linewidth=2)

    # Target (upright, centered)
    ax.scatter([0], [0], color='gold', s=200, marker='*',
               edgecolors='black', linewidths=2, zorder=5, label='Target')

    ax.set_xlabel('Cart Position x (m)', fontsize=12)
    ax.set_ylabel('Pole Angle θ (rad)', fontsize=12)
    ax.set_title('Safe Region for Cartpole', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('stage_2_safe_region.png', dpi=150, bbox_inches='tight')
    print(f"Safe region plot saved to: stage_2_safe_region.png")
    plt.show()


if __name__ == "__main__":
    # Run constrained environment demo
    constrained_environment_demo()

    # Visualize safe region
    visualize_safe_region()

    print("\n" + "=" * 60)
    print("Stage 2 Complete!")
    print("=" * 60)
    print("\nKey observations:")
    print("  - Constraints define safe regions in state/action space")
    print("  - Random policy frequently violates constraints")
    print("  - Penalties in reward encourage safety (but don't guarantee it)")
    print("\nNext steps:")
    print("  - Proceed to Stage 3: Train RL policy that respects constraints")
    print("  - Learn about safety filters (CBF) for hard guarantees")
