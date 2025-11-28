"""
Stage 1 Example: Basic Environment Usage

This example demonstrates:
1. Creating an environment
2. Running a random policy
3. Extracting and visualizing state information
4. Understanding the simulation loop

Run with:
    python stage_1_basic_env.py
"""

import numpy as np
import matplotlib.pyplot as plt
from safe_control_gym.utils.registration import make


def basic_environment_demo():
    """Demonstrate basic environment creation and usage."""

    print("=" * 60)
    print("Stage 1: Basic Environment Demo")
    print("=" * 60)

    # Create cartpole environment
    env = make(
        'cartpole',
        gui=True,  # Show visualization
        verbose=True,
        ctrl_freq=50,  # 50 Hz control
        pyb_freq=1000,  # 1000 Hz physics
        episode_len_sec=10  # 10 second episodes
    )

    print(f"\nEnvironment Information:")
    print(f"  State dimension: {env.state_dim}")
    print(f"  Action dimension: {env.action_dim}")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print(f"  Control frequency: {env.CTRL_FREQ} Hz")
    print(f"  PyBullet frequency: {env.PYB_FREQ} Hz")
    print(f"  Max episode steps: {env.CTRL_STEPS}")

    # Storage for trajectory
    states = []
    actions = []
    rewards = []

    # Run episode
    print(f"\nRunning episode...")
    obs, info = env.reset()
    print(f"Initial state: {obs}")

    done = False
    step = 0

    while not done and step < 200:
        # Random action
        action = env.action_space.sample()

        # Step environment
        obs, reward, terminated, info = env.step(action)
        done = terminated

        # Store data
        states.append(obs.copy())
        actions.append(action.copy())
        rewards.append(reward)

        # Print occasionally
        if step % 20 == 0:
            print(f"Step {step:3d}: x={obs[0]:6.3f}, θ={obs[2]:6.3f}, "
                  f"reward={reward:7.3f}")

        step += 1

    print(f"\nEpisode finished at step {step}")
    print(f"Total reward: {sum(rewards):.2f}")

    env.close()

    # Visualize trajectory
    visualize_trajectory(states, actions, rewards)


def visualize_trajectory(states, actions, rewards):
    """Plot the trajectory data."""

    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)

    time = np.arange(len(states)) * 0.02  # 50 Hz = 0.02s timestep

    fig, axes = plt.subplots(3, 1, figsize=(10, 8))

    # State plots
    ax = axes[0]
    ax.plot(time, states[:, 0], label='x (cart position)', linewidth=2)
    ax.plot(time, states[:, 2], label='θ (pole angle)', linewidth=2)
    ax.set_ylabel('State')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Cartpole Trajectory - Random Policy')

    # Action plot
    ax = axes[1]
    ax.plot(time, actions, label='Force', linewidth=2, color='green')
    ax.set_ylabel('Action (N)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Reward plot
    ax = axes[2]
    ax.plot(time, rewards, label='Reward', linewidth=2, color='orange')
    ax.plot(time, np.cumsum(rewards), label='Cumulative Reward',
            linewidth=2, color='red', linestyle='--')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('stage_1_trajectory.png', dpi=150, bbox_inches='tight')
    print(f"\nTrajectory plot saved to: stage_1_trajectory.png")
    plt.show()


def frequency_comparison():
    """Compare different simulation frequencies."""

    print("\n" + "=" * 60)
    print("Frequency Comparison")
    print("=" * 60)

    import time

    configs = [
        {'pyb_freq': 100, 'ctrl_freq': 50},
        {'pyb_freq': 500, 'ctrl_freq': 50},
        {'pyb_freq': 1000, 'ctrl_freq': 50},
    ]

    for config in configs:
        env = make('cartpole', gui=False, **config)

        # Time 100 steps
        start_time = time.time()
        obs, _ = env.reset()

        for _ in range(100):
            action = env.action_space.sample()
            env.step(action)

        elapsed = time.time() - start_time

        print(f"\nPyBullet freq: {config['pyb_freq']} Hz")
        print(f"  Time for 100 steps: {elapsed:.3f} sec")
        print(f"  Simulated time: {100 * env.CTRL_TIMESTEP:.1f} sec")
        print(f"  Speed-up: {100 * env.CTRL_TIMESTEP / elapsed:.1f}x")

        env.close()


if __name__ == "__main__":
    # Run basic demo
    basic_environment_demo()

    # Compare frequencies
    frequency_comparison()

    print("\n" + "=" * 60)
    print("Stage 1 Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  - Try modifying the control frequency")
    print("  - Implement a simple PD controller instead of random actions")
    print("  - Proceed to Stage 2: Adding safety constraints")
