"""
Stage 3 Example: Safe PPO Training

This example demonstrates:
1. Training PPO with constraint penalties
2. Monitoring safety during training
3. Evaluating learned policy
4. Optional: Using CBF safety filter with learned policy

Run with:
    python stage_3_safe_ppo_training.py --train
    python stage_3_safe_ppo_training.py --eval
    python stage_3_safe_ppo_training.py --eval --use_cbf
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

from safe_control_gym.utils.registration import make as make_registered, get_config
from safe_control_gym.controllers.ppo.ppo import PPO


def create_safe_env(gui=False,
                    constraints=True,
                    constraint_penalty=10.0,
                    seed=None,
                    **kwargs):
    """Create cartpole environment with safety constraints.

    Args:
        gui: Whether to show PyBullet GUI.
        constraints: Whether to add safety constraints.
        constraint_penalty: Weight for constraint violation penalty.
        seed: RNG seed passed in by vectorized env wrapper (optional).
        **kwargs: Catch-all for any extra arguments.
    """

    constraint_list = []

    if constraints:
        constraint_list = [
            {
                "constraint_form": "bounded_constraint",
                "constrained_variable": "state",
                "active_dims": [0],  # Cart position
                "upper_bounds": [0.9],
                "lower_bounds": [-0.9],
            },
            {
                "constraint_form": "bounded_constraint",
                "constrained_variable": "state",
                "active_dims": [2],  # Pole angle
                "upper_bounds": [0.3],
                "lower_bounds": [-0.3],
            },
        ]

    env = make_registered(
            'cartpole',
            gui=gui,
            constraints=constraint_list,
            done_on_violation=False,      # Don't terminate on violation
            use_constraint_penalty=True,  # Penalty in reward
            constraint_penalty=constraint_penalty,
            ctrl_freq=50,
            episode_len_sec=10,
        )

    # Initialize missing attributes expected by CartPole._get_info
    if not hasattr(env, "out_of_bounds"):
        env.out_of_bounds = False

    # Optional: actually use the seed
    if seed is not None:
        try:
            env.reset(seed=seed)
        except TypeError:
            pass

    return env


def train_safe_ppo(output_dir='./results_safe_ppo'):
    """Train PPO policy with safety constraints."""

    print("=" * 60)
    print("Training Safe PPO Policy")
    print("=" * 60)

    # Create environment factory
    env_func = partial(create_safe_env,
                       gui=False,
                       constraints=True,
                       constraint_penalty=10.0)

    # Load default PPO config from ppo.yaml
    cfg = get_config('ppo')  # this reads safe_control_gym/controllers/ppo/ppo.yaml

    # You can override some defaults here if you like, e.g.:
    # cfg.update({
    #     "rollout_batch_size": 2048,
    #     "num_workers": 4,
    #     "norm_obs": True,
    # })

    # Create PPO controller using YAML config
    ppo = PPO(
        env_func=env_func,
        training=True,
        checkpoint_path='model_latest.pt',
        output_dir=output_dir,
        **cfg,
    )

    print(f"\nTraining configuration:")
    print(f"  Output directory: {output_dir}")
    print(f"  Max env steps: {ppo.max_env_steps}")
    print(f"  Rollout batch size: {ppo.rollout_batch_size}")
    print(f"  Rollout steps per update: {ppo.rollout_steps}")
    print(f"  Number of workers: {ppo.num_workers}")

    # Train
    print(f"\nStarting training...")
    ppo.reset()
    ppo.learn()

    print(f"\nTraining complete!")
    print(f"Model saved to: {output_dir}/model_latest.pt")

    ppo.close()


def evaluate_policy(use_cbf=False,
                    model_path='./results_safe_ppo/model_latest.pt'):
    """Evaluate trained policy.

    Args:
        use_cbf: Whether to use CBF safety filter.
        model_path: Path to trained model.
    """

    print("=" * 60)
    if use_cbf:
        print("Evaluating Policy with CBF Safety Filter")
    else:
        print("Evaluating Learned Policy")
    print("=" * 60)

    # Create environment
    env_func = partial(create_safe_env, gui=True, constraints=True)
    env = env_func()

    # Load same PPO config to build the network correctly
    cfg = get_config('ppo')

    # Load PPO policy (evaluation mode)
    ppo = PPO(
        env_func=env_func,
        training=False,
        checkpoint_path=model_path,
        output_dir='./temp',
        **cfg,
    )

    # Load trained weights
    ppo.load(model_path)

    # Optional: Create CBF safety filter
    safety_filter = None
    if use_cbf:
        try:
            from safe_control_gym.safety_filters.cbf import CBF

            safety_filter = CBF(
                env_func=env_func,
                slope=0.1,  # α parameter
                soft_constrained=True,
                slack_weight=10000.0,
            )
            print("CBF safety filter enabled")
        except Exception as e:
            print(f"Warning: Could not create CBF filter: {e}")
            print("Proceeding without CBF...")
            use_cbf = False

    # Run evaluation episodes
    n_episodes = 5
    episode_rewards = []
    episode_violations = []
    cbf_interventions = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        violation_count = 0
        intervention_count = 0
        step = 0

        print(f"\nEpisode {episode + 1}/{n_episodes}")

        while not done and step < 500:
            # Get action from policy
            action = ppo.select_action(obs)

            # Optional: Filter through CBF
            if use_cbf and safety_filter is not None:
                action_safe = safety_filter.compute_action(obs, action)

                # Check if CBF modified action
                if np.linalg.norm(action_safe - action) > 1e-4:
                    intervention_count += 1

                action = action_safe

            # Execute action
            obs, reward, terminated, info = env.step(action)
            done = terminated

            episode_reward += float(reward)

            # Check constraint violations
            if env.constraints.is_violated(env):
                violation_count += 1

            step += 1

        episode_rewards.append(episode_reward)
        episode_violations.append(violation_count)
        if use_cbf:
            cbf_interventions.append(intervention_count)

        print(f"  Steps: {step}")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Violations: {violation_count} "
              f"({(violation_count / max(step, 1)) * 100:.1f}%)")
        if use_cbf:
            print(f"  CBF interventions: {intervention_count} "
                  f"({(intervention_count / max(step, 1)) * 100:.1f}%)")

    # Summary statistics
    print(f"\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"Average reward: {np.mean(episode_rewards):.2f} ± "
          f"{np.std(episode_rewards):.2f}")
    print(f"Average violations: {np.mean(episode_violations):.1f} ± "
          f"{np.std(episode_violations):.1f}")
    if use_cbf:
        print(f"Average CBF interventions: {np.mean(cbf_interventions):.1f} ± "
              f"{np.std(cbf_interventions):.1f}")

    env.close()
    ppo.close()


def plot_training_progress(log_dir='./results_safe_ppo'):
    """Plot training progress from logs."""

    import os
    import pandas as pd

    # Try to load training logs
    log_file = os.path.join(log_dir, 'logs.csv')

    if not os.path.exists(log_file):
        print(f"Warning: Log file not found at {log_file}")
        return

    # Read logs
    df = pd.read_csv(log_file)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Reward
    ax = axes[0, 0]
    if 'eval/ep_reward_mean' in df.columns:
        ax.plot(df['step'], df['eval/ep_reward_mean'], linewidth=2)
        ax.fill_between(df['step'],
                        df['eval/ep_reward_mean'] -
                        df.get('eval/ep_reward_std', 0),
                        df['eval/ep_reward_mean'] +
                        df.get('eval/ep_reward_std', 0),
                        alpha=0.3)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Episode Reward')
    ax.set_title('Learning Curve')
    ax.grid(True, alpha=0.3)

    # Constraint violations
    ax = axes[0, 1]
    if 'eval/constraint_violation' in df.columns:
        ax.plot(df['step'], df['eval/constraint_violation'],
                linewidth=2, color='red')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Constraint Violations')
    ax.set_title('Safety Performance')
    ax.grid(True, alpha=0.3)

    # Episode length
    ax = axes[1, 0]
    if 'eval/ep_length' in df.columns:
        ax.plot(df['step'], df['eval/ep_length'],
                linewidth=2, color='green')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Episode Length')
    ax.set_title('Episode Duration')
    ax.grid(True, alpha=0.3)

    # Learning stats
    ax = axes[1, 1]
    if 'train/actor_loss' in df.columns:
        ax.plot(df['step'], df['train/actor_loss'],
                label='Actor Loss', linewidth=2)
    if 'train/critic_loss' in df.columns:
        ax.plot(df['step'], df['train/critic_loss'],
                label='Critic Loss', linewidth=2)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Loss')
    ax.set_title('Training Losses')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'training_progress.png'),
                dpi=150, bbox_inches='tight')
    print(f"Training progress plot saved to: {log_dir}/training_progress.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Safe PPO Training Example')
    parser.add_argument('--train', action='store_true',
                        help='Train a new policy')
    parser.add_argument('--eval', action='store_true',
                        help='Evaluate trained policy')
    parser.add_argument('--use_cbf', action='store_true',
                        help='Use CBF safety filter during evaluation')
    parser.add_argument('--plot', action='store_true',
                        help='Plot training progress')
    parser.add_argument('--output_dir', type=str,
                        default='./results_safe_ppo',
                        help='Output directory')
    parser.add_argument('--model_path', type=str,
                        default='./results_safe_ppo/model_latest.pt',
                        help='Path to trained model')

    args = parser.parse_args()

    if args.train:
        train_safe_ppo(output_dir=args.output_dir)

    if args.eval:
        evaluate_policy(use_cbf=args.use_cbf,
                        model_path=args.model_path)

    if args.plot:
        plot_training_progress(log_dir=args.output_dir)

    if not (args.train or args.eval or args.plot):
        print("Please specify --train, --eval, or --plot")
        print("\nExamples:")
        print("  python stage_3_safe_ppo_training.py --train")
        print("  python stage_3_safe_ppo_training.py --eval")
        print("  python stage_3_safe_ppo_training.py --eval --use_cbf")
        print("  python stage_3_safe_ppo_training.py --plot")
