"""
Training Script for Dungeon Crawler RL Environment

This script trains tabular RL agents (Q-Learning, SARSA, Expected SARSA)
on the Dungeon Crawler environment with TensorBoard logging.

Usage:
    # Train from scratch on 8×8 grid (fast learning)
    python train.py --algorithm qlearning --episodes 2000 --grid-size 8 --max-steps 100 --run-name ql_8x8

    # Transfer learning: continue training on 16×16 grid (curriculum learning)
    python train.py --algorithm qlearning --episodes 3000 --grid-size 16 --max-steps 200 \
        --load-model models/ql_8x8/final_model.pkl --run-name ql_8x8_to_16x16

    # Train from scratch on 16×16 grid
    python train.py --algorithm qlearning --episodes 5000 --run-name ql_16x16

    # Custom hyperparameters
    python train.py --algorithm sarsa --alpha 0.2 --gamma 0.99 --epsilon-decay 0.997

Features:
- Command-line argument parsing for hyperparameters
- TensorBoard logging of training metrics
- Periodic model checkpointing
- Progress bar with tqdm
- Training statistics tracking
- Curriculum learning support (load pre-trained models)
"""

import argparse
import sys
import os
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from environment.dungeon_env import DungeonCrawlerEnv
from utils.state_encoder import StateEncoder
from agents.base_agent import BaseTabularAgent
from agents.qlearning import QLearningAgent
from agents.sarsa import SARSAAgent
from agents.expected_sarsa import ExpectedSARSAAgent


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train RL agents on Dungeon Crawler environment'
    )

    # Algorithm selection
    parser.add_argument(
        '--algorithm',
        type=str,
        required=True,
        choices=['qlearning', 'sarsa', 'expected_sarsa'],
        help='RL algorithm to use'
    )

    # Training parameters
    parser.add_argument(
        '--episodes',
        type=int,
        default=5000,
        help='Number of training episodes (default: 5000)'
    )

    # Agent hyperparameters
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.1,
        help='Learning rate (default: 0.1)'
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.95,
        help='Discount factor (default: 0.95)'
    )
    parser.add_argument(
        '--epsilon',
        type=float,
        default=1.0,
        help='Initial exploration rate (default: 1.0)'
    )
    parser.add_argument(
        '--epsilon-decay',
        type=float,
        default=0.995,
        help='Epsilon decay rate (default: 0.995)'
    )
    parser.add_argument(
        '--epsilon-min',
        type=float,
        default=0.01,
        help='Minimum epsilon (default: 0.01)'
    )

    # Environment parameters (ULTRA-SIMPLIFIED)
    parser.add_argument(
        '--max-steps',
        type=int,
        default=100,
        help='Maximum steps per episode (default: 100, ultra-simplified for 8×8)'
    )
    parser.add_argument(
        '--grid-size',
        type=int,
        default=8,
        help='Grid size (default: 8, ultra-simplified with global vision)'
    )

    # Logging and checkpointing
    parser.add_argument(
        '--run-name',
        type=str,
        default=None,
        help='Name for this training run (default: algorithm_timestamp)'
    )
    parser.add_argument(
        '--checkpoint-freq',
        type=int,
        default=500,
        help='Save checkpoint every N episodes (default: 500)'
    )
    parser.add_argument(
        '--log-freq',
        type=int,
        default=1,
        help='Log to TensorBoard every N episodes (default: 1)'
    )

    # Other options
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print verbose training information'
    )
    parser.add_argument(
        '--load-model',
        type=str,
        default=None,
        help='Path to pre-trained model to continue training (for curriculum learning)'
    )

    return parser.parse_args()


def create_agent(algorithm: str, n_actions: int, args) -> BaseTabularAgent:
    """
    Create agent based on algorithm name.

    Args:
        algorithm: Algorithm name ('qlearning', 'sarsa', 'expected_sarsa')
        n_actions: Number of actions
        args: Command-line arguments

    Returns:
        Agent instance
    """
    agent_params = {
        'n_actions': n_actions,
        'learning_rate': args.alpha,
        'discount_factor': args.gamma,
        'epsilon': args.epsilon,
        'epsilon_decay': args.epsilon_decay,
        'epsilon_min': args.epsilon_min
    }

    if algorithm == 'qlearning':
        return QLearningAgent(**agent_params)
    elif algorithm == 'sarsa':
        return SARSAAgent(**agent_params)
    elif algorithm == 'expected_sarsa':
        return ExpectedSARSAAgent(**agent_params)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def train_episode(env, agent, encoder, algorithm: str) -> tuple:
    """
    Train agent for one episode.

    Args:
        env: Environment instance
        agent: Agent instance
        encoder: State encoder
        algorithm: Algorithm name (for SARSA special handling)

    Returns:
        tuple: (episode_reward, episode_length, success)
    """
    obs, info = env.reset()
    state = encoder.encode(obs)

    episode_reward = 0
    episode_length = 0
    success = False

    # For SARSA, select initial action
    if algorithm == 'sarsa':
        action = agent.get_action(state, training=True)

    done = False
    while not done:
        # Select action (except for SARSA which already has it)
        if algorithm != 'sarsa':
            action = agent.get_action(state, training=True)

        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        next_state = encoder.encode(obs)
        done = terminated or truncated

        # Select next action (for SARSA and update)
        if algorithm == 'sarsa' and not done:
            next_action = agent.get_action(next_state, training=True)
        else:
            next_action = None

        # Update agent (pass done flag for proper terminal state handling)
        agent.update(state, action, reward, next_state, next_action, done=done)

        # Update for next iteration
        state = next_state
        if algorithm == 'sarsa':
            action = next_action

        episode_reward += reward
        episode_length += 1

    # Check if agent won (SIMPLIFIED - just check if terminated)
    # In simplified version: terminated means reached door
    if terminated:
        success = True

    return episode_reward, episode_length, success


def train(args):
    """
    Main training loop.

    Args:
        args: Parsed command-line arguments
    """
    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)

    # Create run name
    if args.run_name is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_name = f"{args.algorithm}_{timestamp}"
    else:
        run_name = args.run_name

    print("=" * 70)
    print("DUNGEON CRAWLER RL TRAINING")
    print("=" * 70)
    print(f"Algorithm:       {args.algorithm.upper()}")
    print(f"Episodes:        {args.episodes:,}")
    print(f"Learning rate:   {args.alpha}")
    print(f"Discount factor: {args.gamma}")
    print(f"Epsilon:         {args.epsilon} → {args.epsilon_min} (decay: {args.epsilon_decay})")
    print(f"Run name:        {run_name}")
    print("=" * 70)
    print()

    # Create environment and encoder (SIMPLIFIED - no max_enemies)
    env = DungeonCrawlerEnv(
        max_steps=args.max_steps,
        grid_size=args.grid_size
    )
    encoder = StateEncoder(grid_size=args.grid_size)

    print(f"Environment:     {env}")
    print(f"Action space:    {env.action_space}")
    print(f"State space:     {encoder.get_state_space_size():,} states")
    print()

    # Create agent
    agent = create_agent(args.algorithm, env.action_space.n, args)

    # Load pre-trained model if specified (for curriculum learning)
    if args.load_model:
        print(f"Loading pre-trained model from: {args.load_model}")
        try:
            agent.load(args.load_model)
            print(f"✓ Model loaded successfully")
            print(f"  Q-table size:    {len(agent.q_table):,} states")
            print(f"  States visited:  {len(agent.states_visited):,}")
            print(f"  Current epsilon: {agent.epsilon:.4f}")
            print(f"  Continuing training with loaded knowledge...")
        except Exception as e:
            print(f"Error loading model: {e}")
            print(f"Starting training from scratch...")

    print(f"Agent:           {agent}")
    print()

    # Create directories for logs and models
    log_dir = Path("logs") / run_name
    model_dir = Path("models") / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard writer
    writer = SummaryWriter(log_dir=str(log_dir))

    # Training statistics
    episode_rewards = []
    episode_lengths = []
    successes = []

    # Training loop with progress bar
    print("Starting training...")
    print()

    try:
        for episode in tqdm(range(args.episodes), desc="Training"):
            # Train one episode
            episode_reward, episode_length, success = train_episode(
                env, agent, encoder, args.algorithm
            )

            # Store statistics
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            successes.append(int(success))

            # Decay epsilon
            agent.decay_epsilon()

            # Log to TensorBoard
            if (episode + 1) % args.log_freq == 0:
                writer.add_scalar('Episode/Reward', episode_reward, episode + 1)
                writer.add_scalar('Episode/Length', episode_length, episode + 1)
                writer.add_scalar('Episode/Success', int(success), episode + 1)
                writer.add_scalar('Episode/Epsilon', agent.epsilon, episode + 1)

                # Compute moving averages (last 100 episodes)
                if episode >= 99:
                    avg_reward_100 = np.mean(episode_rewards[-100:])
                    avg_length_100 = np.mean(episode_lengths[-100:])
                    success_rate_100 = np.mean(successes[-100:])

                    writer.add_scalar('Episode/Avg_Reward_100', avg_reward_100, episode + 1)
                    writer.add_scalar('Episode/Avg_Length_100', avg_length_100, episode + 1)
                    writer.add_scalar('Episode/Success_Rate_100', success_rate_100, episode + 1)

                # Q-table statistics
                q_stats = agent.get_q_table_stats()
                writer.add_scalar('QTable/Mean_Q', q_stats['mean_q'], episode + 1)
                writer.add_scalar('QTable/Max_Q', q_stats['max_q'], episode + 1)
                writer.add_scalar('QTable/Min_Q', q_stats['min_q'], episode + 1)
                writer.add_scalar('QTable/Num_States', q_stats['num_states'], episode + 1)

            # Checkpoint saving
            if (episode + 1) % args.checkpoint_freq == 0:
                checkpoint_path = model_dir / f"checkpoint_ep{episode + 1}.pkl"
                agent.save(str(checkpoint_path))

                if args.verbose:
                    avg_reward = np.mean(episode_rewards[-100:]) if episode >= 99 else np.mean(episode_rewards)
                    success_rate = np.mean(successes[-100:]) if episode >= 99 else np.mean(successes)
                    print(f"\n[Episode {episode + 1}] "
                          f"Avg Reward: {avg_reward:.2f}, "
                          f"Success Rate: {success_rate:.2%}, "
                          f"Epsilon: {agent.epsilon:.4f}")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")

    # Save final model
    final_model_path = model_dir / "final_model.pkl"
    agent.save(str(final_model_path))

    # Print final statistics
    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total episodes:      {len(episode_rewards)}")
    print(f"Average reward:      {np.mean(episode_rewards):.2f}")
    print(f"Best reward:         {np.max(episode_rewards):.2f}")
    print(f"Average length:      {np.mean(episode_lengths):.1f}")
    print(f"Success rate:        {np.mean(successes):.2%}")

    if len(episode_rewards) >= 100:
        print(f"\nLast 100 episodes:")
        print(f"  Average reward:    {np.mean(episode_rewards[-100:]):.2f}")
        print(f"  Success rate:      {np.mean(successes[-100:]):.2%}")

    print(f"\nAgent statistics:")
    print(f"  States visited:    {len(agent.states_visited):,}")
    print(f"  Q-table size:      {len(agent.q_table):,}")
    print(f"  Final epsilon:     {agent.epsilon:.4f}")

    print(f"\nModel saved to:      {final_model_path}")
    print(f"Logs saved to:       {log_dir}")
    print("=" * 70)

    # Close writer
    writer.close()
    env.close()


if __name__ == "__main__":
    args = parse_args()
    train(args)
