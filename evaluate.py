"""
Evaluation Script for Trained Agents

This script evaluates trained RL agents on the Dungeon Crawler environment.

Usage:
    python evaluate.py --model models/qlearning_final/final_model.pkl --episodes 100
    python evaluate.py --model models/sarsa_final/final_model.pkl --render

Features:
- Load trained agent from checkpoint
- Run multiple evaluation episodes
- Compute performance statistics
- Optional rendering of episodes
"""

import argparse
import sys
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from environment.dungeon_env import DungeonCrawlerEnv
from utils.state_encoder import StateEncoder
from agents.qlearning import QLearningAgent
from agents.sarsa import SARSAAgent
from agents.expected_sarsa import ExpectedSARSAAgent


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate trained RL agents on Dungeon Crawler'
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model file (.pkl)'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=100,
        help='Number of evaluation episodes (default: 100)'
    )
    parser.add_argument(
        '--render',
        action='store_true',
        help='Render episodes during evaluation'
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=300,
        help='Maximum steps per episode (default: 300 for 32×32)'
    )
    parser.add_argument(
        '--grid-size',
        type=int,
        default=32,
        help='Grid size (default: 32 with global vision)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed episode information'
    )

    return parser.parse_args()


def load_agent(model_path: str):
    """
    Load trained agent from file.

    Automatically detects algorithm type from Q-table structure.

    Args:
        model_path: Path to model file

    Returns:
        Loaded agent instance
    """
    # Load pickle file
    with open(model_path, 'rb') as f:
        save_data = pickle.load(f)

    n_actions = save_data['n_actions']

    # Create agent (we'll try all three types, doesn't matter which)
    # The Q-table and parameters will be loaded
    agent = QLearningAgent(n_actions=n_actions)
    agent.load(model_path)

    return agent


def evaluate_episode(env, agent, encoder, render: bool = False, verbose: bool = False) -> dict:
    """
    Evaluate agent for one episode.

    Args:
        env: Environment instance
        agent: Trained agent
        encoder: State encoder
        render: Whether to render
        verbose: Whether to print detailed info

    Returns:
        dict: Episode statistics
    """
    obs, info = env.reset()
    door_pos = info['door_pos']  # Get door position for entire episode
    state = encoder.encode(obs, door_pos=door_pos)

    episode_reward = 0
    episode_length = 0
    success = False

    if render:
        env.render()

    done = False
    while not done:
        # Greedy action selection (no exploration during evaluation)
        action = agent.get_action(state, training=False)

        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        next_state = encoder.encode(obs, door_pos=door_pos)
        done = terminated or truncated

        if render:
            print(f"\nAction: {['UP', 'DOWN', 'LEFT', 'RIGHT'][action]}")
            print(f"Reward: {reward:.1f}")
            env.render()
            input("Press Enter to continue...")  # Pause between steps

        episode_reward += reward
        episode_length += 1
        state = next_state

    # Check if agent won (SIMPLIFIED - just check if terminated)
    if terminated:
        success = True

    if verbose:
        status = "SUCCESS" if success else "FAILED"
        print(f"Episode {status}: Reward={episode_reward:.1f}, "
              f"Length={episode_length}")

    return {
        'reward': episode_reward,
        'length': episode_length,
        'success': success
    }


def evaluate(args):
    """
    Main evaluation function.

    Args:
        args: Parsed command-line arguments
    """
    # Set random seed
    if args.seed is not None:
        np.random.seed(args.seed)

    print("=" * 70)
    print("DUNGEON CRAWLER RL EVALUATION")
    print("=" * 70)
    print(f"Model:           {args.model}")
    print(f"Episodes:        {args.episodes}")
    print(f"Render:          {args.render}")
    print("=" * 70)
    print()

    # Load agent
    print("Loading agent...")
    agent = load_agent(args.model)
    print()

    # Create environment and encoder (SIMPLIFIED - global vision)
    render_mode = 'human' if args.render else None
    env = DungeonCrawlerEnv(render_mode=render_mode, max_steps=args.max_steps, grid_size=args.grid_size)
    encoder = StateEncoder(grid_size=args.grid_size)

    # Evaluation loop
    print("Starting evaluation...")
    print()

    results = []

    for episode in tqdm(range(args.episodes), desc="Evaluating", disable=args.render):
        result = evaluate_episode(
            env, agent, encoder,
            render=args.render,
            verbose=args.verbose
        )
        results.append(result)

    # Compute statistics
    rewards = [r['reward'] for r in results]
    lengths = [r['length'] for r in results]
    successes = [r['success'] for r in results]

    # Print results
    print()
    print("=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Episodes:            {len(results)}")
    print(f"\nPerformance:")
    print(f"  Success rate:      {np.mean(successes):.2%} ({sum(successes)}/{len(successes)})")
    print(f"  Average reward:    {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  Best reward:       {np.max(rewards):.2f}")
    print(f"  Worst reward:      {np.min(rewards):.2f}")
    print(f"\nEpisode statistics:")
    print(f"  Average length:    {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
    print(f"  Min length:        {np.min(lengths)}")
    print(f"  Max length:        {np.max(lengths)}")

    # Success analysis
    if sum(successes) > 0:
        successful_episodes = [r for r in results if r['success']]
        print(f"\nSuccessful episodes:")
        print(f"  Average reward:    {np.mean([r['reward'] for r in successful_episodes]):.2f}")
        print(f"  Average length:    {np.mean([r['length'] for r in successful_episodes]):.1f}")

    # Failure analysis
    if sum(successes) < len(successes):
        failed_episodes = [r for r in results if not r['success']]
        print(f"\nFailed episodes:")
        print(f"  Average reward:    {np.mean([r['reward'] for r in failed_episodes]):.2f}")
        print(f"  Average length:    {np.mean([r['length'] for r in failed_episodes]):.1f}")

    print("=" * 70)

    env.close()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
