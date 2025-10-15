"""
PyGame Demo for Dungeon Crawler RL

Interactive demo that allows:
1. Manual play with keyboard controls
2. AI agent playing (load trained model)
3. Real-time visualization with PyGame

Controls (Manual Mode):
- Arrow Keys: Movement (UP, DOWN, LEFT, RIGHT)
- R: Reset episode
- M: Toggle Manual/AI mode
- Q: Quit
- +/-: Increase/decrease FPS

SIMPLIFIED VERSION:
- No combat (ATTACK removed)
- Goal: Reach exit door

Usage:
    # Manual mode
    python demo_pygame.py --manual

    # AI mode with trained model
    python demo_pygame.py --model models/q_learning_best.pkl

    # Custom FPS
    python demo_pygame.py --manual --fps 30
"""

import argparse
import pickle
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any

import gymnasium as gym
import numpy as np
import pygame

from environment.dungeon_env import DungeonCrawlerEnv
from utils.state_encoder import StateEncoder
from agents.qlearning import QLearningAgent


class DungeonDemo:
    """
    Interactive demo for Dungeon Crawler with PyGame visualization.

    Supports both manual keyboard control and AI agent playback.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        manual_mode: bool = True,
        fps: int = 10,
        grid_size: int = 16
    ):
        """
        Initialize the demo.

        Args:
            model_path: Path to trained model pickle file (optional)
            manual_mode: Start in manual mode (True) or AI mode (False)
            fps: Frames per second for visualization
            grid_size: Size of the dungeon grid
        """
        self.manual_mode = manual_mode
        self.fps = fps
        self.grid_size = grid_size

        # Create environment with pygame rendering (ULTRA-SIMPLIFIED - global vision)
        self.env = DungeonCrawlerEnv(
            render_mode='pygame',
            max_steps=100,
            grid_size=grid_size
        )

        # Create encoder (must match training settings)
        self.encoder = StateEncoder(grid_size=grid_size)

        # Load agent if model path provided
        self.agent = None
        if model_path:
            self.agent = self._load_agent(model_path)
            if self.agent is None:
                print(f"Warning: Failed to load agent from {model_path}")
                print("Starting in manual mode.")
                self.manual_mode = True

        # Stats tracking
        self.episode_count = 0
        self.total_reward = 0.0
        self.episode_rewards = []

        # Current episode state
        self.current_door_pos = None  # Store door position for encoding

        # FPS control
        self.clock = pygame.time.Clock()

        # Action cooldown (to prevent holding keys from spamming actions)
        self.last_action_time = 0
        self.action_cooldown = 0.15  # 150ms between actions

        print("\n" + "=" * 60)
        print("DUNGEON CRAWLER RL - PyGame Demo")
        print("=" * 60)
        print(f"Mode: {'MANUAL' if self.manual_mode else 'AI'}")
        print(f"FPS: {self.fps}")
        print(f"Grid Size: {self.grid_size}×{self.grid_size}")
        print()
        print("CONTROLS:")
        print("  Arrow Keys - Move (UP, DOWN, LEFT, RIGHT)")
        print("  R          - Reset episode")
        print("  M          - Toggle Manual/AI mode")
        print("  +/-        - Increase/Decrease FPS (AI speed)")
        print("  [/]        - Decrease/Increase action cooldown (Manual speed)")
        print("  Q/ESC      - Quit")
        print()
        print("GOAL: Reach exit door")
        print("=" * 60)
        print()

    def _load_agent(self, model_path: str):
        """
        Load a trained agent from pickle file.

        Args:
            model_path: Path to model file

        Returns:
            Agent instance or None if loading fails
        """
        try:
            path = Path(model_path)
            if not path.exists():
                print(f"Error: Model file not found: {model_path}")
                return None

            # Create a fresh agent
            agent = QLearningAgent(n_actions=self.env.action_space.n)

            # Load the saved Q-table and parameters
            agent.load(str(path))

            print(f"✓ Agent loaded successfully")
            print(f"  Q-table size: {len(agent.q_table):,} states")
            print(f"  States visited: {len(agent.states_visited):,}")
            print(f"  Current epsilon: {agent.epsilon:.4f}")

            return agent

        except Exception as e:
            print(f"Error loading agent: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _get_ai_action(self, observation: Dict[str, Any]) -> int:
        """
        Get action from AI agent (greedy policy).

        Args:
            observation: Current observation

        Returns:
            Action index (0-3, SIMPLIFIED - no ATTACK)
        """
        if self.agent is None:
            # Random action as fallback
            return self.env.action_space.sample()

        # Encode observation to state (use stored door_pos)
        state = self.encoder.encode(observation, door_pos=self.current_door_pos)

        # Get best action from agent (greedy, no exploration)
        action = self.agent.get_action(state, training=False)

        return action

    def _get_manual_action(self) -> Optional[int]:
        """
        Get action from keyboard input.

        Returns:
            Action index (0-3, SIMPLIFIED - no ATTACK) or None if no action
        """
        keys = pygame.key.get_pressed()

        if keys[pygame.K_UP]:
            return 0  # UP
        elif keys[pygame.K_DOWN]:
            return 1  # DOWN
        elif keys[pygame.K_LEFT]:
            return 2  # LEFT
        elif keys[pygame.K_RIGHT]:
            return 3  # RIGHT

        return None

    def _handle_events(self) -> bool:
        """
        Handle pygame events (keyboard input, window close).

        Returns:
            True to continue, False to quit
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.KEYDOWN:
                # Quit
                if event.key in [pygame.K_q, pygame.K_ESCAPE]:
                    return False

                # Reset
                if event.key == pygame.K_r:
                    print("\n[RESET] Starting new episode...")
                    self._start_episode()

                # Toggle mode
                if event.key == pygame.K_m:
                    self.manual_mode = not self.manual_mode
                    mode_str = "MANUAL" if self.manual_mode else "AI"
                    print(f"\n[MODE] Switched to {mode_str} mode")
                    if not self.manual_mode and self.agent is None:
                        print("Warning: No agent loaded. Switching back to manual mode.")
                        self.manual_mode = True

                # FPS control (for AI mode)
                if event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    self.fps = min(60, self.fps + 5)
                    print(f"[FPS] Increased to {self.fps}")

                if event.key == pygame.K_MINUS:
                    self.fps = max(1, self.fps - 5)
                    print(f"[FPS] Decreased to {self.fps}")

                # Action speed control (for Manual mode)
                if event.key == pygame.K_RIGHTBRACKET:  # ]
                    self.action_cooldown = max(0.05, self.action_cooldown - 0.05)
                    print(f"[SPEED] Action cooldown: {self.action_cooldown:.2f}s (faster)")

                if event.key == pygame.K_LEFTBRACKET:  # [
                    self.action_cooldown = min(1.0, self.action_cooldown + 0.05)
                    print(f"[SPEED] Action cooldown: {self.action_cooldown:.2f}s (slower)")

        return True

    def _start_episode(self):
        """Start a new episode (ULTRA-SIMPLIFIED)."""
        obs, info = self.env.reset()
        self.episode_count += 1
        self.total_reward = 0.0

        # Store door position for entire episode (needed for state encoding)
        self.current_door_pos = info['door_pos']

        print(f"\n=== Episode {self.episode_count} Started ===")
        print(f"Agent at: {info['agent_pos']}")  # Position is in info, not obs
        print(f"Distance to door: {info['dist_to_door']}")

        return obs, info

    def _print_step_info(self, action: int, reward: float, obs: Dict, info: Dict):
        """Print information about the current step (SIMPLIFIED)."""
        action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        mode = "🎮" if self.manual_mode else "🤖"

        print(f"{mode} Step {info['steps']:3d} | "
              f"Action: {action_names[action]:6s} | "
              f"Reward: {reward:6.1f} | "
              f"Dist to Door: {info['dist_to_door']:2d}")

    def _print_episode_summary(self, terminated: bool, truncated: bool):
        """Print summary at end of episode (SIMPLIFIED)."""
        self.episode_rewards.append(self.total_reward)

        print("\n" + "=" * 60)
        if terminated and self.total_reward > 50:
            print("🎉 VICTORY! You reached the exit door!")
        elif terminated:
            print("❌ Episode ended.")
        else:
            print("⏱️  TIME OUT! Max steps reached.")

        print(f"Episode {self.episode_count} Summary:")
        print(f"  Total Reward: {self.total_reward:.1f}")
        print(f"  Average Reward: {np.mean(self.episode_rewards):.1f}")

        if len(self.episode_rewards) > 1:
            print(f"  Best Reward: {max(self.episode_rewards):.1f}")
            print(f"  Worst Reward: {min(self.episode_rewards):.1f}")

        print("=" * 60)

    def run(self):
        """
        Main demo loop.

        Runs the environment and handles input/rendering.
        """
        try:
            obs, info = self._start_episode()
            self.env.render()

            running = True
            action = None

            while running:
                # Handle events
                running = self._handle_events()
                if not running:
                    break

                current_time = time.time()

                # Get action based on mode
                if self.manual_mode:
                    # Manual: wait for keyboard input with cooldown
                    action = self._get_manual_action()

                    if action is None:
                        # No action pressed, just render
                        self.env.render()
                        self.clock.tick(self.fps)
                        continue

                    # Check cooldown
                    if current_time - self.last_action_time < self.action_cooldown:
                        # Still in cooldown, just render
                        self.env.render()
                        self.clock.tick(self.fps)
                        continue

                    self.last_action_time = current_time
                else:
                    # AI: get action from model with FPS throttling
                    if current_time - self.last_action_time < 1.0 / self.fps:
                        self.env.render()
                        self.clock.tick(self.fps)
                        continue

                    action = self._get_ai_action(obs)
                    self.last_action_time = current_time

                # Execute action
                obs, reward, terminated, truncated, info = self.env.step(action)
                self.total_reward += reward

                # Print step info
                self._print_step_info(action, reward, obs, info)

                # Render
                self.env.render()

                # Check if episode ended
                if terminated or truncated:
                    self._print_episode_summary(terminated, truncated)

                    # Wait a bit before starting new episode
                    time.sleep(2.0)

                    # Start new episode
                    obs, info = self._start_episode()

                # Control FPS
                self.clock.tick(self.fps)

        except KeyboardInterrupt:
            print("\n\nInterrupted by user.")

        finally:
            self.env.close()
            print("\nDemo ended. Thanks for playing!")
            print(f"Total episodes: {self.episode_count}")
            if self.episode_rewards:
                print(f"Average reward: {np.mean(self.episode_rewards):.1f}")


def main():
    """Parse arguments and run demo."""
    parser = argparse.ArgumentParser(
        description="Dungeon Crawler RL - PyGame Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Play manually
  python demo_pygame.py --manual

  # Watch AI play
  python demo_pygame.py --model models/q_learning_best.pkl

  # Manual mode with custom FPS
  python demo_pygame.py --manual --fps 30

  # AI mode with slower playback
  python demo_pygame.py --model models/sarsa_best.pkl --fps 5
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to trained model pickle file'
    )

    parser.add_argument(
        '--manual',
        action='store_true',
        help='Start in manual mode (default: auto-detect based on model)'
    )

    parser.add_argument(
        '--fps',
        type=int,
        default=10,
        help='Frames per second (default: 10)'
    )

    parser.add_argument(
        '--grid-size',
        type=int,
        default=8,
        help='Grid size (default: 8, ultra-simplified with global vision)'
    )

    args = parser.parse_args()

    # Determine initial mode
    if args.model and not args.manual:
        manual_mode = False
    else:
        manual_mode = True

    # Create and run demo
    demo = DungeonDemo(
        model_path=args.model,
        manual_mode=manual_mode,
        fps=args.fps,
        grid_size=args.grid_size
    )

    demo.run()


if __name__ == "__main__":
    main()
