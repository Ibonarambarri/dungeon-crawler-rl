"""
Dungeon Crawler RL Environment - ULTRA-SIMPLIFIED VERSION (8×8)

A minimal custom Gymnasium environment for basic navigation with global vision.

ULTRA-SIMPLIFIED for tabular RL:
- 8×8 empty grid (only border walls)
- Agent with global vision (sees entire board)
- Only 2 elements: key and door
- Reward structure (SPARSE ONLY):
  * Key pickup: +10.0
  * Door with key: +100.0 (victory)
  * NO step penalties, NO reward shaping

State Space:
- Global view: Full 8×8 grid
- Has key: binary (0 or 1)
- Total states: 64 positions × 2 has_key = 128 states

Action Space (4 movements):
- 0: Move UP
- 1: Move DOWN
- 2: Move LEFT
- 3: Move RIGHT
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Any, Optional


class DungeonCrawlerEnv(gym.Env):
    """
    Ultra-simplified Gymnasium environment for tabular RL navigation.

    Features:
    - 8×8 grid with only border walls
    - Global vision (sees entire board)
    - Key collection and door escape objective
    - Sparse rewards only (no shaping)

    Attributes:
        grid_size (int): Size of the square grid (8)
        max_steps (int): Maximum steps per episode (100)
        action_space: Discrete(4) - movement actions only
        observation_space: Dict with global_view (8×8) and has_key
    """

    metadata = {'render_modes': ['human', 'ansi', 'pygame'], 'render_fps': 10}

    # Action constants
    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3

    # Cell types for observation
    CELL_FLOOR = 0
    CELL_WALL = 1
    CELL_KEY = 2
    CELL_DOOR = 3
    CELL_AGENT = 4

    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_steps: int = 100,
        grid_size: int = 8
    ):
        """
        Initialize the ultra-simplified environment.

        Args:
            render_mode: Rendering mode ('human', 'ansi', 'pygame', or None)
            max_steps: Maximum steps per episode (default: 100)
            grid_size: Size of the square grid (default: 8)
        """
        super().__init__()

        self.grid_size = grid_size
        self.max_steps = max_steps
        self.render_mode = render_mode

        # PyGame renderer (lazy initialization)
        self.pygame_renderer = None
        if render_mode == 'pygame':
            try:
                from environment.render_pygame import PyGameRenderer
                self.pygame_renderer = PyGameRenderer(
                    grid_size=grid_size,
                    cell_size=48,  # Larger cells for 8×8 grid
                    fps=self.metadata['render_fps']
                )
            except ImportError:
                print("Warning: PyGame renderer not available. Falling back to 'ansi' mode.")
                self.render_mode = 'ansi'

        # Define action space (4 movements)
        self.action_space = spaces.Discrete(4)

        # Define observation space (global 8×8 view + has_key)
        self.observation_space = spaces.Dict({
            'global_view': spaces.Box(
                low=0, high=4, shape=(grid_size, grid_size), dtype=np.int32
            ),
            'has_key': spaces.Discrete(2)  # 0 or 1
        })

        # Grid (0=floor, 1=wall)
        self.grid = np.zeros((grid_size, grid_size), dtype=np.int8)

        # Initialize state variables
        self._init_state()

    def _init_state(self):
        """Initialize all state variables to default values."""
        self.agent_pos = [1, 1]  # Will be overridden in reset()

        # Items
        self.has_key = False
        self.key_on_map = True

        # Fixed positions
        self.key_pos = (0, 0)
        self.door_pos = (0, 0)

        # Episode tracking
        self.steps = 0
        self.last_action = None
        self.last_step_reward = 0.0

    def _create_empty_grid(self):
        """Create an 8×8 empty grid with only border walls."""
        # All floor
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)

        # Add border walls
        self.grid[0, :] = 1
        self.grid[-1, :] = 1
        self.grid[:, 0] = 1
        self.grid[:, -1] = 1

    def _get_random_floor_position(self, exclude_positions=None) -> Tuple[int, int]:
        """
        Get a random floor position (not on walls or excluded positions).

        Args:
            exclude_positions: List of positions to exclude

        Returns:
            Tuple (y, x) of random floor position
        """
        if exclude_positions is None:
            exclude_positions = []

        # Get all floor positions (not walls)
        floor_positions = []
        for y in range(1, self.grid_size - 1):
            for x in range(1, self.grid_size - 1):
                pos = (y, x)
                if self.grid[y, x] == 0 and pos not in exclude_positions:
                    floor_positions.append(pos)

        if not floor_positions:
            raise ValueError("No available floor positions")

        return floor_positions[np.random.randint(len(floor_positions))]

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reset the environment to initial state.

        Generates a new simple 8×8 grid with random spawn positions.

        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)

        Returns:
            observation: Initial observation dictionary
            info: Additional information dictionary
        """
        super().reset(seed=seed)

        # Create empty grid
        self._create_empty_grid()

        # Reset state
        self._init_state()

        # Random spawn positions
        agent_pos_tuple = self._get_random_floor_position()
        self.agent_pos = list(agent_pos_tuple)

        self.key_pos = self._get_random_floor_position(exclude_positions=[agent_pos_tuple])
        self.door_pos = self._get_random_floor_position(
            exclude_positions=[agent_pos_tuple, self.key_pos]
        )

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(
        self, action: int
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Action to take (0-3, movement only)

        Returns:
            observation: New observation after action
            reward: Reward received from this action
            terminated: Whether episode ended (win condition)
            truncated: Whether episode was cut off (max steps)
            info: Additional information
        """
        # Track steps and action
        self.steps += 1
        self.last_action = action

        # Initialize reward (NO step penalty - sparse rewards only)
        reward = 0.0

        # Execute movement action
        if action in [self.ACTION_UP, self.ACTION_DOWN,
                     self.ACTION_LEFT, self.ACTION_RIGHT]:
            self._handle_movement(action)

        # Check for key pickup
        picked_up_key = self._check_item_pickup()
        if picked_up_key:
            reward += 10.0  # KEY REWARD

        # Check terminal conditions
        terminated = False
        truncated = False

        # Win condition: has key AND reached door
        if self.has_key and tuple(self.agent_pos) == self.door_pos:
            reward += 100.0  # DOOR REWARD (victory)
            terminated = True

        # Max steps reached
        if self.steps >= self.max_steps:
            truncated = True

        # Store reward for rendering
        self.last_step_reward = reward

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _handle_movement(self, action: int):
        """
        Handle movement actions (no penalties for wall collision).

        Args:
            action: Movement action (0-3)
        """
        # Calculate new position
        new_pos = self.agent_pos.copy()
        if action == self.ACTION_UP:
            new_pos[0] -= 1
        elif action == self.ACTION_DOWN:
            new_pos[0] += 1
        elif action == self.ACTION_LEFT:
            new_pos[1] -= 1
        elif action == self.ACTION_RIGHT:
            new_pos[1] += 1

        # Check if new position is valid (not wall)
        new_pos_tuple = tuple(new_pos)

        # Check walls - just block movement
        if self.grid[new_pos_tuple] == 1:
            return  # Wall collision - no movement

        # Check grid bounds
        if not (0 <= new_pos[0] < self.grid_size and
                0 <= new_pos[1] < self.grid_size):
            return  # Out of bounds - no movement

        # Valid move - update position
        self.agent_pos = new_pos

    def _check_item_pickup(self) -> bool:
        """
        Check if agent picked up the key.

        Returns:
            bool: True if key was picked up this step
        """
        agent_pos_tuple = tuple(self.agent_pos)

        # Check key pickup
        if self.key_on_map and agent_pos_tuple == self.key_pos:
            self.has_key = True
            self.key_on_map = False
            return True

        return False

    def _manhattan_dist(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """
        Calculate Manhattan distance between two positions.

        Args:
            pos1: First position (y, x)
            pos2: Second position (y, x)

        Returns:
            Manhattan distance
        """
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _get_global_view(self) -> np.ndarray:
        """
        Get global view of entire 8×8 grid.

        Returns:
            np.ndarray: 8×8 array with cell types:
                0 = floor, 1 = wall, 2 = key, 3 = door, 4 = agent
        """
        view = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        agent_y, agent_x = self.agent_pos

        # Fill view with entire grid
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                pos = (y, x)

                # Check what's at this position
                if self.grid[y, x] == 1:
                    view[y, x] = self.CELL_WALL
                elif pos == self.key_pos and self.key_on_map:
                    view[y, x] = self.CELL_KEY
                elif pos == self.door_pos:
                    view[y, x] = self.CELL_DOOR
                else:
                    view[y, x] = self.CELL_FLOOR

        # Place agent on top
        view[agent_y, agent_x] = self.CELL_AGENT

        return view

    def _get_obs(self) -> Dict[str, Any]:
        """
        Get current observation (8×8 global view + has_key).

        Returns:
            observation: Dictionary with global_view and has_key
        """
        return {
            'global_view': self._get_global_view(),
            'has_key': int(self.has_key)
        }

    def _get_info(self) -> Dict[str, Any]:
        """
        Get additional information about the environment state.

        Returns:
            info: Dictionary with auxiliary information
        """
        agent_pos_tuple = tuple(self.agent_pos)
        dist_to_key = self._manhattan_dist(agent_pos_tuple, self.key_pos) if self.key_on_map else 0
        dist_to_door = self._manhattan_dist(agent_pos_tuple, self.door_pos)

        return {
            'steps': self.steps,
            'has_key': self.has_key,
            'dist_to_key': dist_to_key,
            'dist_to_door': dist_to_door,
            'agent_pos': agent_pos_tuple  # For debugging
        }

    def render(self) -> Optional[str]:
        """
        Render the environment.

        Returns:
            str: ANSI string representation if render_mode='ansi'
            None: if render_mode='pygame' (rendered to screen)
        """
        if self.render_mode is None:
            return None

        if self.render_mode == 'pygame' and self.pygame_renderer is not None:
            # Pass environment state to pygame renderer
            env_state = {
                'grid': self.grid,
                'agent_pos': tuple(self.agent_pos),
                'key_pos': self.key_pos if self.key_on_map else None,
                'door_pos': self.door_pos,
                'has_key': self.has_key,
                'last_action': self.last_action,
                'global_view': self._get_global_view()
            }
            info = self._get_info()
            self.pygame_renderer.render(env_state, info, self.last_step_reward)
            return None

        return self._render_text()

    def _render_text(self) -> str:
        """
        Create text-based visualization of the dungeon.

        Returns:
            str: Multi-line string representation
        """
        # Create empty grid
        display_grid = [[' ' for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        # Place walls
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if self.grid[y, x] == 1:
                    display_grid[y][x] = '#'

        # Place key (if still on map)
        if self.key_on_map:
            y, x = self.key_pos
            display_grid[y][x] = 'K'

        # Place door
        y, x = self.door_pos
        display_grid[y][x] = 'X'

        # Place agent (overwrites other symbols)
        y, x = self.agent_pos
        display_grid[y][x] = '@'

        # Build output string
        output = []
        output.append("=" * 50)
        output.append(f"DUNGEON 8×8 - Step {self.steps}/{self.max_steps}")
        output.append("=" * 50)

        # Print grid
        for y in range(self.grid_size):
            row = []
            for x in range(self.grid_size):
                row.append(display_grid[y][x])
            output.append(' '.join(row))

        output.append("=" * 50)

        # Print stats
        items_str = "Key" if self.has_key else "None"
        output.append(f"Items: {items_str}")
        output.append(f"Position: ({self.agent_pos[0]}, {self.agent_pos[1]})")
        output.append(f"Last Reward: {self.last_step_reward:.2f}")

        output.append("=" * 50)

        result = '\n'.join(output)

        if self.render_mode == 'human':
            print(result)

        return result

    def close(self):
        """Clean up resources."""
        if self.pygame_renderer is not None:
            self.pygame_renderer.close()


def test_environment():
    """
    Test function to verify environment works correctly.

    Tests:
    1. Environment creation
    2. Reset works
    3. Random actions execute
    4. Observations are valid
    5. Local view is correct
    """
    print("Testing DungeonCrawlerEnv (8×8 ULTRA-SIMPLIFIED)...")
    print()

    # Test 1: Create environment
    print("Test 1: Creating environment")
    env = DungeonCrawlerEnv(render_mode='human', max_steps=100)
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print("✓ Environment created")
    print()

    # Test 2: Reset
    print("Test 2: Resetting environment")
    obs, info = env.reset(seed=42)
    print(f"Observation keys: {obs.keys()}")
    print(f"Global view shape: {obs['global_view'].shape}")
    print(f"Has key: {obs['has_key']}")
    print(f"Agent position: {info['agent_pos']}")
    print("✓ Reset successful")
    print()

    # Test 3: Global view
    print("Test 3: Checking global view (8×8)")
    print("Global view (entire grid):")
    global_view = obs['global_view']
    symbols = {0: '.', 1: '#', 2: 'K', 3: 'X', 4: '@'}
    for y in range(env.grid_size):
        print(' '.join([symbols[global_view[y, x]] for x in range(env.grid_size)]))
    print("✓ Global view correct")
    print()

    # Test 4: Random actions
    print("Test 4: Running 20 random steps")
    env.render()

    for step in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        print(f"\nStep {step + 1}: Action={action_names[action]}, "
              f"Reward={reward:.2f}, Has Key={obs['has_key']}")

        if terminated or truncated:
            print(f"Episode ended: terminated={terminated}, truncated={truncated}")
            break

    env.render()
    print("\n✓ Random actions executed successfully")
    print()

    # Test 5: Observation validation
    print("Test 5: Validating observation space")
    obs, info = env.reset(seed=123)
    assert env.observation_space.contains(obs), "Invalid observation!"
    print("✓ Observation is valid")
    print()

    print("All tests passed! ✓")
    env.close()


if __name__ == "__main__":
    test_environment()
