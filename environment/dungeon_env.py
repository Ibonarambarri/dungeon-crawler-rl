"""
Dungeon Crawler RL Environment - 16×16 with Global Vision

A custom Gymnasium environment for navigation with full observability.

Features:
- 16×16 grid with randomly placed interior walls (10% density)
- Agent with GLOBAL vision (sees entire 16×16 grid)
- 2 mobile enemies with random movement (instant death on contact)
- Door/exit objective
- Wall generation with guaranteed path to door (BFS validation)

Reward Structure (SHAPED + SPARSE):
  * Reach door: +200.0 (victory, terminated) 🎯
  * Move closer to door: +1.0 - 0.1 = +0.9 (net positive)
  * Move away from door: -1.0 - 0.1 = -1.1 (net negative)
  * Wall collision: -1.0 - 0.1 = -1.1 (penalty, continues playing)
  * Enemy collision: -100.0 (INSTANT DEATH, terminated) ⚠️
  * Step penalty: -0.1 (always applied)

State Space:
- Global view: Full 16×16 grid
- State encoding: Agent position (14×14) × Door position (14×14) = 38,416 states
- Full observability makes learning easier

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
    - Door escape objective (no key required)
    - Distance-based reward shaping to guide learning

    Attributes:
        grid_size (int): Size of the square grid (8)
        max_steps (int): Maximum steps per episode (100)
        action_space: Discrete(4) - movement actions only
        observation_space: Dict with global_view (8×8) only
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
    CELL_DOOR = 2
    CELL_AGENT = 3
    CELL_AGENT_ON_DOOR = 4  # Agent standing on door position
    CELL_ENEMY = 5  # Mobile enemy

    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_steps: int = 300,
        grid_size: int = 16,
        wall_density: float = 0.10,
        wall_collision_penalty: float = -1.0
    ):
        """
        Initialize the environment with global vision.

        Args:
            render_mode: Rendering mode ('human', 'ansi', 'pygame', or None)
            max_steps: Maximum steps per episode (default: 300)
            grid_size: Size of the square grid (default: 16)
            wall_density: Probability of wall placement (0.0-1.0, default: 0.10)
            wall_collision_penalty: Penalty for attempting to move into a wall (default: -1.0)
        """
        super().__init__()

        self.grid_size = grid_size
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.wall_density = wall_density
        self.wall_collision_penalty = wall_collision_penalty

        # PyGame renderer (lazy initialization)
        self.pygame_renderer = None
        if render_mode == 'pygame':
            try:
                from environment.render_pygame import PyGameRenderer
                self.pygame_renderer = PyGameRenderer(
                    grid_size=grid_size,
                    cell_size=48,  # 48px cells for 16×16 full grid (768×868 window)
                    fps=self.metadata['render_fps']
                )
            except ImportError:
                print("Warning: PyGame renderer not available. Falling back to 'ansi' mode.")
                self.render_mode = 'ansi'

        # Define action space (4 movements)
        self.action_space = spaces.Discrete(4)

        # Define observation space (global 16×16 view)
        self.observation_space = spaces.Dict({
            'global_view': spaces.Box(
                low=0, high=5, shape=(grid_size, grid_size), dtype=np.int32
            )
        })

        # Grid (0=floor, 1=wall)
        self.grid = np.zeros((grid_size, grid_size), dtype=np.int8)

        # Initialize state variables
        self._init_state()

    def _init_state(self):
        """Initialize all state variables to default values."""
        self.agent_pos = [1, 1]  # Will be overridden in reset()

        # Fixed positions
        self.door_pos = (0, 0)

        # Enemy positions (2 enemies)
        self.enemy1_pos = [0, 0]  # Will be overridden in reset()
        self.enemy2_pos = [0, 0]  # Will be overridden in reset()

        # Distance tracking for reward shaping
        self.prev_dist_to_door = 0

        # Episode tracking
        self.steps = 0
        self.last_action = None
        self.last_step_reward = 0.0

    def _create_empty_grid(self):
        """Create a 16×16 empty grid with only border walls."""
        # All floor
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)

        # Add border walls
        self.grid[0, :] = 1
        self.grid[-1, :] = 1
        self.grid[:, 0] = 1
        self.grid[:, -1] = 1

    def _is_path_exists(self, start_pos: Tuple[int, int], end_pos: Tuple[int, int]) -> bool:
        """
        Check if a path exists between start and end positions using BFS.

        Args:
            start_pos: Starting position (y, x)
            end_pos: Target position (y, x)

        Returns:
            True if path exists, False otherwise
        """
        from collections import deque

        # BFS to check connectivity
        queue = deque([start_pos])
        visited = {start_pos}

        while queue:
            current_y, current_x = queue.popleft()

            # Check if we reached the goal
            if (current_y, current_x) == end_pos:
                return True

            # Explore neighbors (UP, DOWN, LEFT, RIGHT)
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_y, new_x = current_y + dy, current_x + dx
                new_pos = (new_y, new_x)

                # Check if valid position
                if (0 <= new_y < self.grid_size and
                    0 <= new_x < self.grid_size and
                    new_pos not in visited and
                    self.grid[new_y, new_x] != 1):  # Not a wall
                    visited.add(new_pos)
                    queue.append(new_pos)

        return False

    def _pathfinding_distance(self, start_pos: Tuple[int, int], end_pos: Tuple[int, int]) -> float:
        """
        Calculate the shortest path distance between two positions using BFS.
        Takes walls into account, so the returned distance is the actual number
        of steps needed to reach the goal, not the straight-line Manhattan distance.

        Args:
            start_pos: Starting position (y, x)
            end_pos: Target position (y, x)

        Returns:
            float: Number of steps in shortest path, or float('inf') if no path exists
        """
        from collections import deque

        # Special case: already at goal
        if start_pos == end_pos:
            return 0

        # BFS with distance tracking
        queue = deque([(start_pos, 0)])  # (position, distance)
        visited = {start_pos}

        while queue:
            (current_y, current_x), dist = queue.popleft()

            # Explore neighbors (UP, DOWN, LEFT, RIGHT)
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_y, new_x = current_y + dy, current_x + dx
                new_pos = (new_y, new_x)

                # Check if we reached the goal
                if new_pos == end_pos:
                    return dist + 1

                # Check if valid position
                if (0 <= new_y < self.grid_size and
                    0 <= new_x < self.grid_size and
                    new_pos not in visited and
                    self.grid[new_y, new_x] != 1):  # Not a wall
                    visited.add(new_pos)
                    queue.append((new_pos, dist + 1))

        # No path exists
        return float('inf')

    def _place_random_walls(self):
        """
        Place random walls in the interior of the grid (not on borders).
        Uses wall_density parameter to determine probability of wall placement.
        """
        # Place walls randomly in interior cells
        for y in range(1, self.grid_size - 1):
            for x in range(1, self.grid_size - 1):
                if np.random.random() < self.wall_density:
                    self.grid[y, x] = 1

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

        Generates a new 16×16 grid with random spawn positions for agent, door, and 2 enemies.
        Ensures that walls are placed such that the door is always reachable from the agent's
        starting position.

        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)

        Returns:
            observation: Initial observation dictionary
            info: Additional information dictionary
        """
        super().reset(seed=seed)

        # Create empty grid with border walls
        self._create_empty_grid()

        # Reset state
        self._init_state()

        # Generate valid dungeon layout (with accessible door)
        max_attempts = 100
        for attempt in range(max_attempts):
            # Start fresh with empty grid
            self._create_empty_grid()

            # Place random walls
            self._place_random_walls()

            # Random spawn positions for agent and door
            agent_pos_tuple = self._get_random_floor_position()
            self.agent_pos = list(agent_pos_tuple)

            self.door_pos = self._get_random_floor_position(
                exclude_positions=[agent_pos_tuple]
            )

            # Check if path exists from agent to door
            if self._is_path_exists(agent_pos_tuple, self.door_pos):
                # Valid layout found!
                break
        else:
            # Fallback: if no valid layout found, use empty grid
            self._create_empty_grid()
            agent_pos_tuple = self._get_random_floor_position()
            self.agent_pos = list(agent_pos_tuple)
            self.door_pos = self._get_random_floor_position(
                exclude_positions=[agent_pos_tuple]
            )

        # Spawn 2 enemies in random positions
        enemy1_pos_tuple = self._get_random_floor_position(
            exclude_positions=[agent_pos_tuple, self.door_pos]
        )
        self.enemy1_pos = list(enemy1_pos_tuple)

        enemy2_pos_tuple = self._get_random_floor_position(
            exclude_positions=[agent_pos_tuple, self.door_pos, enemy1_pos_tuple]
        )
        self.enemy2_pos = list(enemy2_pos_tuple)

        # Initialize distance tracking for reward shaping (use REAL pathfinding distance)
        self.prev_dist_to_door = self._pathfinding_distance(tuple(self.agent_pos), self.door_pos)

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
            terminated: Whether episode ended (win/death condition)
            truncated: Whether episode was cut off (max steps)
            info: Additional information
        """
        # Track steps and action
        self.steps += 1
        self.last_action = action

        # Initialize reward with step penalty (always applied)
        reward = -0.1

        # Execute movement action and check for wall collision
        wall_collision = False
        if action in [self.ACTION_UP, self.ACTION_DOWN,
                     self.ACTION_LEFT, self.ACTION_RIGHT]:
            movement_success = self._handle_movement(action)
            if not movement_success:
                # Wall collision detected - apply penalty but continue playing
                wall_collision = True
                reward += self.wall_collision_penalty  # Penalty for hitting wall

        # Move enemies (random movement)
        self._move_enemies()

        # Check for enemy collision (instant death)
        agent_pos_tuple = tuple(self.agent_pos)
        if agent_pos_tuple == tuple(self.enemy1_pos) or agent_pos_tuple == tuple(self.enemy2_pos):
            reward += -100.0  # DEATH PENALTY
            terminated = True
            self.last_step_reward = reward
            observation = self._get_obs()
            info = self._get_info()
            info['death_reason'] = 'enemy_collision'
            return observation, reward, terminated, False, info

        # Calculate current REAL distance to door (using pathfinding, not straight-line)
        current_dist = self._pathfinding_distance(agent_pos_tuple, self.door_pos)

        # Reward shaping based on REAL distance change (accounts for walls)
        if current_dist < self.prev_dist_to_door:
            # Moved closer on REAL path: +1.0 - 0.1 = +0.9 net
            # This correctly rewards moving backward if that's the shorter path around walls
            reward += 1.0
        elif current_dist > self.prev_dist_to_door:
            # Moved away on REAL path: -1.0 - 0.1 = -1.1 net
            reward -= 1.0
        # If distance stays same (hit wall or lateral move): just -0.1 penalty

        # Update previous distance for next step
        self.prev_dist_to_door = current_dist

        # Check terminal conditions
        terminated = False
        truncated = False

        # Win condition: reached door
        if agent_pos_tuple == self.door_pos:
            reward += 200.0  # DOOR REWARD (victory)
            terminated = True

        # Max steps reached
        if self.steps >= self.max_steps:
            truncated = True

        # Store reward for rendering
        self.last_step_reward = reward

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _handle_movement(self, action: int) -> bool:
        """
        Handle movement actions and detect wall collisions.

        Args:
            action: Movement action (0-3)

        Returns:
            bool: True if movement was successful, False if blocked by wall
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

        # Check grid bounds first
        if not (0 <= new_pos[0] < self.grid_size and
                0 <= new_pos[1] < self.grid_size):
            return False  # Out of bounds - wall collision

        # Check walls - block movement and return collision status
        if self.grid[new_pos_tuple] == 1:
            return False  # Wall collision - no movement

        # Valid move - update position
        self.agent_pos = new_pos
        return True  # Successful movement

    def _move_enemies(self):
        """
        Move both enemies randomly (UP/DOWN/LEFT/RIGHT/STAY).
        Each enemy moves independently with random actions.
        """
        # Move enemy 1
        self._move_single_enemy(self.enemy1_pos)

        # Move enemy 2
        self._move_single_enemy(self.enemy2_pos)

    def _move_single_enemy(self, enemy_pos: list):
        """
        Move a single enemy randomly.

        Args:
            enemy_pos: List [y, x] representing enemy position (modified in place)
        """
        # Random movement: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=STAY
        move_action = np.random.randint(0, 5)

        # Calculate new position
        new_pos = enemy_pos.copy()
        if move_action == 0:  # UP
            new_pos[0] -= 1
        elif move_action == 1:  # DOWN
            new_pos[0] += 1
        elif move_action == 2:  # LEFT
            new_pos[1] -= 1
        elif move_action == 3:  # RIGHT
            new_pos[1] += 1
        # If move_action == 4, stay in place (no change)

        # Check if new position is valid (not wall, within bounds)
        if (0 <= new_pos[0] < self.grid_size and
            0 <= new_pos[1] < self.grid_size and
            self.grid[tuple(new_pos)] != 1):  # Not a wall
            # Valid move - update position
            enemy_pos[0] = new_pos[0]
            enemy_pos[1] = new_pos[1]

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
        Get global view of entire 16×16 grid.

        Returns:
            np.ndarray: 16×16 array with cell types:
                0 = floor, 1 = wall, 2 = door, 3 = agent, 4 = agent on door, 5 = enemy
        """
        view = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        agent_y, agent_x = self.agent_pos
        agent_pos_tuple = tuple(self.agent_pos)

        # Fill view with entire grid
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                pos = (y, x)

                # Check what's at this position
                if self.grid[y, x] == 1:
                    view[y, x] = self.CELL_WALL
                elif pos == tuple(self.enemy1_pos) or pos == tuple(self.enemy2_pos):
                    view[y, x] = self.CELL_ENEMY
                elif pos == self.door_pos:
                    view[y, x] = self.CELL_DOOR
                else:
                    view[y, x] = self.CELL_FLOOR

        # Place agent (special case if agent is on door)
        if agent_pos_tuple == self.door_pos:
            view[agent_y, agent_x] = self.CELL_AGENT_ON_DOOR
        else:
            view[agent_y, agent_x] = self.CELL_AGENT

        return view

    def _get_obs(self) -> Dict[str, Any]:
        """
        Get current observation (16×16 global view).

        Returns:
            observation: Dictionary with global_view
        """
        return {
            'global_view': self._get_global_view()
        }

    def _get_info(self) -> Dict[str, Any]:
        """
        Get additional information about the environment state.

        Returns:
            info: Dictionary with auxiliary information
        """
        agent_pos_tuple = tuple(self.agent_pos)
        dist_to_door = self._manhattan_dist(agent_pos_tuple, self.door_pos)

        return {
            'steps': self.steps,
            'dist_to_door': dist_to_door,
            'agent_pos': agent_pos_tuple,  # For debugging
            'door_pos': self.door_pos  # For state encoding
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
                'door_pos': self.door_pos,
                'enemy1_pos': tuple(self.enemy1_pos),
                'enemy2_pos': tuple(self.enemy2_pos),
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

        # Place door
        y, x = self.door_pos
        display_grid[y][x] = 'X'

        # Place enemies
        y, x = self.enemy1_pos
        display_grid[y][x] = 'E'
        y, x = self.enemy2_pos
        display_grid[y][x] = 'E'

        # Place agent (overwrites other symbols)
        y, x = self.agent_pos
        display_grid[y][x] = '@'

        # Build output string
        output = []
        output.append("=" * 70)
        output.append(f"DUNGEON 16×16 - Step {self.steps}/{self.max_steps}")
        output.append("=" * 70)

        # Print grid
        for y in range(self.grid_size):
            row = []
            for x in range(self.grid_size):
                row.append(display_grid[y][x])
            output.append(' '.join(row))

        output.append("=" * 70)

        # Print stats
        output.append(f"Position: ({self.agent_pos[0]}, {self.agent_pos[1]})")
        output.append(f"Distance to Door: {self._manhattan_dist(tuple(self.agent_pos), self.door_pos)}")
        output.append(f"Enemies: ({self.enemy1_pos[0]}, {self.enemy1_pos[1]}), ({self.enemy2_pos[0]}, {self.enemy2_pos[1]})")
        output.append(f"Last Reward: {self.last_step_reward:.2f}")

        output.append("=" * 70)

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
    print("Testing DungeonCrawlerEnv (16×16 with Global Vision)...")
    print()

    # Test 1: Create environment
    print("Test 1: Creating environment")
    env = DungeonCrawlerEnv(render_mode='human', max_steps=300)
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print("✓ Environment created")
    print()

    # Test 2: Reset
    print("Test 2: Resetting environment")
    obs, info = env.reset(seed=42)
    print(f"Observation keys: {obs.keys()}")
    print(f"Global view shape: {obs['global_view'].shape}")
    print(f"Agent position: {info['agent_pos']}")
    print(f"Distance to door: {info['dist_to_door']}")
    print("✓ Reset successful")
    print()

    # Test 3: Global view
    print("Test 3: Checking global view (16×16)")
    print("Global view (entire grid, showing only agent and door for clarity):")
    global_view = obs['global_view']
    symbols = {0: '.', 1: '#', 2: 'X', 3: '@', 4: '$', 5: 'E'}  # $ = agent on door, E = enemy
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
              f"Reward={reward:.2f}, Dist to Door={info['dist_to_door']}")

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
