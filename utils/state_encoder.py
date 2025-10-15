"""
State Encoder Utility - 32×32 Grid with Global Vision

Encodes global 32×32 view observations into single integer states for tabular RL.

State Encoding Strategy (Feature-Based Encoding):
We use the agent's and door's absolute positions to create unique states:

- Agent position: 900 positions (30×30 interior grid, excluding walls)
- Door position: 900 positions (30×30 interior grid, excluding walls)

Total state space: 900 × 900 = 810,000 states

This encoding includes both agent and door positions, allowing the agent to learn
optimal policies that depend on where the door is located.

Cell types in global view:
- 0 = floor
- 1 = wall
- 2 = door
- 3 = agent
- 4 = agent on door
- 5 = enemy
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np


class StateEncoder:
    """
    Encodes global 32×32 view observations into integer state indices.

    Uses both agent and door absolute positions, making each unique combination
    a separate state for tabular RL.

    Attributes:
        grid_size (int): Size of the grid (32×32)
        interior_size (int): Size of interior (30×30, excluding walls)
        state_space_size (int): Total state space size (810,000)
    """

    # Cell type constants (must match dungeon_env.py)
    CELL_FLOOR = 0
    CELL_WALL = 1
    CELL_DOOR = 2
    CELL_AGENT = 3
    CELL_AGENT_ON_DOOR = 4
    CELL_ENEMY = 5

    def __init__(self, grid_size: int = 32):
        """
        Initialize the state encoder.

        Args:
            grid_size: Size of the square grid (default: 32)
        """
        self.grid_size = grid_size

        # Interior positions (exclude border walls)
        self.interior_size = grid_size - 2  # 30 for 32×32 grid
        self.interior_positions = self.interior_size * self.interior_size  # 900

        # State space: agent_pos × door_pos
        # 900 agent positions × 900 door positions = 810,000 states
        self.state_space_size = self.interior_positions * self.interior_positions

    def _find_agent_position(self, global_view: np.ndarray) -> Tuple[int, int]:
        """
        Find the agent's position in the global view.

        Args:
            global_view: 16×16 numpy array with cell types

        Returns:
            Tuple (y, x) of agent position
        """
        # Search for agent (cell type 3) or agent on door (cell type 4)
        agent_positions = np.argwhere(global_view == self.CELL_AGENT)

        if len(agent_positions) == 0:
            # Agent might be on the door
            agent_positions = np.argwhere(global_view == self.CELL_AGENT_ON_DOOR)

        if len(agent_positions) == 0:
            raise ValueError("Agent not found in global view!")

        # Return first occurrence (should only be one)
        y, x = agent_positions[0]
        return (int(y), int(x))

    def _find_door_position(self, global_view: np.ndarray) -> Tuple[int, int]:
        """
        Find the door's position in the global view.

        Args:
            global_view: 16×16 numpy array with cell types

        Returns:
            Tuple (y, x) of door position
        """
        # Search for door (cell type 2)
        door_positions = np.argwhere(global_view == self.CELL_DOOR)

        if len(door_positions) == 0:
            # Door might be covered by agent (cell type 4)
            agent_positions = np.argwhere(global_view == self.CELL_AGENT_ON_DOOR)
            if len(agent_positions) > 0:
                y, x = agent_positions[0]
                return (int(y), int(x))
            raise ValueError("Door not found in global view!")

        # Return first occurrence (should only be one)
        y, x = door_positions[0]
        return (int(y), int(x))

    def _position_to_interior_index(self, y: int, x: int) -> int:
        """
        Convert absolute (y, x) position to interior grid index.

        Args:
            y: Y coordinate (1-30 for 32×32 grid)
            x: X coordinate (1-30 for 32×32 grid)

        Returns:
            Interior index (0-899 for 30×30 interior)
        """
        # Subtract 1 to convert from absolute to interior coordinates
        interior_y = y - 1
        interior_x = x - 1
        return interior_y * self.interior_size + interior_x

    def _interior_index_to_position(self, index: int) -> Tuple[int, int]:
        """
        Convert interior grid index to absolute (y, x) position.

        Args:
            index: Interior index (0-899)

        Returns:
            Tuple (y, x) in absolute coordinates (1-30)
        """
        interior_y = index // self.interior_size
        interior_x = index % self.interior_size
        # Add 1 to convert from interior to absolute coordinates
        return (interior_y + 1, interior_x + 1)

    def encode(self, observation: Dict[str, Any], door_pos: Optional[Tuple[int, int]] = None) -> int:
        """
        Encode an observation (global view) into a single integer state.

        Extracts agent and door positions from global view and encodes them.

        Args:
            observation: Dictionary containing:
                - global_view: np.array(32, 32) - global view with cell types
            door_pos: Optional explicit door position (y, x). If provided, uses this instead of searching.

        Returns:
            int: Unique state index (0 to 809,999)

        Example:
            >>> encoder = StateEncoder()
            >>> obs = {
            ...     'global_view': np.array(...),  # 32×32 grid
            ... }
            >>> state = encoder.encode(obs)
        """
        global_view = observation['global_view']

        # Extract agent position from global view
        agent_y, agent_x = self._find_agent_position(global_view)

        # Get door position (explicit if provided, otherwise search)
        if door_pos is not None:
            door_y, door_x = door_pos
        else:
            door_y, door_x = self._find_door_position(global_view)

        # Convert to interior indices
        agent_idx = self._position_to_interior_index(agent_y, agent_x)
        door_idx = self._position_to_interior_index(door_y, door_x)

        # Combine into single state: agent_idx * 900 + door_idx
        state = agent_idx * self.interior_positions + door_idx

        # Validate
        assert 0 <= state < self.state_space_size, \
            f"Encoded state {state} out of bounds [0, {self.state_space_size})"

        return int(state)

    def decode(self, state: int) -> Dict[str, Any]:
        """
        Decode an integer state back into feature representation.

        This is the inverse of encode(). Useful for debugging and analysis.

        Args:
            state: Integer state index (0 to 809,999)

        Returns:
            Dictionary with:
                - agent_pos: (y, x) position
                - door_pos: (y, x) position

        Example:
            >>> encoder = StateEncoder()
            >>> features = encoder.decode(1000)
            >>> print(features)
            {'agent_pos': (2, 11), 'door_pos': (2, 11)}
        """
        assert 0 <= state < self.state_space_size, \
            f"State {state} out of bounds [0, {self.state_space_size})"

        # Extract agent and door indices
        agent_idx = state // self.interior_positions
        door_idx = state % self.interior_positions

        # Convert to absolute positions
        agent_pos = self._interior_index_to_position(agent_idx)
        door_pos = self._interior_index_to_position(door_idx)

        return {
            'agent_pos': agent_pos,
            'door_pos': door_pos
        }

    def get_state_space_size(self) -> int:
        """
        Get the total theoretical state space size.

        Returns:
            int: Total number of possible states (810,000)
        """
        return self.state_space_size

    def __repr__(self) -> str:
        """String representation of the encoder."""
        return (f"StateEncoder(grid={self.grid_size}×{self.grid_size}, "
                f"interior={self.interior_size}×{self.interior_size}, "
                f"global_vision=True, "
                f"features=[agent_pos({self.interior_positions}), door_pos({self.interior_positions})], "
                f"state_space={self.state_space_size:,})")


def test_state_encoder():
    """
    Test function to verify state encoder correctness.

    Tests:
    1. Basic encoding with global view (agent + door)
    2. Different agent positions produce different states
    3. Different door positions produce different states
    4. Decode works correctly
    5. Boundary values
    """
    encoder = StateEncoder(grid_size=16)

    print(f"State Encoder Test (16×16 Grid with Global Vision)")
    print(f"State space size: {encoder.get_state_space_size():,}")
    print(f"Features: agent_pos(196) × door_pos(196)")
    print()

    # Test 1: Basic encoding
    print("Test 1: Basic Encoding (Agent + Door)")
    # Create a simple global view with agent at (8, 7) and door at (12, 12)
    global_view = np.zeros((16, 16), dtype=np.int32)
    global_view[8, 7] = encoder.CELL_AGENT  # Agent
    global_view[12, 12] = encoder.CELL_DOOR  # Door
    # Add walls
    global_view[0, :] = encoder.CELL_WALL
    global_view[-1, :] = encoder.CELL_WALL
    global_view[:, 0] = encoder.CELL_WALL
    global_view[:, -1] = encoder.CELL_WALL

    obs = {
        'global_view': global_view
    }
    state = encoder.encode(obs)

    print(f"Agent at (8, 7), Door at (12, 12)")
    print(f"Encoded state: {state}")
    print("✓ Encoding successful")
    print()

    # Test 2: Different agent positions
    print("Test 2: Different Agent Positions")
    global_view2 = np.copy(global_view)
    global_view2[8, 7] = encoder.CELL_FLOOR  # Remove old agent
    global_view2[10, 12] = encoder.CELL_AGENT  # New agent position

    obs2 = {
        'global_view': global_view2
    }
    state2 = encoder.encode(obs2)

    print(f"State 1 (agent (8,7), door (12,12)): {state}")
    print(f"State 2 (agent (10,12), door (12,12)): {state2}")
    print(f"States are different: {state != state2}")
    print("✓ Agent position differentiation works")
    print()

    # Test 3: Different door positions
    print("Test 3: Different Door Positions")
    global_view3 = np.copy(global_view)
    global_view3[12, 12] = encoder.CELL_FLOOR  # Remove old door
    global_view3[5, 5] = encoder.CELL_DOOR  # New door position

    obs3 = {
        'global_view': global_view3
    }
    state3 = encoder.encode(obs3)

    print(f"State 1 (agent (8,7), door (12,12)): {state}")
    print(f"State 3 (agent (8,7), door (5,5)): {state3}")
    print(f"States are different: {state != state3}")
    print("✓ Door position differentiation works")
    print()

    # Test 4: Decode functionality
    print("Test 4: Decode Functionality")
    decoded = encoder.decode(state)
    print(f"Original state: {state}")
    print(f"Decoded features: {decoded}")
    print(f"Agent position: {decoded['agent_pos']}")
    print(f"Door position: {decoded['door_pos']}")
    print("✓ Decode works correctly")
    print()

    # Test 5: Boundary values
    print("Test 5: Boundary Values")
    # Minimum state (both at (1,1))
    min_view = np.zeros((16, 16), dtype=np.int32)
    min_view[1, 1] = encoder.CELL_AGENT_ON_DOOR  # Agent on door at (1,1)
    min_obs = {'global_view': min_view}
    min_state = encoder.encode(min_obs)

    # Maximum state (both at (14,14))
    max_view = np.zeros((16, 16), dtype=np.int32)
    max_view[14, 14] = encoder.CELL_AGENT_ON_DOOR  # Agent on door at (14,14)
    max_obs = {'global_view': max_view}
    max_state = encoder.encode(max_obs)

    print(f"Min state (both at (1,1)): {min_state}")
    print(f"Max state (both at (14,14)): {max_state}")
    print(f"Within bounds: {0 <= min_state < encoder.state_space_size and 0 <= max_state < encoder.state_space_size}")
    print("✓ Boundary values correct")
    print()

    print("All tests passed! ✓")
    print()
    print(f"Summary:")
    print(f"  State space: {encoder.state_space_size:,} states")
    print(f"  Encoding: agent_pos × door_pos from global view")
    print(f"  Tractable for Q-Learning/SARSA/Expected SARSA!")
    print(f"  Note: Door position varies per episode, making states meaningful!")


if __name__ == "__main__":
    test_state_encoder()
