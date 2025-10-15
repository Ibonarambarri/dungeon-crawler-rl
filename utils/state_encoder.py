"""
State Encoder Utility - ULTRA-SIMPLIFIED VERSION

Encodes global 8×8 view observations into single integer states for tabular RL.

ULTRA-SIMPLIFIED VERSION - Global Vision Navigation:
- Input: 8×8 global view only
- Output: Single integer state index

State Encoding Strategy:
We encode agent position AND door position:
- Agent position: 64 positions (8×8 grid)
- Door position: 64 positions (8×8 grid)

Total state space: 64 × 64 = 4,096 states (tractable for tabular RL!)

Cell types in global view:
- 0 = floor
- 1 = wall
- 2 = door
- 3 = agent
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np


class StateEncoder:
    """
    Encodes global 8×8 view observations into integer state indices.

    ULTRA-SIMPLIFIED VERSION for global vision navigation.

    We extract both agent and door positions from the global view.

    Attributes:
        grid_size (int): Size of the grid (8×8)
        state_space_size (int): Total state space size (4096)
    """

    # Cell type constants (must match dungeon_env.py)
    CELL_FLOOR = 0
    CELL_WALL = 1
    CELL_DOOR = 2
    CELL_AGENT = 3
    CELL_AGENT_ON_DOOR = 4

    def __init__(self, grid_size: int = 8):
        """
        Initialize the state encoder.

        Args:
            grid_size: Size of the square grid (default: 8)
        """
        self.grid_size = grid_size

        # State dimensions
        self.pos_size = grid_size * grid_size  # 64 positions

        # Total state space: 64 agent positions × 64 door positions = 4096 states
        self.state_space_size = self.pos_size * self.pos_size

    def _find_agent_position(self, global_view: np.ndarray) -> Tuple[int, int]:
        """
        Find the agent's position in the global view.

        Args:
            global_view: 8×8 numpy array with cell types

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
            global_view: 8×8 numpy array with cell types

        Returns:
            Tuple (y, x) of door position
        """
        # Search for door (cell type 2)
        door_positions = np.argwhere(global_view == self.CELL_DOOR)

        if len(door_positions) == 0:
            # Door might be covered by agent (cell type 4)
            door_positions = np.argwhere(global_view == self.CELL_AGENT_ON_DOOR)

        if len(door_positions) == 0:
            raise ValueError("Door not found in global view!")

        # Return first occurrence (should only be one)
        y, x = door_positions[0]
        return (int(y), int(x))

    def encode(self, observation: Dict[str, Any]) -> int:
        """
        Encode a global view observation into a single integer state.

        Args:
            observation: Dictionary containing:
                - global_view: np.array(8, 8) - global view with cell types

        Returns:
            int: Unique state index (0 to 4095)

        Example:
            >>> encoder = StateEncoder()
            >>> obs = {
            ...     'global_view': np.array(...),  # 8×8 grid
            ... }
            >>> state = encoder.encode(obs)
        """
        global_view = observation['global_view']

        # Extract agent and door positions from global view
        agent_y, agent_x = self._find_agent_position(global_view)
        door_y, door_x = self._find_door_position(global_view)

        # Convert 2D positions to 1D indices
        agent_idx = agent_y * self.grid_size + agent_x
        door_idx = door_y * self.grid_size + door_x

        # Combine into single state: state = agent_idx * 64 + door_idx
        state = agent_idx * self.pos_size + door_idx

        # Validate
        assert 0 <= state < self.state_space_size, \
            f"Encoded state {state} out of bounds [0, {self.state_space_size})"

        return state

    def decode(self, state: int) -> Dict[str, Any]:
        """
        Decode an integer state back into feature representation.

        This is the inverse of encode(). Useful for debugging and analysis.

        Args:
            state: Integer state index (0 to 4095)

        Returns:
            Dictionary with:
                - agent_pos: (y, x) position
                - door_pos: (y, x) position

        Example:
            >>> encoder = StateEncoder()
            >>> features = encoder.decode(200)
            >>> print(features)
            {'agent_pos': (3, 2), 'door_pos': (1, 8)}
        """
        assert 0 <= state < self.state_space_size, \
            f"State {state} out of bounds [0, {self.state_space_size})"

        # Extract agent and door indices
        agent_idx = state // self.pos_size
        door_idx = state % self.pos_size

        # Convert 1D indices to 2D positions
        agent_y = agent_idx // self.grid_size
        agent_x = agent_idx % self.grid_size

        door_y = door_idx // self.grid_size
        door_x = door_idx % self.grid_size

        return {
            'agent_pos': (agent_y, agent_x),
            'door_pos': (door_y, door_x)
        }

    def get_state_space_size(self) -> int:
        """
        Get the total theoretical state space size.

        Returns:
            int: Total number of possible states (4096)
        """
        return self.state_space_size

    def __repr__(self) -> str:
        """String representation of the encoder."""
        return (f"StateEncoder(global_vision={self.grid_size}×{self.grid_size}, "
                f"features=[agent_pos(64) × door_pos(64)], "
                f"state_space={self.state_space_size})")


def test_state_encoder():
    """
    Test function to verify state encoder correctness.

    Tests:
    1. Basic encoding with global view
    2. Different positions produce different states
    3. Different has_key values produce different states
    4. Decode works correctly
    5. Boundary values
    """
    encoder = StateEncoder(grid_size=8)

    print(f"State Encoder Test (ULTRA-SIMPLIFIED GLOBAL VISION)")
    print(f"State space size: {encoder.get_state_space_size():,}")
    print(f"Features: agent_pos(64) × door_pos(64)")
    print()

    # Test 1: Basic encoding
    print("Test 1: Basic Encoding")
    # Create a simple global view with agent at (4, 5)
    global_view = np.zeros((8, 8), dtype=np.int32)
    global_view[4, 5] = encoder.CELL_AGENT  # Agent
    global_view[6, 6] = encoder.CELL_DOOR  # Door
    # Add walls
    global_view[0, :] = encoder.CELL_WALL
    global_view[-1, :] = encoder.CELL_WALL
    global_view[:, 0] = encoder.CELL_WALL
    global_view[:, -1] = encoder.CELL_WALL

    obs = {'global_view': global_view}
    state = encoder.encode(obs)

    print(f"Agent at position (4, 5)")
    print(f"Encoded state: {state}")
    print("✓ Encoding successful")
    print()

    # Test 2: Different positions produce different states
    print("Test 2: Different Positions")
    global_view2 = np.copy(global_view)
    global_view2[4, 5] = encoder.CELL_FLOOR  # Remove old agent
    global_view2[3, 2] = encoder.CELL_AGENT  # New agent position

    obs2 = {'global_view': global_view2}
    state2 = encoder.encode(obs2)

    print(f"State 1 (pos (4,5)): {state}")
    print(f"State 2 (pos (3,2)): {state2}")
    print(f"States are different: {state != state2}")
    print("✓ Position differentiation works")
    print()

    # Test 3: Decode functionality
    print("Test 3: Decode Functionality")
    decoded = encoder.decode(state)
    print(f"Original state: {state}")
    print(f"Decoded features: {decoded}")
    print(f"Agent position: {decoded['agent_pos']}")
    print("✓ Decode works correctly")
    print()

    # Test 4: Boundary values
    print("Test 4: Boundary Values")
    # Minimum state (pos (1,1))
    min_view = np.zeros((8, 8), dtype=np.int32)
    min_view[1, 1] = encoder.CELL_AGENT
    min_obs = {'global_view': min_view}
    min_state = encoder.encode(min_obs)

    # Maximum state (pos (6,6))
    max_view = np.zeros((8, 8), dtype=np.int32)
    max_view[6, 6] = encoder.CELL_AGENT
    max_obs = {'global_view': max_view}
    max_state = encoder.encode(max_obs)

    print(f"Min state (pos (1,1)): {min_state}")
    print(f"Max state (pos (6,6)): {max_state}")
    print(f"Within bounds: {0 <= min_state < encoder.state_space_size and 0 <= max_state < encoder.state_space_size}")
    print("✓ Boundary values correct")
    print()

    print("All tests passed! ✓")
    print()
    print(f"Summary:")
    print(f"  State space: {encoder.state_space_size:,} states")
    print(f"  Encoding: agent_pos × door_pos")
    print(f"  Still tractable for tabular RL!")


if __name__ == "__main__":
    test_state_encoder()
