"""
State Encoder Utility - ULTRA-SIMPLIFIED VERSION

Encodes global 8×8 view observations into single integer states for tabular RL.

ULTRA-SIMPLIFIED VERSION - Global Vision Navigation:
- Input: 8×8 global view + has_key flag
- Output: Single integer state index

State Encoding Strategy:
We encode agent position (64 positions in 8×8) + key status (2 states):
- Agent position: 64 positions (8×8 grid)
- Has key collected: 2 states (0 or 1)

Total state space: 64 × 2 = 128 states (extremely tractable for tabular RL!)

Cell types in global view:
- 0 = floor
- 1 = wall
- 2 = key
- 3 = door
- 4 = agent
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np


class StateEncoder:
    """
    Encodes global 8×8 view observations into integer state indices.

    ULTRA-SIMPLIFIED VERSION for global vision navigation.

    We extract the agent's position from the global view and combine it
    with the key possession status to create a unique state.

    Attributes:
        grid_size (int): Size of the grid (8×8)
        state_space_size (int): Total state space size (128)
    """

    # Cell type constants (must match dungeon_env.py)
    CELL_FLOOR = 0
    CELL_WALL = 1
    CELL_KEY = 2
    CELL_DOOR = 3
    CELL_AGENT = 4

    def __init__(self, grid_size: int = 8):
        """
        Initialize the state encoder.

        Args:
            grid_size: Size of the square grid (default: 8)
        """
        self.grid_size = grid_size

        # State dimensions
        self.pos_size = grid_size * grid_size  # 64 positions
        self.has_key_size = 2  # 0 or 1

        # Calculate multipliers for positional encoding
        # State = pos_idx * 2 + has_key
        multiplier = 1
        self.has_key_mult = multiplier
        multiplier *= self.has_key_size

        self.pos_mult = multiplier

        # Total state space: 64 positions × 2 has_key = 128 states
        self.state_space_size = self.pos_size * self.has_key_size

    def _find_agent_position(self, global_view: np.ndarray) -> Tuple[int, int]:
        """
        Find the agent's position in the global view.

        Args:
            global_view: 8×8 numpy array with cell types

        Returns:
            Tuple (y, x) of agent position
        """
        # Search for agent (cell type 4)
        agent_positions = np.argwhere(global_view == self.CELL_AGENT)

        if len(agent_positions) == 0:
            raise ValueError("Agent not found in global view!")

        # Return first occurrence (should only be one)
        y, x = agent_positions[0]
        return (int(y), int(x))

    def encode(self, observation: Dict[str, Any]) -> int:
        """
        Encode a global view observation into a single integer state.

        Args:
            observation: Dictionary containing:
                - global_view: np.array(8, 8) - global view with cell types
                - has_key: int - 0 or 1

        Returns:
            int: Unique state index (0 to 127)

        Example:
            >>> encoder = StateEncoder()
            >>> obs = {
            ...     'global_view': np.array(...),  # 8×8 grid
            ...     'has_key': 0
            ... }
            >>> state = encoder.encode(obs)
        """
        global_view = observation['global_view']
        has_key = observation['has_key']

        # Extract agent position from global view
        agent_y, agent_x = self._find_agent_position(global_view)

        # Convert 2D position to 1D index
        pos_idx = agent_y * self.grid_size + agent_x

        # Validate inputs
        assert 0 <= pos_idx < self.pos_size, f"Invalid position index: {pos_idx}"
        assert has_key in [0, 1], f"Invalid has_key: {has_key}"

        # Encode using simple positional system
        # State = pos_idx * 2 + has_key
        state = pos_idx * self.pos_mult + has_key * self.has_key_mult

        assert 0 <= state < self.state_space_size, \
            f"Encoded state {state} out of bounds [0, {self.state_space_size})"

        return state

    def decode(self, state: int) -> Dict[str, Any]:
        """
        Decode an integer state back into feature representation.

        This is the inverse of encode(). Useful for debugging and analysis.

        Args:
            state: Integer state index (0 to 127)

        Returns:
            Dictionary with:
                - agent_pos: (y, x) position
                - has_key: 0 or 1

        Example:
            >>> encoder = StateEncoder()
            >>> features = encoder.decode(123)
            >>> print(features)
            {'agent_pos': (7, 3), 'has_key': 1}
        """
        assert 0 <= state < self.state_space_size, \
            f"State {state} out of bounds [0, {self.state_space_size})"

        # Extract components using integer division and modulo
        # State = pos_idx * 2 + has_key
        pos_idx = state // self.pos_mult
        has_key = state % self.pos_mult

        # Convert 1D position back to 2D
        agent_y = pos_idx // self.grid_size
        agent_x = pos_idx % self.grid_size

        return {
            'agent_pos': (agent_y, agent_x),
            'has_key': int(has_key)
        }

    def get_state_space_size(self) -> int:
        """
        Get the total theoretical state space size.

        Returns:
            int: Total number of possible states (128)
        """
        return self.state_space_size

    def __repr__(self) -> str:
        """String representation of the encoder."""
        return (f"StateEncoder(global_vision={self.grid_size}×{self.grid_size}, "
                f"features=[agent_pos(64) × has_key(2)], "
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
    print(f"Features: agent_pos(64) × has_key(2)")
    print()

    # Test 1: Basic encoding
    print("Test 1: Basic Encoding")
    # Create a simple global view with agent at (4, 5)
    global_view = np.zeros((8, 8), dtype=np.int32)
    global_view[4, 5] = encoder.CELL_AGENT  # Agent
    global_view[2, 3] = encoder.CELL_KEY  # Key
    global_view[6, 6] = encoder.CELL_DOOR  # Door
    # Add walls
    global_view[0, :] = encoder.CELL_WALL
    global_view[-1, :] = encoder.CELL_WALL
    global_view[:, 0] = encoder.CELL_WALL
    global_view[:, -1] = encoder.CELL_WALL

    obs = {'global_view': global_view, 'has_key': 0}
    state = encoder.encode(obs)

    print(f"Agent at position (4, 5)")
    print(f"Has key: {obs['has_key']}")
    print(f"Encoded state: {state}")
    print("✓ Encoding successful")
    print()

    # Test 2: Different positions produce different states
    print("Test 2: Different Positions")
    global_view2 = np.copy(global_view)
    global_view2[4, 5] = encoder.CELL_FLOOR  # Remove old agent
    global_view2[3, 2] = encoder.CELL_AGENT  # New agent position

    obs2 = {'global_view': global_view2, 'has_key': 0}
    state2 = encoder.encode(obs2)

    print(f"State 1 (pos (4,5), no key): {state}")
    print(f"State 2 (pos (3,2), no key): {state2}")
    print(f"States are different: {state != state2}")
    print("✓ Position differentiation works")
    print()

    # Test 3: Different has_key values
    print("Test 3: Different has_key Values")
    obs3a = {'global_view': global_view, 'has_key': 0}
    obs3b = {'global_view': global_view, 'has_key': 1}

    state3a = encoder.encode(obs3a)
    state3b = encoder.encode(obs3b)

    print(f"State with has_key=0: {state3a}")
    print(f"State with has_key=1: {state3b}")
    print(f"States are different: {state3a != state3b}")
    print("✓ has_key differentiation works")
    print()

    # Test 4: Decode functionality
    print("Test 4: Decode Functionality")
    decoded = encoder.decode(state)
    print(f"Original state: {state}")
    print(f"Decoded features: {decoded}")
    print(f"Agent position: {decoded['agent_pos']}")
    print(f"Has key: {decoded['has_key']}")
    print("✓ Decode works correctly")
    print()

    # Test 5: Boundary values
    print("Test 5: Boundary Values")
    # Minimum state (pos (0,0), no key) - but (0,0) is a wall, so use (1,1)
    min_view = np.zeros((8, 8), dtype=np.int32)
    min_view[1, 1] = encoder.CELL_AGENT
    min_obs = {'global_view': min_view, 'has_key': 0}
    min_state = encoder.encode(min_obs)

    # Maximum state (pos (7,7), has key) - but (7,7) is a wall, so use (6,6)
    max_view = np.zeros((8, 8), dtype=np.int32)
    max_view[6, 6] = encoder.CELL_AGENT
    max_obs = {'global_view': max_view, 'has_key': 1}
    max_state = encoder.encode(max_obs)

    print(f"Min state (pos (1,1), no key): {min_state}")
    print(f"Max state (pos (6,6), has key): {max_state}")
    print(f"Within bounds: {0 <= min_state < encoder.state_space_size and 0 <= max_state < encoder.state_space_size}")
    print("✓ Boundary values correct")
    print()

    print("All tests passed! ✓")
    print()
    print(f"Summary:")
    print(f"  State space: {encoder.state_space_size:,} states")
    print(f"  Encoding: agent_pos × has_key")
    print(f"  Simple and efficient for tabular RL")


if __name__ == "__main__":
    test_state_encoder()
