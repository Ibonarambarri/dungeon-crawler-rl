"""
Procedural Dungeon Generator - ULTRA-SIMPLIFIED VERSION

Generates simple 8×8 empty dungeons with only border walls.
Random spawn positions for agent and door.

ULTRA-SIMPLIFIED for tabular RL:
- Grid size: 8×8
- Only border walls (no internal walls)
- Only 2 spawn positions: agent, door

Design:
- ~56% floor (36 of 64 cells) - only borders are walls
- ~44% walls (28 border cells)
- Completely open navigation space
"""

import numpy as np
import random
from typing import Tuple


class DungeonGenerator:
    """
    Generates simple 8×8 empty dungeons.

    ULTRA-SIMPLIFIED VERSION for tabular RL navigation.

    The generator creates an 8×8 grid with:
    - Border walls only
    - Random spawn positions for agent, key, door
    - Completely open interior (6×6 navigable space)
    """

    def __init__(self, width: int = 8, height: int = 8, seed: int = None):
        """
        Initialize the dungeon generator.

        Args:
            width: Grid width (default: 8)
            height: Grid height (default: 8)
            seed: Random seed for reproducibility
        """
        self.width = width
        self.height = height
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.grid = np.zeros((height, width), dtype=np.int8)

    def generate(self) -> dict:
        """
        Generate a complete dungeon.

        ULTRA-SIMPLIFIED VERSION - empty grid with border walls only.

        Returns:
            dict: Dungeon data including:
                - grid: 8×8 numpy array (0=floor, 1=wall)
                - walls: list of wall positions
                - floor: list of floor positions
                - agent_start: agent starting position (y, x)
                - door_pos: door/exit position (y, x)
        """
        # Create empty grid with border walls
        self._create_empty_grid()

        # Find random spawn positions
        spawn_positions = self._find_spawn_positions()

        return {
            'grid': self.grid,
            'walls': self._get_wall_positions(),
            'floor': self._get_floor_positions(),
            **spawn_positions
        }

    def _create_empty_grid(self):
        """Create 8×8 grid with only border walls."""
        # All floor
        self.grid = np.zeros((self.height, self.width), dtype=np.int8)

        # Add border walls
        self.grid[0, :] = 1  # Top border
        self.grid[-1, :] = 1  # Bottom border
        self.grid[:, 0] = 1  # Left border
        self.grid[:, -1] = 1  # Right border

    def _get_wall_positions(self) -> list:
        """Get list of all wall positions."""
        walls = []
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y, x] == 1:
                    walls.append((y, x))
        return walls

    def _get_floor_positions(self) -> list:
        """Get list of all floor positions (interior only, excluding borders)."""
        floors = []
        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                if self.grid[y, x] == 0:
                    floors.append((y, x))
        return floors

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

        # Get all floor positions (interior only)
        floor_positions = []
        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                pos = (y, x)
                if self.grid[y, x] == 0 and pos not in exclude_positions:
                    floor_positions.append(pos)

        if not floor_positions:
            raise ValueError("No available floor positions")

        return floor_positions[np.random.randint(len(floor_positions))]

    def _find_spawn_positions(self) -> dict:
        """
        Find random positions for agent and door.

        All positions are guaranteed to be:
        - On floor tiles (not walls)
        - Unique (no overlaps)
        - In the interior (not on borders)

        Returns:
            dict: Positions for agent_start, door_pos
        """
        # Agent spawn
        agent_pos = self._get_random_floor_position()

        # Door spawn (different from agent)
        door_pos = self._get_random_floor_position(
            exclude_positions=[agent_pos]
        )

        return {
            'agent_start': agent_pos,
            'door_pos': door_pos
        }


def test_generator():
    """Test the ultra-simplified dungeon generator."""
    print("Testing Dungeon Generator (8×8 ULTRA-SIMPLIFIED)...")
    print()

    generator = DungeonGenerator(width=8, height=8, seed=42)
    dungeon = generator.generate()

    print(f"Generated dungeon:")
    print(f"  Grid size: {dungeon['grid'].shape}")
    print(f"  Walls: {len(dungeon['walls'])}")
    print(f"  Floor tiles: {len(dungeon['floor'])}")
    print(f"  Agent start: {dungeon['agent_start']}")
    print(f"  Door position: {dungeon['door_pos']}")
    print()

    # Visualize (simple ASCII)
    print("ASCII Visualization (8×8):")
    grid = dungeon['grid']
    for y in range(grid.shape[0]):
        row = ""
        for x in range(grid.shape[1]):
            if (y, x) == dungeon['agent_start']:
                row += '@'
            elif (y, x) == dungeon['door_pos']:
                row += 'X'
            elif grid[y, x] == 1:
                row += '#'
            else:
                row += '.'
        print(row)

    print()
    print("✓ Dungeon generation successful!")


if __name__ == "__main__":
    test_generator()
