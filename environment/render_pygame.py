"""
PyGame Renderer for Dungeon Crawler - ULTRA-SIMPLIFIED

Renders the 8×8 dungeon with simple visual elements using PyGame.
No camera system needed (grid is small enough to show full), minimal UI.
"""

import pygame
import numpy as np
from typing import Dict, Tuple, Optional, Any
import sys


class PyGameRenderer:
    """
    Ultra-simplified PyGame renderer for 8×8 navigation environment.

    Features:
    - Full 8×8 grid visible at once (no camera needed)
    - Simple sprites for agent, key, door
    - Minimal UI showing stats
    - No vision overlay (agent has global vision)

    """

    # Color palette
    COLOR_BACKGROUND = (20, 20, 30)
    COLOR_FLOOR = (40, 40, 50)
    COLOR_WALL = (100, 100, 110)
    COLOR_AGENT = (50, 150, 255)         # Blue
    COLOR_KEY = (255, 255, 100)          # Yellow
    COLOR_DOOR = (50, 255, 100)          # Green
    COLOR_TEXT = (255, 255, 255)         # White
    COLOR_UI_BG = (30, 30, 40)
    COLOR_VISION = (100, 200, 255, 80)   # Semi-transparent blue for vision overlay

    def __init__(self, grid_size: int = 8, cell_size: int = 48, fps: int = 10):
        """
        Initialize the PyGame renderer.

        Args:
            grid_size: Size of the game grid (default: 8)
            cell_size: Size of each cell in pixels (default: 48)
            fps: Target frames per second (default: 10)
        """
        pygame.init()

        self.grid_size = grid_size
        self.cell_size = cell_size
        self.fps = fps

        # Calculate window size
        self.game_width = grid_size * cell_size
        self.game_height = grid_size * cell_size

        # UI dimensions
        self.ui_height = 80

        # Total window size
        self.window_width = self.game_width
        self.window_height = self.game_height + self.ui_height

        # Create window
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Dungeon Crawler RL - 8×8")

        # Clock for FPS
        self.clock = pygame.time.Clock()

        # Fonts
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 22)

        # Generate sprites
        self.sprites = self._generate_sprites()

    def _generate_sprites(self) -> Dict[str, pygame.Surface]:
        """
        Generate all game sprites procedurally.

        Returns:
            Dictionary mapping sprite names to surfaces
        """
        sprites = {}

        # Agent sprite (blue circle with eyes)
        agent_surf = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
        pygame.draw.circle(
            agent_surf,
            self.COLOR_AGENT,
            (self.cell_size // 2, self.cell_size // 2),
            self.cell_size // 2 - 2
        )
        # White outline
        pygame.draw.circle(
            agent_surf,
            (255, 255, 255),
            (self.cell_size // 2, self.cell_size // 2),
            self.cell_size // 2 - 2,
            3
        )
        # Eyes
        eye_size = 6
        eye_y = self.cell_size // 2 - 6
        pygame.draw.circle(agent_surf, (255, 255, 255), (self.cell_size // 2 - 8, eye_y), eye_size)
        pygame.draw.circle(agent_surf, (255, 255, 255), (self.cell_size // 2 + 8, eye_y), eye_size)
        pygame.draw.circle(agent_surf, (0, 0, 0), (self.cell_size // 2 - 8, eye_y), eye_size - 3)
        pygame.draw.circle(agent_surf, (0, 0, 0), (self.cell_size // 2 + 8, eye_y), eye_size - 3)
        sprites['agent'] = agent_surf

        # Key sprite (yellow key shape)
        key_surf = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
        center = self.cell_size // 2
        # Key head (circle)
        pygame.draw.circle(key_surf, self.COLOR_KEY, (center, center - 8), 10)
        pygame.draw.circle(key_surf, self.COLOR_BACKGROUND, (center, center - 8), 6)
        # Key shaft
        pygame.draw.rect(key_surf, self.COLOR_KEY, (center - 3, center, 6, 16))
        # Teeth
        pygame.draw.rect(key_surf, self.COLOR_KEY, (center + 3, center + 6, 6, 4))
        pygame.draw.rect(key_surf, self.COLOR_KEY, (center + 3, center + 12, 6, 4))
        sprites['key'] = key_surf

        # Door sprite (green door)
        door_surf = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
        door_margin = 8
        pygame.draw.rect(door_surf, self.COLOR_DOOR,
                        (door_margin, door_margin,
                         self.cell_size - 2*door_margin, self.cell_size - 2*door_margin))
        pygame.draw.rect(door_surf, (30, 180, 60),
                        (door_margin + 2, door_margin + 2,
                         self.cell_size - 2*door_margin - 4, self.cell_size - 2*door_margin - 4))
        # Door handle
        pygame.draw.circle(door_surf, (255, 215, 0), (self.cell_size - door_margin - 6, center), 4)
        sprites['door'] = door_surf

        # Wall sprite (gray brick)
        wall_surf = pygame.Surface((self.cell_size, self.cell_size))
        wall_surf.fill(self.COLOR_WALL)
        # Brick texture
        brick_height = self.cell_size // 2
        pygame.draw.rect(wall_surf, (85, 85, 95), (2, 2, self.cell_size - 4, brick_height - 4))
        pygame.draw.rect(wall_surf, (85, 85, 95), (2, brick_height + 2, self.cell_size - 4, brick_height - 4))
        # Grid lines
        pygame.draw.line(wall_surf, (60, 60, 70), (0, brick_height), (self.cell_size, brick_height), 3)
        pygame.draw.line(wall_surf, (60, 60, 70), (self.cell_size // 2, 0), (self.cell_size // 2, self.cell_size), 3)
        sprites['wall'] = wall_surf

        # Floor sprite (dark tile)
        floor_surf = pygame.Surface((self.cell_size, self.cell_size))
        floor_surf.fill(self.COLOR_FLOOR)
        # Subtle grid
        pygame.draw.rect(floor_surf, (45, 45, 55), (2, 2, self.cell_size - 4, self.cell_size - 4), 1)
        sprites['floor'] = floor_surf

        return sprites

    def render(self, env_state: Dict[str, Any], info: Dict[str, Any], last_reward: float = 0):
        """
        Render the current game state.

        Args:
            env_state: Dictionary containing:
                - grid: 8×8 numpy array of dungeon layout
                - agent_pos: (y, x) position
                - key_pos: (y, x) or None if collected
                - door_pos: (y, x)
                - has_key: boolean
                - global_view: 8×8 numpy array (optional)
            info: Dictionary with game stats
            last_reward: Last reward received
        """
        # Handle pygame events (prevent freezing)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                sys.exit()

        # Clear screen
        self.screen.fill(self.COLOR_BACKGROUND)

        # Render game view (full 8×8 grid)
        self._render_game_view(env_state)

        # Render UI
        self._render_ui(env_state, info, last_reward)

        # Update display
        pygame.display.flip()
        self.clock.tick(self.fps)

    def _render_game_view(self, env_state: Dict[str, Any]):
        """
        Render the full 8×8 game grid.

        Args:
            env_state: Current environment state
        """
        grid = env_state.get('grid')
        if grid is None:
            return

        # Render all tiles
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                screen_x = x * self.cell_size
                screen_y = y * self.cell_size

                # Render tile (wall or floor)
                if grid[y, x] == 1:  # Wall
                    self.screen.blit(self.sprites['wall'], (screen_x, screen_y))
                else:  # Floor
                    self.screen.blit(self.sprites['floor'], (screen_x, screen_y))

        # Render key (if still on map)
        key_pos = env_state.get('key_pos')
        if key_pos is not None:
            self._render_entity(key_pos, 'key')

        # Render door
        door_pos = env_state.get('door_pos')
        if door_pos is not None:
            self._render_entity(door_pos, 'door')

        # Render agent (always on top)
        agent_pos = env_state.get('agent_pos')
        if agent_pos is not None:
            self._render_entity(agent_pos, 'agent')

    def _render_entity(self, position: Tuple[int, int], sprite_name: str):
        """
        Render an entity at a given position.

        Args:
            position: (y, x) position
            sprite_name: Name of sprite to render
        """
        y, x = position
        screen_x = x * self.cell_size
        screen_y = y * self.cell_size
        self.screen.blit(self.sprites[sprite_name], (screen_x, screen_y))

    def _render_ui(self, env_state: Dict[str, Any], info: Dict[str, Any], last_reward: float):
        """
        Render UI overlay at bottom of screen.

        Args:
            env_state: Current environment state
            info: Game statistics
            last_reward: Last reward received
        """
        ui_y = self.game_height
        ui_rect = pygame.Rect(0, ui_y, self.window_width, self.ui_height)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, ui_rect)
        pygame.draw.line(
            self.screen,
            self.COLOR_TEXT,
            (0, ui_y),
            (self.window_width, ui_y),
            2
        )

        # Stats text
        stats_x = 20
        stats_y = ui_y + 15

        # Steps
        steps_text = self.font_medium.render(
            f"Steps: {info.get('steps', 0)}",
            True,
            self.COLOR_TEXT
        )
        self.screen.blit(steps_text, (stats_x, stats_y))

        # Has key indicator
        key_status = "Yes" if env_state.get('has_key', False) else "No"
        key_text = self.font_medium.render(
            f"Has Key: {key_status}",
            True,
            self.COLOR_KEY if env_state.get('has_key', False) else (150, 150, 150)
        )
        self.screen.blit(key_text, (stats_x + 150, stats_y))

        # Last reward
        reward_color = (50, 255, 100) if last_reward > 0 else (255, 50, 50) if last_reward < 0 else self.COLOR_TEXT
        reward_text = self.font_small.render(
            f"Reward: {last_reward:+.2f}",
            True,
            reward_color
        )
        self.screen.blit(reward_text, (stats_x, stats_y + 35))

        # Distance to goal
        dist_text = self.font_small.render(
            f"Dist to Key: {info.get('dist_to_key', 0)}  |  Dist to Door: {info.get('dist_to_door', 0)}",
            True,
            (180, 180, 180)
        )
        self.screen.blit(dist_text, (stats_x + 150, stats_y + 35))

    def close(self):
        """Clean up pygame resources."""
        pygame.quit()


def test_renderer():
    """Test the PyGame renderer with a simple 8×8 scene."""
    print("Testing PyGame Renderer (8×8)...")

    renderer = PyGameRenderer(grid_size=8, cell_size=48, fps=10)

    # Create a simple test grid (8×8)
    test_grid = np.zeros((8, 8), dtype=np.int8)
    # Add borders
    test_grid[0, :] = 1
    test_grid[-1, :] = 1
    test_grid[:, 0] = 1
    test_grid[:, -1] = 1

    test_state = {
        'grid': test_grid,
        'agent_pos': (4, 4),
        'key_pos': (2, 2),
        'door_pos': (6, 6),
        'has_key': False
    }

    test_info = {
        'steps': 0,
        'dist_to_key': 4,
        'dist_to_door': 4
    }

    print("Rendering test scene (close window to exit)...")
    print("Agent will move around the grid. Press close button to stop.")

    running = True
    frame = 0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Move agent in a circle
        frame += 1
        if frame % 10 == 0:
            y, x = test_state['agent_pos']
            # Simple movement pattern
            if x < 6:
                x += 1
            elif y < 6:
                y += 1
            elif x > 1:
                x -= 1
            elif y > 1:
                y -= 1
            test_state['agent_pos'] = (y, x)

        renderer.render(test_state, test_info, last_reward=0.0)

    renderer.close()
    print("✓ Renderer test completed")


if __name__ == "__main__":
    test_renderer()
