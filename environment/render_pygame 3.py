"""
PyGame Renderer for Dungeon Crawler - 16×16 with Local Vision

Renders the 16×16 dungeon with camera system centered on agent.
Features:
- Camera system following the agent
- Local vision (5×5) overlay highlighting visible area
- Enemy sprites with distinct appearance
- Smooth rendering with proper layering
"""

import pygame
import numpy as np
from typing import Dict, Tuple, Optional, Any
import sys


class PyGameRenderer:
    """
    PyGame renderer for 16×16 dungeon with local vision and camera system.

    Features:
    - Camera system centered on agent (shows portion of 16×16 grid)
    - Local vision (5×5) overlay highlighting visible area
    - Simple sprites for agent, door, and enemies
    - UI showing stats and information

    """

    # Color palette
    COLOR_BACKGROUND = (20, 20, 30)
    COLOR_FLOOR = (40, 40, 50)
    COLOR_WALL = (100, 100, 110)
    COLOR_AGENT = (50, 150, 255)         # Blue
    COLOR_DOOR = (50, 255, 100)          # Green
    COLOR_ENEMY = (255, 50, 50)          # Red
    COLOR_TEXT = (255, 255, 255)         # White
    COLOR_UI_BG = (30, 30, 40)
    COLOR_VISION = (100, 200, 255, 60)   # Semi-transparent blue for vision overlay

    def __init__(self, grid_size: int = 16, cell_size: int = 32, local_view_size: int = 5, fps: int = 10):
        """
        Initialize the PyGame renderer.

        Args:
            grid_size: Size of the game grid (default: 16)
            cell_size: Size of each cell in pixels (default: 32)
            local_view_size: Size of local vision window (default: 5)
            fps: Target frames per second (default: 10)
        """
        pygame.init()

        self.grid_size = grid_size
        self.cell_size = cell_size
        self.local_view_size = local_view_size
        self.local_view_radius = local_view_size // 2
        self.fps = fps

        # Camera viewport size (show portion of the grid)
        self.camera_tiles_width = min(12, grid_size)  # Show 12x12 tiles max
        self.camera_tiles_height = min(12, grid_size)

        # Calculate window size
        self.game_width = self.camera_tiles_width * cell_size
        self.game_height = self.camera_tiles_height * cell_size

        # UI dimensions
        self.ui_height = 80

        # Total window size
        self.window_width = self.game_width
        self.window_height = self.game_height + self.ui_height

        # Create window
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Dungeon Crawler RL - 16×16 (Local Vision 5×5)")

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

        # Enemy sprite (red circle with angry eyes)
        enemy_surf = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
        pygame.draw.circle(
            enemy_surf,
            self.COLOR_ENEMY,
            (self.cell_size // 2, self.cell_size // 2),
            self.cell_size // 2 - 2
        )
        # Dark outline
        pygame.draw.circle(
            enemy_surf,
            (150, 0, 0),
            (self.cell_size // 2, self.cell_size // 2),
            self.cell_size // 2 - 2,
            3
        )
        # Angry eyes
        eye_size = 6
        eye_y = self.cell_size // 2 - 6
        pygame.draw.circle(enemy_surf, (255, 255, 0), (self.cell_size // 2 - 8, eye_y), eye_size)
        pygame.draw.circle(enemy_surf, (255, 255, 0), (self.cell_size // 2 + 8, eye_y), eye_size)
        pygame.draw.circle(enemy_surf, (0, 0, 0), (self.cell_size // 2 - 8, eye_y), eye_size - 3)
        pygame.draw.circle(enemy_surf, (0, 0, 0), (self.cell_size // 2 + 8, eye_y), eye_size - 3)
        # Angry mouth
        pygame.draw.arc(enemy_surf, (0, 0, 0),
                       (self.cell_size // 2 - 8, self.cell_size // 2 + 4, 16, 10),
                       3.14, 6.28, 3)
        sprites['enemy'] = enemy_surf

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

    def _get_camera_bounds(self, agent_pos: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """
        Calculate camera bounds centered on agent.

        Args:
            agent_pos: (y, x) position of agent

        Returns:
            Tuple (min_y, max_y, min_x, max_x) of tiles to render
        """
        agent_y, agent_x = agent_pos

        # Center camera on agent
        half_width = self.camera_tiles_width // 2
        half_height = self.camera_tiles_height // 2

        min_x = agent_x - half_width
        max_x = agent_x + half_width + 1
        min_y = agent_y - half_height
        max_y = agent_y + half_height + 1

        # Clamp to grid bounds
        min_x = max(0, min_x)
        max_x = min(self.grid_size, max_x)
        min_y = max(0, min_y)
        max_y = min(self.grid_size, max_y)

        # Adjust if near edges to keep camera size constant
        if max_x - min_x < self.camera_tiles_width:
            if min_x == 0:
                max_x = min(self.camera_tiles_width, self.grid_size)
            elif max_x == self.grid_size:
                min_x = max(0, self.grid_size - self.camera_tiles_width)

        if max_y - min_y < self.camera_tiles_height:
            if min_y == 0:
                max_y = min(self.camera_tiles_height, self.grid_size)
            elif max_y == self.grid_size:
                min_y = max(0, self.grid_size - self.camera_tiles_height)

        return min_y, max_y, min_x, max_x

    def _render_game_view(self, env_state: Dict[str, Any]):
        """
        Render the game grid with camera system centered on agent.

        Args:
            env_state: Current environment state
        """
        grid = env_state.get('grid')
        agent_pos = env_state.get('agent_pos')
        if grid is None or agent_pos is None:
            return

        # Get camera bounds
        min_y, max_y, min_x, max_x = self._get_camera_bounds(agent_pos)

        # Render visible tiles
        for y in range(min_y, max_y):
            for x in range(min_x, max_x):
                screen_x = (x - min_x) * self.cell_size
                screen_y = (y - min_y) * self.cell_size

                # Render tile (wall or floor)
                if grid[y, x] == 1:  # Wall
                    self.screen.blit(self.sprites['wall'], (screen_x, screen_y))
                else:  # Floor
                    self.screen.blit(self.sprites['floor'], (screen_x, screen_y))

        # Render door (if visible)
        door_pos = env_state.get('door_pos')
        if door_pos is not None:
            self._render_entity_with_camera(door_pos, 'door', min_y, min_x)

        # Render enemies (if visible)
        enemy1_pos = env_state.get('enemy1_pos')
        if enemy1_pos is not None:
            self._render_entity_with_camera(enemy1_pos, 'enemy', min_y, min_x)

        enemy2_pos = env_state.get('enemy2_pos')
        if enemy2_pos is not None:
            self._render_entity_with_camera(enemy2_pos, 'enemy', min_y, min_x)

        # Render agent (always on top, always visible)
        self._render_entity_with_camera(agent_pos, 'agent', min_y, min_x)

        # Render local vision overlay (5×5 around agent)
        self._render_vision_overlay(agent_pos, min_y, min_x)

    def _render_entity_with_camera(self, position: Tuple[int, int], sprite_name: str,
                                    camera_min_y: int, camera_min_x: int):
        """
        Render an entity at a given position with camera offset.

        Args:
            position: (y, x) position in grid coordinates
            sprite_name: Name of sprite to render
            camera_min_y: Camera's minimum y coordinate
            camera_min_x: Camera's minimum x coordinate
        """
        y, x = position
        screen_x = (x - camera_min_x) * self.cell_size
        screen_y = (y - camera_min_y) * self.cell_size

        # Only render if within camera bounds
        if 0 <= screen_x < self.game_width and 0 <= screen_y < self.game_height:
            self.screen.blit(self.sprites[sprite_name], (screen_x, screen_y))

    def _render_vision_overlay(self, agent_pos: Tuple[int, int],
                               camera_min_y: int, camera_min_x: int):
        """
        Render semi-transparent overlay highlighting the agent's 5×5 local vision.

        Args:
            agent_pos: (y, x) position of agent
            camera_min_y: Camera's minimum y coordinate
            camera_min_x: Camera's minimum x coordinate
        """
        agent_y, agent_x = agent_pos
        radius = self.local_view_radius

        # Create semi-transparent overlay surface
        overlay = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
        overlay.fill(self.COLOR_VISION)

        # Render overlay for each cell in the 5×5 local view
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                grid_y = agent_y + dy
                grid_x = agent_x + dx

                # Check if within grid bounds
                if 0 <= grid_y < self.grid_size and 0 <= grid_x < self.grid_size:
                    screen_x = (grid_x - camera_min_x) * self.cell_size
                    screen_y = (grid_y - camera_min_y) * self.cell_size

                    # Only render if within camera view
                    if 0 <= screen_x < self.game_width and 0 <= screen_y < self.game_height:
                        self.screen.blit(overlay, (screen_x, screen_y))

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

        # Agent position
        agent_pos = env_state.get('agent_pos', (0, 0))
        pos_text = self.font_medium.render(
            f"Pos: ({agent_pos[0]}, {agent_pos[1]})",
            True,
            self.COLOR_TEXT
        )
        self.screen.blit(pos_text, (stats_x + 150, stats_y))

        # Last reward
        reward_color = (50, 255, 100) if last_reward > 0 else (255, 50, 50) if last_reward < 0 else self.COLOR_TEXT
        reward_text = self.font_small.render(
            f"Reward: {last_reward:+.2f}",
            True,
            reward_color
        )
        self.screen.blit(reward_text, (stats_x, stats_y + 35))

        # Distance to door
        dist_text = self.font_small.render(
            f"Dist to Door: {info.get('dist_to_door', 0)}",
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
