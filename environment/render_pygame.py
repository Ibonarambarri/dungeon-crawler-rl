"""
PyGame Renderer for Dungeon Crawler - 32×32 with Global Vision

Renders the 32×32 dungeon with camera system centered on agent.
Features:
- Camera system following the agent (shows viewport of full grid)
- Agent has GLOBAL vision (sees entire 32×32 grid)
- Camera viewport shows portion around agent for rendering
- Enemy sprites with distinct appearance
- Smooth rendering with proper layering
"""

import pygame
import numpy as np
from typing import Dict, Tuple, Optional, Any
import sys


class PyGameRenderer:
    """
    PyGame renderer for 32×32 dungeon with global vision.

    Features:
    - Camera system centered on agent showing viewport of grid
    - Agent has GLOBAL vision (sees entire 32×32 grid, not just viewport)
    - Viewport shows 16×16 tiles around agent for comfortable display
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

    def __init__(self, grid_size: int = 32, cell_size: int = 32, fps: int = 10):
        """
        Initialize the PyGame renderer with camera system.

        Args:
            grid_size: Size of the full game grid (default: 32)
            cell_size: Size of each cell in pixels (default: 32 for comfortable view)
            fps: Target frames per second (default: 10)
        """
        pygame.init()

        self.grid_size = grid_size
        self.cell_size = cell_size
        self.fps = fps

        # Camera viewport size (show 16×16 tiles centered on agent)
        self.viewport_tiles_width = 16
        self.viewport_tiles_height = 16

        # Calculate window size based on viewport
        self.game_width = self.viewport_tiles_width * cell_size
        self.game_height = self.viewport_tiles_height * cell_size

        # UI dimensions
        self.ui_height = 100

        # Total window size
        self.window_width = self.game_width
        self.window_height = self.game_height + self.ui_height

        # Create window (512×612 for 16×16 viewport with 32px cells)
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Dungeon Crawler RL - 32×32 (Global Vision, Camera View)")

        # Camera position (top-left of viewport in grid coordinates)
        self.camera_x = 0
        self.camera_y = 0

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

        # Agent sprite (blue circle with simple eyes and smile)
        agent_surf = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
        center = self.cell_size // 2
        radius = center - 3

        # Main body with shadow
        pygame.draw.circle(agent_surf, (30, 100, 200), (center + 1, center + 1), radius)  # Shadow
        pygame.draw.circle(agent_surf, self.COLOR_AGENT, (center, center), radius)  # Main body

        # White outline
        pygame.draw.circle(agent_surf, (255, 255, 255), (center, center), radius, 2)

        # Eyes (smaller and proportional to 32px cell)
        eye_size = 3  # Much smaller
        eye_spacing = self.cell_size // 6
        eye_y = center - self.cell_size // 8

        # Simple black dots for eyes
        pygame.draw.circle(agent_surf, (0, 0, 0), (center - eye_spacing, eye_y), eye_size)
        pygame.draw.circle(agent_surf, (0, 0, 0), (center + eye_spacing, eye_y), eye_size)

        # Small white highlights
        pygame.draw.circle(agent_surf, (255, 255, 255), (center - eye_spacing + 1, eye_y - 1), 1)
        pygame.draw.circle(agent_surf, (255, 255, 255), (center + eye_spacing + 1, eye_y - 1), 1)

        # Small smile (proportional)
        smile_rect = pygame.Rect(center - 6, center + 2, 12, 8)
        pygame.draw.arc(agent_surf, (255, 255, 255), smile_rect, 3.14, 2 * 3.14, 2)

        sprites['agent'] = agent_surf

        # Enemy sprite (red circle with simple angry face)
        enemy_surf = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
        center = self.cell_size // 2
        radius = center - 3

        # Main body with shadow
        pygame.draw.circle(enemy_surf, (150, 20, 20), (center + 1, center + 1), radius)  # Shadow
        pygame.draw.circle(enemy_surf, self.COLOR_ENEMY, (center, center), radius)  # Main body

        # Dark red outline
        pygame.draw.circle(enemy_surf, (150, 0, 0), (center, center), radius, 2)

        # Eyes (smaller and proportional to 32px cell)
        eye_size = 3  # Much smaller
        eye_spacing = self.cell_size // 6
        eye_y = center - self.cell_size // 8

        # Simple yellow dots for angry eyes
        pygame.draw.circle(enemy_surf, (255, 200, 0), (center - eye_spacing, eye_y), eye_size)
        pygame.draw.circle(enemy_surf, (255, 200, 0), (center + eye_spacing, eye_y), eye_size)

        # Dark pupils (very small)
        pygame.draw.circle(enemy_surf, (100, 0, 0), (center - eye_spacing, eye_y), 1)
        pygame.draw.circle(enemy_surf, (100, 0, 0), (center + eye_spacing, eye_y), 1)

        # Small angry eyebrows (proportional)
        brow_y = eye_y - 3
        pygame.draw.line(enemy_surf, (100, 0, 0),
                        (center - eye_spacing - 3, brow_y + 1),
                        (center - eye_spacing + 3, brow_y), 2)
        pygame.draw.line(enemy_surf, (100, 0, 0),
                        (center + eye_spacing - 3, brow_y),
                        (center + eye_spacing + 3, brow_y + 1), 2)

        # Small frown (proportional)
        frown_rect = pygame.Rect(center - 5, center + 3, 10, 6)
        pygame.draw.arc(enemy_surf, (100, 0, 0), frown_rect, 0, 3.14, 2)

        sprites['enemy'] = enemy_surf

        # Door sprite (medieval dungeon door - more realistic)
        door_surf = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
        center = self.cell_size // 2
        margin = 6

        # Door frame (stone/wood)
        frame_color = (80, 60, 40)  # Dark brown
        pygame.draw.rect(door_surf, frame_color, (margin - 2, margin - 2,
                        self.cell_size - 2*margin + 4, self.cell_size - 2*margin + 4))

        # Main door (wooden planks)
        door_color = (120, 80, 40)  # Medium brown
        door_rect = pygame.Rect(margin, margin,
                               self.cell_size - 2*margin, self.cell_size - 2*margin)
        pygame.draw.rect(door_surf, door_color, door_rect)

        # Wooden planks (horizontal lines)
        plank_color = (100, 65, 30)
        num_planks = 4
        plank_height = (self.cell_size - 2*margin) // num_planks
        for i in range(num_planks):
            y = margin + i * plank_height
            pygame.draw.line(door_surf, plank_color,
                           (margin, y), (self.cell_size - margin, y), 2)

        # Vertical reinforcement bars
        bar_color = (60, 60, 60)  # Dark grey
        bar_width = 4
        pygame.draw.rect(door_surf, bar_color,
                        (margin + 8, margin, bar_width, self.cell_size - 2*margin))
        pygame.draw.rect(door_surf, bar_color,
                        (self.cell_size - margin - 12, margin, bar_width, self.cell_size - 2*margin))

        # Metal rivets (decorative)
        rivet_color = (180, 180, 180)  # Silver
        rivet_positions = [
            (margin + 10, margin + 6),
            (margin + 10, self.cell_size - margin - 6),
            (self.cell_size - margin - 10, margin + 6),
            (self.cell_size - margin - 10, self.cell_size - margin - 6),
        ]
        for pos in rivet_positions:
            pygame.draw.circle(door_surf, rivet_color, pos, 3)
            pygame.draw.circle(door_surf, (100, 100, 100), pos, 3, 1)

        # Door handle/lock (golden)
        handle_x = self.cell_size - margin - 10
        handle_y = center
        pygame.draw.circle(door_surf, (255, 215, 0), (handle_x, handle_y), 6)  # Gold ring
        pygame.draw.circle(door_surf, (200, 160, 0), (handle_x, handle_y), 6, 2)  # Darker outline

        # Keyhole in handle
        pygame.draw.circle(door_surf, (50, 40, 20), (handle_x, handle_y), 2)

        # Hinges (left side)
        hinge_color = (80, 80, 80)
        hinge_y_positions = [margin + 10, self.cell_size - margin - 10]
        for hinge_y in hinge_y_positions:
            pygame.draw.rect(door_surf, hinge_color, (margin - 2, hinge_y - 4, 6, 8))

        # Green glow (exit indication)
        glow_surf = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, (50, 255, 100, 40),
                        (margin - 4, margin - 4,
                         self.cell_size - 2*margin + 8, self.cell_size - 2*margin + 8))
        door_surf.blit(glow_surf, (0, 0))

        sprites['door'] = door_surf

        # Wall sprite (stone brick texture)
        wall_surf = pygame.Surface((self.cell_size, self.cell_size))
        wall_surf.fill(self.COLOR_WALL)

        # Brick pattern (2 rows)
        brick_height = self.cell_size // 2
        brick_width = self.cell_size // 2

        # Top row - 2 bricks
        for i in range(2):
            x = i * brick_width
            pygame.draw.rect(wall_surf, (90, 90, 100),
                           (x + 2, 2, brick_width - 4, brick_height - 4))
            pygame.draw.rect(wall_surf, (110, 110, 120),
                           (x + 3, 3, brick_width - 6, brick_height - 6), 2)

        # Bottom row - offset bricks
        for i in range(3):
            x = (i - 0.5) * brick_width
            if x < -brick_width // 2:
                continue
            if x > self.cell_size - brick_width // 2:
                continue
            pygame.draw.rect(wall_surf, (85, 85, 95),
                           (int(x) + 2, brick_height + 2,
                            brick_width - 4, brick_height - 4))
            pygame.draw.rect(wall_surf, (105, 105, 115),
                           (int(x) + 3, brick_height + 3,
                            brick_width - 6, brick_height - 6), 2)

        # Mortar lines (darker)
        mortar_color = (60, 60, 70)
        # Horizontal
        pygame.draw.line(wall_surf, mortar_color, (0, brick_height),
                        (self.cell_size, brick_height), 3)
        # Vertical in top row
        pygame.draw.line(wall_surf, mortar_color, (brick_width, 0),
                        (brick_width, brick_height), 3)
        # Vertical in bottom row (offset)
        pygame.draw.line(wall_surf, mortar_color, (brick_width // 2, brick_height),
                        (brick_width // 2, self.cell_size), 3)
        pygame.draw.line(wall_surf, mortar_color, (brick_width + brick_width // 2, brick_height),
                        (brick_width + brick_width // 2, self.cell_size), 3)

        sprites['wall'] = wall_surf

        # Floor sprite (stone tile)
        floor_surf = pygame.Surface((self.cell_size, self.cell_size))
        floor_surf.fill(self.COLOR_FLOOR)

        # Tile with beveled edges
        tile_margin = 3
        pygame.draw.rect(floor_surf, (45, 45, 55),
                        (tile_margin, tile_margin,
                         self.cell_size - 2*tile_margin, self.cell_size - 2*tile_margin))

        # Lighter center
        inner_margin = tile_margin + 2
        pygame.draw.rect(floor_surf, (48, 48, 58),
                        (inner_margin, inner_margin,
                         self.cell_size - 2*inner_margin, self.cell_size - 2*inner_margin))

        # Grout lines (darker borders)
        pygame.draw.rect(floor_surf, (30, 30, 38),
                        (0, 0, self.cell_size, self.cell_size), tile_margin)

        sprites['floor'] = floor_surf

        return sprites

    def render(self, env_state: Dict[str, Any], info: Dict[str, Any], last_reward: float = 0):
        """
        Render the current game state.

        Args:
            env_state: Dictionary containing:
                - grid: 16×16 numpy array of dungeon layout
                - agent_pos: (y, x) position
                - door_pos: (y, x)
                - enemy1_pos: (y, x)
                - enemy2_pos: (y, x)
                - global_view: 16×16 numpy array (optional)
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

        # Render game view (full 16×16 grid)
        self._render_game_view(env_state)

        # Render UI
        self._render_ui(env_state, info, last_reward)

        # Update display
        pygame.display.flip()
        self.clock.tick(self.fps)

    def _update_camera(self, agent_pos: Tuple[int, int]):
        """
        Update camera position to center on agent.

        Args:
            agent_pos: (y, x) position of agent
        """
        agent_y, agent_x = agent_pos

        # Center camera on agent
        self.camera_x = agent_x - self.viewport_tiles_width // 2
        self.camera_y = agent_y - self.viewport_tiles_height // 2

        # Clamp camera to grid bounds
        self.camera_x = max(0, min(self.camera_x, self.grid_size - self.viewport_tiles_width))
        self.camera_y = max(0, min(self.camera_y, self.grid_size - self.viewport_tiles_height))

    def _render_game_view(self, env_state: Dict[str, Any]):
        """
        Render the game grid with camera viewport centered on agent.
        Note: Agent has GLOBAL vision (sees entire 32×32 grid),
        but we only RENDER a 16×16 viewport for display.

        Args:
            env_state: Current environment state
        """
        grid = env_state.get('grid')
        agent_pos = env_state.get('agent_pos')
        if grid is None or agent_pos is None:
            return

        # Update camera to follow agent
        self._update_camera(agent_pos)

        # Calculate viewport bounds in grid coordinates
        viewport_start_x = int(self.camera_x)
        viewport_start_y = int(self.camera_y)
        viewport_end_x = min(viewport_start_x + self.viewport_tiles_width, self.grid_size)
        viewport_end_y = min(viewport_start_y + self.viewport_tiles_height, self.grid_size)

        # Render tiles within viewport
        for grid_y in range(viewport_start_y, viewport_end_y):
            for grid_x in range(viewport_start_x, viewport_end_x):
                # Screen position (relative to viewport)
                screen_x = (grid_x - viewport_start_x) * self.cell_size
                screen_y = (grid_y - viewport_start_y) * self.cell_size

                # Render tile (wall or floor)
                if grid[grid_y, grid_x] == 1:  # Wall
                    self.screen.blit(self.sprites['wall'], (screen_x, screen_y))
                else:  # Floor
                    self.screen.blit(self.sprites['floor'], (screen_x, screen_y))

        # Render door (if in viewport)
        door_pos = env_state.get('door_pos')
        if door_pos is not None and self._is_in_viewport(door_pos):
            self._render_entity_viewport(door_pos, 'door')

        # Render enemies (if in viewport)
        enemy1_pos = env_state.get('enemy1_pos')
        if enemy1_pos is not None and self._is_in_viewport(enemy1_pos):
            self._render_entity_viewport(enemy1_pos, 'enemy')

        enemy2_pos = env_state.get('enemy2_pos')
        if enemy2_pos is not None and self._is_in_viewport(enemy2_pos):
            self._render_entity_viewport(enemy2_pos, 'enemy')

        # Render agent (always on top, if in viewport - should always be visible)
        if self._is_in_viewport(agent_pos):
            self._render_entity_viewport(agent_pos, 'agent')

    def _is_in_viewport(self, position: Tuple[int, int]) -> bool:
        """
        Check if a position is within the current camera viewport.

        Args:
            position: (y, x) position in grid coordinates

        Returns:
            True if position is visible in current viewport
        """
        y, x = position
        viewport_start_x = int(self.camera_x)
        viewport_start_y = int(self.camera_y)
        viewport_end_x = viewport_start_x + self.viewport_tiles_width
        viewport_end_y = viewport_start_y + self.viewport_tiles_height

        return (viewport_start_x <= x < viewport_end_x and
                viewport_start_y <= y < viewport_end_y)

    def _render_entity_viewport(self, position: Tuple[int, int], sprite_name: str):
        """
        Render an entity at a given position relative to viewport.

        Args:
            position: (y, x) position in grid coordinates
            sprite_name: Name of sprite to render
        """
        y, x = position

        # Convert grid position to screen position (relative to viewport)
        viewport_start_x = int(self.camera_x)
        viewport_start_y = int(self.camera_y)

        screen_x = (x - viewport_start_x) * self.cell_size
        screen_y = (y - viewport_start_y) * self.cell_size

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
    """Test the PyGame renderer with a simple 16×16 scene."""
    print("Testing PyGame Renderer (16×16 Full Grid with Global Vision)...")
    print("Window size: 768×868 px (48px per cell)")

    renderer = PyGameRenderer(grid_size=16, cell_size=48, fps=10)

    # Create a simple test grid (16×16)
    test_grid = np.zeros((16, 16), dtype=np.int8)
    # Add borders
    test_grid[0, :] = 1
    test_grid[-1, :] = 1
    test_grid[:, 0] = 1
    test_grid[:, -1] = 1

    test_state = {
        'grid': test_grid,
        'agent_pos': (8, 8),
        'door_pos': (12, 12),
        'enemy1_pos': (5, 5),
        'enemy2_pos': (10, 6)
    }

    test_info = {
        'steps': 0,
        'dist_to_door': 8
    }

    print("Rendering test scene (close window to exit)...")
    print("Agent will move around the grid. Press close button to stop.")

    running = True
    frame = 0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Move agent in a square pattern
        frame += 1
        if frame % 10 == 0:
            y, x = test_state['agent_pos']
            # Simple movement pattern
            if x < 12 and y == 8:
                x += 1
            elif x == 12 and y < 12:
                y += 1
            elif x > 8 and y == 12:
                x -= 1
            elif x == 8 and y > 8:
                y -= 1
            test_state['agent_pos'] = (y, x)

        renderer.render(test_state, test_info, last_reward=0.0)

    renderer.close()
    print("✓ Renderer test completed")


if __name__ == "__main__":
    test_renderer()
