These are pre-trained q-models with these configs:

```python
grid_layout = [
        ['S', ' ', 'R', ' '],
        [' ', 'O', ' ', ' '],
        [' ', ' ', ' ', 'O'],
        ['O', ' ', ' ', 'G']
    ]

# Q-Learning hyperparameters
alpha = 0.8  # learning rate
gamma = 0.93  # discount factor

# Epsilon-greedy policy parameters
epsilon = 1.0
min_epsilon = 0.05
decay = 0.01

# Exponential decay of epsilon
epsilon = max(min_epsilon, epsilon * np.exp(-decay))
```
'''
Custom Gym environment with configurable grid layout
'''
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import numpy as np
import pygame
import sys
from os import path
from enum import Enum

# Register this module as a gym environment
register(
    id='custom-grid-v0',
    entry_point='custom_grid_env:CustomGridEnv',
)

class GridAction(Enum):
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3

class GridTile(Enum):
    EMPTY = 0      # Regular floor
    OBSTACLE = 1   # Obstacle (robot can't move here)
    SAFE_ZONE = 2  # Safe zone (gives reward)
    GOAL = 3       # Goal (terminal state)

class CustomGridEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], 'render_fps': 4}

    def __init__(self, 
                 grid_layout=None,
                 render_mode=None,
                 agent_sprite='sprites/robot.png', # change this to robot for presentation woi
                 goal_sprite='sprites/house.png',
                 floor_sprite='sprites/tiles.jpg',
                 obstacle_sprite='sprites/pedestrian.png', # change this to pedestrian for presentation woi
                 safe_zone_sprite='sprites/safe.jpg',
                 cell_size=64):
        
        # Default 4x4 layout if none provided
        if grid_layout is None:
            # 'S' = Start, 'G' = Goal, 'O' = Obstacle, 'R' = Safe zone (Reward), ' ' = Empty
            grid_layout = [
                ['S', ' ', 'R', ' '],
                [' ', 'O', ' ', ' '],
                [' ', ' ', ' ', 'O'],
                ['O', ' ', ' ', 'G']
            ]
        
        self.grid_layout = grid_layout
        self.grid_rows = len(grid_layout)
        self.grid_cols = len(grid_layout[0])
        self.render_mode = render_mode
        self.cell_size = cell_size
        
        # Parse grid layout
        self._parse_grid_layout()
        
        # Sprite paths
        self.agent_sprite_path = agent_sprite
        self.goal_sprite_path = goal_sprite
        self.floor_sprite_path = floor_sprite
        self.obstacle_sprite_path = obstacle_sprite
        self.safe_zone_sprite_path = safe_zone_sprite
        
        # Action space: 4 directions
        self.action_space = spaces.Discrete(4)
        
        # Observation space: [agent_row, agent_col, goal_row, goal_col]
        self.observation_space = spaces.Box(
            low=0,
            high=np.array([self.grid_rows-1, self.grid_cols-1, self.grid_rows-1, self.grid_cols-1]),
            shape=(4,),
            dtype=np.int32
        )
        
        # Initialize positions
        self.agent_pos = None
        self.visited_safe_zones = set()
        
        # Pygame initialization
        self.window_surface = None
        self.clock = None
        self.sprites_loaded = False
        
    def _parse_grid_layout(self):
        """Parse the grid layout to find start, goal, obstacles, and safe zones"""
        self.start_pos = None
        self.goal_pos = None
        self.obstacles = []
        self.safe_zones = []
        
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                cell = self.grid_layout[r][c]
                
                if cell == 'S':
                    self.start_pos = [r, c]
                elif cell == 'G':
                    self.goal_pos = [r, c]
                elif cell == 'O':
                    self.obstacles.append([r, c])
                elif cell == 'R':
                    self.safe_zones.append([r, c])
        
        # Ensure start and goal are defined
        if self.start_pos is None:
            self.start_pos = [0, 0]
        if self.goal_pos is None:
            self.goal_pos = [self.grid_rows-1, self.grid_cols-1]
        
    def _init_pygame(self):
        """Initialize pygame and load sprites"""
        if self.sprites_loaded:
            return
            
        pygame.init()
        pygame.display.init()
        
        self.clock = pygame.time.Clock()
        
        # Font for action and info display
        self.action_font = pygame.font.SysFont("Calibri", 30)
        self.info_font = pygame.font.SysFont("Calibri", 24)
        self.action_info_height = self.action_font.get_height() + self.info_font.get_height() + 10
        
        # Window size
        self.window_size = (
            self.cell_size * self.grid_cols, 
            self.cell_size * self.grid_rows + self.action_info_height
        )
        
        self.window_surface = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption("Custom Grid Environment")
        
        # Load sprites
        try:
            # Load agent sprite
            if path.exists(self.agent_sprite_path):
                img = pygame.image.load(self.agent_sprite_path)
                self.agent_img = pygame.transform.scale(img, (self.cell_size, self.cell_size))
            else:
                self.agent_img = self._create_default_sprite((0, 100, 255), "A")
            
            # Load goal sprite
            if path.exists(self.goal_sprite_path):
                img = pygame.image.load(self.goal_sprite_path)
                self.goal_img = pygame.transform.scale(img, (self.cell_size, self.cell_size))
            else:
                self.goal_img = self._create_default_sprite((0, 255, 0), "G")
            
            # Load floor sprite
            if path.exists(self.floor_sprite_path):
                img = pygame.image.load(self.floor_sprite_path)
                self.floor_img = pygame.transform.scale(img, (self.cell_size, self.cell_size))
            else:
                self.floor_img = self._create_default_floor()
            
            # Load obstacle sprite
            if path.exists(self.obstacle_sprite_path):
                img = pygame.image.load(self.obstacle_sprite_path)
                self.obstacle_img = pygame.transform.scale(img, (self.cell_size, self.cell_size))
            else:
                self.obstacle_img = self._create_default_sprite((100, 100, 100), "O")
            
            # Load safe zone sprite
            if path.exists(self.safe_zone_sprite_path):
                img = pygame.image.load(self.safe_zone_sprite_path)
                self.safe_zone_img = pygame.transform.scale(img, (self.cell_size, self.cell_size))
            else:
                self.safe_zone_img = self._create_default_sprite((255, 215, 0), "R")
                
        except Exception as e:
            print(f"Error loading sprites: {e}")
            print("Using default sprites instead")
            self.agent_img = self._create_default_sprite((0, 100, 255), "A")
            self.goal_img = self._create_default_sprite((0, 255, 0), "G")
            self.floor_img = self._create_default_floor()
            self.obstacle_img = self._create_default_sprite((100, 100, 100), "O")
            self.safe_zone_img = self._create_default_sprite((255, 215, 0), "R")
        
        self.sprites_loaded = True
        self.last_action = "None"
        self.total_reward = 0
    
    def _create_default_sprite(self, color, letter=""):
        """Create a default colored sprite with letter"""
        surface = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
        pygame.draw.rect(surface, color, (5, 5, self.cell_size-10, self.cell_size-10))
        
        if letter:
            font = pygame.font.SysFont("Arial", self.cell_size // 2, bold=True)
            text = font.render(letter, True, (255, 255, 255))
            text_rect = text.get_rect(center=(self.cell_size // 2, self.cell_size // 2))
            surface.blit(text, text_rect)
        
        return surface
    
    def _create_default_floor(self):
        """Create a default floor tile"""
        surface = pygame.Surface((self.cell_size, self.cell_size))
        surface.fill((240, 240, 240))
        pygame.draw.rect(surface, (200, 200, 200), 
                        (0, 0, self.cell_size, self.cell_size), 1)
        return surface
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset agent position to start position
        self.agent_pos = np.array(self.start_pos.copy())
        self.visited_safe_zones = set()
        self.total_reward = 0
        
        obs = np.concatenate((self.agent_pos, self.goal_pos))
        info = {}
        
        if self.render_mode == 'human':
            self.render()
        
        return obs, info
    
    def step(self, action):
        # Store old position
        old_pos = self.agent_pos.copy()
        
        # Perform action
        if action == GridAction.LEFT.value:
            if self.agent_pos[1] > 0:
                self.agent_pos[1] -= 1
        elif action == GridAction.RIGHT.value:
            if self.agent_pos[1] < self.grid_cols - 1:
                self.agent_pos[1] += 1
        elif action == GridAction.UP.value:
            if self.agent_pos[0] > 0:
                self.agent_pos[0] -= 1
        elif action == GridAction.DOWN.value:
            if self.agent_pos[0] < self.grid_rows - 1:
                self.agent_pos[0] += 1
        
        # Check if hit obstacle - revert move
        if list(self.agent_pos) in self.obstacles:
            self.agent_pos = old_pos
            reward = -1  # Penalty for hitting obstacle
            terminated = False
        # Check if reached goal
        elif np.array_equal(self.agent_pos, self.goal_pos):
            reward = 10  # Big reward for reaching goal
            terminated = True
        # Check if reached safe zone
        elif list(self.agent_pos) in self.safe_zones:
            pos_tuple = tuple(self.agent_pos)
            if pos_tuple not in self.visited_safe_zones:
                self.visited_safe_zones.add(pos_tuple)
                reward = 1  # Reward for first visit to safe zone
            else:
                reward = 0  # No reward for revisiting
            terminated = False
        else:
            reward = 0  # No reward for regular move
            terminated = False
        
        self.total_reward += reward
        
        obs = np.concatenate((self.agent_pos, self.goal_pos))
        info = {'total_reward': self.total_reward}
        
        if self.render_mode == 'human':
            self.last_action = GridAction(action).name
            self.render()
        
        return obs, reward, terminated, False, info
    
    def render(self):
        if self.render_mode is None:
            return
        
        if self.render_mode == "human" or self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if not self.sprites_loaded:
            self._init_pygame()
        
        self._process_events()
        
        # 1. Clear screen
        self.window_surface.fill((255, 255, 255))
        
        # 2. Draw grid and objects
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                pos = (c * self.cell_size, r * self.cell_size)
                
                # Draw floor
                self.window_surface.blit(self.floor_img, pos)
                
                # Draw safe zones
                if [r, c] in self.safe_zones:
                    self.window_surface.blit(self.safe_zone_img, pos)
                    if tuple([r, c]) in self.visited_safe_zones:
                        dim_overlay = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
                        dim_overlay.fill((0, 0, 0, 100))
                        self.window_surface.blit(dim_overlay, pos)
                
                # Draw obstacles
                if [r, c] in self.obstacles:
                    self.window_surface.blit(self.obstacle_img, pos)
                
                # Draw goal
                if np.array_equal([r, c], self.goal_pos):
                    self.window_surface.blit(self.goal_img, pos)
                
                # Draw agent
                if np.array_equal([r, c], self.agent_pos):
                    self.window_surface.blit(self.agent_img, pos)
        
        # 3. Draw action and reward text
        text_y = self.window_size[1] - self.action_info_height
        action_text = self.action_font.render(f'Action: {self.last_action}', True, (0, 0, 0))
        self.window_surface.blit(action_text, (10, text_y))
        
        reward_text = self.info_font.render(
            f'Total Reward: {self.total_reward:.1f} | Safe Zones: {len(self.visited_safe_zones)}/{len(self.safe_zones)}', 
            True, (0, 100, 0)
        )
        self.window_surface.blit(reward_text, (10, text_y + self.action_font.get_height() + 5))

        # 4. Handle Return Modes
        if self.render_mode == "human":
            pygame.display.update()
            self.clock.tick(self.metadata['render_fps'])
            return None
        else: # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2)
            )
    
    def _process_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
    
    def close(self):
        if self.window_surface is not None:
            pygame.quit()

# Test the environment
if __name__ == "__main__":
    # Define custom grid layout
    custom_layout = [
        ['S', ' ', 'R', ' '],
        [' ', 'O', ' ', ' '],
        [' ', ' ', ' ', 'O'],
        ['O', ' ', ' ', 'G']
    ]
    
    # Create environment with custom layout
    env = gym.make('custom-grid-v0', 
                   grid_layout=custom_layout,
                   render_mode='human',
                   cell_size=80)
    
    obs = env.reset()[0]
    print("Environment created!")
    print(f"Observation: {obs}")
    # Access the unwrapped environment to get custom attributes
    unwrapped_env = env.unwrapped
    print(f"Grid size: {unwrapped_env.grid_rows}x{unwrapped_env.grid_cols}")
    print(f"Start: {unwrapped_env.start_pos}, Goal: {unwrapped_env.goal_pos}")
    print(f"Obstacles: {unwrapped_env.obstacles}")
    print(f"Safe zones: {unwrapped_env.safe_zones}")

    # Take random actions
    for i in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step {i}: Reward = {reward}, Total = {info['total_reward']}")
        
        if terminated:
            print("Goal reached! Resetting...")
            obs = env.reset()[0]
    
    env.close()
