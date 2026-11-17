import gymnasium as gym
import numpy as np
from typing import Any

class MazeEnv(gym.Env):
    
    MOVEMENTS = {
        0: ( 0, -1), # go up
        1: ( 1,  0), # go right
        2: ( 0,  1), # go down
        3: (-1,  0), # go left
    }
    
    def __init__(self):
        self.grid = [
            [0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0],
        ]
        
        self.start = (0, 0)
        self.goal = (4, 4)
        self.current_pos = [0, 0]
        self.step_counter = 0
        self.step_limit = 100
        
        self.grid_height = len(self.grid)
        self.grid_width = len(self.grid[0])
        
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.MultiDiscrete(
            [self.grid_width, self.grid_height]
        )
    
    def get_cell(self, x: int, y: int) -> int:
        # Check x,y types
        if not type(x) is int:
            raise TypeError
        if not type(y) is int:
            raise TypeError
        
        return self.grid[y][x]
    
    def is_valid_position(self, position: tuple[int, int]) -> bool:
        
        # Check if position exists
        if not position:
            return False
        if len(position) != 2:
            return False
        
        x = position[0]
        y = position[1]
        
        # Check if position is in bounds
        if x < 0:
            return False
        if y < 0:
            return False
        if x >= self.grid_width:
            return False
        if y >= self.grid_height:
            return False

        # Check if position collides
        if self.get_cell(x, y) == 1:
            return False

        return True
    
    def move_agent(self, action: int) -> bool:
        if not action in self.MOVEMENTS:
            raise ValueError

        current_x = self.current_pos[0]
        current_y = self.current_pos[1]
        
        move_x = self.MOVEMENTS[action][0]
        move_y = self.MOVEMENTS[action][1]
        
        new_x = current_x + move_x
        new_y = current_y + move_y
        
        if self.is_valid_position((new_x, new_y)):
            self.current_pos = [new_x, new_y]
            return True
        
        return False

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed = seed)
        self.current_pos = list(self.start)
        
        self.step_counter = 0
        
        observation = np.array(self.current_pos.copy())
        info = {}
        
        return observation, info

    def step(self, action):
        self.move_agent(action)
        self.step_counter += 1
        
        observation = np.array(self.current_pos.copy())
        
        reward = 5 if self.current_pos == list(self.goal) else -1
        
        terminated = True if self.current_pos == list(self.goal) else False
        
        truncated = True if self.step_counter >= self.step_limit else False
        
        info = {}
        
        return observation, reward, terminated, truncated, info