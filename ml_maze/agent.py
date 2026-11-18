from ml_maze.q_network import QNetwork
import torch
import random

class Agent():
    def __init__(self, env):
        self.env = env
        self.q_network = QNetwork()

    def select_action(self, state, epsilon) -> int:
        
        if random.random() < epsilon:
            return self.env.action_space.sample()
        
        state_tensor = torch.as_tensor(state, dtype = torch.float32)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            action = torch.argmax(q_values).item()
        
        return action