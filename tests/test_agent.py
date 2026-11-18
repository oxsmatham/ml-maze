from ml_maze.environment import MazeEnv
from ml_maze.agent import Agent
import torch

def test_select_action_epsilon_1():
    env = MazeEnv()
    agent = Agent(env)
    
    action = agent.select_action(state = [0, 0], epsilon = 1) 
    
    assert env.action_space.contains(action)

def test_select_action_epsilon_0():
    env = MazeEnv()
    agent = Agent(env)
    state = [0, 0]
    
    state_tensor = torch.as_tensor(state, dtype = torch.float32)
    with torch.no_grad():
        expected_action = torch.argmax(agent.q_network(state_tensor)).item()
    
    action = agent.select_action(state = state, epsilon = 0)
    
    assert action == expected_action
    assert env.action_space.contains(action)