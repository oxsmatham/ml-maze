from ml_maze.environment import MazeEnv
from ml_maze.agent import Agent
import torch
import math

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

def test_learn_updates_online_network_only():
    torch.manual_seed(0)
    env = MazeEnv()
    agent = Agent(env, gamma = 0.99, learning_rate = 0.001)
    batch = [
        ([0, 0], 1, -1.0, [1, 0], False),
        ([3, 4], 1, 5.0, [4, 4], True)
    ]
    
    q_network_before = [
        parameter.detach().clone()
        for parameter in agent.q_network.parameters()
    ]
    target_network_before = [
        parameter.detach().clone()
        for parameter in agent.target_network.parameters()
    ]
    
    assert all(
        torch.equal(q_network, target)
        for q_network, target in zip(q_network_before, target_network_before)
    )
    
    loss = agent.learn(batch)
    
    q_network_changed = any(
        not torch.equal(before, after)
        for before, after in zip(q_network_before, agent.q_network.parameters())
    )
    
    target_network_changed = any(
        not torch.equal(before, after)
        for before, after in zip(target_network_before, agent.target_network.parameters())
    )
    
    assert q_network_changed # verändert
    assert not target_network_changed # unverändert
    
    assert math.isfinite(loss) == True
    assert loss >= 0