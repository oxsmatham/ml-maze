import torch
from ml_maze.q_network import QNetwork

def test_q_network_returns_four_values_per_state():
    model = QNetwork()
    states = torch.tensor(
        [[0, 0], [1, 0]],
        dtype = torch.float32
    )
    
    q_values = model(states)
    
    assert q_values.shape == (2, 4)