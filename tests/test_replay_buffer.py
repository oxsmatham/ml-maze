import pytest
from ml_maze.replay_buffer import ReplayBuffer

def test_push_overflow():
    buffer = ReplayBuffer(max_capacity = 2)
    
    assert len(buffer) == 0
    
    buffer.push(("state", 1, 0, "state_1", False))
    
    assert len(buffer) == 1
    
    buffer.push(("state_1", 1, 0, "state_2", False))
    
    assert len(buffer) == 2
    
    buffer.push(("state_2", 1, 0, "state_3", False))
    
    assert len(buffer) == 2
    
    assert buffer.transitions == [("state_1", 1, 0, "state_2", False), ("state_2", 1, 0, "state_3", False)]

def test_sample_buffer_too_small():
    buffer = ReplayBuffer(max_capacity = 2)
    buffer.push(("state", 1, 0, "next_state", False))
    
    with pytest.raises(ValueError):
        buffer.sample(batch_size = 2)

def test_sample_return():
    buffer = ReplayBuffer(max_capacity = 3)
    transitions = [
        ("s0", 0, 0, "s1", False),
        ("s1", 1, 0, "s2", False),
        ("s2", 2, 1, "s3", False),
    ]
    for transition in transitions:
        buffer.push(transition)
    
    sample = buffer.sample(batch_size = 2)
    
    assert len(sample) == 2
    assert all(transition in transitions for transition in sample)