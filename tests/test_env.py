import pytest
import numpy as np
from ml_maze.environment import MazeEnv

def test_valid_positions():
    env = MazeEnv()
    assert env.is_valid_position((0, 0)) is True
    assert env.is_valid_position((4, 4)) is True

def test_wall_collisions():
    env = MazeEnv()
    assert env.is_valid_position((3, 0)) is False

def test_out_of_bounds():
    env = MazeEnv()
    assert env.is_valid_position((-1, 2)) is False
    assert env.is_valid_position((5, 2)) is False

def test_invalid_position_input():
    env = MazeEnv()
    assert env.is_valid_position(()) is False
    assert env.is_valid_position((1, 2, 3)) is False
    assert env.is_valid_position((1, )) is False

    with pytest.raises(TypeError):
        env.is_valid_position((1.5, 2))


@pytest.mark.parametrize("action, assertion, expected_position", [
    (0, False, [0, 0]),
    (1, True, [1, 0]),
    (2, False, [0, 0]),
    (3, False, [0, 0]),
])
def test_basic_movements(action, assertion, expected_position):
    env = MazeEnv()
    assert env.move_agent(action) is assertion
    assert env.current_pos == expected_position

@pytest.mark.parametrize("action", [5, -1, (1, 1), 1.5])
def test_invalid_movements_valueerror(action):
    env = MazeEnv()
    with pytest.raises(ValueError):
        env.move_agent(action)

def test_movement_chain():
    env = MazeEnv()
    assert env.move_agent(1) is True
    assert env.current_pos == [1, 0]
    assert env.move_agent(1) is True
    assert env.current_pos == [2, 0]
    assert env.move_agent(1) is False
    assert env.current_pos == [2, 0]

@pytest.mark.parametrize("x, y", [
    (1.5, 3),
    (0, 2.1),
])
def test_getcell_typeerror(x, y):
    env = MazeEnv()
    with pytest.raises(TypeError):
        env.get_cell(x, y)

def test_reset():
    env = MazeEnv()
    assert env.move_agent(1) is True
    assert env.current_pos == [1, 0]
    
    observation, info = env.reset()
    assert isinstance(observation, np.ndarray)
    np.testing.assert_array_equal(observation, np.array([0, 0]))
    assert env.observation_space.contains(observation)
    assert info == {}
    assert env.current_pos == [0, 0]

def test_step_valid():
    env = MazeEnv()
    env.reset()
    
    observation, reward, terminated, truncated, info = env.step(1)
    
    np.testing.assert_array_equal(observation, np.array([1, 0]))
    assert env.observation_space.contains(observation)
    assert reward == -1
    assert terminated is False
    assert truncated is False
    assert info == {}
    assert env.step_counter == 1

def test_step_blocked_move():
    env = MazeEnv()
    env.reset()
    
    observation, reward, terminated, truncated, info = env.step(0)
    
    np.testing.assert_array_equal(observation, np.array([0, 0]))
    assert env.observation_space.contains(observation)
    assert reward == -1
    assert terminated is False
    assert truncated is False
    assert info == {}
    assert env.step_counter == 1

def test_step_terminated():
    env = MazeEnv()
    env.reset()
    for i in [1, 1, 2, 2, 3, 3, 2, 2, 1, 1, 1]:
        env.step(i)
    
    observation, reward, terminated, truncated, info = env.step(1)
    
    np.testing.assert_array_equal(observation, np.array([4, 4]))
    assert env.observation_space.contains(observation)
    assert reward == +5
    assert terminated is True
    assert truncated is False
    assert info == {}
    assert env.step_counter == 12


def test_step_truncated():
    env = MazeEnv()
    env.reset()
    for i in range(49):
        env.step(1)
        env.step(3)
    env.step(1) ## 99th step
    
    observation, reward, terminated, truncated, info = env.step(3) # 100th step
    
    np.testing.assert_array_equal(observation, np.array([0, 0]))
    assert env.observation_space.contains(observation)
    assert reward == -1
    assert terminated is False
    assert truncated is True
    assert info == {}
    assert env.step_counter == 100
