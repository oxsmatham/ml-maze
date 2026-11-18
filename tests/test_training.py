from ml_maze.environment import MazeEnv
from ml_maze.agent import Agent
from ml_maze.replay_buffer import ReplayBuffer
from ml_maze.training import run_episode

def test_run_episode_stores_one_transition_per_step():
    env = MazeEnv()
    env.step_limit = 3
    agent = Agent(env)
    buffer = ReplayBuffer(max_capacity = 10)
    
    total_reward, steps = run_episode(env, agent, buffer, epsilon = 1.0)
    
    assert steps == 3
    assert len(buffer) == 3
    assert total_reward == -3

    last_transition = buffer.transitions[-1]
    state, action, reward, next_state, terminated = last_transition
    
    assert terminated is False
    assert env.observation_space.contains(next_state)