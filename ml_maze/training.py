def run_episode(env, agent, buffer, epsilon):

    state, info = env.reset()
    total_reward = 0
    steps = 0
    episode_done = False

    while not episode_done:
        action = agent.select_action(state, epsilon)
        next_state, reward, terminated, truncated, info = env.step(action)
        
        buffer.push((state, action, reward, next_state, terminated))
        
        state = next_state
        
        total_reward += reward
        steps += 1
        
        episode_done = terminated or truncated
    
    return total_reward, steps