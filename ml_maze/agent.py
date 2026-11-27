from ml_maze.q_network import QNetwork
import torch
import random

class Agent():
    def __init__(self, env, gamma = 0.99, learning_rate = 0.001):
        self.env = env
        self.q_network = QNetwork()
        self.target_network = QNetwork()
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(),
            lr = self.learning_rate
        )
        
        self.gamma = gamma
        
        

    def select_action(self, state, epsilon) -> int:
        
        if random.random() < epsilon:
            return self.env.action_space.sample()
        
        state_tensor = torch.as_tensor(state, dtype = torch.float32)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            action = torch.argmax(q_values).item()
        
        return action

    def learn(self, batch):
        self.optimizer.zero_grad()
        
        state, action, reward, next_state, terminated = zip(*batch)
        state_tensor = torch.as_tensor(state, dtype = torch.float32)
        action_tensor = torch.as_tensor(action, dtype = torch.long)
        reward_tensor = torch.as_tensor(reward, dtype = torch.float32)
        next_state_tensor = torch.as_tensor(next_state, dtype = torch.float32)
        terminated_tensor = torch.as_tensor(terminated, dtype = torch.bool)
        
        all_q_values = self.q_network(state_tensor)
        actions = action_tensor.unsqueeze(1)
        predicted_q_values = all_q_values.gather(1, actions).squeeze(1)
        
        with torch.no_grad():
            next_q_values = self.target_network(next_state_tensor)
            best_next_q_values = next_q_values.max(dim = 1).values
        
        targets = reward_tensor + self.gamma * (1 - terminated_tensor.float()) * best_next_q_values
        
        loss = torch.nn.functional.smooth_l1_loss(
            predicted_q_values,
            targets
        )
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()