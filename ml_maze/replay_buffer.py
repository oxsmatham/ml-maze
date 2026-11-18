import random

class ReplayBuffer():
    def __init__(self, max_capacity):
        self.max_capacity = max_capacity
        self.transitions = []
    
    def __len__(self) -> int:
        return len(self.transitions)
    
    def push(self, transition) -> None:
        '''transition = (state, action, reward, next_state, terminated)'''
        if len(self.transitions) >= self.max_capacity:
            self.transitions.pop(0)
        
        self.transitions.append(transition)

    def sample(self, batch_size: int):
        sample = random.sample(population = self.transitions, k = batch_size)
        
        return sample