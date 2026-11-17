import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.linear1 = nn.Linear(in_features = 2, out_features = 64)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features = 64, out_features = 4)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x