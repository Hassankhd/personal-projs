import torch
import torch.nn as nn
import torch.nn.functional as F

class ChessCNN(nn.Module):
    def __init__(self):
        super(ChessCNN, self).__init__()
        self.fc1 = nn.Linear(832, 832)
        self.fc2 = nn.Linear(832, 416)
        self.fc3 = nn.Linear(416, 208)
        self.fc4 = nn.Linear(208, 104)
        self.fc5 = nn.Linear(104, 1)
    
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        
        x = F.relu(self.fc2(x))
        
        x = F.relu(self.fc3(x))
        
        x = F.relu(self.fc4(x))
        
        x = self.fc5(x)
        return x

