import torch 
import torch.nn as nn 


class FullNet(nn.Module):

    def __init__(self):
        super(FullNet,self).__init__()

        self.fc1 = nn.Linear(96,64)
        self.fc2 = nn.Linear(64,32)
        self.relu = nn.ReLU()

    def forward(self,tensor):
        
        x = self.fc1(tensor)
        x = self.relu(x)
        x = self.fc2(x)

        return x 