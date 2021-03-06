import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1   = nn.Conv2d(1, 6, 3)
#        self.pool    = nn.MaxPool2d(2, 2)
        self.conv2   = nn.Conv2d(6, 16, 3)
        self.fc1     = nn.Linear(16*5*5, 120)
        self.fc2     = nn.Linear(120, 84)
        self.fc3     = nn.Linear(84, 10)
        self.fc4     = nn.Linear(10, 2)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
    
        return x



