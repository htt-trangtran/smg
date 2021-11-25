############################
# written by Trang H. Tran and Lam M. Nguyen
############################

"""
Define Neural Networks 
"""

import torch.nn as nn
import torch.nn.functional as F

#-------------------------(LeNet5 for Cifar10)-------------------------------
class LeNet5 (nn.Module):
    def __init__ (self):
        super().__init__()

        # This network is for images of size 32x32, with 3 color channels  (Cifar10 dataset)
        self.conv1 = nn.Conv2d (3, 6, 5)
        self.pool = nn.MaxPool2d (2,2)
        self.conv2 = nn.Conv2d (6, 16, 5)
        self.fc1 = nn.Linear (16* 5* 5, 120)
        self.fc2 = nn.Linear (120, 84)
        self.fc3 = nn.Linear (84, 10)
        self.lastbias = 'fc3.bias'
    
    def forward (self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16* 5* 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
#-------------------------------------------------------------------------------


#---------------(LeNet_300_100 for (Fashion) MNIST)-----------------------------
class LeNet_300_100 (nn.Module):
    def __init__ (self):
        super().__init__()

        # This network is for images of size 28x28, with 1 color channels 
        self.fc1 = nn.Linear (28* 28, 300)
        self.fc2 = nn.Linear (300, 100)
        self.fc3 = nn.Linear (100, 10)
        self.relu = nn.ReLU()
        self.lastbias = 'fc3.bias'
    
    def forward (self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
#-------------------------------------------------------------------------------