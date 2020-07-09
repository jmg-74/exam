import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ===== Ex. from tutorial  =====
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)    #arg2: 6=> 64
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 16, 5)   #arg1: 6=> 64
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
# Trainable parameters:
"""
conv1.weight 	 4725
conv1.bias 	 63 # or 6
conv2.weight 	 25200
conv2.bias 	 16
fc1.weight 	 48000
fc1.bias 	 120
fc2.weight 	 10080
fc2.bias 	 84
fc3.weight 	 840
fc3.bias 	 10
------------------------
Total 	 89138
"""

# ===== Ex. from Bytepawn =====
# http://bytepawn.com/solving-cifar-10-with-pytorch-and-skl.html
class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(3,   64,  3)
        self.conv2 = nn.Conv2d(64,  128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)

# Trainable parameters:
"""
conv1.weight 	 1728
conv1.bias 	 64
conv2.weight 	 73728
conv2.bias 	 128
conv3.weight 	 294912
conv3.bias 	 256
fc1.weight 	 131072
fc1.bias 	 128
fc2.weight 	 32768
fc2.bias 	 256
fc3.weight 	 2560
fc3.bias 	 10
-----------------------
Total 	 537610
"""


def param_show(model):
    """Prints how many parameters the net has"""
    total = 0
    print('Trainable parameters:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, '\t', param.numel())
            total += param.numel()
    print("-----------------------")
    print('Total', '\t', total, '\n')
#    for p in model.parameters():
#        print(p.shape)

