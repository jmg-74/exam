#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import *
from collections import OrderedDict

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
        super(Net, self).__init__()
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
    #for param in model.parameters():
        if param.requires_grad:
            print(name, '\t', param.numel())
            #print("\t", param.numel())
            total += param.numel()
    print("-----------------------")
    print('Total', '\t', total, '_n')
    for p in model.parameters():
        print(p.shape)


def train(model, optimizer, criterion, trainloader, epochs,
          device="cpu", save=False, testloader=None):
    for epoch in range(1, epochs+1):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch, i + 1, running_loss / 2000))
                running_loss = 0.0

        print(f'Finished training (epoch {epoch}/{epochs})')

        if testloader:
            acc, _ = test(model, testloader, categories=False, device=device)
            print(f'Accuracy of the network on the 10000 test images: {acc:.2f} %')

    if save:
        torch.save(model.state_dict(), 'models/cifar10.pth')


def test(model, testloader, categories=False, device="cpu"):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if not categories:
        return 100 * correct / total, None

    # Detail of accuracy for each category
    class_correct = [0.]*10
    class_total = [0.] * 10
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    categ_acc = [0.] * 10
    for i in range(10):
        categ_acc[i] = 100 * class_correct[i] / class_total[i]

    return 100 * correct / total, categ_acc




def main():
    # ===== DETECT CUDA IF AVAILABLE =====
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    print("Running on", device_name.upper())

    # ===== LOAD DATA =====
    # PIL [0, 1] images to [-1, 1] Tensors
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # ===== BUILD NET MODEL =====

    # REM: to restart from a saved model
    #net = Net() # Or another choice, then
    #net.load_state_dict(torch.load(PATH))

# ***************
    # 0: home-made net
    # 1: not pre-trained VGG16
    # 2: pre-trained VGG16
    # 3: (not frozen) pre-trained VGG16 + fully connected layer
    # 4: frozen pre-trained VGG16 + fully connected layer
####    # 5: not pre-trained VGG16 + fully connected layer ????
    MODE = 3
# ***************

    if MODE == 0:
        # Local definition : Net or Net2
        net = Net()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    elif MODE == 1:
        net = vgg16(pretrained=False)
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    else: #pre-trained
        net = vgg16(pretrained=True)

        if MODE == 4:
            # Freeze existing model parameters for training
            for param in net.parameters():
                param.requires_grad = False
        if MODE > 2:
            # Add some neww layers to train
            last_child = list(net.children())[-1]
            input_features = last_child[0].in_features
            hidden_units = 512
            classifier = nn.Sequential(OrderedDict([      ### vgg16 : input_features = 25088
                                            ('fc1', nn.Linear(input_features, hidden_units)),
                                            ('relu', nn.ReLU()),
                                            ###('dropout', nn.Dropout(p=0.5)),
                                            ('fc2', nn.Linear(hidden_units, 102)),
                                            ###('relu2', nn.ReLU()),          ## Traces of
                                            ###('fc3', nn.Linear(256, 102)),  ##  experiments.
                                            ('output', nn.LogSoftmax(dim=1))
                                            ]))
            net.classifier = classifier
#            optimizer = optim.Adam(net.classifier.parameters(), lr=0.001)
            optimizer = optim.SGD(net.classifier.parameters(), lr=0.001)


        else: # MODE == 2
            optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    net.to(device)
    criterion = nn.CrossEntropyLoss()
    param_show(net)

    # ===== TRAIN MODEL =====
# ***************
    EPOCHS = 10
# ***************
    print(f"Dataset and network are ready (mode {MODE}), let's train our model "
          f"(×{EPOCHS} epoch" + ("s" if EPOCHS > 1 else "") + ")...")
    # Remove last parameter (testloader) to avoid tests after each epoch
    train(net, optimizer, criterion, trainloader, EPOCHS, device,
          save=True, testloader=testloader)

#    dataiter = iter(testloader)
#    data = dataiter.next()
#    images, labels = data[0].to(device), data[1].to(device)

    #===== TEST MODEL =====
    CATEGORIES = True

    acc, categ_acc = test(net, testloader, categories=CATEGORIES, device=device)

    print(f'Accuracy of the network on the 10000 test images: {acc:.2f} %')

    if CATEGORIES:
        for i in range(10):
            print(f'Accuracy of {classes[i]:5s} : {categ_acc[i]:.2f} %')



if __name__ == "__main__":
    main()
