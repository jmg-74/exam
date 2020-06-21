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

from my_nets import Net, Net2

# ***************
    # 0: home-made net
    # 1: not pre-trained VGG16
    # 2: pre-trained VGG16
    # 3: (not frozen) pre-trained VGG16 + fully connected layer
    # 4: frozen pre-trained VGG16 + fully connected layer
MODE = 4
# ***************
EPOCHS = 20
# ***************


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


def train(model, optimizer, criterion, trainloader, epochs,
          device="cpu", save=False, testloader=None):

    if testloader:
            acc, _ = test(model, testloader, categories=False, device=device)
            print(f'Accuracy of the network on the 10000 test images: {acc:.2f} %')

    for epoch in range(1, epochs+1):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs, `data` is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics every 2000 mini-batches
            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'->{i+1} loss: {running_loss / 2000:.3f}')
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

    # Detail of accuracy for each category if expected
    class_correct = [0.] * 10
    class_total = [0] * 10
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            print(c, ' / ', (predicted==labels)) ####################################
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
    # (PIL [0, 1] images to [-1, 1] Tensors)
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, # *** Better as a param ? ***
                                              shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=4)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # ===== BUILD NET MODEL =====

    # REM: to restart from a saved model
    #net = Net() # Or another choice, then
    #net.load_state_dict(torch.load(PATH))

    if MODE == 0:
        # Home-made, local definition : (Net or) Net2
        net = Net2()
        optimizer = optim.SGD(net.parameters(), lr=0.001) #, momentum=0.9)

    elif MODE == 1:
        net = vgg16(pretrained=False)
        # Adapt output to 10 classes
        input_features = net.classifier[6].in_features
        net.classifier[6] = nn.Linear(input_features, 10)

        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    else: #pre-trained
        net = vgg16(pretrained=True)

        if MODE == 4:
            # Freeze existing model parameters for training
            #   (or just first convolutional layers != "classifier")
            for name, param in net.named_parameters():
                if name[:10] != "classifier": param.requires_grad = False
                #param.requires_grad = False

        if MODE > 2:
            # Add a new layer to train
            last_child = list(net.children())[-1]
#            print("\tLAST CHILD:", last_child)
#            input_features = last_child[0].in_features
            input_features = net.classifier[6].in_features
            net.classifier[6] = nn.Linear(input_features, 10)

            # Old version, not just a simple adaptation of last layer
#            hidden_units = 512                                 # *** Better as a param ***
#            classifier = nn.Sequential(OrderedDict([
#                                            ('fc1', nn.Linear(input_features, hidden_units)),
#                                            ('relu', nn.ReLU()),
#                                            ('fc2', nn.Linear(hidden_units, 10)),
#                                            ('output', nn.LogSoftmax(dim=1))
#                                            ]))
#            net.classifier = nn.Sequential( nn.Linear(input_features, hidden_units),
#                                          nn.ReLU(),
#                                          nn.Dropout(0.4),
#                                          nn.Linear(hidden_units, 10),
#                                          nn.Linear(input_features, 10),
#                                          nn.LogSoftmax(dim=1))

            net.classifier[6].requires_grad = True

            #
            last_child = list(net.children())[-1]
            print("\tLAST CHILD (2) :", last_child)

#            net.classifier = classifier
            optimizer = optim.SGD(net.classifier.parameters(), lr=0.001, momentum=0.9)


        else: # MODE == 2
            # Adapt output to 10 classes
            input_features = net.classifier[6].in_features
            net.classifier[6] = nn.Linear(input_features, 10)
            optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    net.to(device)
    criterion = nn.CrossEntropyLoss()
#    param_show(net)

    # ===== TRAIN MODEL =====
    print(f"Dataset and network are ready (mode {MODE}), let's train our model "
          f"(×{EPOCHS} epoch" + ("s" if EPOCHS > 1 else "") + ")...")

    # (Just remove last parameter (testloader) to avoid tests after each epoch)
    train(net, optimizer, criterion, trainloader, EPOCHS, device,
          save=False, testloader=testloader)

    #===== TEST MODEL (detail categories) =====
    CATEGORIES = True
    acc, categ_acc = test(net, testloader, categories=CATEGORIES, device=device)

    print(f'Accuracy of the network on the 10000 test images: {acc:.2f} %')

    if CATEGORIES:
        for i in range(10):
            print(f'Accuracy of {classes[i]:5s} : {categ_acc[i]:.2f} %')



if __name__ == "__main__":
    main()

