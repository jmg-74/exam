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

from my_nets import Net, Net2, param_show
from torch.cuda import memory_allocated
# To be able to fetch next packages from parent folder
import sys
sys.path.append("..")
from torchdp import PrivacyEngine, utils




# ===== Parameters (will be shell args) ===============================================
class Argu():
    def __init__(self):
        self.batch_size = 4
        self.learning_rate = 0.001
        self.epochs = 20
        self.disable_dp = False
        self.hidden_units = 512
        self.noise = 0.5
        self.clip = 0.5
        self.delta = 1e-4
        self.categories = False

args = Argu()

args.mode = 2

# 0: home-made net
# 1: not pre-trained VGG16
# 2: pre-trained VGG16
# 3: (not frozen) pre-trained VGG16 + fully connected layer
# 4: frozen pre-trained VGG16 + fully connected layer
# ====================================================================================

def train(model, optimizer, criterion, trainloader, epochs,
          device="cpu", save=False, testloader=None, args=None):


    accur = []
    for epoch in range(1, epochs+1):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs. `data` is a list of [inputs, labels]
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
                print(f'->{i+1} loss: {running_loss / 2000:.3f}', end='\t')
                running_loss = 0.0

                # ===== DP ===================================================
                if not args.disable_dp:
                    epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(args.delta)
                    print(f' [DP : ε = {epsilon:.2f}, δ = {args.delta} for α = {best_alpha}]')
                else:
                    print()
                # ============================================================

        print(f'Finished training (epoch {epoch}/{epochs})')

        if testloader:
            # Detailled stats with categories only for last epoch
            acc, _ = test(model, testloader, categories=(i == epochs), device=device)
            accur.append(acc)
            print(f'Accuracy of the network on the 10000 test images: {acc:.2f} %')
            if (i == epochs):
                for i in range(10):
                    print(f'Accuracy of {classes[i]:5s} : {categ_acc[i]:.2f} %')

    if save:
        torch.save(model.state_dict(), 'models/cifar10.pth')

    return accur



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
    class_correct = [0.] * 10
    class_total = [0] * 10
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            # There are 4 images per batch for testloader
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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=4)

    args.sample_size = len(trainloader.dataset)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # ===== BUILD NET MODEL =====

    # REM: to restart from a saved model
    #net = Net() # Or another choice, then
    #net.load_state_dict(torch.load(PATH))

    # *********************************************************
    # 0: home-made net
    # 1: not pre-trained VGG16
    # 2: pre-trained VGG16
    # 3: (not frozen) pre-trained VGG16 + fully connected layer
    # 4: frozen pre-trained VGG16 + fully connected layer
    # *********************************************************

    if args.mode == 0:
        # Home made, local definition : Net or Net2
        net = Net()
        optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9)

    elif args.mode in [1, 2]:
        net = vgg16(pretrained=(args.mode == 2))

        # Adapt output to 10 classes
        input_features = net.classifier[6].in_features
        net.classifier[6] = nn.Linear(input_features, 10)

        optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9)

    else: # Pre-trained
        net = vgg16(pretrained=True)

        if args.mode == 4:
            # Freeze existing model parameters for training
            #  (or juste first convolutional layers != "classifier")
            #for param in net.parameters():
            #    param.requires_grad = False
            for name, param in net.named_parameters():
                if name[:10] != "classifier": param.requires_grad = False

        if args.mode > 2:
            # Add some neww layers to train
            # Verification (before)
#            last_child = list(net.children())[-1]
#            print("\tLAST CHILD:", last_child)

            # Just adapt to 10 categories
            #input_features = last_child[0].in_features
            input_features = net.classifier[6].in_features
            net.classifier[6] = nn.Linear(input_features, 10)

            # Old version (adding several layers)
            #classifier = nn.Sequential(OrderedDict([
            #                                ('fc1', nn.Linear(input_features, args.hidden_units)),
            #                                ('relu', nn.ReLU()),
            #                                ###('dropout', nn.Dropout(p=0.5)),
            #                                ('fc2', nn.Linear(args.hidden_units, 10)),
            #                                ###('relu2', nn.ReLU()),          ## Traces of
            #                                ###('fc3', nn.Linear(256, 10)),   ##  experiments.
            #                                ('output', nn.LogSoftmax(dim=1))
            #                                ]))
            #net.classifier = nn.Sequential( nn.Linear(input_features, hidden_units),
            #                                          nn.ReLU(),
            #                                          nn.Dropout(0.4),
            #                                          nn.Linear(hidden_units, 10),
            #                                          nn.Linear(input_features, 10),
            #                                          nn.LogSoftmax(dim=1))
            #optimizer = optim.SGD(net.classifier.parameters(), lr=0.args.learning_rate, momentum=0.9)

            #                       net???
            dp_mod = net.classifier if args.mode == 4 else net
            optimizer = optim.SGD(dp_mod.parameters(), lr=args.learning_rate, momentum=0.9) ##################################

            # Verification
#            last_child = list(net.children())[-1]
#            print("\tLAST CHILD:", last_child)

#        else: # args.mode == 2
#            # Adapt output to 10 classes
#            input_features = net.classifier[6].in_features
#            net.classifier[6] = nn.Linear(input_features, 10)

#            optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9)

    net.to(device)


# ===== DP ===============================================
    dp_mod = net.classifier if args.mode == 4 else net
    if not args.disable_dp:
            privacy_engine = PrivacyEngine(
                dp_mod,
                batch_size=args.batch_size,
                sample_size=args.sample_size,
                alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
                noise_multiplier=args.noise,
                max_grad_norm=args.clip,
            )
            privacy_engine.attach(optimizer)
# ========================================================

    criterion = nn.CrossEntropyLoss()

    # Structure of network
#    param_show(net)

    # ===== TRAIN MODEL =====
    print(f"Dataset and network are ready (mode {args.mode}), let's train our model "
          f"(×{args.epochs} epoch" + ("s" if args.epochs > 1 else "") + ")...")

    # (Just use `testloader=None` to avoid tests after each epoch)
    accur = train(net, optimizer, criterion, trainloader, args.epochs, device,
                  save=False, testloader=testloader, args=args)

    #===== TEST MODEL =====
    #
    # Already done during training (except details)

#    acc, categ_acc = test(net, testloader, categories=args.categories, device=device)
#    accur.append(acc)

#    print(f'Accuracy of the network on the 10000 test images: {acc:.2f} %')
#    if args.categories:
#        for i in range(10):
#            print(f'Accuracy of {classes[i]:5s} : {categ_acc[i]:.2f} %')

    print(f'size={args.sample_size}, '
          f'bs={args.batch_size}, '
          f'nm={args.noise}, '
          f'ep={args.epochs}, '
          f'd={args.delta}, '
          f'cl={args.clip}, '
          f'lr={args.learning_rate}, '
          f'hu={args.hidden_units}, '
          f'M={args.mode}\n'
          f'acc={accur}')


if __name__ == "__main__":
    main()
