#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Runs MNIST training with differential privacy.

  ADAPTED VERSION TO MONITOR GPU MEMORY

"""

import argparse
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda import memory_allocated
from torchdp import PrivacyEngine
from torchvision import datasets, transforms
from tqdm import tqdm


# Precomputed characteristics of the MNIST dataset
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


class SampleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, 2, padding=3)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.fc1 = nn.Linear(32 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        # x of shape [B, 1, 28, 28]
        x = F.relu(self.conv1(x))   # -> [B, 16, 14, 14]
        x = F.max_pool2d(x, 2, 1)   # -> [B, 16, 13, 13]
        x = F.relu(self.conv2(x))   # -> [B, 32, 5, 5]
        x = F.max_pool2d(x, 2, 1)   # -> [B, 32, 4, 4]
        x = x.view(-1, 32 * 4 * 4)  # -> [B, 512]
        x = F.relu(self.fc1(x))     # -> [B, 32]
        x = self.fc2(x)             # -> [B, 10]
        return x

    def name(self):
        return "SampleConvNet"


# `infos` param. added for monitoring
def train(args, model, device, train_loader, optimizer, epoch,   infos=dict()):
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []
    for _batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
##
        _mem_monitor("2. (train) images loaded", device)   # ===== Monitoring =====
        infos['2.IL']= f'{memory_allocated(device)}' # Bytes

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
##
        _mem_monitor("3. (train) loss computed and backpropagated", device)   # ===== Monitoring =====
        infos['3.LG']= f'{memory_allocated(device)}'

        optimizer.step()
##
        _mem_monitor("4. (train) model updated from gradients", device)   # ===== Monitoring =====
        infos['4.MU']= f'{memory_allocated(device)}'

        losses.append(loss.item())

    if not args.disable_dp:
        epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(args.delta)
        print(
            f"Train Epoch: {epoch} \t"
            f"Loss: {np.mean(losses):.6f} "
            f"(ε = {epsilon:.2f}, δ = {args.delta}) for α = {best_alpha}"
        )
    else:
        print(f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f}")


def test(args, model, device, test_loader):
##
    _mem_monitor("TEST 0", device)  # ===== Monitoring =====
    model.eval()
    _mem_monitor("TEST 0.1", device)  # ===== Monitoring =====
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
##
            _mem_monitor("TEST 2.: images loaded", device)  # ===== Monitoring =====
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
##
    _mem_monitor("TEST 3.: end", device)  # ===== Monitoring =====

    return correct / len(test_loader.dataset)


# Switch on/off `silent` value to avoid messages printing globaly
def _mem_monitor(msg="", device=None, silent=True):    # ===== MONITORING =====
    """
    Print a message showing memory allocated on device.
    """
    if silent: return
    print("\t>>>", msg, ">>>",
          f'{memory_allocated(device)/1024/1024:.3f} MB')



def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=64,
        metavar="B",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1024,
        metavar="TB",
        help="input batch size for testing (default: 1024)",
    )
    parser.add_argument(
        "-n",
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "-r",
        "--n-runs",
        type=int,
        default=1,
        metavar="R",
        help="number of runs to average on (default: 1)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=.1,
        metavar="LR",
        help="learning rate (default: .1)",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        metavar="S",
        help="Noise multiplier (default 1.0)",
    )
    parser.add_argument(
        "-c",
        "--max-per-sample-grad_norm",
        type=float,
        default=1.0,
        metavar="C",
        help="Clip per-sample gradients to this norm (default 1.0)",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        metavar="D",
        help="Target delta (default: 1e-5)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="GPU ID for this process (default: 'cuda')",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="Save the trained model (default: false)",
    )
    parser.add_argument(
        "--disable-dp",
        action="store_true",
        default=False,
        help="Disable privacy training and just train with vanilla SGD",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="../mnist",
        help="Where MNIST is/will be stored",
    )
    args = parser.parse_args()
    device = torch.device(args.device)

    kwargs = {"num_workers": 1, "pin_memory": True}

    file_name = 'mem_stats_mnist.csv'
    if not os.path.exists(file_name):
        with open(file_name, 'w') as f:
            f.write('Batch_size, Differential_Privacy, '
                    'Mem_Model_Loaded, Mem_Images_Loaded, Mem_Loss_Gradient, '
                    'Mem_Model_Computed\n')

    infos = dict()

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            args.data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))]
            ),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            args.data_root,
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))]
            ),
        ),
        batch_size=args.test_batch_size,
        shuffle=True,
        **kwargs,
    )
    run_results = []
    for _ in range(args.n_runs):
        model = SampleConvNet().to(device)
        _mem_monitor("1. HYBRID_MODEL : model loaded ", args.device)   # ===== Monitoring =====

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0)
        if not args.disable_dp:
            privacy_engine = PrivacyEngine(
                model,
                batch_size=args.batch_size,
                sample_size=len(train_loader.dataset),
                alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
                noise_multiplier=args.sigma,
                max_grad_norm=args.max_per_sample_grad_norm,
            )
            privacy_engine.attach(optimizer)
        _mem_monitor("1.1 HYBRID_MODEL : model loaded + DP ", args.device)   # ===== Monitoring =====
        infos['1.ML']= f'{memory_allocated(device)}'

        for epoch in range(1, args.epochs + 1):
            # Add `infos` parameter for monitoring
            train(args, model, device, train_loader, optimizer, epoch,  infos)
        run_results.append(test(args, model, device, test_loader))

    if len(run_results) > 1:
        print("Accuracy averaged over {} runs: {:.2f}% ± {:.2f}%".format(
            len(run_results),
            np.mean(run_results) * 100,
            np.std(run_results) * 100
        )
        )

##    # Memory stats
    infos['BS'] = args.batch_size
    infos['DP'] = 'N' if args.disable_dp else 'Y'
    print(f'\n>>>>> Train / Batch size = {infos["BS"]}, DP={infos["DP"]} >>>>>')
    print('\t', infos)

    # Store stats in .cvs file
    with open(file_name, 'a') as f:
                    for k in ('BS', 'DP', '1.ML', '2.IL', '3.LG', '4.MU'):
                        f.write(str(infos[k]))
                        if k != '4.MU': f.write(', ')
                    f.write('\n')


if __name__ == "__main__":
    main()
