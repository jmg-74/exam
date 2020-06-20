#!/usr/bin/env python3
"""
 Experiment with different parameters the CNN training and collect accuracy

 Quick and dirty adaptation from train.py.
 Search `TUNING` in this code for hard coded parameters to try.
"""

import os
import pickle
import argparse
import datetime
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from train import test, train, get_data, hybrid_model

def main():
     # Pre-trained model
    VALID_ARCH_CHOICES = ("vgg16", "vgg13", "densenet121")

    # Parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir",
                    help="Directory containing the dataset (default: data)",
                    default="data",
                    nargs="?")
    ap.add_argument("--arch",
                    help="Model architecture from 'torchvision.models' (default: vgg16)",
                    choices=VALID_ARCH_CHOICES, default=VALID_ARCH_CHOICES[0])
#    ap.add_argument("--hidden-units",
#                    help="Number of units the hidden layer should consist of (default: 512)",
#                    default=512,
#                    type=int)
    ap.add_argument("--cpu",
                    help="Use CPU (else GPU) for training (default if not set: GPU)",
                    action="store_true")
    args = ap.parse_args()

    device = "cpu" if args.cpu else "cuda"
    args.device = device
    args.noise = 0.25
    args.clip = 1.0
    args.batch_size = 64
    args.hidden_units = 256
    args.delta = 1e-4

    # Build model: chose loss function, optimizer, processor support
#    # Done later to reset the model
#    model = hybrid_model(arch=args.arch, hidden_units=args.hidden_units)
    criterion = nn.NLLLoss()
    device = "cpu" if args.cpu else "cuda"

    # ===== TUNING ===========================================================
    # Hyperparameters to test
    lr_range = [1e-4] #, 1e-5]                    #####  <== choice (enumeration)
    batch_size_range = [64] #, 32, 128, 8, 4,  1] #####  <== choice (enumeration)
    epochs = 6                                 #####  <== choice (1 value=max)
    # Number of iteration for each parameter
    iter = 3                                   #####  <== choice (single value)

    # DP or not DP, that is the question
    args.disable_dp = True                     #####  <== choice (boolean)
    # ========================================================================

    # File to export results
    dp_or_not = "noDP_" if args.disable_dp else "DP_"
    file = "experiment_stats/accuracy_data_" + dp_or_not
    file += str(datetime.datetime.today()).replace(' ','_') + ".csv"

    steps = len(lr_range) * len(batch_size_range) * iter
    step = 0

    # Write column titles
    with open(file, 'w') as f:
        f.write('learning_rate, batch_size, n_epochs, accuracy, n_times_for_avg\n')

    # Experiment loops
    for lr in lr_range:
        args.learning_rate = lr

        for bs in batch_size_range:
            args.batch_size = bs
            # Load the dataset into a dataloader  ### default test batch size ###
            trainloader, testloader, mapping = get_data(data_folder=args.data_dir,
                                                        batch_size=bs)
            args.sample_size = len(trainloader.dataset)

            #for epochs in epochs_range:
            accuracy_sum = []

            for _ in range(iter):
                    # Reset the model
                    model, optimizer = hybrid_model(arch=args.arch,
                                                    hidden_units=args.hidden_units,
                                                    args=args)
                    step += 1
                    _, acc = train(model=model,
                                   trainloader=trainloader,
                                   testloader=testloader,
                                   epochs=epochs,
                                   print_every=None,
                                   criterion=criterion,
                                   optimizer=optimizer,
                                   device=device,
                                   arch=args.arch,
                                   model_dir='',
                                   serialize=False,
                                   detail=False,
                                   args=args,
                                  )
                    acc = np.multiply(acc, 100)
                    accuracy_sum.append(acc)
                    print(f' {step}/{steps}\tlr={lr}, bs={bs},')
                    for  n_epoch, accur in enumerate(acc, start=1):
                        line = f'{lr}, {bs}, {n_epoch}, {accur:.2f}, 1\n'
                        with open(file, 'a') as f:
                            f.write(line)
                        print(f'\t. ×{n_epoch} epoch{"s" if n_epoch > 1 else " "}'
                              f' => accuracy = {accur:.2f}%')

            # Sum up for identical settings, repeted `iter` times
            acc_avg = np.average(accuracy_sum, axis=0)
            for n_epoch, accur in enumerate(acc_avg, start=1):
                    line = f'{lr}, {bs}, {n_epoch}, {accur:.2f}, {iter}\n'
                    with open(file, 'a') as f:
                        f.write(line)
                    print(f'\t\t>>> Average on {iter} iterations >>>\tlr={lr}, bs={bs},'
                          f' ×{n_epoch} epoch{"s" if n_epoch > 1 else " "}'
                          f' => accuracy = {accur:.2f}%')


if __name__ == "__main__":
    main()
