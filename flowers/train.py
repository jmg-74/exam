#!/usr/bin/env python3
"""
Script callable from command line, to train and save a model.
               (Differential Private version)

Usage:   python3 train.py
   or:   ./train.py

For optional parameters, try:    ./train.py -h

To (un)set GPU memory monitoring (evaluations: see ## heading lines),
 see _mem_monitor()'s `silent` default boolean value in this code.
 Just printed on std output, no file created here from these stats.
"""
import argparse
import os
from shutil import copyfile
from tqdm import tqdm

import torch
from torch import nn, optim
from torchvision import datasets, models, transforms

# To be able to fetch next packages from parent folder
import sys
sys.path.append("..")

from torch.utils.data import DataLoader
from torch.cuda import memory_allocated
from collections import OrderedDict

from torchdp import PrivacyEngine, utils


def test(model, testloader, criterion, device):
    """
    Returns (loss, accuracy) of model w.r.t. the given  testloader.
    """
    # Switch to evaluation mode, and CUDA if possible
##    _mem_monitor("TEST 0", device)
    model.eval()
    model.to(device)
##    _mem_monitor("TEST 1 (model loaded)", device)

    losses = 0
    correct = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
##            _mem_monitor("TEST 2 (images loaded)", device) # ===== monitoring =====

            # Forward step and loss computation
            output = model(images)    # Is Torch([b, 102])
            losses += criterion(output, labels).item()
            _, predicted = torch.max(output.data, 1)  # (values, indices)
            correct +=  (predicted == labels).sum().item()

    # Switch back to training mode
    model.train()

    testloader_size = len(testloader.dataset)
    accuracy = correct / testloader_size
    loss = losses / len(testloader) # consistent with training loss

##    _mem_monitor("TEST 3 (END)", device)

    return loss, accuracy


def train(model, trainloader, testloader, epochs, print_every, criterion, optimizer, device,
          arch="vgg16", model_dir="models", serialize=False, detail=False, args=None):
    """
    Trains the model with given parameters then saves model state (parameters).

    . These files are serialized (like pickle) if `serialize` is True:
      checkpoint.pth represents current epoch, best_model.pth is the best one.
    . Details are printed if boolean parameter `detail` is True.

    """
    # Change to train mode, load on GPU if possible
    model.train()
    model.to(device) # In fact, already done

    best_accuracy = 0
    steps_nb = len(trainloader)
    accuracy_list = [0 for _ in range(epochs)]
    loss_list = [0 for _ in range(epochs)]

    # Epoch loop
    for epoch in range(1, epochs+1):
        running_loss, running_step = 0, 0

        # Batch loop
        for step, (images, labels) in enumerate(tqdm(trainloader), start=1):
            images, labels = images.to(device), labels.to(device)
##            _mem_monitor("2. TRAIN : images loaded", device)  # ===== Monitoring =====

            # Forward and backward passes
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
##            _mem_monitor("2.1. TRAIN : forward pass done", device)   # ===== Monitoring =====
            loss.backward()
##            _mem_monitor("3. TRAIN : loss gradient backpropagated", device)   # ===== Monitoring =====
            optimizer.step()
##            _mem_monitor("4. TRAIN :gradient computed", device)   # ===== Monitoring =====

            running_loss += loss.item()
            running_step += 1

            # Print perf. each `print_every` step, or last one
            if (detail and (step % print_every == 0 or step == steps_nb)
                       # Cancel printing if it is near the end
                       and not(0.94 * steps_nb < step < steps_nb)):
                testloss, accuracy = test(model, testloader, criterion, device)
                # ===== DP ===================================================
                if not args.disable_dp:
                    epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(args.delta)
                    print(f' [DP : ε = {epsilon:.2f}, δ = {args.delta} for α = {best_alpha}]')
                # ============================================================
                print(f'>>> {step}/{steps_nb}, epoch {epoch}/{epochs} >>>\t'
                      f'Training loss: {running_loss/running_step:.3f} -- '
                      f'Test loss: {testloss:.3f} -- '
                      f'Test accuracy: {accuracy*100:.1f}%')

                running_loss, running_step = 0, 0

        # End of an epoch
        if detail:
            print()
        else:
            testloss, accuracy = test(model, testloader, criterion, device)
            # One last print if `not detail`
            if not args.disable_dp:
                    epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(args.delta)
                    print(f' [DP : ε = {epsilon:.2f}, δ = {args.delta} for α = {best_alpha}]')
            # ============================================================
            print(f'>>> {step}/{steps_nb}, epoch {epoch}/{epochs} >>>\t'
                      f'Training loss: {running_loss/running_step:.3f} -- '
                      f'Test loss: {testloss:.3f} -- '
                      f'Test accuracy: {accuracy*100:.1f}%')

        accuracy_list[epoch-1] = accuracy
        loss_list[epoch-1] = testloss

        # Serialize model state
        if serialize:
            torch.save({'epoch': epochs,
                        'classifier': model.classifier,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'class_idx_mapping': model.class_idx_mapping,
                        'arch': arch,
                        'best_accuracy': best_accuracy*100},
                       os.path.join(model_dir, 'checkpoint.pth'))
            if accuracy > best_accuracy:
                copyfile(os.path.join(model_dir,'checkpoint.pth'),
                         os.path.join(model_dir,'best_model.pth'))
            best_accuracy = max(accuracy, best_accuracy)

    return testloss, accuracy_list


def get_data(data_folder, batch_size, test_batch_size=1):
    """
    Returns the dataset as a dataloader.

    Arguments:
        data_folder: Path to the folder where data resides.
            Should have two subdirectories named "train" and "valid".
        batch_size: size of batch for Stochastic Gradient Descent.

    Returns tuple of:
        train_dataloader: Train dataloader iterator.
        test_dataloader: Validation dataloader iterator.
        train_dataset.class_to_id: dict to map classes to indexes.
    """
    ### TODO: download dataset and unzip if folder is empty ###
    train_dir = os.path.join(data_folder, "train")
    valid_dir = os.path.join(data_folder, "valid")

    # Define transforms for the training and validation sets
    # Divide side of images by FACT
    FACT = 2
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(size=224 // FACT),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    validation_transforms = transforms.Compose([
        transforms.Resize(256 // FACT),
        transforms.CenterCrop(224 // FACT),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load the datasets with ImageFolder and previous transforms
    # (ImageFolder is a generic data loader, for a specific organisation
    #  of images in folders)
    train_dataset = datasets.ImageFolder(train_dir,
                                         transform=train_transforms)
    test_dataset = datasets.ImageFolder(valid_dir,
                                        transform=validation_transforms)

    # Using the image datasets, define the dataloaders
    # (DataLoader provides an iterator over a sampled dataset, see
    #   https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
    train_dataloader = DataLoader(train_dataset,
                                  shuffle=True,
                                  batch_size=batch_size,
                                  num_workers=4)
    test_dataloader = DataLoader(test_dataset,
                                 shuffle=True,
                                 batch_size=test_batch_size,
                                 num_workers=4)

    # (class_to_idx is an attribute of any ImageFolder)
    return train_dataloader, test_dataloader, train_dataset.class_to_idx


def hybrid_model(arch="vgg16", hidden_units=4096, class_idx_mapping=None, args=None):
    """
    Return a model based on `arch` pre-trained one and 2 new fully connected layers.
    """
    # Model adapted to chosen architecture, thanks to dynamic execution
    my_local = dict()
    exec(f'model = models.{arch}(pretrained=True)', globals(), my_local)
    model =  my_local['model']
##    model = utils.convert_batchnorm_modules(model)   # ===== Monitoring =====

    # Freeze existing model parameters for training
    for param in model.parameters():
        param.requires_grad = False

    # Get last child module of imported model
    last_child = list(model.children())[-1]

    if type(last_child) == torch.nn.modules.linear.Linear:
        input_features = last_child.in_features
    elif type(last_child) == torch.nn.modules.container.Sequential:
        input_features = last_child[0].in_features

    # Add some neww layers to train
    classifier = nn.Sequential(OrderedDict([      ### vgg16 : input_features = 25088
                                            ('fc1', nn.Linear(input_features, hidden_units)),
                                            ('relu', nn.ReLU()),
                                            ###('dropout', nn.Dropout(p=0.5)),
                                            ('fc2', nn.Linear(hidden_units, 102)),
                                            ###('relu2', nn.ReLU()),          ## Traces of 
                                            ###('fc3', nn.Linear(256, 102)),  ##  experiments.
                                            ('output', nn.LogSoftmax(dim=1))
                                            ]))
    model.classifier = classifier
    model.class_idx_mapping = class_idx_mapping
    model = model.to(args.device)
##    _mem_monitor("1. HYBRID_MODEL : model loaded ", args.device)    # ===== Monitoring =====

    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    if not args.disable_dp:
            privacy_engine = PrivacyEngine(
                classifier, ### = model. Idem with classifier
                batch_size=args.batch_size,
                sample_size=args.sample_size,
                alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
                noise_multiplier=args.noise,
                max_grad_norm=args.clip,
            )
            privacy_engine.attach(optimizer)
##    _mem_monitor("HYBRID_MODEL after DP tranfo. ", args.device)   # ===== Monitoring =====

    return model, optimizer


def _mem_monitor(msg="", device=None, silent=False): ###  ===== MONITORING ===== ###
    """
    Print a message showing memory allocated on device.
    """
    if silent: return
    print("\t>>>", msg, ">>>",
          f'{memory_allocated(device)/1024/1024:.3f} MB')


def main():
    # Pre-trained model
    VALID_ARCH_CHOICES = ("vgg16", "vgg13", "densenet121")
    # Print perf. PRINT_PER_EPOCH or (PRINT_PER_EPOCH + 1) times per epoch
    PRINT_PER_EPOCH = 4

    # Parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir",
                    help="Directory containing the dataset (default: data)",
                    default="data",
                    nargs="?")
    ap.add_argument("--arch",
                    help="Model architecture from 'torchvision.models' (default: vgg16)",
                    choices=VALID_ARCH_CHOICES, default=VALID_ARCH_CHOICES[0])
    ap.add_argument("--hidden-units",
                    help="Number of units the hidden layer should consist of (default: 512)",
                    default=512,
                    type=int)
    ap.add_argument("--batch-size",
                    help="Batch size during training (default: 64)",
                    default=64,
                    type=int)
    ap.add_argument("--test-batch-size",
                    help="Batch size for test, validation (default: 64)",
                    default=64,
                    type=int)
    ap.add_argument("--learning-rate",
                    help="Learning rate for Adam optimizer (default: 0.001)",
                    default=0.001,
                    type=float)
    ap.add_argument("--epochs",
                    help="Number of iterations over the whole dataset (default: 3)",
                    default=3,
                    type=int)
    ap.add_argument("--cpu",
                    help="Use CPU (else GPU) for training (default if not set: GPU)",
                    action="store_true")
    ap.add_argument("--model-dir",
                    help="Directory which will contain the model checkpoints (default: models)",
                    default="models")
    ap.add_argument("--serialize",
                    help="Serialize, save the trained model if set (default: not set)",
                    default=False,
                    action="store_true")
    ap.add_argument("--no-detail",
                    help="Print details during training if not set (default: not set - slows down training)",
                    default=False,
                    action="store_true")
    # DP specific
    ap.add_argument("--disable-dp",
                    help="Disable 'Diffential Privacy' mode if set (default: not set)",
                    default=False,
                    action="store_true")
    ap.add_argument("--noise",
                    help="Noise multiplier for Gaussian noise added (default: 0.25)",
                    default=0.25,
                    type=float)
    ap.add_argument("--clip",
                    help="Clip per-sample gradients to this l2-norm (default: 1.0)",
                    default=1.0,
                    type=float)
    ap.add_argument("--delta",
                    help="Target delta for DP (default: 1e-4)",
                    default=1e-4,
                    type=float)
    args = ap.parse_args()

    args.device = "cpu" if args.cpu else "cuda"

##    _mem_monitor("INIT", args.device)   # ===== Monitoring =====

    # Create directory for model files: checkpoint.pth and best_model.pth
    os.system("mkdir -p " + args.model_dir)

    # Load the dataset into a dataloader
    trainloader, testloader, mapping = get_data(data_folder=args.data_dir,
                                         batch_size=args.batch_size,
                                         test_batch_size=args.test_batch_size)
    args.sample_size = len(trainloader.dataset)

    # Build model: chose loss function, optimizer, processor support
    model , optimizer = hybrid_model(arch=args.arch,
                                     hidden_units=args.hidden_units,
                                     class_idx_mapping=mapping,
                                     args=args)
    criterion = nn.NLLLoss()
##    _mem_monitor("1.1 BEFORE TRAINING", args.device)   # ===== Monitoring =====
    # Launch training
    train(model=model,
        trainloader=trainloader,
        testloader=testloader,
        epochs=args.epochs,
        print_every=int(len(trainloader)/PRINT_PER_EPOCH),
        criterion=criterion,
        optimizer=optimizer,
        device=args.device,
        arch=args.arch,
        model_dir=args.model_dir,
        serialize=args.serialize,
        detail=not(args.no_detail),
        args=args,
    )


if __name__ == '__main__':
    main()
