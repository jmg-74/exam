#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Runs DCGAN training with differential privacy.

"""
from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchdp import PrivacyEngine, utils
from torchdp import autograd_grad_sample

parser = argparse.ArgumentParser()
parser.add_argument('--data-root', required=False, help='path to dataset')
parser.add_argument(
    '--workers',
    type=int,
    help='number of data loading workers',
    default=2
)
parser.add_argument('--batch-size', type=int, default=64, help='input batch size')
parser.add_argument(
    '--imageSize',
    type=int,
    default=64,
    help='the height / width of the input image to network'
)
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=128)
parser.add_argument('--ndf', type=int, default=128)
parser.add_argument(
    '--epochs',
    type=int,
    default=25,
    help='number of epochs to train for'
)
parser.add_argument(
    '--lr',
    type=float,
    default=0.0002,
    help='learning rate, default=0.0002'
)
parser.add_argument(
    '--beta1',
    type=float,
    default=0.5,
    help='beta1 for adam. default=0.5'
)
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument(
    '--outf',
    default='.',
    help='folder to output images and model checkpoints'
)
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument(
    '--target-digit',
    type=int,
    default=8,
    help='the target digit(0~9) for MNIST training'
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda",
    help="GPU ID for this process (default: 'cuda')",
)
parser.add_argument(
    "--disable-dp",
    action="store_true",
    default=False,
    help="Disable privacy training and just train with vanilla SGD",
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

opt = parser.parse_args()

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if opt.data_root is None and str(opt.dataset).lower() != 'fake':
    raise ValueError("`data-root` parameter is \
    required for dataset \"%s\"" % opt.dataset)

try:
    dataset = dset.MNIST(root=opt.data_root, download=False,
                        transform=transforms.Compose([
                            transforms.Resize(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,)),
                        ]))
    idx = dataset.targets == opt.target_digit
    dataset.targets = dataset.targets[idx]
    dataset.data = dataset.data[idx]
    nc = 1
except ValueError:
    print("Cannot load dataset")

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                         shuffle=True, num_workers=int(opt.workers))

device = torch.device(opt.device)
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


netG = Generator(ngpu)
if not opt.disable_dp:
    netG = utils.convert_batchnorm_modules(netG)
netG = netG.to(device)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


netD = Discriminator(ngpu)
if not opt.disable_dp:
    netD = utils.convert_batchnorm_modules(netD)
    netG = utils.convert_batchnorm_modules(netG)
netD = netD.to(device)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))

criterion = nn.BCELoss()

FIXED_NOISE = torch.randn(opt.batch_size, nz, 1, 1, device=device)
REAL_LABEL = 1
FAKE_LABEL = 0

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

privacy_engine = PrivacyEngine(
    netD,
    batch_size=opt.batch_size,
    sample_size=len(dataloader.dataset),
    alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
    noise_multiplier=opt.sigma,
    max_grad_norm=opt.max_per_sample_grad_norm
)
if not opt.disable_dp:
    privacy_engine.attach(optimizerD)
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

for epoch in range(opt.epochs):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        netD.zero_grad()

        real_data = data[0].to(device)
        batch_size = real_data.size(0)

        # train with fake
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(noise)
        label_fake = torch.full((batch_size,), FAKE_LABEL, device=device)
        output = netD(fake.detach())
        errD_fake = criterion(output, label_fake)
        errD_fake.backward()
        optimizerD.step()

        # train with real
        label_true = torch.full((batch_size,), REAL_LABEL, device=device)
        output = netD(real_data)
        errD_real = criterion(output, label_true)
        errD_real.backward()
        optimizerD.step()
        D_x = output.mean().item()

        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label_g = torch.full((batch_size,), REAL_LABEL, device=device)
        output_g = netD(fake)
        errG = criterion(output_g, label_g)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()
        # When training the Generator, the gradients backpropagate through netD to netG
        # The hook attached to netD then got triggerd
        # which add grad_output to backpropagate lists in each layer in netD
        # However the list is not cleared in optimizerG.step()
        # thus we have to clear the backprop list in netD manually
        # we will make this cleaner in the API in the near future,
        # but for now this works and it is correct
        # (even though it's slower than it should be because we generate
        # per-sample gradients and then throw them away)
        autograd_grad_sample.clear_backprops(netD)
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opt.epochs, i, len(dataloader),
                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        if not opt.disable_dp:
            epsilon, best_alpha = optimizerD.privacy_engine.get_privacy_spent(opt.delta)
            print('(ε = %.2f, δ = %.2f) for α = %.2f'
            % (epsilon, opt.delta, best_alpha))

        if i % 100 == 0:
            vutils.save_image(real_data,
                    '%s/real_samples.png' % opt.outf,
                    normalize=True)
            fake = netG(FIXED_NOISE)
            vutils.save_image(fake.detach(),
                    '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                    normalize=True)

    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
