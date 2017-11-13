from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from model.Generator import Generator


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--imageSize', type=int, default=256, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--netG', default='dcgan/netG_epoch_507.pth', help="path to netG (to continue training)")
parser.add_argument('--outf', default='dcgan/', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--ngf', type=int, default=16)
parser.add_argument('--dataset', default='CIFAR', help='which dataset to train on, CIFAR|MNIST')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

###########   Load netG   ###########
assert opt.netG != '', "netG must be provided!"
nc = 3
if(opt.dataset == 'CIFAR'):
    nc = 3
netG = Generator(nc, opt.ngf, opt.nz)
netG.load_state_dict(torch.load(opt.netG))

###########   Generate   ###########
noise = torch.FloatTensor(opt.batchSize, opt.nz, 1, 1)
noise = Variable(noise)

if(opt.cuda):
    netG.cuda()
    noise = noise.cuda()


for i in range(1,100):
  noise.data.normal_(0, 1)
  fake = netG(noise)
  vutils.save_image(fake.data[1],
                    '%s/generate_fake_samples1_i_%03d.png' % (opt.outf, i),
                    normalize=True)
  vutils.save_image(fake.data[2],
                    '%s/generate_fake_samples2_i_%03d.png' % (opt.outf, i),
                    normalize=True)
  vutils.save_image(fake.data[3],
                    '%s/generate_fake_samples3_i_%03d.png' % (opt.outf, i),
                    normalize=True)
  vutils.save_image(fake.data[4],
                    '%s/generate_fake_samples4_i_%03d.png' % (opt.outf, i),
                    normalize=True)
  vutils.save_image(fake.data[5],
                    '%s/generate_fake_samples5_i_%03d.png' % (opt.outf, i),
                    normalize=True)
  vutils.save_image(fake.data[6],
                    '%s/generate_fake_samples6_i_%03d.png' % (opt.outf, i),
                    normalize=True)
  vutils.save_image(fake.data[7],
                    '%s/generate_fake_samples7_i_%03d.png' % (opt.outf, i),
                    normalize=True)
  vutils.save_image(fake.data[8],
                    '%s/generate_fake_samples8_i_%03d.png' % (opt.outf, i),
                    normalize=True)
  vutils.save_image(fake.data[9],
                    '%s/generate_fake_samples9_i_%03d.png' % (opt.outf, i),
                    normalize=True)
  vutils.save_image(fake.data[10],
                    '%s/generate_fake_samples10_i_%03d.png' % (opt.outf, i),
                    normalize=True)
  vutils.save_image(fake.data[11],
                    '%s/generate_fake_samples11_i_%03d.png' % (opt.outf, i),
                    normalize=True)
  vutils.save_image(fake.data[12],
                    '%s/generate_fake_samples12_i_%03d.png' % (opt.outf, i),
                    normalize=True)
  vutils.save_image(fake.data[13],
                    '%s/generate_fake_samples13_i_%03d.png' % (opt.outf, i),
                    normalize=True)
  vutils.save_image(fake.data[14],
                    '%s/generate_fake_samples14_i_%03d.png' % (opt.outf, i),
                    normalize=True)
  vutils.save_image(fake.data[15],
                    '%s/generate_fake_samples15_i_%03d.png' % (opt.outf, i),
                    normalize=True)
  vutils.save_image(fake.data[0],
                    '%s/generate_fake_samples16_i_%03d.png' % (opt.outf, i),
                    normalize=True)