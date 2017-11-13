"""
Created on 2017.6.17

@author: tfygg
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
import os
import argparse



def train(epoch):
    model_ft.train()
    for batch_idx, (data, target) in enumerate(trainloader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        optimizer.zero_grad()
        output = model_ft(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                100. * batch_idx / len(trainloader), loss.data[0]))

def test(epoch):
    model_ft.eval()
    test_loss = 0
    correct = 0
    i=0
    for data, target in testloader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model_ft(data)
        test_loss = criterion(output, target)

        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

        i=i+1
        if pred.eq(target.data).cpu().sum() == 0:
            print(i,target.data[0], pred[0][0])


    test_loss = test_loss
    test_loss /= len(testloader)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss.data[0], correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))

def test1(epoch):
    model_ft.eval()
    test_loss = 0
    correct = 0
    i=0
    for data, target in testloader1:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model_ft(data)
        test_loss = criterion(output, target)

        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()
        i=i+1
        if pred.eq(target.data).cpu().sum() == 0:
            print(i,target.data[0], pred[0][0])


    test_loss = test_loss
    test_loss /= len(testloader1)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss.data[0], correct, len(testloader1.dataset),
        100. * correct / len(testloader1.dataset)))



if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=80, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    print('==> Preparing data..')
    transform_train = transforms.Compose([
        # transforms.RandomCrop(224, padding=4),
        transforms.Scale(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Scale(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.ImageFolder('E:/plant disease/data/Apple/80train3/train', transform=transform_train)
 #   trainset = torchvision.datasets.CIFAR10(root='../data/cifar', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.ImageFolder('E:/plant disease/data/Apple/80train3/test', transform=transform_test)
#    testset = torchvision.datasets.CIFAR10(root='../data/cifar', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)
    testset1 = torchvision.datasets.ImageFolder('E:/plant disease/data/Apple/test3nn', transform=transform_test)
    #    testset = torchvision.datasets.CIFAR10(root='../data/cifar', train=False, download=True, transform=transform_test)
    testloader1 = torch.utils.data.DataLoader(testset1, batch_size=args.test_batch_size, shuffle=False, num_workers=2)


    # ConvNet
    model_ft = models.resnet18(pretrained=True)
    print(model_ft)

    for i, param in enumerate(model_ft.parameters()):
        param.requires_grad = False

    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 10)
    print(model_ft)

    criterion = nn.CrossEntropyLoss()

    if args.cuda:
        model_ft.cuda()
        criterion.cuda()

    optimizer = optim.SGD(model_ft.fc.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4)

    for epoch in range(1, args.epochs + 1):
       train(epoch)
       test(epoch)
       test1(epoch)
