
import os
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision

import matplotlib.pyplot as plt
import numpy as np


# Settable parameters
batch_size_train = 128
batch_size_test = 1000
learning_rate = 0.6
momentum = 0.9
n_epochs = 20
log_interval = 10

torch.backends.cudnn.enabled = False

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                      download=True, transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize((0.1307,), (0.3081,))]))

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                     download=True, transform=torchvision.transforms.Compose([
                                      torchvision.transforms.ToTensor(),
                                      torchvision.transforms.Normalize((0.1307,), (0.3081,))]))

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test,
                                          shuffle=True, num_workers=0)


class Flatten(nn.Module):
   def forward(self, input):
       return input.view(input.size(0), -1)


class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.linear_relu_stack = nn.Sequential(
            Flatten(),
            nn.Linear(28*28, 32),
            nn.Sigmoid(),
            nn.Linear(32, 32),
            nn.Sigmoid(),
            nn.Linear(32, 10),
            nn.Softmax()
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
criterion = nn.CrossEntropyLoss()


def train(epoch):

    model.train()
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        output = model(inputs)

        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()


def test():
    model.eval()
    test_loss = 0
    correct = 0

    for data, target in testloader:
        output = model(data)
        test_loss += F.nll_loss(output, target).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()

    print("{}/{}".format(correct, len(testloader.dataset)))


test()
for epoch in range(1, n_epochs+1):
    train(epoch)
    test()

