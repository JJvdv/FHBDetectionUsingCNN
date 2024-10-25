import torch

import torch.nn as nn

from torch.nn import Module, Conv2d, Linear, MaxPool2d, ReLU, Dropout, Sigmoid

###########################################################
# class NeuralNetwork()
# A convolutional neural network using the Pytorch library
############################################################
class NeuralNetwork(Module):
    def __init__(self, num_channels):
        super(NeuralNetwork, self).__init__()

        self.conv1 = Conv2d(in_channels = num_channels, out_channels = 20, kernel_size = (5, 5))
        self.bn1 = nn.BatchNorm2d(20)
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size = (2, 2), stride = (2, 2))

        self.conv2 = Conv2d(in_channels = 20, out_channels = 50, kernel_size = (5, 5))
        self.bn2 = nn.BatchNorm2d(50)
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size = (2, 2), stride = (2, 2))
        self.dropout = Dropout(p = 0.5)

        self.fc1 = Linear(in_features = 1250, out_features = 800)
        self.relu3 = ReLU()
        self.fc2 = Linear(in_features = 800, out_features = 1)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = torch.flatten(x, 1)
        #print(x.shape) - debugging
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu3(x)

        #print(x.shape) - debugging
        x = self.fc2(x)
        out = self.sigmoid(x)
        
        return out