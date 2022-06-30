# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self, num_classes=5):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96,
                               kernel_size=11, stride=4, padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256,
                               kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384,
                               kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384,
                               kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256,
                               kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features=9216, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))       # input[3, 224, 224]  output[96, 55, 55]
        x = self.maxpool(x)             # output[96, 27, 27]
        x = F.relu(self.conv2(x))       # output[256, 27, 27]
        x = self.maxpool(x)             # output[256, 13, 13]
        x = F.relu(self.conv3(x))       # output[384, 13, 13]
        x = F.relu(self.conv4(x))       # output[384, 13, 13]
        x = F.relu(self.conv5(x))       # output[256, 13, 13]
        x = self.maxpool(x)             # output[256, 6, 6]
        x = x.reshape(x.shape[0], -1)  # reshape to 256*6*6 = 9246
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Reduced AlexNet
class AlexNet_without_conv1(nn.Module):
    def __init__(self,num_classes=5):
        super(AlexNet_without_conv1, self).__init__()
        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=96,
        #                        kernel_size=11, stride=4, padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2a = nn.Conv2d(in_channels=3, out_channels=256,
                                kernel_size=5, stride=9, padding=6)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384,
                               kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384,
                               kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256,
                               kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features=9216, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=num_classes)

    def forward(self, x):
        # x = F.relu(self.conv1(x))
        # x = self.maxpool(x)
        x = F.relu(self.conv2a(x))   # input[3, 224, 224]  output[96, 55, 55]
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.maxpool(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class AlexNet_without_conv2(nn.Module):
    def __init__(self,num_classes=5):
        super(AlexNet_without_conv2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96,
                               kernel_size=11, stride=4, padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        # self.conv2 = nn.Conv2d(in_channels=96, out_channels=256,
        #                        kernel_size=5, stride=1, padding=2)
        self.conv3a = nn.Conv2d(in_channels=96, out_channels=384,
                               kernel_size=3, stride=2, padding=0)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384,
                               kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256,
                               kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features=9216, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        # x = F.relu(self.conv2a(x))
        # x = self.maxpool(x)
        x = F.relu(self.conv3a(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.maxpool(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class AlexNet_without_conv3(nn.Module):
    def __init__(self, num_classes=5):
        super(AlexNet_without_conv3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96,
                               kernel_size=11, stride=4, padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256,
                               kernel_size=5, stride=1, padding=2)
        # self.conv3 = nn.Conv2d(in_channels=256, out_channels=384,
        #                        kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(in_channels=256, out_channels=384,
                                kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256,
                               kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features=9216, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        # x = F.relu(self.conv3(x))
        x = F.relu(self.conv4a(x))
        x = F.relu(self.conv5(x))
        x = self.maxpool(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class AlexNet_without_conv4(nn.Module):
    def __init__(self, num_classes=5):
        super(AlexNet_without_conv4, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96,
                               kernel_size=11, stride=4, padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256,
                               kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384,
                               kernel_size=3, stride=1, padding=1)
        # self.conv4 = nn.Conv2d(in_channels=384, out_channels=384,
        #                        kernel_size=3, stride=1, padding=1)
        self.conv5a = nn.Conv2d(in_channels=384, out_channels=256,
                                kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features=9216, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        # x = F.relu(self.conv4(x))
        x = F.relu(self.conv5a(x))
        x = self.maxpool(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class AlexNet_without_conv5(nn.Module):
    def __init__(self, num_classes=5):
        super(AlexNet_without_conv5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96,
                               kernel_size=11, stride=4, padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256,
                               kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384,
                               kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384,
                               kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256,
                                kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=384, out_channels=256,
                               kernel_size=1, stride=1, padding=0)
        self.fc1 = nn.Linear(in_features=9216, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv6(x))
        x = self.maxpool(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class AlexNet_without_BothFC(nn.Module):
    def __init__(self, num_classes=5):
        super(AlexNet_without_BothFC, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96,
                               kernel_size=11, stride=4, padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256,
                               kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384,
                               kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384,
                               kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256,
                               kernel_size=3, stride=1, padding=1)
        self.fc3 = nn.Linear(in_features=9216, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))       # input[3, 224, 224]  output[96, 55, 55]
        x = self.maxpool(x)             # output[96, 27, 27]
        x = F.relu(self.conv2(x))       # output[256, 27, 27]
        x = self.maxpool(x)             # output[256, 13, 13]
        x = F.relu(self.conv3(x))       # output[384, 13, 13]
        x = F.relu(self.conv4(x))       # output[384, 13, 13]
        x = F.relu(self.conv5(x))       # output[256, 13, 13]
        x = self.maxpool(x)             # output[256, 6, 6]
        x = x.reshape(x.shape[0], -1)  # reshape to 256*6*6 = 9246
        x = self.fc3(x)
        return x

