# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

# Original AlexNet
class AlexNet(nn.Module):
    layerToTrain = "all"
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
        x = x.reshape(x.shape[0], -1)   # reshape to 256*6*6 = 9246
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Reduced AlexNet
class AlexNet_without_conv1(nn.Module):
    layerToTrain = ["conv2a", "fc"]
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
    layerToTrain = ["conv3a", "fc"]
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
    layerToTrain = ["conv4a", "fc"]
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
    layerToTrain = ["conv5a", "fc"]
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
    layerToTrain = ["conv6","fc"]
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
    layerToTrain = ["fc"]
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



class AlexNet_Extreme(nn.Module):
    layerToTrain = ["fc"]
    def __init__(self, num_classes=5):
        super(AlexNet_Extreme, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96,
                               kernel_size=11, stride=4, padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fc3 = nn.Linear(in_features=69984, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc3(x)
        return x


# Concatenate Model
class AlexNet_ConcatenateConv1to2(nn.Module):
    def __init__(self, num_classes=5):
        super(AlexNet_ConcatenateConv1to2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96,
                               kernel_size=11, stride=4, padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256,
                               kernel_size=5, stride=1, padding=2)

        self.fc3 = nn.Linear(in_features=113248, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))       # input[3, 224, 224]  output[96, 55, 55]
        x = self.maxpool(x)             # output[96, 27, 27]

        a1 = x.reshape(x.shape[0], -1) # Flatten and Save: A[1] # [96, 27, 27] => [69984]

        x = F.relu(self.conv2(x))       # output[256, 27, 27]
        x = self.maxpool(x)             # output[256, 13, 13]

        x = torch.cat((a1, x.reshape(x.shape[0], -1)),1)  # reshape 256*13*13 = 43264 concatenate 69984 == 113248
        x = self.fc3(x)
        return x

class AlexNet_ConcatenateConv1to3(nn.Module):
    def __init__(self, num_classes=5):
        super(AlexNet_ConcatenateConv1to3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96,
                               kernel_size=11, stride=4, padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256,
                               kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384,
                               kernel_size=3, stride=1, padding=1)
        self.fc3 = nn.Linear(in_features=178144, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))       # input[3, 224, 224]  output[96, 55, 55]
        x = self.maxpool(x)             # output[96, 27, 27]

        a1 = x.reshape(x.shape[0], -1)   # Flatten and Save: A[1] # [96, 27, 27] => [69984]

        x = F.relu(self.conv2(x))       # output[256, 27, 27]
        x = self.maxpool(x)             # output[256, 13, 13]

        a2 = x.reshape(x.shape[0], -1)   # Flatten and Save: A[2] # [256, 13, 13] => [43264]

        x = F.relu(self.conv3(x))       # output[384, 13, 13]

        x = torch.cat((a1, a2, x.reshape(x.shape[0], -1)),1)  # reshape 384*13*13 => 64896 + 69984 + 43264 + == 178144
        x = self.fc3(x)
        return x

class AlexNet_ConcatenateConv1to4(nn.Module):
    def __init__(self, num_classes=5):
        super(AlexNet_ConcatenateConv1to4, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96,
                               kernel_size=11, stride=4, padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256,
                               kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384,
                               kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384,
                               kernel_size=3, stride=1, padding=1)
        self.fc3 = nn.Linear(in_features=243040, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))       # input[3, 224, 224]  output[96, 55, 55]
        x = self.maxpool(x)             # output[96, 27, 27]

        a1 = x.reshape(x.shape[0], -1)   # Flatten and Save: A[1] # [96, 27, 27] => [69984]

        x = F.relu(self.conv2(x))       # output[256, 27, 27]
        x = self.maxpool(x)             # output[256, 13, 13]

        a2 = x.reshape(x.shape[0], -1)   # Flatten and Save: A[2] # [256, 13, 13] => [43264]

        x = F.relu(self.conv3(x))       # output[384, 13, 13]

        a3 = x.reshape(x.shape[0], -1)   # Flatten and Save: A[3] # [384, 13, 13] => [64896]

        x = F.relu(self.conv4(x))       # output[384, 13, 13]

        x = torch.cat((a1, a2, a3, x.reshape(x.shape[0], -1)),1)  # reshape 384*13*13 => 64896 + 64896 + 69984 + 43264 == 243040
        x = self.fc3(x)
        return x

class AlexNet_CatAllConv(nn.Module):
    def __init__(self, num_classes=5):
        super(AlexNet_CatAllConv, self).__init__()
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

        self.fc3 = nn.Linear(in_features=252256, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))       # input[3, 224, 224]  output[96, 55, 55]
        x = self.maxpool(x)             # output[96, 27, 27]

        a1 = x.reshape(x.shape[0], -1)   # Flatten and Save: A[1] # [96, 27, 27] => [69984]

        x = F.relu(self.conv2(x))       # output[256, 27, 27]
        x = self.maxpool(x)             # output[256, 13, 13]

        a2 = x.reshape(x.shape[0], -1)   # Flatten and Save: A[2] # [256, 13, 13] => [43264]

        x = F.relu(self.conv3(x))       # output[384, 13, 13]

        a3 = x.reshape(x.shape[0], -1)   # Flatten and Save: A[3] # [384, 13, 13] => [64896]

        x = F.relu(self.conv4(x))       # output[384, 13, 13]

        a4 = x.reshape(x.shape[0], -1)  # Flatten and Save: A[3] # [384, 13, 13] => [64896]

        x = F.relu(self.conv5(x))       # output[256, 13, 13]
        x = self.maxpool(x)             # output[256, 6, 6]

        x = torch.cat((a1, a2, a3, a4, x.reshape(x.shape[0], -1)),1)  # reshape 256, 6, 6 => 9216 + 64896 + 64896 + 69984 + 43264 == 243040
        x = self.fc3(x)
        return x










