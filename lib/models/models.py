#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from torch import nn
import torch.nn.functional as F
import torchvision.models as models
from collections import OrderedDict
import torch

class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)

class CNNFemnist(nn.Module):
    def __init__(self, args):
        super(CNNFemnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, args.out_channels, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(int(16820/20*args.out_channels), 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x1 = F.relu(self.fc1(x))
        x = F.dropout(x1, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1), x1

class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(int(320/20*20), 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x1 = F.relu(self.fc1(x))
        x = F.dropout(x1, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1), x1

class CNNFashion_Mnist(nn.Module):
    def __init__(self, args):
        super(CNNFashion_Mnist, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# class DigitModel(nn.Module):
#     """
#     Model for benchmark experiment on Digits.
#     """
#
#     def __init__(self, args):
#         super(DigitModel, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, 5, 1, 2)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.conv2 = nn.Conv2d(64, 64, 5, 1, 2)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.conv3 = nn.Conv2d(64, 128, 5, 1, 2)
#         self.bn3 = nn.BatchNorm2d(128)
#
#         self.fc1 = nn.Linear(6272, 2048)
#         self.bn4 = nn.BatchNorm1d(2048)
#         self.fc2 = nn.Linear(2048, 512)
#         self.bn5 = nn.BatchNorm1d(512)
#         self.fc3 = nn.Linear(512, args.num_classes)
#
#     def forward(self, x):
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.max_pool2d(x, 2)
#
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.max_pool2d(x, 2)
#
#         x = F.relu(self.bn3(self.conv3(x)))
#
#         x = x.view(x.shape[0], -1)
#
#         x = self.fc1(x)
#         x = self.bn4(x)
#         x = F.relu(x)
#
#         x = self.fc2(x)
#         x1 = self.bn5(x)
#         x = F.relu(x1)
#
#         x = self.fc3(x)
#         return F.log_softmax(x, dim=1), x1

class DigitModel(nn.Module):
    def __init__(self, pretrained):
        super(DigitModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(int(320), 50)
        self.fc2 = nn.Linear(50, 10)


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.bn1(self.conv1(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x1 = F.relu(self.fc1(x))
        x = F.dropout(x1, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1), x1

class ProjHead(nn.Module):
    def __init__(self, in_d, out_d):
        super(ProjHead, self).__init__()
        self.fc1 = nn.Linear(in_d, out_d)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.normalize(x, dim=1)
        return x

class ProjandProj(nn.Module):
    def __init__(self, in_d, prj1_d, prj2_d):
        super(ProjandProj, self).__init__()
        self.fc1 = nn.Linear(in_d, prj1_d)
        self.fc2 = nn.Linear(prj1_d, prj2_d)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x1 = F.normalize(x, dim=1)
        x = F.relu(self.fc2(x1))
        x2 = F.normalize(x, dim=1)
        return x1, x2

class ProjandDeci(nn.Module):
    def __init__(self, in_d, out_d, num_classes):
        super(ProjandDeci, self).__init__()
        self.fc1 = nn.Linear(in_d, out_d)
        self.fc2 = nn.Linear(out_d, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x1 = F.normalize(x, dim=1)
        x = F.relu(self.fc2(x1))
        x = F.normalize(x, dim=1)
        return F.log_softmax(x, dim=1), x1

class ProjandDeci_vit(nn.Module):
    def __init__(self, in_d, dim, out_d, num_classes):
        super(ProjandDeci_vit, self).__init__()
        self.fc1 = nn.Linear(in_d, dim)
        self.fc2 = nn.Linear(dim, out_d)
        self.fc3 = nn.Linear(out_d, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.normalize(x, dim=1)
        x = F.relu(self.fc2(x))
        x1 = F.normalize(x, dim=1)
        x = F.relu(self.fc3(x1))
        x = F.normalize(x, dim=1)
        return F.log_softmax(x, dim=1), x1

class ProjandDeci_Dyn(nn.Module):
    # num of fc layers is 5
    # def __init__(self, in_d, out_d, num_classes):
    #     super(ProjandDeci_Dyn, self).__init__()
    #     self.fc1 = nn.Linear(512 * 3, 512 * 2)
    #     self.fc2 = nn.Linear(512 * 2, 512 * 1)
    #     self.fc3 = nn.Linear(512 * 1, 256)
    #     self.fc4 = nn.Linear(256, 256)
    #     self.fc5 = nn.Linear(256, 256)
    #     self.fc6 = nn.Linear(256, num_classes)
    #
    # def forward(self, x):
    #     x = F.relu(self.fc1(x))
    #     x = F.normalize(x, dim=1)
    #     x = F.relu(self.fc2(x))
    #     x = F.normalize(x, dim=1)
    #     x = F.relu(self.fc3(x))
    #     x = F.normalize(x, dim=1)
    #     x = F.relu(self.fc4(x))
    #     x = F.normalize(x, dim=1)
    #     x = F.relu(self.fc5(x))
    #     x1 = F.normalize(x, dim=1)
    #     x = F.relu(self.fc6(x1))
    #     x = F.normalize(x, dim=1)
    #     return F.log_softmax(x, dim=1), x1

    # # num of fc layers is 4
    # def __init__(self, in_d, out_d, num_classes):
    #     super(ProjandDeci_Dyn, self).__init__()
    #     self.fc1 = nn.Linear(512 * 3, 512 * 2)
    #     self.fc2 = nn.Linear(512 * 2, 512 * 1)
    #     self.fc3 = nn.Linear(512 * 1, 256)
    #     self.fc4 = nn.Linear(256, 256)
    #     self.fc5 = nn.Linear(256, num_classes)
    #
    # def forward(self, x):
    #     x = F.relu(self.fc1(x))
    #     x = F.normalize(x, dim=1)
    #     x = F.relu(self.fc2(x))
    #     x = F.normalize(x, dim=1)
    #     x = F.relu(self.fc3(x))
    #     x = F.normalize(x, dim=1)
    #     x = F.relu(self.fc4(x))
    #     x1 = F.normalize(x, dim=1)
    #     x = F.relu(self.fc5(x1))
    #     x = F.normalize(x, dim=1)
    #     return F.log_softmax(x, dim=1), x1

    # # num of fc layers is 3
    # def __init__(self, in_d, out_d, num_classes):
    #     super(ProjandDeci_Dyn, self).__init__()
    #     self.fc1 = nn.Linear(512*3, 512*2)
    #     self.fc2 = nn.Linear(512 * 2, 512 * 1)
    #     self.fc3 = nn.Linear(512*1, 256)
    #     self.fc4 = nn.Linear(256, num_classes)
    #
    # def forward(self, x):
    #     x = F.relu(self.fc1(x))
    #     x = F.normalize(x, dim=1)
    #     x = F.relu(self.fc2(x))
    #     x = F.normalize(x, dim=1)
    #     x = F.relu(self.fc3(x))
    #     x1 = F.normalize(x, dim=1)
    #     x = F.relu(self.fc4(x1))
    #     x = F.normalize(x, dim=1)
    #     return F.log_softmax(x, dim=1), x1

    # # num of fc layers is 2
    # def __init__(self, in_d, out_d, num_classes):
    #     super(ProjandDeci_Dyn, self).__init__()
    #     self.fc1 = nn.Linear(512*3, 512*2)
    #     self.fc2 = nn.Linear(512 * 2, 256)
    #     self.fc3 = nn.Linear(256, num_classes)
    #
    # def forward(self, x):
    #     x = F.relu(self.fc1(x))
    #     x = F.normalize(x, dim=1)
    #     x = F.relu(self.fc2(x))
    #     x1 = F.normalize(x, dim=1)
    #     x = F.relu(self.fc3(x1))
    #     x = F.normalize(x, dim=1)
    #     return F.log_softmax(x, dim=1), x1

    # num of fc layers is 1
    def __init__(self, in_d, out_d, num_classes):
        super(ProjandDeci_Dyn, self).__init__()
        self.fc1 = nn.Linear(512 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x1 = F.normalize(x, dim=1)
        x = F.relu(self.fc2(x1))
        x = F.normalize(x, dim=1)
        return F.log_softmax(x, dim=1), x1


# class OfficeModel(nn.Module):
#     """
#     used for domainnet and Office-Caltech10
#     """
#     def __init__(self, args):
#         super(OfficeModel, self).__init__()
#         self.conv1 = nn.Conv2d(3, 10, 5)
#         self.bn1 = nn.BatchNorm2d(10)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(10, 16, 5)
#         self.fc0 = nn.Linear(16 * 61 * 61, 120)
#         self.fc1 = nn.Linear(120, 84)
#         self.fc2 = nn.Linear(84, args.num_classes)
#
#     def forward(self, x):
#         x = F.relu(self.pool(self.bn1(self.conv1(x))))
#         x = F.relu(self.pool(self.conv2(x)))
#         x = x.view(-1, 16 * 61 * 61)
#         x1 = F.relu(self.fc0(x))
#         x = F.relu(self.fc1(x1))
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1), x1

# class OfficeModel(nn.Module):
#     """
#     used for DomainNet and Office-Caltech10
#     """
#
#     def __init__(self, args):
#         super(OfficeModel, self).__init__()
#         self.features1 = nn.Sequential(
#             OrderedDict([
#                 ('conv1', nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)),
#                 ('bn1', nn.BatchNorm2d(64)),
#                 ('relu1', nn.ReLU(inplace=True)),
#                 ('maxpool1', nn.MaxPool2d(kernel_size=3, stride=2)),
#
#                 ('conv2', nn.Conv2d(64, 192, kernel_size=5, padding=2)),
#                 ('bn2', nn.BatchNorm2d(192)),
#                 ('relu2', nn.ReLU(inplace=True)),
#                 ('maxpool2', nn.MaxPool2d(kernel_size=3, stride=2)),
#
#                 ('conv3', nn.Conv2d(192, 384, kernel_size=3, padding=1)),
#                 ('bn3', nn.BatchNorm2d(384)),
#                 ('relu3', nn.ReLU(inplace=True)),
#
#                 ('conv4', nn.Conv2d(384, 256, kernel_size=3, padding=1)),
#                 ('bn4', nn.BatchNorm2d(256)),
#                 ('relu4', nn.ReLU(inplace=True)),
#
#                 ('conv5', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
#                 ('bn5', nn.BatchNorm2d(256)),
#             ])
#         )
#         self.features2 = nn.Sequential(
#             OrderedDict([
#                 ('relu5', nn.ReLU(inplace=True)),
#                 ('maxpool5', nn.MaxPool2d(kernel_size=3, stride=2)),
#             ])
#         )
#
#         self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
#
#         self.classifier1 = nn.Sequential(
#             OrderedDict([
#                 ('fc1', nn.Linear(256 * 6 * 6, 4096)),
#                 ('bn6', nn.BatchNorm1d(4096)),
#                 ('relu6', nn.ReLU(inplace=True)),
#
#                 ('fc2', nn.Linear(4096, 4096)),
#                 ('bn7', nn.BatchNorm1d(4096)),
#             ])
#         )
#
#         self.classifier2 = nn.Sequential(
#             OrderedDict([
#                 ('relu7', nn.ReLU(inplace=True)),
#                 ('fc3', nn.Linear(4096, args.num_classes)),
#             ])
#         )
#
#     def forward(self, x):
#         x = self.features1(x)
#         x0 = self.features2(x)
#         x = self.avgpool(x0)
#         x = torch.flatten(x, 1)
#         x1 = self.classifier1(x)
#         x = self.classifier2(x1)
#         return F.log_softmax(x, dim=1), x0[:,:,0,0]

class OfficeModel(nn.Module):
    """
    used for DomainNet and Office-Caltech10
    """

    def __init__(self, args):
        super(OfficeModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(192)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(384)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.relu5 = nn.ReLU(inplace=True)
        self.maxpool5 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.fc1 = nn.Linear(256 * 6 * 6, 2048)
        self.bn6 = nn.BatchNorm1d(2048)
        self.relu6 = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(2048, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.relu7 = nn.ReLU(inplace=True)

        self.fc3 = nn.Linear(256, args.num_classes)

    def forward(self, x):

        x = self.maxpool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.maxpool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.maxpool5(self.relu5(self.bn5(self.conv5(x))))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.relu6(self.bn6(self.fc1(x)))
        x1 = self.bn7(self.fc2(x))
        x = self.relu7(x1)
        x = self.fc3(x)

        return F.log_softmax(x, dim=1), x1

# class Cifar10Model(nn.Module):
#     """
#     used for domainnet and Office-Caltech10
#     """
#     def __init__(self, args):
#         super(Cifar10Model, self).__init__()
#         self.conv1 = nn.Conv2d(3, 10, 5)
#         self.bn1 = nn.BatchNorm2d(10)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(10, 16, 5)
#         self.bn2 = nn.BatchNorm2d(16)
#         self.fc0 = nn.Linear(16 * 5 * 5, 120)
#         self.bn3 = nn.BatchNorm2d(120)
#         self.fc1 = nn.Linear(120, 84)
#         self.bn4 = nn.BatchNorm2d(84)
#         self.fc2 = nn.Linear(84, args.num_classes)
#
#     def forward(self, x):
#         x = F.relu(self.pool(self.bn1(self.conv1(x))))
#         x = F.relu(self.pool(self.bn2(self.conv2(x))))
#         x = x.view(-1, 16 * 5 * 5)
#         x1 = F.relu(self.bn3(self.fc0(x)))
#         x = F.relu(self.bn4(self.fc1(x1)))
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1), x1

class Cifar10Model(nn.Module):
    """
    used for CIFAR-10 and Office-Caltech10
    """

    def __init__(self, args):
        super(Cifar10Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(192)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(192, 128, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 384, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(384)

        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.maxpool6 = nn.MaxPool2d(3, 2)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(9216, 4096)),
                ('bn6', nn.BatchNorm1d(4096)),
                ('relu6', nn.ReLU(inplace=True)),

                ('fc2', nn.Linear(4096, 4096)),
                ('bn7', nn.BatchNorm1d(4096)),
                ('relu7', nn.ReLU(inplace=True)),

                ('fc3', nn.Linear(4096, args.num_classes)),
            ])
        )

    def forward(self, x):
        x = self.maxpool1(F.relu(self.bn1(self.conv1(x))))
        x = self.maxpool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x1 = F.relu(self.bn5(self.conv5(x)))
        # x1 = self.maxpool6(F.relu(self.bn6(self.conv6(x))))
        x = self.avgpool(x1)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        # x1 = self.features(x)
        # x = self.avgpool(x1)
        # x = torch.flatten(x, 1)
        # x = self.classifier(x)
        return F.log_softmax(x, dim=1), x1

class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc0 = nn.Linear(16 * 5 * 5, 120)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, args.num_classes)

    # def forward(self, x):
    #     x = self.pool(F.relu(self.conv1(x)))
    #     x = self.pool(F.relu(self.conv2(x)))
    #     x = x.view(-1, 16 * 5 * 5)
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     x = self.fc3(x)
    #     return F.log_softmax(x, dim=1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x1 = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x1))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1), x1

class Lenet(nn.Module):
    def __init__(self, args):
        super(Lenet, self).__init__()
        self.n_cls = 10
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 5 * 5, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, self.n_cls)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x1 = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x1))
        x = self.fc3(x)

        return F.log_softmax(x, dim=1), x1

class AlexNet(nn.Module):
    """
    used for DomainNet and Office-Caltech10
    """

    def __init__(self, args):
        super(AlexNet, self).__init__()
        self.features1 = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('maxpool1', nn.MaxPool2d(kernel_size=3, stride=2)),

                ('conv2', nn.Conv2d(64, 192, kernel_size=5, padding=2)),
                ('bn2', nn.BatchNorm2d(192)),
                ('relu2', nn.ReLU(inplace=True)),
                ('maxpool2', nn.MaxPool2d(kernel_size=3, stride=2)),

                ('conv3', nn.Conv2d(192, 384, kernel_size=3, padding=1)),
                ('bn3', nn.BatchNorm2d(384)),
                ('relu3', nn.ReLU(inplace=True)),

                ('conv4', nn.Conv2d(384, 256, kernel_size=3, padding=1)),
                ('bn4', nn.BatchNorm2d(256)),
                ('relu4', nn.ReLU(inplace=True)),

                ('conv5', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
                ('bn5', nn.BatchNorm2d(256)),
            ])
        )
        self.features2 = nn.Sequential(
            OrderedDict([
                ('relu5', nn.ReLU(inplace=True)),
                ('maxpool5', nn.MaxPool2d(kernel_size=3, stride=2)),
            ])
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier1 = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(256 * 6 * 6, 2048)),
                ('bn6', nn.BatchNorm1d(2048)),
                ('relu6', nn.ReLU(inplace=True)),

                ('fc2', nn.Linear(2048, 256)),
                ('bn7', nn.BatchNorm1d(256)),
            ])
        )

        self.classifier2 = nn.Sequential(
            OrderedDict([
                ('relu7', nn.ReLU(inplace=True)),
                ('fc3', nn.Linear(256, args.num_classes)),
            ])
        )

    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x1 = self.classifier1(x)
        x = self.classifier2(x1)

        return F.log_softmax(x, dim=1), x1