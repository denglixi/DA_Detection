from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torchvision.models as models
import pdb


"""

"""


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False)


class netD_pixel(nn.Module):
    def __init__(self, input_dim, output_dim, context=False):
        super(netD_pixel, self).__init__()
        self.conv1 = conv1x1(input_dim, input_dim)
        # self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = conv1x1(input_dim, output_dim)
        # self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = conv1x1(output_dim, 1)

        self.context = context

    def forward(self, x):
        x = F.relu(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        if self.context:
            feat = F.avg_pool2d(x, (x.size(2), x.size(3)))
            # feat = x
            x = F.sigmoid(self.conv3(x))
            return x.view(-1, 1), feat  # torch.cat((feat1,feat2),1)#F
        else:
            x = F.sigmoid(self.conv3(x))
            return x.view(-1, 1)  # F.sigmoid(x)


class netD(nn.Module):
    def __init__(self, input_dim, context=False):
        super(netD, self).__init__()
        self.conv1 = conv3x3(input_dim, 512, stride=2)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = conv3x3(512, 128, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = conv3x3(128, 128, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128, 2)
        self.context = context

    def forward(self, x):
        x = F.dropout(F.relu(self.bn1(self.conv1(x))), training=self.training)
        x = F.dropout(F.relu(self.bn2(self.conv2(x))), training=self.training)
        x = F.dropout(F.relu(self.bn3(self.conv3(x))), training=self.training)
        x = F.avg_pool2d(x, (x.size(2), x.size(3)))
        x = x.view(-1, 128)
        if self.context:
            feat = x
        x = self.fc(x)
        if self.context:
            return x, feat  # torch.cat((feat1,feat2),1)#F
        else:
            return x


class netD_dc(nn.Module):
    def __init__(self):
        super(netD_dc, self).__init__()
        self.fc1 = nn.Linear(2048, 100)
        self.bn1 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, 100)
        self.bn2 = nn.BatchNorm1d(100)
        self.fc3 = nn.Linear(100, 2)

    def forward(self, x):
        x = F.dropout(F.relu(self.bn1(self.fc1(x))), training=self.training)
        x = F.dropout(F.relu(self.bn2(self.fc2(x))), training=self.training)
        x = self.fc3(x)
        return x
