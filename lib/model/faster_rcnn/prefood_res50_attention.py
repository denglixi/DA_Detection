from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.utils.config import cfg
from model.faster_rcnn.faster_rcnn_global_local_attention import _fasterRCNNAttention

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo
import pdb

import numpy as np

__weights_dict = dict()


def load_weights(weight_file):
    if weight_file is None:
        return

    try:
        weights_dict = np.load(weight_file).item()
    except:
        weights_dict = np.load(weight_file, encoding='bytes').item()

    return weights_dict

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False)

class netD_pixel(nn.Module):
    def __init__(self, context=False):
        super(netD_pixel, self).__init__()
        self.conv1 = nn.Conv2d(256, 256, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.conv3 = nn.Conv2d(128, 1, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.context = context
        self._init_weights()

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(
                    mean)  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                # m.bias.data.zero_()
        normal_init(self.conv1, 0, 0.01)
        normal_init(self.conv2, 0, 0.01)
        normal_init(self.conv3, 0, 0.01)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        if self.context:
            feat = F.avg_pool2d(x, (x.size(2), x.size(3)))
            x = self.conv3(x)
            return F.sigmoid(x), feat
        else:
            x = self.conv3(x)
            return F.sigmoid(x)


class netD(nn.Module):
    def __init__(self, context=False):
        super(netD, self).__init__()
        self.conv1 = conv3x3(1024, 512, stride=2)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = conv3x3(512, 128, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = conv3x3(128, 128, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128, 2)
        self.context = context
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

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
            return x, feat
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

class KitModule(nn.Module):
    def __init__(self, weight_file=None):
        super(KitModule, self).__init__()
        global __weights_dict
        __weights_dict = load_weights(weight_file)

    def forward(self, x):
        raise NotImplemented

    # @staticmethod
    def batch_normalization(self, dim, name, **kwargs):
        if dim == 1:
            layer = nn.BatchNorm1d(**kwargs)
        elif dim == 2:
            layer = nn.BatchNorm2d(**kwargs)
        elif dim == 3:
            layer = nn.BatchNorm3d(**kwargs)
        else:
            raise NotImplementedError()

        try:
            if 'scale' in __weights_dict[name]:
                layer.state_dict()['weight'].copy_(
                    torch.from_numpy(__weights_dict[name]['scale']))
            else:
                layer.weight.data.fill_(1)

            if 'bias' in __weights_dict[name]:
                layer.state_dict()['bias'].copy_(
                    torch.from_numpy(__weights_dict[name]['bias']))
            else:
                layer.bias.data.fill_(0)

            layer.state_dict()['running_mean'].copy_(
                torch.from_numpy(__weights_dict[name]['mean']))
            layer.state_dict()['running_var'].copy_(
                torch.from_numpy(__weights_dict[name]['var']))
        except (KeyError, TypeError):
            print("no weight of {} from pretrained model".format(name))
        return layer

    # @staticmethod
    def conv(self, dim, name, **kwargs):
        if dim == 1:
            layer = nn.Conv1d(**kwargs)
        elif dim == 2:
            layer = nn.Conv2d(**kwargs)
        elif dim == 3:
            layer = nn.Conv3d(**kwargs)
        else:
            raise NotImplementedError()

        try:
            layer.state_dict()['weight'].copy_(
                torch.from_numpy(__weights_dict[name]['weights']))
            if 'bias' in __weights_dict[name]:
                layer.state_dict()['bias'].copy_(
                    torch.from_numpy(__weights_dict[name]['bias']))
        except (KeyError, TypeError):
            print("no weight of {} from pretrained model".format(name))

        return layer

    # @staticmethod
    def dense(self, name, **kwargs):
        layer = nn.Linear(**kwargs)
        try:
            layer.state_dict()['weight'].copy_(
                torch.from_numpy(__weights_dict[name]['weights']))
            if 'bias' in __weights_dict[name]:
                layer.state_dict()['bias'].copy_(
                    torch.from_numpy(__weights_dict[name]['bias']))
        except (KeyError, TypeError):
            print("no weight of {} from pretrained model".format(name))
        return layer

# class res50_b1(KitModule):
#    def __init__(self, weight_file):
#        super(res50_base, self).__init__(weight_file)
#        self.conv1 = self.conv(2, name='conv1', in_channels=3, out_channels=64, kernel_size=(
#            7, 7), stride=(2, 2), groups=1, bias=True)
#        self.bn_conv1 = self.batch_normalization(
#            2, 'bn_conv1', num_features=64, eps=9.999999747378752e-06, momentum=0.0)
#    def forward(self):


class res50_layer1(KitModule):
    def __init__(self, weight_file=None):
        super(res50_layer1, self).__init__(weight_file)

        self.conv1 = self.conv(2, name='conv1', in_channels=3, out_channels=64, kernel_size=(
            7, 7), stride=(2, 2), groups=1, bias=True)
        self.bn_conv1 = self.batch_normalization(
            2, 'bn_conv1', num_features=64, eps=9.999999747378752e-06, momentum=0.0)

    def forward(self, x):
        conv1_pad = F.pad(x, (3, 3, 3, 3))
        conv1 = self.conv1(conv1_pad)
        bn_conv1 = self.bn_conv1(conv1)
        conv1_relu = F.relu(bn_conv1)
        pool1_pad = F.pad(conv1_relu, (0, 1, 0, 1), value=float('-inf'))
        pool1 = F.max_pool2d(pool1_pad, kernel_size=(
            3, 3), stride=(2, 2), padding=0, ceil_mode=False)
        return pool1


class res50_layer2(KitModule):
    def __init__(self, weight_file=None):
        super(res50_layer2, self).__init__(weight_file)

        self.res2a_branch1 = self.conv(2, name='res2a_branch1', in_channels=64, out_channels=256, kernel_size=(
            1, 1), stride=(1, 1), groups=1, bias=False)
        self.res2a_branch2a = self.conv(2, name='res2a_branch2a', in_channels=64, out_channels=64, kernel_size=(
            1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn2a_branch1 = self.batch_normalization(
            2, 'bn2a_branch1', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.bn2a_branch2a = self.batch_normalization(
            2, 'bn2a_branch2a', num_features=64, eps=9.999999747378752e-06, momentum=0.0)
        self.res2a_branch2b = self.conv(2, name='res2a_branch2b', in_channels=64, out_channels=64, kernel_size=(
            3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn2a_branch2b = self.batch_normalization(
            2, 'bn2a_branch2b', num_features=64, eps=9.999999747378752e-06, momentum=0.0)
        self.res2a_branch2c = self.conv(2, name='res2a_branch2c', in_channels=64, out_channels=256, kernel_size=(
            1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn2a_branch2c = self.batch_normalization(
            2, 'bn2a_branch2c', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.res2b_branch2a = self.conv(2, name='res2b_branch2a', in_channels=256, out_channels=64, kernel_size=(
            1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn2b_branch2a = self.batch_normalization(
            2, 'bn2b_branch2a', num_features=64, eps=9.999999747378752e-06, momentum=0.0)
        self.res2b_branch2b = self.conv(2, name='res2b_branch2b', in_channels=64, out_channels=64, kernel_size=(
            3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn2b_branch2b = self.batch_normalization(
            2, 'bn2b_branch2b', num_features=64, eps=9.999999747378752e-06, momentum=0.0)
        self.res2b_branch2c = self.conv(2, name='res2b_branch2c', in_channels=64, out_channels=256, kernel_size=(
            1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn2b_branch2c = self.batch_normalization(
            2, 'bn2b_branch2c', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.res2c_branch2a = self.conv(2, name='res2c_branch2a', in_channels=256, out_channels=64, kernel_size=(
            1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn2c_branch2a = self.batch_normalization(
            2, 'bn2c_branch2a', num_features=64, eps=9.999999747378752e-06, momentum=0.0)
        self.res2c_branch2b = self.conv(2, name='res2c_branch2b', in_channels=64, out_channels=64, kernel_size=(
            3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn2c_branch2b = self.batch_normalization(
            2, 'bn2c_branch2b', num_features=64, eps=9.999999747378752e-06, momentum=0.0)
        self.res2c_branch2c = self.conv(2, name='res2c_branch2c', in_channels=64, out_channels=256, kernel_size=(
            1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn2c_branch2c = self.batch_normalization(
            2, 'bn2c_branch2c', num_features=256, eps=9.999999747378752e-06, momentum=0.0)

    def forward(self, x):
        res2a_branch1 = self.res2a_branch1(x)
        res2a_branch2a = self.res2a_branch2a(x)
        bn2a_branch1 = self.bn2a_branch1(res2a_branch1)
        bn2a_branch2a = self.bn2a_branch2a(res2a_branch2a)
        res2a_branch2a_relu = F.relu(bn2a_branch2a)
        res2a_branch2b_pad = F.pad(res2a_branch2a_relu, (1, 1, 1, 1))
        res2a_branch2b = self.res2a_branch2b(res2a_branch2b_pad)
        bn2a_branch2b = self.bn2a_branch2b(res2a_branch2b)
        res2a_branch2b_relu = F.relu(bn2a_branch2b)
        res2a_branch2c = self.res2a_branch2c(res2a_branch2b_relu)
        bn2a_branch2c = self.bn2a_branch2c(res2a_branch2c)
        res2a = bn2a_branch1 + bn2a_branch2c
        res2a_relu = F.relu(res2a)
        res2b_branch2a = self.res2b_branch2a(res2a_relu)
        bn2b_branch2a = self.bn2b_branch2a(res2b_branch2a)
        res2b_branch2a_relu = F.relu(bn2b_branch2a)
        res2b_branch2b_pad = F.pad(res2b_branch2a_relu, (1, 1, 1, 1))
        res2b_branch2b = self.res2b_branch2b(res2b_branch2b_pad)
        bn2b_branch2b = self.bn2b_branch2b(res2b_branch2b)
        res2b_branch2b_relu = F.relu(bn2b_branch2b)
        res2b_branch2c = self.res2b_branch2c(res2b_branch2b_relu)
        bn2b_branch2c = self.bn2b_branch2c(res2b_branch2c)
        res2b = res2a_relu + bn2b_branch2c
        res2b_relu = F.relu(res2b)
        res2c_branch2a = self.res2c_branch2a(res2b_relu)
        bn2c_branch2a = self.bn2c_branch2a(res2c_branch2a)
        res2c_branch2a_relu = F.relu(bn2c_branch2a)
        res2c_branch2b_pad = F.pad(res2c_branch2a_relu, (1, 1, 1, 1))
        res2c_branch2b = self.res2c_branch2b(res2c_branch2b_pad)
        bn2c_branch2b = self.bn2c_branch2b(res2c_branch2b)
        res2c_branch2b_relu = F.relu(bn2c_branch2b)
        res2c_branch2c = self.res2c_branch2c(res2c_branch2b_relu)
        bn2c_branch2c = self.bn2c_branch2c(res2c_branch2c)
        res2c = res2b_relu + bn2c_branch2c
        res2c_relu = F.relu(res2c)
        return res2c_relu


class res50_layer3(KitModule):
    def __init__(self, weight_file=None):
        super(res50_layer3, self).__init__(weight_file)

        self.res3a_branch1 = self.conv(2, name='res3a_branch1', in_channels=256, out_channels=512, kernel_size=(
            1, 1), stride=(2, 2), groups=1, bias=False)
        self.res3a_branch2a = self.conv(2, name='res3a_branch2a', in_channels=256, out_channels=128, kernel_size=(
            1, 1), stride=(2, 2), groups=1, bias=False)
        self.bn3a_branch1 = self.batch_normalization(
            2, 'bn3a_branch1', num_features=512, eps=9.999999747378752e-06, momentum=0.0)
        self.bn3a_branch2a = self.batch_normalization(
            2, 'bn3a_branch2a', num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.res3a_branch2b = self.conv(2, name='res3a_branch2b', in_channels=128, out_channels=128, kernel_size=(
            3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn3a_branch2b = self.batch_normalization(
            2, 'bn3a_branch2b', num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.res3a_branch2c = self.conv(2, name='res3a_branch2c', in_channels=128, out_channels=512, kernel_size=(
            1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn3a_branch2c = self.batch_normalization(
            2, 'bn3a_branch2c', num_features=512, eps=9.999999747378752e-06, momentum=0.0)
        self.res3b_branch2a = self.conv(2, name='res3b_branch2a', in_channels=512, out_channels=128, kernel_size=(
            1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn3b_branch2a = self.batch_normalization(
            2, 'bn3b_branch2a', num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.res3b_branch2b = self.conv(2, name='res3b_branch2b', in_channels=128, out_channels=128, kernel_size=(
            3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn3b_branch2b = self.batch_normalization(
            2, 'bn3b_branch2b', num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.res3b_branch2c = self.conv(2, name='res3b_branch2c', in_channels=128, out_channels=512, kernel_size=(
            1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn3b_branch2c = self.batch_normalization(
            2, 'bn3b_branch2c', num_features=512, eps=9.999999747378752e-06, momentum=0.0)
        self.res3c_branch2a = self.conv(2, name='res3c_branch2a', in_channels=512, out_channels=128, kernel_size=(
            1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn3c_branch2a = self.batch_normalization(
            2, 'bn3c_branch2a', num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.res3c_branch2b = self.conv(2, name='res3c_branch2b', in_channels=128, out_channels=128, kernel_size=(
            3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn3c_branch2b = self.batch_normalization(
            2, 'bn3c_branch2b', num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.res3c_branch2c = self.conv(2, name='res3c_branch2c', in_channels=128, out_channels=512, kernel_size=(
            1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn3c_branch2c = self.batch_normalization(
            2, 'bn3c_branch2c', num_features=512, eps=9.999999747378752e-06, momentum=0.0)
        self.res3d_branch2a = self.conv(2, name='res3d_branch2a', in_channels=512, out_channels=128, kernel_size=(
            1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn3d_branch2a = self.batch_normalization(
            2, 'bn3d_branch2a', num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.res3d_branch2b = self.conv(2, name='res3d_branch2b', in_channels=128, out_channels=128, kernel_size=(
            3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn3d_branch2b = self.batch_normalization(
            2, 'bn3d_branch2b', num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.res3d_branch2c = self.conv(2, name='res3d_branch2c', in_channels=128, out_channels=512, kernel_size=(
            1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn3d_branch2c = self.batch_normalization(
            2, 'bn3d_branch2c', num_features=512, eps=9.999999747378752e-06, momentum=0.0)

    def forward(self, x):
        res3a_branch1 = self.res3a_branch1(x)
        res3a_branch2a = self.res3a_branch2a(x)
        bn3a_branch1 = self.bn3a_branch1(res3a_branch1)
        bn3a_branch2a = self.bn3a_branch2a(res3a_branch2a)
        res3a_branch2a_relu = F.relu(bn3a_branch2a)
        res3a_branch2b_pad = F.pad(res3a_branch2a_relu, (1, 1, 1, 1))
        res3a_branch2b = self.res3a_branch2b(res3a_branch2b_pad)
        bn3a_branch2b = self.bn3a_branch2b(res3a_branch2b)
        res3a_branch2b_relu = F.relu(bn3a_branch2b)
        res3a_branch2c = self.res3a_branch2c(res3a_branch2b_relu)
        bn3a_branch2c = self.bn3a_branch2c(res3a_branch2c)
        res3a = bn3a_branch1 + bn3a_branch2c
        res3a_relu = F.relu(res3a)
        res3b_branch2a = self.res3b_branch2a(res3a_relu)
        bn3b_branch2a = self.bn3b_branch2a(res3b_branch2a)
        res3b_branch2a_relu = F.relu(bn3b_branch2a)
        res3b_branch2b_pad = F.pad(res3b_branch2a_relu, (1, 1, 1, 1))
        res3b_branch2b = self.res3b_branch2b(res3b_branch2b_pad)
        bn3b_branch2b = self.bn3b_branch2b(res3b_branch2b)
        res3b_branch2b_relu = F.relu(bn3b_branch2b)
        res3b_branch2c = self.res3b_branch2c(res3b_branch2b_relu)
        bn3b_branch2c = self.bn3b_branch2c(res3b_branch2c)
        res3b = res3a_relu + bn3b_branch2c
        res3b_relu = F.relu(res3b)
        res3c_branch2a = self.res3c_branch2a(res3b_relu)
        bn3c_branch2a = self.bn3c_branch2a(res3c_branch2a)
        res3c_branch2a_relu = F.relu(bn3c_branch2a)
        res3c_branch2b_pad = F.pad(res3c_branch2a_relu, (1, 1, 1, 1))
        res3c_branch2b = self.res3c_branch2b(res3c_branch2b_pad)
        bn3c_branch2b = self.bn3c_branch2b(res3c_branch2b)
        res3c_branch2b_relu = F.relu(bn3c_branch2b)
        res3c_branch2c = self.res3c_branch2c(res3c_branch2b_relu)
        bn3c_branch2c = self.bn3c_branch2c(res3c_branch2c)
        res3c = res3b_relu + bn3c_branch2c
        res3c_relu = F.relu(res3c)
        res3d_branch2a = self.res3d_branch2a(res3c_relu)
        bn3d_branch2a = self.bn3d_branch2a(res3d_branch2a)
        res3d_branch2a_relu = F.relu(bn3d_branch2a)
        res3d_branch2b_pad = F.pad(res3d_branch2a_relu, (1, 1, 1, 1))
        res3d_branch2b = self.res3d_branch2b(res3d_branch2b_pad)
        bn3d_branch2b = self.bn3d_branch2b(res3d_branch2b)
        res3d_branch2b_relu = F.relu(bn3d_branch2b)
        res3d_branch2c = self.res3d_branch2c(res3d_branch2b_relu)
        bn3d_branch2c = self.bn3d_branch2c(res3d_branch2c)
        res3d = res3c_relu + bn3d_branch2c
        res3d_relu = F.relu(res3d)
        return res3d_relu


class res50_layer4(KitModule):
    def __init__(self, weight_file=None):
        super(res50_layer4, self).__init__(weight_file)

        self.res4a_branch1 = self.conv(2, name='res4a_branch1', in_channels=512, out_channels=1024, kernel_size=(
            1, 1), stride=(2, 2), groups=1, bias=False)
        self.res4a_branch2a = self.conv(2, name='res4a_branch2a', in_channels=512, out_channels=256, kernel_size=(
            1, 1), stride=(2, 2), groups=1, bias=False)
        self.bn4a_branch1 = self.batch_normalization(
            2, 'bn4a_branch1', num_features=1024, eps=9.999999747378752e-06, momentum=0.0)
        self.bn4a_branch2a = self.batch_normalization(
            2, 'bn4a_branch2a', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.res4a_branch2b = self.conv(2, name='res4a_branch2b', in_channels=256, out_channels=256, kernel_size=(
            3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn4a_branch2b = self.batch_normalization(
            2, 'bn4a_branch2b', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.res4a_branch2c = self.conv(2, name='res4a_branch2c', in_channels=256, out_channels=1024, kernel_size=(
            1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4a_branch2c = self.batch_normalization(
            2, 'bn4a_branch2c', num_features=1024, eps=9.999999747378752e-06, momentum=0.0)
        self.res4b_branch2a = self.conv(2, name='res4b_branch2a', in_channels=1024, out_channels=256, kernel_size=(
            1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4b_branch2a = self.batch_normalization(
            2, 'bn4b_branch2a', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.res4b_branch2b = self.conv(2, name='res4b_branch2b', in_channels=256, out_channels=256, kernel_size=(
            3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn4b_branch2b = self.batch_normalization(
            2, 'bn4b_branch2b', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.res4b_branch2c = self.conv(2, name='res4b_branch2c', in_channels=256, out_channels=1024, kernel_size=(
            1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4b_branch2c = self.batch_normalization(
            2, 'bn4b_branch2c', num_features=1024, eps=9.999999747378752e-06, momentum=0.0)
        self.res4c_branch2a = self.conv(2, name='res4c_branch2a', in_channels=1024, out_channels=256, kernel_size=(
            1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4c_branch2a = self.batch_normalization(
            2, 'bn4c_branch2a', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.res4c_branch2b = self.conv(2, name='res4c_branch2b', in_channels=256, out_channels=256, kernel_size=(
            3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn4c_branch2b = self.batch_normalization(
            2, 'bn4c_branch2b', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.res4c_branch2c = self.conv(2, name='res4c_branch2c', in_channels=256, out_channels=1024, kernel_size=(
            1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4c_branch2c = self.batch_normalization(
            2, 'bn4c_branch2c', num_features=1024, eps=9.999999747378752e-06, momentum=0.0)
        self.res4d_branch2a = self.conv(2, name='res4d_branch2a', in_channels=1024, out_channels=256, kernel_size=(
            1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4d_branch2a = self.batch_normalization(
            2, 'bn4d_branch2a', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.res4d_branch2b = self.conv(2, name='res4d_branch2b', in_channels=256, out_channels=256, kernel_size=(
            3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn4d_branch2b = self.batch_normalization(
            2, 'bn4d_branch2b', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.res4d_branch2c = self.conv(2, name='res4d_branch2c', in_channels=256, out_channels=1024, kernel_size=(
            1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4d_branch2c = self.batch_normalization(
            2, 'bn4d_branch2c', num_features=1024, eps=9.999999747378752e-06, momentum=0.0)
        self.res4e_branch2a = self.conv(2, name='res4e_branch2a', in_channels=1024, out_channels=256, kernel_size=(
            1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4e_branch2a = self.batch_normalization(
            2, 'bn4e_branch2a', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.res4e_branch2b = self.conv(2, name='res4e_branch2b', in_channels=256, out_channels=256, kernel_size=(
            3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn4e_branch2b = self.batch_normalization(
            2, 'bn4e_branch2b', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.res4e_branch2c = self.conv(2, name='res4e_branch2c', in_channels=256, out_channels=1024, kernel_size=(
            1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4e_branch2c = self.batch_normalization(
            2, 'bn4e_branch2c', num_features=1024, eps=9.999999747378752e-06, momentum=0.0)
        self.res4f_branch2a = self.conv(2, name='res4f_branch2a', in_channels=1024, out_channels=256, kernel_size=(
            1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4f_branch2a = self.batch_normalization(
            2, 'bn4f_branch2a', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.res4f_branch2b = self.conv(2, name='res4f_branch2b', in_channels=256, out_channels=256, kernel_size=(
            3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn4f_branch2b = self.batch_normalization(
            2, 'bn4f_branch2b', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.res4f_branch2c = self.conv(2, name='res4f_branch2c', in_channels=256, out_channels=1024, kernel_size=(
            1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4f_branch2c = self.batch_normalization(
            2, 'bn4f_branch2c', num_features=1024, eps=9.999999747378752e-06, momentum=0.0)

    def forward(self, x):
        res4a_branch1 = self.res4a_branch1(x)
        res4a_branch2a = self.res4a_branch2a(x)
        bn4a_branch1 = self.bn4a_branch1(res4a_branch1)
        bn4a_branch2a = self.bn4a_branch2a(res4a_branch2a)
        res4a_branch2a_relu = F.relu(bn4a_branch2a)
        res4a_branch2b_pad = F.pad(res4a_branch2a_relu, (1, 1, 1, 1))
        res4a_branch2b = self.res4a_branch2b(res4a_branch2b_pad)
        bn4a_branch2b = self.bn4a_branch2b(res4a_branch2b)
        res4a_branch2b_relu = F.relu(bn4a_branch2b)
        res4a_branch2c = self.res4a_branch2c(res4a_branch2b_relu)
        bn4a_branch2c = self.bn4a_branch2c(res4a_branch2c)
        res4a = bn4a_branch1 + bn4a_branch2c
        res4a_relu = F.relu(res4a)
        res4b_branch2a = self.res4b_branch2a(res4a_relu)
        bn4b_branch2a = self.bn4b_branch2a(res4b_branch2a)
        res4b_branch2a_relu = F.relu(bn4b_branch2a)
        res4b_branch2b_pad = F.pad(res4b_branch2a_relu, (1, 1, 1, 1))
        res4b_branch2b = self.res4b_branch2b(res4b_branch2b_pad)
        bn4b_branch2b = self.bn4b_branch2b(res4b_branch2b)
        res4b_branch2b_relu = F.relu(bn4b_branch2b)
        res4b_branch2c = self.res4b_branch2c(res4b_branch2b_relu)
        bn4b_branch2c = self.bn4b_branch2c(res4b_branch2c)
        res4b = res4a_relu + bn4b_branch2c
        res4b_relu = F.relu(res4b)
        res4c_branch2a = self.res4c_branch2a(res4b_relu)
        bn4c_branch2a = self.bn4c_branch2a(res4c_branch2a)
        res4c_branch2a_relu = F.relu(bn4c_branch2a)
        res4c_branch2b_pad = F.pad(res4c_branch2a_relu, (1, 1, 1, 1))
        res4c_branch2b = self.res4c_branch2b(res4c_branch2b_pad)
        bn4c_branch2b = self.bn4c_branch2b(res4c_branch2b)
        res4c_branch2b_relu = F.relu(bn4c_branch2b)
        res4c_branch2c = self.res4c_branch2c(res4c_branch2b_relu)
        bn4c_branch2c = self.bn4c_branch2c(res4c_branch2c)
        res4c = res4b_relu + bn4c_branch2c
        res4c_relu = F.relu(res4c)
        res4d_branch2a = self.res4d_branch2a(res4c_relu)
        bn4d_branch2a = self.bn4d_branch2a(res4d_branch2a)
        res4d_branch2a_relu = F.relu(bn4d_branch2a)
        res4d_branch2b_pad = F.pad(res4d_branch2a_relu, (1, 1, 1, 1))
        res4d_branch2b = self.res4d_branch2b(res4d_branch2b_pad)
        bn4d_branch2b = self.bn4d_branch2b(res4d_branch2b)
        res4d_branch2b_relu = F.relu(bn4d_branch2b)
        res4d_branch2c = self.res4d_branch2c(res4d_branch2b_relu)
        bn4d_branch2c = self.bn4d_branch2c(res4d_branch2c)
        res4d = res4c_relu + bn4d_branch2c
        res4d_relu = F.relu(res4d)
        res4e_branch2a = self.res4e_branch2a(res4d_relu)
        bn4e_branch2a = self.bn4e_branch2a(res4e_branch2a)
        res4e_branch2a_relu = F.relu(bn4e_branch2a)
        res4e_branch2b_pad = F.pad(res4e_branch2a_relu, (1, 1, 1, 1))
        res4e_branch2b = self.res4e_branch2b(res4e_branch2b_pad)
        bn4e_branch2b = self.bn4e_branch2b(res4e_branch2b)
        res4e_branch2b_relu = F.relu(bn4e_branch2b)
        res4e_branch2c = self.res4e_branch2c(res4e_branch2b_relu)
        bn4e_branch2c = self.bn4e_branch2c(res4e_branch2c)
        res4e = res4d_relu + bn4e_branch2c
        res4e_relu = F.relu(res4e)
        res4f_branch2a = self.res4f_branch2a(res4e_relu)
        bn4f_branch2a = self.bn4f_branch2a(res4f_branch2a)
        res4f_branch2a_relu = F.relu(bn4f_branch2a)
        res4f_branch2b_pad = F.pad(res4f_branch2a_relu, (1, 1, 1, 1))
        res4f_branch2b = self.res4f_branch2b(res4f_branch2b_pad)
        bn4f_branch2b = self.bn4f_branch2b(res4f_branch2b)
        res4f_branch2b_relu = F.relu(bn4f_branch2b)
        res4f_branch2c = self.res4f_branch2c(res4f_branch2b_relu)
        bn4f_branch2c = self.bn4f_branch2c(res4f_branch2c)
        res4f = res4e_relu + bn4f_branch2c
        res4f_relu = F.relu(res4f)
        return res4f_relu


class res50_base(KitModule):
    def __init__(self, weight_file=None):
        super(res50_base, self).__init__(weight_file)

        self.conv1 = self.conv(2, name='conv1', in_channels=3, out_channels=64, kernel_size=(
            7, 7), stride=(2, 2), groups=1, bias=True)
        self.bn_conv1 = self.batch_normalization(
            2, 'bn_conv1', num_features=64, eps=9.999999747378752e-06, momentum=0.0)
        self.res2a_branch1 = self.conv(2, name='res2a_branch1', in_channels=64, out_channels=256, kernel_size=(
            1, 1), stride=(1, 1), groups=1, bias=False)
        self.res2a_branch2a = self.conv(2, name='res2a_branch2a', in_channels=64, out_channels=64, kernel_size=(
            1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn2a_branch1 = self.batch_normalization(
            2, 'bn2a_branch1', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.bn2a_branch2a = self.batch_normalization(
            2, 'bn2a_branch2a', num_features=64, eps=9.999999747378752e-06, momentum=0.0)
        self.res2a_branch2b = self.conv(2, name='res2a_branch2b', in_channels=64, out_channels=64, kernel_size=(
            3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn2a_branch2b = self.batch_normalization(
            2, 'bn2a_branch2b', num_features=64, eps=9.999999747378752e-06, momentum=0.0)
        self.res2a_branch2c = self.conv(2, name='res2a_branch2c', in_channels=64, out_channels=256, kernel_size=(
            1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn2a_branch2c = self.batch_normalization(
            2, 'bn2a_branch2c', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.res2b_branch2a = self.conv(2, name='res2b_branch2a', in_channels=256, out_channels=64, kernel_size=(
            1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn2b_branch2a = self.batch_normalization(
            2, 'bn2b_branch2a', num_features=64, eps=9.999999747378752e-06, momentum=0.0)
        self.res2b_branch2b = self.conv(2, name='res2b_branch2b', in_channels=64, out_channels=64, kernel_size=(
            3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn2b_branch2b = self.batch_normalization(
            2, 'bn2b_branch2b', num_features=64, eps=9.999999747378752e-06, momentum=0.0)
        self.res2b_branch2c = self.conv(2, name='res2b_branch2c', in_channels=64, out_channels=256, kernel_size=(
            1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn2b_branch2c = self.batch_normalization(
            2, 'bn2b_branch2c', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.res2c_branch2a = self.conv(2, name='res2c_branch2a', in_channels=256, out_channels=64, kernel_size=(
            1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn2c_branch2a = self.batch_normalization(
            2, 'bn2c_branch2a', num_features=64, eps=9.999999747378752e-06, momentum=0.0)
        self.res2c_branch2b = self.conv(2, name='res2c_branch2b', in_channels=64, out_channels=64, kernel_size=(
            3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn2c_branch2b = self.batch_normalization(
            2, 'bn2c_branch2b', num_features=64, eps=9.999999747378752e-06, momentum=0.0)
        self.res2c_branch2c = self.conv(2, name='res2c_branch2c', in_channels=64, out_channels=256, kernel_size=(
            1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn2c_branch2c = self.batch_normalization(
            2, 'bn2c_branch2c', num_features=256, eps=9.999999747378752e-06, momentum=0.0)

        self.res3a_branch1 = self.conv(2, name='res3a_branch1', in_channels=256, out_channels=512, kernel_size=(
            1, 1), stride=(2, 2), groups=1, bias=False)
        self.res3a_branch2a = self.conv(2, name='res3a_branch2a', in_channels=256, out_channels=128, kernel_size=(
            1, 1), stride=(2, 2), groups=1, bias=False)
        self.bn3a_branch1 = self.batch_normalization(
            2, 'bn3a_branch1', num_features=512, eps=9.999999747378752e-06, momentum=0.0)
        self.bn3a_branch2a = self.batch_normalization(
            2, 'bn3a_branch2a', num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.res3a_branch2b = self.conv(2, name='res3a_branch2b', in_channels=128, out_channels=128, kernel_size=(
            3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn3a_branch2b = self.batch_normalization(
            2, 'bn3a_branch2b', num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.res3a_branch2c = self.conv(2, name='res3a_branch2c', in_channels=128, out_channels=512, kernel_size=(
            1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn3a_branch2c = self.batch_normalization(
            2, 'bn3a_branch2c', num_features=512, eps=9.999999747378752e-06, momentum=0.0)
        self.res3b_branch2a = self.conv(2, name='res3b_branch2a', in_channels=512, out_channels=128, kernel_size=(
            1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn3b_branch2a = self.batch_normalization(
            2, 'bn3b_branch2a', num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.res3b_branch2b = self.conv(2, name='res3b_branch2b', in_channels=128, out_channels=128, kernel_size=(
            3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn3b_branch2b = self.batch_normalization(
            2, 'bn3b_branch2b', num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.res3b_branch2c = self.conv(2, name='res3b_branch2c', in_channels=128, out_channels=512, kernel_size=(
            1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn3b_branch2c = self.batch_normalization(
            2, 'bn3b_branch2c', num_features=512, eps=9.999999747378752e-06, momentum=0.0)
        self.res3c_branch2a = self.conv(2, name='res3c_branch2a', in_channels=512, out_channels=128, kernel_size=(
            1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn3c_branch2a = self.batch_normalization(
            2, 'bn3c_branch2a', num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.res3c_branch2b = self.conv(2, name='res3c_branch2b', in_channels=128, out_channels=128, kernel_size=(
            3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn3c_branch2b = self.batch_normalization(
            2, 'bn3c_branch2b', num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.res3c_branch2c = self.conv(2, name='res3c_branch2c', in_channels=128, out_channels=512, kernel_size=(
            1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn3c_branch2c = self.batch_normalization(
            2, 'bn3c_branch2c', num_features=512, eps=9.999999747378752e-06, momentum=0.0)
        self.res3d_branch2a = self.conv(2, name='res3d_branch2a', in_channels=512, out_channels=128, kernel_size=(
            1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn3d_branch2a = self.batch_normalization(
            2, 'bn3d_branch2a', num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.res3d_branch2b = self.conv(2, name='res3d_branch2b', in_channels=128, out_channels=128, kernel_size=(
            3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn3d_branch2b = self.batch_normalization(
            2, 'bn3d_branch2b', num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.res3d_branch2c = self.conv(2, name='res3d_branch2c', in_channels=128, out_channels=512, kernel_size=(
            1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn3d_branch2c = self.batch_normalization(
            2, 'bn3d_branch2c', num_features=512, eps=9.999999747378752e-06, momentum=0.0)

        self.res4a_branch1 = self.conv(2, name='res4a_branch1', in_channels=512, out_channels=1024, kernel_size=(
            1, 1), stride=(2, 2), groups=1, bias=False)
        self.res4a_branch2a = self.conv(2, name='res4a_branch2a', in_channels=512, out_channels=256, kernel_size=(
            1, 1), stride=(2, 2), groups=1, bias=False)
        self.bn4a_branch1 = self.batch_normalization(
            2, 'bn4a_branch1', num_features=1024, eps=9.999999747378752e-06, momentum=0.0)
        self.bn4a_branch2a = self.batch_normalization(
            2, 'bn4a_branch2a', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.res4a_branch2b = self.conv(2, name='res4a_branch2b', in_channels=256, out_channels=256, kernel_size=(
            3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn4a_branch2b = self.batch_normalization(
            2, 'bn4a_branch2b', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.res4a_branch2c = self.conv(2, name='res4a_branch2c', in_channels=256, out_channels=1024, kernel_size=(
            1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4a_branch2c = self.batch_normalization(
            2, 'bn4a_branch2c', num_features=1024, eps=9.999999747378752e-06, momentum=0.0)
        self.res4b_branch2a = self.conv(2, name='res4b_branch2a', in_channels=1024, out_channels=256, kernel_size=(
            1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4b_branch2a = self.batch_normalization(
            2, 'bn4b_branch2a', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.res4b_branch2b = self.conv(2, name='res4b_branch2b', in_channels=256, out_channels=256, kernel_size=(
            3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn4b_branch2b = self.batch_normalization(
            2, 'bn4b_branch2b', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.res4b_branch2c = self.conv(2, name='res4b_branch2c', in_channels=256, out_channels=1024, kernel_size=(
            1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4b_branch2c = self.batch_normalization(
            2, 'bn4b_branch2c', num_features=1024, eps=9.999999747378752e-06, momentum=0.0)
        self.res4c_branch2a = self.conv(2, name='res4c_branch2a', in_channels=1024, out_channels=256, kernel_size=(
            1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4c_branch2a = self.batch_normalization(
            2, 'bn4c_branch2a', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.res4c_branch2b = self.conv(2, name='res4c_branch2b', in_channels=256, out_channels=256, kernel_size=(
            3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn4c_branch2b = self.batch_normalization(
            2, 'bn4c_branch2b', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.res4c_branch2c = self.conv(2, name='res4c_branch2c', in_channels=256, out_channels=1024, kernel_size=(
            1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4c_branch2c = self.batch_normalization(
            2, 'bn4c_branch2c', num_features=1024, eps=9.999999747378752e-06, momentum=0.0)
        self.res4d_branch2a = self.conv(2, name='res4d_branch2a', in_channels=1024, out_channels=256, kernel_size=(
            1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4d_branch2a = self.batch_normalization(
            2, 'bn4d_branch2a', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.res4d_branch2b = self.conv(2, name='res4d_branch2b', in_channels=256, out_channels=256, kernel_size=(
            3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn4d_branch2b = self.batch_normalization(
            2, 'bn4d_branch2b', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.res4d_branch2c = self.conv(2, name='res4d_branch2c', in_channels=256, out_channels=1024, kernel_size=(
            1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4d_branch2c = self.batch_normalization(
            2, 'bn4d_branch2c', num_features=1024, eps=9.999999747378752e-06, momentum=0.0)
        self.res4e_branch2a = self.conv(2, name='res4e_branch2a', in_channels=1024, out_channels=256, kernel_size=(
            1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4e_branch2a = self.batch_normalization(
            2, 'bn4e_branch2a', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.res4e_branch2b = self.conv(2, name='res4e_branch2b', in_channels=256, out_channels=256, kernel_size=(
            3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn4e_branch2b = self.batch_normalization(
            2, 'bn4e_branch2b', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.res4e_branch2c = self.conv(2, name='res4e_branch2c', in_channels=256, out_channels=1024, kernel_size=(
            1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4e_branch2c = self.batch_normalization(
            2, 'bn4e_branch2c', num_features=1024, eps=9.999999747378752e-06, momentum=0.0)
        self.res4f_branch2a = self.conv(2, name='res4f_branch2a', in_channels=1024, out_channels=256, kernel_size=(
            1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4f_branch2a = self.batch_normalization(
            2, 'bn4f_branch2a', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.res4f_branch2b = self.conv(2, name='res4f_branch2b', in_channels=256, out_channels=256, kernel_size=(
            3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn4f_branch2b = self.batch_normalization(
            2, 'bn4f_branch2b', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.res4f_branch2c = self.conv(2, name='res4f_branch2c', in_channels=256, out_channels=1024, kernel_size=(
            1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn4f_branch2c = self.batch_normalization(
            2, 'bn4f_branch2c', num_features=1024, eps=9.999999747378752e-06, momentum=0.0)

    def forward(self, x):
        conv1_pad = F.pad(x, (3, 3, 3, 3))
        conv1 = self.conv1(conv1_pad)
        bn_conv1 = self.bn_conv1(conv1)
        conv1_relu = F.relu(bn_conv1)
        pool1_pad = F.pad(conv1_relu, (0, 1, 0, 1), value=float('-inf'))
        pool1 = F.max_pool2d(pool1_pad, kernel_size=(
            3, 3), stride=(2, 2), padding=0, ceil_mode=False)
        res2a_branch1 = self.res2a_branch1(pool1)
        res2a_branch2a = self.res2a_branch2a(pool1)
        bn2a_branch1 = self.bn2a_branch1(res2a_branch1)
        bn2a_branch2a = self.bn2a_branch2a(res2a_branch2a)
        res2a_branch2a_relu = F.relu(bn2a_branch2a)
        res2a_branch2b_pad = F.pad(res2a_branch2a_relu, (1, 1, 1, 1))
        res2a_branch2b = self.res2a_branch2b(res2a_branch2b_pad)
        bn2a_branch2b = self.bn2a_branch2b(res2a_branch2b)
        res2a_branch2b_relu = F.relu(bn2a_branch2b)
        res2a_branch2c = self.res2a_branch2c(res2a_branch2b_relu)
        bn2a_branch2c = self.bn2a_branch2c(res2a_branch2c)
        res2a = bn2a_branch1 + bn2a_branch2c
        res2a_relu = F.relu(res2a)
        res2b_branch2a = self.res2b_branch2a(res2a_relu)
        bn2b_branch2a = self.bn2b_branch2a(res2b_branch2a)
        res2b_branch2a_relu = F.relu(bn2b_branch2a)
        res2b_branch2b_pad = F.pad(res2b_branch2a_relu, (1, 1, 1, 1))
        res2b_branch2b = self.res2b_branch2b(res2b_branch2b_pad)
        bn2b_branch2b = self.bn2b_branch2b(res2b_branch2b)
        res2b_branch2b_relu = F.relu(bn2b_branch2b)
        res2b_branch2c = self.res2b_branch2c(res2b_branch2b_relu)
        bn2b_branch2c = self.bn2b_branch2c(res2b_branch2c)
        res2b = res2a_relu + bn2b_branch2c
        res2b_relu = F.relu(res2b)
        res2c_branch2a = self.res2c_branch2a(res2b_relu)
        bn2c_branch2a = self.bn2c_branch2a(res2c_branch2a)
        res2c_branch2a_relu = F.relu(bn2c_branch2a)
        res2c_branch2b_pad = F.pad(res2c_branch2a_relu, (1, 1, 1, 1))
        res2c_branch2b = self.res2c_branch2b(res2c_branch2b_pad)
        bn2c_branch2b = self.bn2c_branch2b(res2c_branch2b)
        res2c_branch2b_relu = F.relu(bn2c_branch2b)
        res2c_branch2c = self.res2c_branch2c(res2c_branch2b_relu)
        bn2c_branch2c = self.bn2c_branch2c(res2c_branch2c)
        res2c = res2b_relu + bn2c_branch2c
        res2c_relu = F.relu(res2c)
        res3a_branch1 = self.res3a_branch1(res2c_relu)
        res3a_branch2a = self.res3a_branch2a(res2c_relu)
        bn3a_branch1 = self.bn3a_branch1(res3a_branch1)
        bn3a_branch2a = self.bn3a_branch2a(res3a_branch2a)
        res3a_branch2a_relu = F.relu(bn3a_branch2a)
        res3a_branch2b_pad = F.pad(res3a_branch2a_relu, (1, 1, 1, 1))
        res3a_branch2b = self.res3a_branch2b(res3a_branch2b_pad)
        bn3a_branch2b = self.bn3a_branch2b(res3a_branch2b)
        res3a_branch2b_relu = F.relu(bn3a_branch2b)
        res3a_branch2c = self.res3a_branch2c(res3a_branch2b_relu)
        bn3a_branch2c = self.bn3a_branch2c(res3a_branch2c)
        res3a = bn3a_branch1 + bn3a_branch2c
        res3a_relu = F.relu(res3a)
        res3b_branch2a = self.res3b_branch2a(res3a_relu)
        bn3b_branch2a = self.bn3b_branch2a(res3b_branch2a)
        res3b_branch2a_relu = F.relu(bn3b_branch2a)
        res3b_branch2b_pad = F.pad(res3b_branch2a_relu, (1, 1, 1, 1))
        res3b_branch2b = self.res3b_branch2b(res3b_branch2b_pad)
        bn3b_branch2b = self.bn3b_branch2b(res3b_branch2b)
        res3b_branch2b_relu = F.relu(bn3b_branch2b)
        res3b_branch2c = self.res3b_branch2c(res3b_branch2b_relu)
        bn3b_branch2c = self.bn3b_branch2c(res3b_branch2c)
        res3b = res3a_relu + bn3b_branch2c
        res3b_relu = F.relu(res3b)
        res3c_branch2a = self.res3c_branch2a(res3b_relu)
        bn3c_branch2a = self.bn3c_branch2a(res3c_branch2a)
        res3c_branch2a_relu = F.relu(bn3c_branch2a)
        res3c_branch2b_pad = F.pad(res3c_branch2a_relu, (1, 1, 1, 1))
        res3c_branch2b = self.res3c_branch2b(res3c_branch2b_pad)
        bn3c_branch2b = self.bn3c_branch2b(res3c_branch2b)
        res3c_branch2b_relu = F.relu(bn3c_branch2b)
        res3c_branch2c = self.res3c_branch2c(res3c_branch2b_relu)
        bn3c_branch2c = self.bn3c_branch2c(res3c_branch2c)
        res3c = res3b_relu + bn3c_branch2c
        res3c_relu = F.relu(res3c)
        res3d_branch2a = self.res3d_branch2a(res3c_relu)
        bn3d_branch2a = self.bn3d_branch2a(res3d_branch2a)
        res3d_branch2a_relu = F.relu(bn3d_branch2a)
        res3d_branch2b_pad = F.pad(res3d_branch2a_relu, (1, 1, 1, 1))
        res3d_branch2b = self.res3d_branch2b(res3d_branch2b_pad)
        bn3d_branch2b = self.bn3d_branch2b(res3d_branch2b)
        res3d_branch2b_relu = F.relu(bn3d_branch2b)
        res3d_branch2c = self.res3d_branch2c(res3d_branch2b_relu)
        bn3d_branch2c = self.bn3d_branch2c(res3d_branch2c)
        res3d = res3c_relu + bn3d_branch2c
        res3d_relu = F.relu(res3d)
        res4a_branch1 = self.res4a_branch1(res3d_relu)
        res4a_branch2a = self.res4a_branch2a(res3d_relu)
        bn4a_branch1 = self.bn4a_branch1(res4a_branch1)
        bn4a_branch2a = self.bn4a_branch2a(res4a_branch2a)
        res4a_branch2a_relu = F.relu(bn4a_branch2a)
        res4a_branch2b_pad = F.pad(res4a_branch2a_relu, (1, 1, 1, 1))
        res4a_branch2b = self.res4a_branch2b(res4a_branch2b_pad)
        bn4a_branch2b = self.bn4a_branch2b(res4a_branch2b)
        res4a_branch2b_relu = F.relu(bn4a_branch2b)
        res4a_branch2c = self.res4a_branch2c(res4a_branch2b_relu)
        bn4a_branch2c = self.bn4a_branch2c(res4a_branch2c)
        res4a = bn4a_branch1 + bn4a_branch2c
        res4a_relu = F.relu(res4a)
        res4b_branch2a = self.res4b_branch2a(res4a_relu)
        bn4b_branch2a = self.bn4b_branch2a(res4b_branch2a)
        res4b_branch2a_relu = F.relu(bn4b_branch2a)
        res4b_branch2b_pad = F.pad(res4b_branch2a_relu, (1, 1, 1, 1))
        res4b_branch2b = self.res4b_branch2b(res4b_branch2b_pad)
        bn4b_branch2b = self.bn4b_branch2b(res4b_branch2b)
        res4b_branch2b_relu = F.relu(bn4b_branch2b)
        res4b_branch2c = self.res4b_branch2c(res4b_branch2b_relu)
        bn4b_branch2c = self.bn4b_branch2c(res4b_branch2c)
        res4b = res4a_relu + bn4b_branch2c
        res4b_relu = F.relu(res4b)
        res4c_branch2a = self.res4c_branch2a(res4b_relu)
        bn4c_branch2a = self.bn4c_branch2a(res4c_branch2a)
        res4c_branch2a_relu = F.relu(bn4c_branch2a)
        res4c_branch2b_pad = F.pad(res4c_branch2a_relu, (1, 1, 1, 1))
        res4c_branch2b = self.res4c_branch2b(res4c_branch2b_pad)
        bn4c_branch2b = self.bn4c_branch2b(res4c_branch2b)
        res4c_branch2b_relu = F.relu(bn4c_branch2b)
        res4c_branch2c = self.res4c_branch2c(res4c_branch2b_relu)
        bn4c_branch2c = self.bn4c_branch2c(res4c_branch2c)
        res4c = res4b_relu + bn4c_branch2c
        res4c_relu = F.relu(res4c)
        res4d_branch2a = self.res4d_branch2a(res4c_relu)
        bn4d_branch2a = self.bn4d_branch2a(res4d_branch2a)
        res4d_branch2a_relu = F.relu(bn4d_branch2a)
        res4d_branch2b_pad = F.pad(res4d_branch2a_relu, (1, 1, 1, 1))
        res4d_branch2b = self.res4d_branch2b(res4d_branch2b_pad)
        bn4d_branch2b = self.bn4d_branch2b(res4d_branch2b)
        res4d_branch2b_relu = F.relu(bn4d_branch2b)
        res4d_branch2c = self.res4d_branch2c(res4d_branch2b_relu)
        bn4d_branch2c = self.bn4d_branch2c(res4d_branch2c)
        res4d = res4c_relu + bn4d_branch2c
        res4d_relu = F.relu(res4d)
        res4e_branch2a = self.res4e_branch2a(res4d_relu)
        bn4e_branch2a = self.bn4e_branch2a(res4e_branch2a)
        res4e_branch2a_relu = F.relu(bn4e_branch2a)
        res4e_branch2b_pad = F.pad(res4e_branch2a_relu, (1, 1, 1, 1))
        res4e_branch2b = self.res4e_branch2b(res4e_branch2b_pad)
        bn4e_branch2b = self.bn4e_branch2b(res4e_branch2b)
        res4e_branch2b_relu = F.relu(bn4e_branch2b)
        res4e_branch2c = self.res4e_branch2c(res4e_branch2b_relu)
        bn4e_branch2c = self.bn4e_branch2c(res4e_branch2c)
        res4e = res4d_relu + bn4e_branch2c
        res4e_relu = F.relu(res4e)
        res4f_branch2a = self.res4f_branch2a(res4e_relu)
        bn4f_branch2a = self.bn4f_branch2a(res4f_branch2a)
        res4f_branch2a_relu = F.relu(bn4f_branch2a)
        res4f_branch2b_pad = F.pad(res4f_branch2a_relu, (1, 1, 1, 1))
        res4f_branch2b = self.res4f_branch2b(res4f_branch2b_pad)
        bn4f_branch2b = self.bn4f_branch2b(res4f_branch2b)
        res4f_branch2b_relu = F.relu(bn4f_branch2b)
        res4f_branch2c = self.res4f_branch2c(res4f_branch2b_relu)
        bn4f_branch2c = self.bn4f_branch2c(res4f_branch2c)
        res4f = res4e_relu + bn4f_branch2c
        res4f_relu = F.relu(res4f)

        return res4f_relu


class res50_top(KitModule):
    def __init__(self, weight_file=None):
        super(res50_top, self).__init__(weight_file)

        self.res5a_branch1 = self.conv(2, name='res5a_branch1', in_channels=1024, out_channels=2048, kernel_size=(
            1, 1), stride=(2, 2), groups=1, bias=False)
        self.res5a_branch2a = self.conv(2, name='res5a_branch2a', in_channels=1024, out_channels=512, kernel_size=(
            1, 1), stride=(2, 2), groups=1, bias=False)
        self.bn5a_branch1 = self.batch_normalization(
            2, 'bn5a_branch1', num_features=2048, eps=9.999999747378752e-06, momentum=0.0)
        self.bn5a_branch2a = self.batch_normalization(
            2, 'bn5a_branch2a', num_features=512, eps=9.999999747378752e-06, momentum=0.0)
        self.res5a_branch2b = self.conv(2, name='res5a_branch2b', in_channels=512, out_channels=512, kernel_size=(
            3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn5a_branch2b = self.batch_normalization(
            2, 'bn5a_branch2b', num_features=512, eps=9.999999747378752e-06, momentum=0.0)
        self.res5a_branch2c = self.conv(2, name='res5a_branch2c', in_channels=512, out_channels=2048, kernel_size=(
            1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn5a_branch2c = self.batch_normalization(
            2, 'bn5a_branch2c', num_features=2048, eps=9.999999747378752e-06, momentum=0.0)
        self.res5b_branch2a = self.conv(2, name='res5b_branch2a', in_channels=2048, out_channels=512, kernel_size=(
            1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn5b_branch2a = self.batch_normalization(
            2, 'bn5b_branch2a', num_features=512, eps=9.999999747378752e-06, momentum=0.0)
        self.res5b_branch2b = self.conv(2, name='res5b_branch2b', in_channels=512, out_channels=512, kernel_size=(
            3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn5b_branch2b = self.batch_normalization(
            2, 'bn5b_branch2b', num_features=512, eps=9.999999747378752e-06, momentum=0.0)
        self.res5b_branch2c = self.conv(2, name='res5b_branch2c', in_channels=512, out_channels=2048, kernel_size=(
            1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn5b_branch2c = self.batch_normalization(
            2, 'bn5b_branch2c', num_features=2048, eps=9.999999747378752e-06, momentum=0.0)
        self.res5c_branch2a = self.conv(2, name='res5c_branch2a', in_channels=2048, out_channels=512, kernel_size=(
            1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn5c_branch2a = self.batch_normalization(
            2, 'bn5c_branch2a', num_features=512, eps=9.999999747378752e-06, momentum=0.0)
        self.res5c_branch2b = self.conv(2, name='res5c_branch2b', in_channels=512, out_channels=512, kernel_size=(
            3, 3), stride=(1, 1), groups=1, bias=False)
        self.bn5c_branch2b = self.batch_normalization(
            2, 'bn5c_branch2b', num_features=512, eps=9.999999747378752e-06, momentum=0.0)
        self.res5c_branch2c = self.conv(2, name='res5c_branch2c', in_channels=512, out_channels=2048, kernel_size=(
            1, 1), stride=(1, 1), groups=1, bias=False)
        self.bn5c_branch2c = self.batch_normalization(
            2, 'bn5c_branch2c', num_features=2048, eps=9.999999747378752e-06, momentum=0.0)
        self.fc798_1 = self.dense(
            name='fc798_1', in_features=2048, out_features=798, bias=True)

    def forward(self, x):
        res5a_branch1 = self.res5a_branch1(x)
        res5a_branch2a = self.res5a_branch2a(x)
        bn5a_branch1 = self.bn5a_branch1(res5a_branch1)
        bn5a_branch2a = self.bn5a_branch2a(res5a_branch2a)
        res5a_branch2a_relu = F.relu(bn5a_branch2a)
        res5a_branch2b_pad = F.pad(res5a_branch2a_relu, (1, 1, 1, 1))
        res5a_branch2b = self.res5a_branch2b(res5a_branch2b_pad)
        bn5a_branch2b = self.bn5a_branch2b(res5a_branch2b)
        res5a_branch2b_relu = F.relu(bn5a_branch2b)
        res5a_branch2c = self.res5a_branch2c(res5a_branch2b_relu)
        bn5a_branch2c = self.bn5a_branch2c(res5a_branch2c)
        res5a = bn5a_branch1 + bn5a_branch2c
        res5a_relu = F.relu(res5a)
        res5b_branch2a = self.res5b_branch2a(res5a_relu)
        bn5b_branch2a = self.bn5b_branch2a(res5b_branch2a)
        res5b_branch2a_relu = F.relu(bn5b_branch2a)
        res5b_branch2b_pad = F.pad(res5b_branch2a_relu, (1, 1, 1, 1))
        res5b_branch2b = self.res5b_branch2b(res5b_branch2b_pad)
        bn5b_branch2b = self.bn5b_branch2b(res5b_branch2b)
        res5b_branch2b_relu = F.relu(bn5b_branch2b)
        res5b_branch2c = self.res5b_branch2c(res5b_branch2b_relu)
        bn5b_branch2c = self.bn5b_branch2c(res5b_branch2c)
        res5b = res5a_relu + bn5b_branch2c
        res5b_relu = F.relu(res5b)
        res5c_branch2a = self.res5c_branch2a(res5b_relu)
        bn5c_branch2a = self.bn5c_branch2a(res5c_branch2a)
        res5c_branch2a_relu = F.relu(bn5c_branch2a)
        res5c_branch2b_pad = F.pad(res5c_branch2a_relu, (1, 1, 1, 1))
        res5c_branch2b = self.res5c_branch2b(res5c_branch2b_pad)
        bn5c_branch2b = self.bn5c_branch2b(res5c_branch2b)
        res5c_branch2b_relu = F.relu(bn5c_branch2b)
        res5c_branch2c = self.res5c_branch2c(res5c_branch2b_relu)
        bn5c_branch2c = self.bn5c_branch2c(res5c_branch2c)
        res5c = res5b_relu + bn5c_branch2c
        res5c_relu = F.relu(res5c)
        # pool5 = F.avg_pool2d(res5c_relu, kernel_size=(
        #    7, 7), stride=(1, 1), padding=(0,), ceil_mode=False)
        return res5c_relu

        # fc798_0 = pool5.view(pool5.size(0), -1)
        # fc798_1 = self.fc798_1(fc798_0)
        # prob = F.softmax(fc798_1)


class PreResNet50Attention(_fasterRCNNAttention):
    def __init__(self, classes, pretrained=False, class_agnostic=False, weight_file=None, fixed_layer=0, lc=False, gc=False):

        if weight_file == 'imagenet':
            self.model_path = "data/pretrained_model/resnet50_caffe.pth"
        elif weight_file == 'prefood':
            self.model_path = "data/pretrained_model/prefood_res50.pth"
        else:
            self.model_path = "data/pretrained_model/prefood_res50.pth"
        if not pretrained:
            self.model_path = None
        self.dout_base_model = 1024
        self.class_agnostic = class_agnostic
        self.fixed_layer = fixed_layer
        self.lc = lc
        self.gc = gc

        _fasterRCNNAttention.__init__(self, classes, class_agnostic,lc, gc)

    def _init_modules(self):
        #self.RCNN_base = res50_base(self.model_path)
        #self.RCNN_base = nn.Sequential(
        #    res50_layer1(self.model_path),
        #    res50_layer2(self.model_path),
        #    res50_layer3(self.model_path),
        #    res50_layer4(self.model_path))
        self.RCNN_base1 = nn.Sequential(res50_layer1(self.model_path),
                                        res50_layer2(self.model_path))
        self.RCNN_base2 = nn.Sequential(res50_layer3(self.model_path),
                                        res50_layer4(self.model_path))
        self.netD_pixel = netD_pixel(context=self.lc)
        self.netD = netD(context=self.gc)


        assert self.fixed_layer >= 0 and self.fixed_layer <= 4

        for layer_i in range(self.fixed_layer):
            for p in self.RCNN_base[layer_i].parameters():
                p.requires_grad = False

        self.RCNN_top = res50_top(self.model_path)

        feat_d = 2048
        if self.lc:
            feat_d += 128
        if self.gc:
            feat_d += 128


        self.RCNN_cls_score = nn.Linear(feat_d, self.n_classes)

        #self.RCNN_cls_score = nn.Linear(1024, self.n_classes)
        if self.class_agnostic:
            self.RCNN_bbox_pred = nn.Linear(feat_d, 4)
        else:
            self.RCNN_bbox_pred = nn.Linear(feat_d, 4 * self.n_classes)

        # Fix blocks
        for p in self.RCNN_base1[0].parameters():
            p.requires_grad = False
        for p in self.RCNN_base1[1].parameters():
            p.requires_grad = False

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters():
                    p.requires_grad = False

        self.RCNN_base1.apply(set_bn_fix)
        self.RCNN_base2.apply(set_bn_fix)
        self.RCNN_top.apply(set_bn_fix)

    def train(self, mode=True):
        # Override train so that the training mode is set as we want
        nn.Module.train(self, mode)
        if mode:
            # Set fixed blocks to be in eval mode
            # self.RCNN_base.eval()
            # self.RCNN_base[5].train()
            # self.RCNN_base[6].train()

            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            self.RCNN_base1.apply(set_bn_eval)
            self.RCNN_base2.apply(set_bn_eval)
            self.RCNN_top.apply(set_bn_eval)

    def _head_to_tail(self, pool5):
        fc7 = self.RCNN_top(pool5).mean(3).mean(2)
        return fc7
