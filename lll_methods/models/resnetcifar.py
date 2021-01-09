'''
Taken with some modifications from the code written by Yerlan Idelbayev
https://github.com/akamaster/pytorch_resnet_cifar10

Properly implemented ResNet-s for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
'''
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

__all__ = ['ResNetCIFAR']


def _weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, nonlinearity="relu")
    elif isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, nonlinearity="sigmoid")


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='B', relu_output=True):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu_output = relu_output

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], [0, 0, 0, 0, planes // 4, planes // 4], "constant",
                                                  0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        if self.relu_output:
            out = F.relu(out)
        return out


class ResNetCIFARModule(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, relu_last_hidden=False):
        super(ResNetCIFARModule, self).__init__()
        self.in_planes = 16
        self.latent_dim = 64
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], 1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], 2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], 2, relu_last_hidden)
        self.output_layer = nn.Linear(self.latent_dim, self.num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, relu_last_hidden=True):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(len(strides)):
            if i == (len(strides) - 1):
                layers.append(block(self.in_planes, planes, strides[i], relu_output=relu_last_hidden))
            else:
                layers.append(block(self.in_planes, planes, strides[i]))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.avg_pool2d(x, x.size()[3])
        x = x.view(x.size(0), -1)
        out = self.output_layer(x)
        return out, x


class ResNetCIFAR(nn.Module):
    def __init__(self, num_classes=10, num_layers=20, relu_last_hidden=False):
        super(ResNetCIFAR, self).__init__()
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.relu_last_hidden = relu_last_hidden

        if num_layers not in [20, 32, 44, 56, 110]:
            raise ValueError("For ResNetCifar, choose a number of layers out of 20, 32, 44, 56, and 110")
        elif num_layers == 20:
            self.model = ResNetCIFARModule(BasicBlock, [3, 3, 3], num_classes, relu_last_hidden)
        elif num_layers == 32:
            self.model = ResNetCIFARModule(BasicBlock, [5, 5, 5], num_classes, relu_last_hidden)
        elif num_layers == 44:
            self.model = ResNetCIFARModule(BasicBlock, [7, 7, 7], num_classes, relu_last_hidden)
        elif num_layers == 56:
            self.model = ResNetCIFARModule(BasicBlock, [9, 9, 9], num_classes, relu_last_hidden)
        elif num_layers == 110:
            self.model = ResNetCIFARModule(BasicBlock, [18, 18, 18], num_classes, relu_last_hidden)

        self.apply(_weights_init)

    def forward(self, input_):
        return self.model(input_)
