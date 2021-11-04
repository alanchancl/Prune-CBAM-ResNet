#!/usr/bin/env python3

from typing import List

import torch.nn as nn

from ..modules.utils import Flatten


class ResNet(nn.Module):
    def __init__(
        self,
        block: nn.Module,
        layers: List[int],
        num_classes: int,
    ) -> None:
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(7)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.flatten = Flatten()
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._init_layers()

    def _init_layers(self):
        nn.init.kaiming_normal_(self.fc.weight)
        for key in self.state_dict():
            if key.split(".")[-1] == "weight":
                if "conv" in key:
                    nn.init.kaiming_normal_(
                        self.state_dict()[key], mode="fan_out"
                    )
                if "bn" in key:
                    self.state_dict()[key][...] = 1
            elif key.split(".")[-1] == "bias":
                self.state_dict()[key][...] = 0

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def get_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x, return_features=False):
        x = self.get_features(x)
        if return_features:
            return x

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
