#!/usr/bin/env python3

from typing import List

import torch.nn as nn

from .resnet import ResNet


class CBAMResNet(ResNet):
    def __init__(
        self,
        block: nn.Module,
        layers: List[int],
        num_classes: int,
    ) -> None:
        r"""CBAM based ResNet"""
        super().__init__(block=block, layers=layers, num_classes=num_classes)

    def _init_layers(self):
        nn.init.kaiming_normal_(self.fc.weight)
        for key in self.state_dict():
            if key.split(".")[-1] == "weight":
                if "conv" in key:
                    nn.init.kaiming_normal_(
                        self.state_dict()[key], mode="fan_out"
                    )
                if "bn" in key:
                    if "spatial_gate" in key:
                        self.state_dict()[key][...] = 0
                    else:
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
                use_cbam=True,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_cbam=True))

        return nn.Sequential(*layers)
