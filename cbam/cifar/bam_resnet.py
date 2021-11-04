#!/usr/bin/env python3

from typing import List

import torch.nn as nn

from .resnet import ResNet
from ..modules.bam import BAM


class BAMResNet(ResNet):
    def __init__(
        self,
        block: nn.Module,
        layers: List[int],
        num_classes: int,
    ) -> None:
        self.bam1 = BAM(64 * block.expansion)
        self.bam2 = BAM(128 * block.expansion)
        self.bam3 = BAM(256 * block.expansion)
        super().__init__(block=block, layers=layers, num_classes=num_classes)

    def get_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.bam1(x)
        x = self.layer2(x)
        x = self.bam2(x)
        x = self.layer3(x)
        x = self.bam3(x)
        x = self.layer4(x)
        return x
