#!/usr/bin/env python3

from .bam_resnet import BAMResNet
from .cbam_resnet import CBAMResNet
from .resnet import ResNet
from ..modules.blocks import BasicBlock, Bottleneck

__all__ = ["create_resnet"]


def create_resnet(depth, num_classes, att_type=None):
    r"""Create a resnet model for CIFAR (10 and 100)"""
    assert depth in [
        18,
        34,
        50,
        101,
    ], "network depth should be 18, 34, 50 or 101"

    if depth == 18:
        if att_type == "BAM":
            model = BAMResNet(BasicBlock, [2, 2, 2, 2], num_classes)
        elif att_type == "CBAM":
            model = CBAMResNet(BasicBlock, [2, 2, 2, 2], num_classes)
        else:
            model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes)
    elif depth == 34:
        if att_type == "BAM":
            model = BAMResNet(BasicBlock, [3, 4, 6, 3], num_classes)
        elif att_type == "CBAM":
            model = CBAMResNet(BasicBlock, [3, 4, 6, 3], num_classes)
        else:
            model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes)
    elif depth == 50:
        if att_type == "BAM":
            model = BAMResNet(Bottleneck, [3, 4, 6, 3], num_classes)
        elif att_type == "CBAM":
            model = CBAMResNet(Bottleneck, [3, 4, 6, 3], num_classes)
        else:
            model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes)
    elif depth == 101:
        if att_type == "BAM":
            model = BAMResNet(Bottleneck, [3, 4, 23, 3], num_classes)
        elif att_type == "CBAM":
            model = CBAMResNet(Bottleneck, [3, 4, 23, 3], num_classes)
        else:
            model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes)
    return model
