#!/usr/bin/env python3

from .bam_resnet import BAMResNet
from .cbam_resnet import CBAMResNet
from .resnet import ResNet
from ..modules.blocks import BasicBlock, Bottleneck
from .resnet_attn import ResNet_Self_Attn
from ..kWTA.resnet import SparseResNet18, SparseResNet34, SparseResNet50, SparseResNet101
import torch.utils.model_zoo as model_zoo

__all__ = ["create_resnet"]

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def load_pretrain_model(model, model_name):
    model_dict = model.state_dict()
    # print(model_dict)
    pretrained_dict = model_zoo.load_url(model_urls[model_name])
    keys = []
    for k, v in pretrained_dict.items():
        keys.append(k)
    i = 0
    for k, v in model_dict.items():
        # print(k)
        if v.size() == pretrained_dict[keys[i]].size():
            model_dict[k] = pretrained_dict[keys[i]]
            # print(model_dict[k])
            i = i + 1
    # print(i)
    model.load_state_dict(model_dict)
    return model


def create_resnet(depth, num_classes, att_type=None, pretrained=True):
    r"""Create a resnet model for ImageNet"""
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
        elif att_type == "SelfAttn":
            model = ResNet_Self_Attn(depth, num_classes)
            pretrained = False
        elif att_type == "kWTA":
            model = SparseResNet18(num_classes, sparsities=[0.2, 0.2, 0.2, 0.2], sparse_func='vol')
        else:
            model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

        if pretrained == True:
            model = load_pretrain_model(model, 'resnet18')
            print('load pretrain model done!~')

    elif depth == 34:
        if att_type == "BAM":
            model = BAMResNet(BasicBlock, [3, 4, 6, 3], num_classes)
        elif att_type == "CBAM":
            model = CBAMResNet(BasicBlock, [3, 4, 6, 3], num_classes)
        elif att_type == "SelfAttn":
            model = ResNet_Self_Attn(depth, num_classes)
            pretrained = False
        elif att_type == "kWTA":
            model = SparseResNet34(num_classes, sparsities=[0.2, 0.2, 0.2, 0.2], sparse_func='vol')
        else:
            model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes)

        if pretrained == True:
            model = load_pretrain_model(model, 'resnet34')
            print('load pretrain model done!~')

    elif depth == 50:
        if att_type == "BAM":
            model = BAMResNet(Bottleneck, [3, 4, 6, 3], num_classes)
        elif att_type == "CBAM":
            model = CBAMResNet(Bottleneck, [3, 4, 6, 3], num_classes)
        elif att_type == "SelfAttn":
            model = ResNet_Self_Attn(depth, num_classes)
            pretrained = False
        elif att_type == "kWTA":
            model = SparseResNet50(num_classes, sparsities=[0.2, 0.2, 0.2, 0.2], sparse_func='vol')
        else:
            model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes)

        if pretrained == True:
            model = load_pretrain_model(model, 'resnet50')
            print('load pretrain model done!~')

    elif depth == 101:
        if att_type == "BAM":
            model = BAMResNet(Bottleneck, [3, 4, 23, 3], num_classes)
        elif att_type == "CBAM":
            model = CBAMResNet(Bottleneck, [3, 4, 23, 3], num_classes)
        elif att_type == "SelfAttn":
            model = ResNet_Self_Attn(depth, num_classes)
            pretrained = False
        elif att_type == "kWTA":
            model = SparseResNet101(num_classes, sparsities=[0.2, 0.2, 0.2, 0.2], sparse_func='vol')
        else:
            model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes)

        if pretrained == True:
            model = load_pretrain_model(model, 'resnet101')
            print('load pretrain model done!~')
    return model
