#!/usr/bin/env python3

from detectron2.layers import CNNBlockBase
from detectron2.modeling import (
    BACKBONE_REGISTRY,
    ResNet,
)
from detectron2.modeling.backbone.resnet import (
    BasicBlock,
    BasicStem,
    BottleneckBlock,
)

from ..modules.bam import BAM


class BAMBlock(CNNBlockBase):
    def __init__(self, in_channels, out_channels, *, stride=1, **kwargs):
        super().__init__(in_channels, out_channels, stride)
        self.bam = BAM(out_channels)

    def forward(self, x):
        return self.bam(x)


@BACKBONE_REGISTRY.register()
def build_bam_resnet_backbone(cfg, input_shape):
    r"""Create a ResNet with BAM instance from config

    Returns:
        ResNet+CBAM: a :class:`ResNet` instance
    """
    # need registeration of new blocks/stems?
    # need registration of new blocks/stems?
    norm = cfg.MODEL.RESNETS.NORM
    stem = BasicStem(
        in_channels=input_shape.channels,
        out_channels=cfg.MODEL.RESNETS.STEM_OUT_CHANNELS,
        norm=norm,
    )

    # fmt: off
    freeze_at           = cfg.MODEL.BACKBONE.FREEZE_AT
    out_features        = cfg.MODEL.RESNETS.OUT_FEATURES
    depth               = cfg.MODEL.RESNETS.DEPTH
    num_groups          = cfg.MODEL.RESNETS.NUM_GROUPS
    width_per_group     = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
    bottleneck_channels = num_groups * width_per_group
    in_channels         = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
    out_channels        = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    stride_in_1x1       = cfg.MODEL.RESNETS.STRIDE_IN_1X1
    res5_dilation       = cfg.MODEL.RESNETS.RES5_DILATION
    # fmt: on
    assert res5_dilation in {1, 2}, "res5_dilation cannot be {}.".format(
        res5_dilation
    )

    num_blocks_per_stage = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
    }[depth]

    if depth in [18, 34]:
        assert (
            out_channels == 64
        ), "Must set MODEL.RESNETS.RES2_OUT_CHANNELS = 64 for R18/R34"
        assert (
            res5_dilation == 1
        ), "Must set MODEL.RESNETS.RES5_DILATION = 1 for R18/R34"
        assert (
            num_groups == 1
        ), "Must set MODEL.RESNETS.NUM_GROUPS = 1 for R18/R34"

    stages = []

    # Avoid creating variables without gradients
    # It consumes extra memory and may cause allreduce to fail
    out_stage_idx = [
        {"res2": 2, "res3": 3, "res4": 4, "res5": 5}[f] for f in out_features
    ]
    max_stage_idx = max(out_stage_idx)
    for idx, stage_idx in enumerate(range(2, max_stage_idx + 1)):
        dilation = res5_dilation if stage_idx == 5 else 1
        first_stride = (
            1 if idx == 0 or (stage_idx == 5 and dilation == 2) else 2
        )
        stage_kargs = {
            "num_blocks": num_blocks_per_stage[idx],
            "stride_per_block": [first_stride]
            + [1] * (num_blocks_per_stage[idx] - 1),
            "in_channels": in_channels,
            "out_channels": out_channels,
            "norm": norm,
        }
        # Use BasicBlock for R18 and R34.
        if depth in [18, 34]:
            stage_kargs["block_class"] = BasicBlock
        else:
            stage_kargs["bottleneck_channels"] = bottleneck_channels
            stage_kargs["stride_in_1x1"] = stride_in_1x1
            stage_kargs["dilation"] = dilation
            stage_kargs["num_groups"] = num_groups
            stage_kargs["block_class"] = BottleneckBlock
        blocks = ResNet.make_stage(**stage_kargs)
        if stage_idx in [2, 3, 4]:
            bam_kargs = {
                "block_class": BAMBlock,
                "stride_per_block": [1],
                "in_channels": in_channels,
                "out_channels": out_channels,
                "num_blocks": 1,
            }
            bam_block = ResNet.make_stage(**bam_kargs)
        in_channels = out_channels
        out_channels *= 2
        bottleneck_channels *= 2
        stages.append(blocks)
        if stage_idx in [2, 3, 4]:
            stages.append(bam_block)

    return ResNet(stem, stages, out_features=out_features).freeze(freeze_at)
