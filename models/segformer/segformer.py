# The SegFormer code was heavily based on https://github.com/NVlabs/SegFormer
# Users should be careful about adopting these functions in any commercial matters.
# https://github.com/NVlabs/SegFormer#license

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mix_transformer import BACKBONES

class MLP(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).permute([0, 2, 1])
        x = self.proj(x)
        return x


class SegFormer(nn.Module):
    """
    The SegFormer implementation based on PaddlePaddle.

    The original article refers to
    Xie, Enze, et al. "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers." arXiv preprint arXiv:2105.15203 (2021).

    Args:
        num_classes (int): The unique number of target classes.
        backbone (Paddle.nn.Module): A backbone network.
        embedding_dim (int): The MLP decoder channel dimension.
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature.
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769. Default: False.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(self,
                 num_classes,
                 backbone,
                 embedding_dim,
                 align_corners=False,
                 pretrained=None):
        super(SegFormer, self).__init__()

        self.pretrained = pretrained
        self.align_corners = align_corners
        self.backbone = backbone
        self.num_classes = num_classes
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.backbone.feat_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.dropout = nn.Dropout2d(0.1)
        self.linear_fuse = nn.intrinsic.ConvBnReLU2d(
            nn.Conv2d(in_channels=embedding_dim * 4,
            out_channels=embedding_dim,
            kernel_size=1,
            bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU()
            )

        self.linear_pred = nn.Conv2d(
            embedding_dim, self.num_classes, kernel_size=1)

        # self.init_weight()

    # def init_weight(self):
    #     if self.pretrained is not None:
    #         utils.load_entire_model(self, self.pretrained)

    def forward(self, x):
        feats = self.backbone(x)
        c1, c2, c3, c4 = feats

        ############## MLP decoder on C1-C4 ###########
        c1_shape = c1.shape
        c2_shape = c2.shape
        c3_shape = c3.shape
        c4_shape = c4.shape

        c4_ = self.linear_c4(c4).permute([0, 2, 1])
        _c4 = c4_.reshape(*c4_.shape[:2], c4_shape[2], c4_shape[3])
        _c4 = F.interpolate(
            _c4,
            size=c1_shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        c3_ = self.linear_c3(c3).permute([0, 2, 1])
        _c3 = c3_.reshape(*c3_.shape[:2], c3_shape[2], c3_shape[3])
        _c3 = F.interpolate(
            _c3,
            size=c1_shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        c2_ = self.linear_c2(c2).permute([0, 2, 1])
        _c2 = c2_.reshape(*c2_.shape[:2], c2_shape[2], c2_shape[3])
        _c2 = F.interpolate(
            _c2,
            size=c1_shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        c1_ = self.linear_c1(c1).permute([0, 2, 1])
        _c1 = c1_.reshape(*c2_.shape[:2], c1_shape[2], c1_shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], axis=1))

        logit = self.dropout(_c)
        logit = self.linear_pred(logit)
        return [
            F.interpolate(
                logit,
                size=x.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        ]


def SegFormer_B0(**kwargs):
    return SegFormer(
        backbone=BACKBONES['MixVisionTransformer_B0'](),
        embedding_dim=256,
        **kwargs)


def SegFormer_B1(**kwargs):
    return SegFormer(
        backbone=BACKBONES['MixVisionTransformer_B1'](),
        embedding_dim=256,
        **kwargs)


def SegFormer_B2(**kwargs):
    return SegFormer(
        backbone=BACKBONES['MixVisionTransformer_B2'](),
        embedding_dim=768,
        **kwargs)


def SegFormer_B3(**kwargs):
    return SegFormer(
        backbone=BACKBONES['MixVisionTransformer_B3'](),
        embedding_dim=768,
        **kwargs)


def SegFormer_B4(**kwargs):
    return SegFormer(
        backbone=BACKBONES['MixVisionTransformer_B4'](),
        embedding_dim=768,
        **kwargs)


def SegFormer_B5(**kwargs):
    return SegFormer(
        backbone=BACKBONES['MixVisionTransformer_B5'](),
        embedding_dim=768,
        **kwargs)
