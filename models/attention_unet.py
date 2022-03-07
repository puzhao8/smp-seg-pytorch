# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
# from paddleseg.cvlibs import manager
# from paddleseg.models import layers
# from paddleseg import utils
import numpy as np


# nn.Module (Paddle) -> nn.Module (pytorch)
# MaxPool2D -> MaxPool2d
# nn.Conv2d -> nn.Conv2d

# @manager.MODELS.add_component
class AttentionUNet(nn.Module):
    """
    The Attention-UNet implementation based on PaddlePaddle.
    As mentioned in the original paper, author proposes a novel attention gate (AG)
    that automatically learns to focus on target structures of varying shapes and sizes.
    Models trained with AGs implicitly learn to suppress irrelevant regions in an input image while
    highlighting salient features useful for a specific task.

    The original article refers to
    Oktay, O, et, al. "Attention u-net: Learning where to look for the pancreas."
    (https://arxiv.org/pdf/1804.03999.pdf).

    Args:
        num_classes (int): The unique number of target classes.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(self, input_channels=3, num_classes=2, topo=[64, 128, 256, 512], pretrained=None, use_deconv=False):
        super().__init__()
        n_channels = input_channels
        # topo = [64, 128, 256, 512]
        self.encoder = Encoder(n_channels, topo)
        # filters = np.array([64, 128, 256, 512, 1024])
        # print(topo)
        # print(topo + [2*topo[-1]])
        # print(topo.append(topo[-1]*2))

        filters = np.array(topo + [2*topo[-1]])
        self.up5 = UpConv(ch_in=filters[4], ch_out=filters[3])
        self.att5 = AttentionBlock(
            F_g=filters[3], F_l=filters[3], F_out=filters[2])
        self.up_conv5 = ConvBlock(ch_in=filters[4], ch_out=filters[3])

        self.up4 = UpConv(ch_in=filters[3], ch_out=filters[2])
        self.att4 = AttentionBlock(
            F_g=filters[2], F_l=filters[2], F_out=filters[1])
        self.up_conv4 = ConvBlock(ch_in=filters[3], ch_out=filters[2])

        self.up3 = UpConv(ch_in=filters[2], ch_out=filters[1])
        self.att3 = AttentionBlock(
            F_g=filters[1], F_l=filters[1], F_out=filters[0])
        self.up_conv3 = ConvBlock(ch_in=filters[2], ch_out=filters[1])

        self.up2 = UpConv(ch_in=filters[1], ch_out=filters[0])
        self.att2 = AttentionBlock(
            F_g=filters[0], F_l=filters[0], F_out=filters[0] // 2)
        self.up_conv2 = ConvBlock(ch_in=filters[1], ch_out=filters[0])

        self.conv_1x1 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, stride=1, padding=0)
        self.pretrained = pretrained
        # self.init_weight()

    def forward(self, x):
        x = torch.cat(x, dim=1) # added by puzhao

        x5, (x1, x2, x3, x4) = self.encoder(x)
        d5 = self.up5(x5)
        x4 = self.att5(g=d5, x=x4)
        d5 = torch.cat([x4, d5], axis=1)
        d5 = self.up_conv5(d5)

        d4 = self.up4(d5)
        x3 = self.att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), axis=1)
        d4 = self.up_conv4(d4)

        d3 = self.up3(d4)
        x2 = self.att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), axis=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        x1 = self.att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.up_conv2(d2)

        logit = self.conv_1x1(d2)
        logit_list = [logit]
        return logit_list

    # def init_weight(self):
    #     if self.pretrained is not None:
    #         utils.load_entire_model(self, self.pretrained)


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_out):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_out, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F_out))

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_out, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F_out))

        self.psi = nn.Sequential(
            nn.Conv2d(F_out, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1), nn.Sigmoid())

        self.relu = nn.ReLU()

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        res = x * psi
        return res


class UpConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_out), nn.ReLU())

    def forward(self, x):
        return self.up(x)


class Encoder(nn.Module):
    def __init__(self, input_channels, filters):
        super().__init__()
        self.double_conv = nn.Sequential(
            ConvBNReLU(input_channels, filters[0], 3),
            ConvBNReLU(filters[0], filters[0], 3))
        down_channels = filters
        self.down_sample_list = nn.ModuleList([
            self.down_sampling(channel, channel * 2)
            for channel in down_channels
        ])

    def down_sampling(self, in_channels, out_channels):
        modules = []
        modules.append(nn.MaxPool2d(kernel_size=2, stride=2))
        modules.append(ConvBNReLU(in_channels, out_channels, 3))
        modules.append(ConvBNReLU(out_channels, out_channels, 3))
        return nn.Sequential(*modules)

    def forward(self, x):
        short_cuts = []
        x = self.double_conv(x)
        for down_sample in self.down_sample_list:
            short_cuts.append(x)
            x = down_sample(x)
        return x, short_cuts


class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_out), nn.ReLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_out), nn.ReLU())

    def forward(self, x):
        return self.conv(x)


class ConvBNReLU(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding='same',
                 **kwargs):
        super().__init__()

        self._conv = nn.Conv2d(
            # in_channels, out_channels, kernel_size, **kwargs)
            in_channels, out_channels, kernel_size, padding=1, **kwargs)

        self._batch_norm = nn.BatchNorm2d(out_channels)
        self._relu = nn.ReLU()
        # self._dropout = nn.Dropout2d(p=0.1) # added by puzhao

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        x = self._relu(x)
        # x = self._dropout(x) # added by puzhao
        return x



    

if __name__ == "__main__":

    input_channels = 6
    x = torch.rand(10, input_channels, 256, 256)
    model = AttentionUNet(input_channels=input_channels, num_classes=2)

    print(model)
    print(model(x)[-1].shape)