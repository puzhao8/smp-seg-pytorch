# https://sourcegraph.com/github.com/PaddlePaddle/PaddleSeg/-/blob/paddleseg/models/unet.py?L155:16#tab=def

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

# import paddle
# import paddle.nn as nn
# import paddle.nn.functional as F

# from paddleseg import utils
# from paddleseg.cvlibs import manager
# from paddleseg.models import layers

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


# @manager.MODELS.add_component
class UNet(nn.Module):
    """
    The UNet implementation based on PaddlePaddle.

    The original article refers to
    Olaf Ronneberger, et, al. "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    (https://arxiv.org/abs/1505.04597).

    Args:
        num_classes (int): The unique number of target classes.
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.  Default: False.
        use_deconv (bool, optional): A bool value indicates whether using deconvolution in upsampling.
            If False, use resize_bilinear. Default: False.
        pretrained (str, optional): The path or url of pretrained model for fine tuning. Default: None.
    """

    def __init__(self,
            input_channels=6,
            num_classes=2,
            # topo=[64, 128, 256, 512],
            topo=[16, 32, 64, 128],
            # topo=[16, 32, 64],
            align_corners=False,
            use_deconv=False,
            pretrained=None):
        super().__init__()

        self.encode = Encoder(input_channels, topo=topo)
        decoder_topo = topo[::-1]
        self.decode = Decoder(align_corners, use_deconv=use_deconv, topo=decoder_topo)
        self.cls = self.conv = nn.Conv2d(
            in_channels=topo[0],
            out_channels=num_classes,
            kernel_size=3,
            stride=1,
            padding=1)

        # self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax(dim=1)

        # self.pretrained = pretrained
        # self.init_weight()

    def forward(self, x):
        ''' x should be a list or tuple '''
        x = torch.cat(x, dim=1) # concat all input tensors

        logit_list = []
        xc, short_cuts = self.encode(x)
        # logit_list.append(xc) # most center features

        # for shortcut in short_cuts:
        #     print(shortcut.shape)

        x = self.decode(xc, short_cuts)
        x = self.cls(x) # output
        logit_list.append(x)

        return logit_list



class Encoder(nn.Module):
    def __init__(self, input_channels=3, topo=[16, 32, 64, 128]):
        super().__init__()

        self.double_conv = nn.Sequential(
            ConvBNReLU(input_channels, topo[0], 3), 
            ConvBNReLU(topo[0], topo[0], 3)
        )

        down_channels = []
        for i in range(0, len(topo)):
            if i < len(topo) - 1: down_channels.append([topo[i], topo[i+1]])
            else: down_channels.append([topo[i], topo[i]])
        # print(down_channels)
        # down_channels = [[64, 128], [128, 256], [256, 512], [512, 512]]

        self.downLayerList = []
        for channel in down_channels:
            self.downLayerList.append(self.down_sampling(channel[0], channel[1]))
        self.down_stages = nn.Sequential(*self.downLayerList)
        

    def down_sampling(self, in_channels, out_channels):
        modules = []
        modules.append(nn.MaxPool2d(kernel_size=2, stride=2))
        modules.append(ConvBNReLU(in_channels, out_channels, 3))
        modules.append(ConvBNReLU(out_channels, out_channels, 3))
        return nn.Sequential(*modules)

    def forward(self, x):
        short_cuts = []

        # print("---- Encoder ----")
        x = self.double_conv(x)
        for down_sample in self.downLayerList:
            short_cuts.append(x)
            x = down_sample(x)
            # print(x.shape)
        return x, short_cuts


class Decoder(nn.Module):
    def __init__(self, align_corners, use_deconv=False, topo=[16, 32, 64, 128]):
        super().__init__()

        # up_channels = [[512, 256], [256, 128], [128, 64], [64, 64]]
        # [512, 256, 128, 64512]
        up_channels = []
        for i in range(0, len(topo)):
            if i < len(topo)-1: up_channels.append([topo[i], topo[i+1]])
            else: up_channels.append([topo[i], topo[i]])
        print(up_channels)

        self.upLayerList = [
            UpSampling(channel[0], channel[1], align_corners, use_deconv)
            for channel in up_channels
        ]

        self.up_stages = nn.Sequential(*self.upLayerList)

    def forward(self, x, short_cuts):
        # print("---- UNet Dncoder ----")
        for i in range(len(short_cuts)):
            # print(x.shape, short_cuts[-(i+1)].shape)
            x = self.up_stages[i](x, short_cuts[-(i + 1)])
            # print(x.shape)

        return x

class UpSampling(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 align_corners,
                 use_deconv=False):
        super().__init__()

        self.align_corners = align_corners

        self.use_deconv = use_deconv
        if self.use_deconv:
            self.deconv = nn.ConvTranspose2d(
                in_channels,
                out_channels // 2,
                kernel_size=2,
                stride=2,
                padding=0)
            in_channels = in_channels + out_channels // 2
        else:
            in_channels *= 2

        self.double_conv = nn.Sequential(
            ConvBNReLU(in_channels, out_channels, 3),
            ConvBNReLU(out_channels, out_channels, 3))

    def forward(self, x, short_cut):
        # print("before inter: ", x.shape, short_cut.shape)

        if self.use_deconv:
            x = self.deconv(x)
        else:
            x = F.interpolate(
                x,
                (short_cut.shape)[2:],
                mode='bilinear',
                align_corners=self.align_corners)

        # print(x.shape, short_cut.shape)
        x = torch.cat([x, short_cut], dim=1)
        x = self.double_conv(x)
        # print(x.shape)
        return x


# https://sourcegraph.com/github.com/PaddlePaddle/PaddleSeg/-/blob/paddleseg/models/layers/layer_libs.py?L98
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

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        x = self._relu(x)
        return x



if __name__ == "__main__":

    import numpy as np
    from torchsummary import summary

    x1 = np.random.rand(10,6,256,256)
    # x2 = np.random.rand(10,3,256,256)
    x1 = torch.from_numpy(x1).type(torch.FloatTensor)#.cuda().type(torch.cuda.FloatTensor)
    # x2 = torch.from_numpy(x2).type(torch.FloatTensor)#.cuda().type(torch.cuda.FloatTensor)

    myunet = UNet(input_channels=6)
    # myunet.cuda()

    # print(myunet)
    print(myunet.forward([x1])[-1].shape)
    # summary(myunet, (3,256,256))