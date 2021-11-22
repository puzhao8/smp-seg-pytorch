from typing import Optional, Union, List
from smp.unet.decoder import UnetDecoder
from smp.encoders.__init__ import get_encoder
# # from ..base import SegmentationModel
from smp.base.heads import SegmentationHead, ClassificationHead
from easydict import EasyDict as edict


import torch
from smp.base import initialization as init


class SegmentationModel(torch.nn.Module):

    def initialize(self):
        init.initialize_decoder(self.decoder1)
        init.initialize_head(self.segmentation_head1)

        init.initialize_decoder(self.decoder2)
        init.initialize_head(self.segmentation_head2)

        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        x1, x2 = x
        features1 = self.encoder1(x1)
        decoder1 = self.decoder1(*features1)
        decoder1 = self.segmentation_head1(decoder1)

        features2 = self.encoder2(x2)
        decoder2 = self.decoder2(*features2)
        decoder2 = self.segmentation_head2(decoder2)

        # decoder_output = torch.cat([decoder1, decoder2], dim=1)
        masks = self.segmentation_head(decoder1)

        # if self.classification_head is not None:
        #     labels = self.classification_head(features[-1])
        #     return masks, labels

        return masks, [decoder1, decoder2]

    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        with torch.no_grad():
            x, _ = self.forward(x)

        return x



class FuseUnet(SegmentationModel):
    """Unet_ is a fully convolution neural network for image semantic segmentation. Consist of *encoder* 
    and *decoder* parts connected with *skip connections*. Encoder extract features of different spatial 
    resolution (skip connections) which are used by decoder to define accurate segmentation mask. Use *concatenation*
    for fusing decoder blocks with skip connections.

    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features 
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and 
            other pretrained weights (see table with available weights for each encoder_name)
        decoder_channels: List of integers which specify **in_channels** parameter for convolutions used in decoder.
            Length of the list should be the same as **encoder_depth**
        decoder_use_batchnorm: If **True**, BatchNorm2d layer between Conv2D and Activation layers
            is used. If **"inplace"** InplaceABN will be used, allows to decrease memory consumption.
            Available options are **True, False, "inplace"**
        decoder_attention_type: Attention module used in decoder of the model. Available options are **None** and **scse**.
            SCSE paper - https://arxiv.org/abs/1808.08127
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.
            Default is **None**
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build 
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax" (could be **None** to return logits)

    Returns:
        ``torch.nn.Module``: Unet

    .. _Unet:
        https://arxiv.org/abs/1505.04597

    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: List[int] = (3, 3), # *(S1, S2) by puzhao
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()

        def get_UNet(input_channels):
            encoder = get_encoder(
                encoder_name,
                in_channels=input_channels,
                depth=encoder_depth,
                weights=encoder_weights,
            )

            decoder = UnetDecoder(
                encoder_channels=encoder.out_channels,
                decoder_channels=decoder_channels,
                n_blocks=encoder_depth,
                use_batchnorm=decoder_use_batchnorm,
                center=True if encoder_name.startswith("vgg") else False,
                attention_type=decoder_attention_type,
            )

            segmentation_head = SegmentationHead(
                in_channels=decoder_channels[-1], 
                out_channels=classes,
                activation=activation,
                kernel_size=3,
            )
            return encoder, decoder, segmentation_head
        
        self.encoder1, self.decoder1, self.segmentation_head1= get_UNet(in_channels[0])
        self.encoder2, self.decoder2, self.segmentation_head2= get_UNet(in_channels[1])
        
        self.segmentation_head = SegmentationHead(
                # in_channels=classes*len(in_channels), # puzhao
                # out_channels=classes,
                in_channels=len(in_channels), # puzhao
                out_channels=classes,
                activation=activation,
                kernel_size=3,
            )


        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()
