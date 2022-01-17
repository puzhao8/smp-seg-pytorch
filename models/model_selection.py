
from logging import error
import smp
from models.unet import UNet
from models.siam_unet import SiamUnet_conc, SiamUnet_diff
from models.unet_cdc import UNet as cdc_unet
from models.siam_unet_resnet import SiamResUnet

from models.unet_distill import UNet as distill_unet
from models.segformer import segformer_models

def get_model(cfg):

    ########################### COMPUTE INPUT & OUTPUT CHANNELS ############################
    print("Satellites: ", cfg.DATA.SATELLITES)
    print("NUM_CLASSES:", cfg.MODEL.NUM_CLASSES)

    # cfg.MODEL.NUM_CLASSES = cfg.MODEL.cfg.MODEL.NUM_CLASSES

    INPUT_CHANNELS_DICT = {}
    INPUT_CHANNELS_LIST = []
    for sat in cfg.DATA.SATELLITES:
        INPUT_CHANNELS_DICT[sat] = len(list(cfg.DATA.INPUT_BANDS[sat]))
        if cfg.DATA.STACKING: INPUT_CHANNELS_DICT[sat] = len(cfg.DATA.PREPOST) * INPUT_CHANNELS_DICT[sat]
        INPUT_CHANNELS_LIST.append(INPUT_CHANNELS_DICT[sat])
    
    ########################### MODEL SELECTION ############################
    if cfg.MODEL.ARCH == "UNet":
        INPUT_CHANNELS = sum(INPUT_CHANNELS_LIST)
        model = UNet(INPUT_CHANNELS, 
                        num_classes=cfg.MODEL.NUM_CLASSES, 
                        topo=cfg.MODEL.TOPO,
                        use_deconv=cfg.MODEL.USE_DECONV) #'FC-EF'

    if cfg.MODEL.ARCH == 'distill_unet':
        INPUT_CHANNELS = INPUT_CHANNELS_LIST[0] # defined by the first sensor
        model = UNet(INPUT_CHANNELS, 
                        num_classes=cfg.MODEL.NUM_CLASSES, 
                        topo=cfg.MODEL.TOPO, 
                    ) #'FC-Siam-diff'

    if cfg.MODEL.ARCH in segformer_models.keys():
        INPUT_CHANNELS = sum(INPUT_CHANNELS_LIST)
        model = segformer_models[cfg.MODEL.ARCH](in_channels=INPUT_CHANNELS, num_classes=cfg.MODEL.NUM_CLASSES)
    
    if "SiamUnet" in cfg.MODEL.ARCH:
        if len(INPUT_CHANNELS_LIST) == 1: # single sensor
            INPUT_CHANNELS = INPUT_CHANNELS_LIST[0]
        elif len(INPUT_CHANNELS_LIST) == 2: # two sensors
            if INPUT_CHANNELS_LIST[0] == INPUT_CHANNELS_LIST[1]: 
                INPUT_CHANNELS = INPUT_CHANNELS_LIST[0]
            else:
                INPUT_CHANNELS = INPUT_CHANNELS_LIST
                print("INPUT_CHANNELS is a list, pleae fix it!")

        if cfg.MODEL.ARCH == "SiamUnet_conc":
            model = SiamUnet_conc(INPUT_CHANNELS, cfg.MODEL.NUM_CLASSES, \
                                topo=cfg.MODEL.TOPO, 
                                share_encoder=cfg.MODEL.SHARE_ENCODER,
                                use_deconv=cfg.MODEL.USE_DECONV
                            ) #'FC-Siam-conc'

        if cfg.MODEL.ARCH == "SiamUnet_diff":
            model = SiamUnet_diff(INPUT_CHANNELS, cfg.MODEL.NUM_CLASSES, 
                                topo=cfg.MODEL.TOPO, 
                                share_encoder=cfg.MODEL.SHARE_ENCODER,
                                use_deconv=cfg.MODEL.USE_DECONV
                            ) #'FC-Siam-diff'

    
    if cfg.MODEL.ARCH == "SiamUnet_minDiff":
        model = SiamUnet_diff(INPUT_CHANNELS, cfg.MODEL.NUM_CLASSES, topo=cfg.MODEL.TOPO) #'FC-Siam-diff'

    if cfg.MODEL.ARCH == "cdc_unet":
        model = cdc_unet(INPUT_CHANNELS, cfg.MODEL.NUM_CLASSES) #'FC-EF'

    ########################### Residual UNet ############################
    if cfg.MODEL.ARCH == f'UNet_{cfg.MODEL.ENCODER}':
        INPUT_CHANNELS = sum(INPUT_CHANNELS_LIST)
        print(f"===> Network Architecture: {cfg.MODEL.ARCH}")
        # create segmentation model with pretrained encoder

        model = smp.Unet(
            encoder_name = cfg.MODEL.ENCODER, 
            encoder_weights = cfg.MODEL.ENCODER_WEIGHTS, 
            encoder_depth=cfg.MODEL.ENCODER_DEPTH,
            decoder_channels=cfg.MODEL.TOPO[::-1],
            # decoder_attention_type="scse",
            in_channels = INPUT_CHANNELS,
            classes = cfg.MODEL.NUM_CLASSES, 
            activation = None,
        )

    # DeepLabV3+
    if cfg.MODEL.ARCH == 'DeepLabV3+':
        print(f"===> Network Architecture: {cfg.MODEL.ARCH}")
        # create segmentation model with pretrained encoder
        model = smp.DeepLabV3Plus(
            encoder_name = cfg.MODEL.ENCODER, 
            encoder_weights = cfg.MODEL.ENCODER_WEIGHTS, 
            classes = cfg.MODEL.NUM_CLASSES, 
            activation = cfg.MODEL.ACTIVATION,
            in_channels = INPUT_CHANNELS
        )

    
    if cfg.MODEL.ARCH == 'SiamResUnet':
        print(f"===> Network Architecture: {cfg.MODEL.ARCH}")
        # create segmentation model with pretrained encoder

        input_channels = []
        for sat in cfg.DATA.SATELLITES:
            if cfg.DATA.STACKING: tmp = len(cfg.DATA.PREPOST) * INPUT_CHANNELS_DICT[sat]
            else: tmp = INPUT_CHANNELS_DICT[sat]
            input_channels.append(tmp)

        from models.siam_unet_resnet import SiamResUnet
        model = SiamResUnet(
            encoder_name = cfg.MODEL.ENCODER, 
            encoder_weights = cfg.MODEL.ENCODER_WEIGHTS, 
            in_channels = input_channels,
            classes = cfg.MODEL.NUM_CLASSES, 
            activation = cfg.MODEL.ACTIVATION,
        )

    # print("==================================")
    # print(model)
    # print("==================================")
    print("INPUT_CHANNELS: ", INPUT_CHANNELS)
    print()

    return model





if __name__ == "__main__":
    from easydict import EasyDict as edict
    cfg = edict({
        "data": {
            "stacking": True,
            "CLASSES": ['burned'],
            'prepost': ['pre', 'post'],
            "satellites": ['S2'],
            "INPUT_BANDS": {
                'S1': ['ND','VH','VV'],
                'S2': ['B4', 'B8', 'B12'],
            }
        },
        "model": {
            'ARCH': 'SiamUnet_conc',
            'TOPO': [16,32,64,128],
            'SHARE_ENCODER': True
        }
        })

    get_model(cfg)