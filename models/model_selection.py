
from logging import error
import smp
from models.unet import UNet
from models.siam_unet import SiamUnet_conc, SiamUnet_diff
from models.unet_cdc import UNet as cdc_unet
from models.siam_unet_resnet import SiamResUnet

from models.unet_distill import UNet as distill_unet

def get_model(cfg):

    ########################### COMPUTE INPUT & OUTPUT CHANNELS ############################
    print("satellites: ", cfg.data.satellites)
    print("CLASSES:", cfg.data.CLASSES)

    OUT_CHANNELS = len(cfg.data.CLASSES)

    INPUT_CHANNELS_DICT = {}
    INPUT_CHANNELS_LIST = []
    for sat in cfg.data.satellites:
        INPUT_CHANNELS_DICT[sat] = len(list(cfg.data.INPUT_BANDS[sat]))
        if cfg.data.stacking: INPUT_CHANNELS_DICT[sat] = len(cfg.data.prepost) * INPUT_CHANNELS_DICT[sat]
        INPUT_CHANNELS_LIST.append(INPUT_CHANNELS_DICT[sat])
    
    ########################### MODEL SELECTION ############################
    if cfg.model.ARCH == "UNet":
        INPUT_CHANNELS = sum(INPUT_CHANNELS_LIST)
        model = UNet(INPUT_CHANNELS, OUT_CHANNELS, topo=cfg.model.TOPO) #'FC-EF'

    if cfg.model.ARCH == 'distill_unet':
        INPUT_CHANNELS = INPUT_CHANNELS_LIST[0] # defined by the first sensor
        model = distill_unet(INPUT_CHANNELS, OUT_CHANNELS, 
                            topo=cfg.model.TOPO, 
                        ) #'FC-Siam-diff'
    
    if "SiamUnet" in cfg.model.ARCH:
        if len(INPUT_CHANNELS_LIST) == 1: # single sensor
            INPUT_CHANNELS = INPUT_CHANNELS_LIST[0]
        elif len(INPUT_CHANNELS_LIST) == 2: # two sensors
            if INPUT_CHANNELS_LIST[0] == INPUT_CHANNELS_LIST[1]: 
                INPUT_CHANNELS = INPUT_CHANNELS_LIST[0]
            else:
                INPUT_CHANNELS = INPUT_CHANNELS_LIST
                print("INPUT_CHANNELS is a list, pleae fix it!")

        if cfg.model.ARCH == "SiamUnet_conc":
            model = SiamUnet_conc(INPUT_CHANNELS, OUT_CHANNELS, \
                                topo=cfg.model.TOPO, 
                                share_encoder=cfg.model.SHARE_ENCODER
                            ) #'FC-Siam-conc'

        if cfg.model.ARCH == "SiamUnet_diff":
            model = SiamUnet_diff(INPUT_CHANNELS, OUT_CHANNELS, 
                                topo=cfg.model.TOPO, 
                                share_encoder=cfg.model.SHARE_ENCODER
                            ) #'FC-Siam-diff'

    
    if cfg.model.ARCH == "SiamUnet_minDiff":
        model = SiamUnet_diff(INPUT_CHANNELS, OUT_CHANNELS, topo=cfg.model.TOPO) #'FC-Siam-diff'

    if cfg.model.ARCH == "cdc_unet":
        model = cdc_unet(INPUT_CHANNELS, OUT_CHANNELS) #'FC-EF'

    ########################### Residual UNet ############################
    if cfg.model.ARCH == 'ResUNet':
        print(f"===> Network Architecture: {cfg.model.ARCH}")
        # create segmentation model with pretrained encoder

        model = smp.Unet(
            encoder_name = cfg.model.ENCODER, 
            encoder_weights = cfg.model.ENCODER_WEIGHTS, 
            in_channels = INPUT_CHANNELS,
            classes = OUT_CHANNELS, 
            activation = cfg.model.ACTIVATION,
        )

    # DeepLabV3+
    if cfg.model.ARCH == 'DeepLabV3+':
        print(f"===> Network Architecture: {cfg.model.ARCH}")
        # create segmentation model with pretrained encoder
        model = smp.DeepLabV3Plus(
            encoder_name = cfg.model.ENCODER, 
            encoder_weights = cfg.model.ENCODER_WEIGHTS, 
            classes = OUT_CHANNELS, 
            activation = cfg.model.ACTIVATION,
            in_channels = INPUT_CHANNELS
        )

    
    if cfg.model.ARCH == 'SiamResUnet':
        print(f"===> Network Architecture: {cfg.model.ARCH}")
        # create segmentation model with pretrained encoder

        input_channels = []
        for sat in cfg.data.satellites:
            if cfg.data.stacking: tmp = len(cfg.data.prepost) * INPUT_CHANNELS_DICT[sat]
            else: tmp = INPUT_CHANNELS_DICT[sat]
            input_channels.append(tmp)

        from models.siam_unet_resnet import SiamResUnet
        model = SiamResUnet(
            encoder_name = cfg.model.ENCODER, 
            encoder_weights = cfg.model.ENCODER_WEIGHTS, 
            in_channels = input_channels,
            classes = OUT_CHANNELS, 
            activation = cfg.model.ACTIVATION,
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