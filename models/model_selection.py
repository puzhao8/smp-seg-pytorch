
import smp
from models.unet import UNet
from models.siam_unet import SiamUnet_conc, SiamUnet_diff
from models.unet_cdc import UNet as cdc_unet
from models.siam_unet_resnet import SiamResUnet

def get_model(cfg):

    ########################### COMPUTE INPUT & OUTPUT CHANNELS ############################
    INPUT_CHANNELS_DICT = {}
    for sat in cfg.data.satellites:
        INPUT_CHANNELS_DICT[sat] = len(list(cfg.data.INPUT_BANDS[sat]))

    # single sensor
    if cfg.data.stacking:
        INPUT_CHANNELS = len(cfg.data.prepost) * INPUT_CHANNELS_DICT[cfg.data.satellites[0]]
    else:
        INPUT_CHANNELS = INPUT_CHANNELS_DICT[cfg.data.satellites[0]]
    
    print("INPUT_CHANNELS: ", INPUT_CHANNELS)
    OUT_CHANNELS = len(cfg.data.CLASSES)
    
    ########################### MODEL SELECTION ############################
    if cfg.model.ARCH == "UNet":
        model = UNet(INPUT_CHANNELS, OUT_CHANNELS) #'FC-EF'
    
    if cfg.model.ARCH == "SiamUnet_conc":
        model = SiamUnet_conc(INPUT_CHANNELS, OUT_CHANNELS, topo=cfg.model.TOPO) #'FC-Siam-conc'

    if cfg.model.ARCH == "SiamUnet_diff":
        model = SiamUnet_diff(INPUT_CHANNELS, OUT_CHANNELS, topo=cfg.model.TOPO) #'FC-Siam-diff'

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
    return model