
import smp
from fcnn4cd.unet import Unet as Vanilla_unet
from fcnn4cd.paddle_unet import UNet as Paddle_unet
from fcnn4cd.paddle_unet_cdc import UNet as cdc_unet
from fcnn4cd.siamunet_conc import SiamUnet_conc
from fcnn4cd.siamunet_diff import SiamUnet_diff
from fcnn4cd.siamunet_min_diff import SiamUnet_minDiff

def init_model(cfg):

    INPUT_CHANNELS_DICT = {}
    for sat in cfg.data.satellites:
        INPUT_CHANNELS_DICT[sat] = len(list(cfg.data.INPUT_BANDS[sat]))

    # single sensor
    if cfg.data.stacking and (1==len(cfg.data.satellites)): 
        INPUT_CHANNELS = len(cfg.data.prepost) * INPUT_CHANNELS_DICT[cfg.data.satellites[0]]
    else:
        INPUT_CHANNELS = INPUT_CHANNELS_DICT[cfg.data.satellites[0]]
    
    print("INPUT_CHANNELS: ", INPUT_CHANNELS)
    
    # UNet
    if cfg.model.ARCH == 'UNet':
        print(f"===> Network Architecture: {cfg.model.ARCH}")
        # create segmentation model with pretrained encoder

        model = smp.Unet(
            encoder_name = cfg.model.ENCODER, 
            encoder_weights = cfg.model.ENCODER_WEIGHTS, 
            in_channels = INPUT_CHANNELS,
            classes = len(cfg.data.CLASSES), 
            activation = cfg.model.ACTIVATION,
        )

    # DeepLabV3+
    if cfg.model.ARCH == 'DeepLabV3+':
        print(f"===> Network Architecture: {cfg.model.ARCH}")
        # create segmentation model with pretrained encoder
        model = smp.DeepLabV3Plus(
            encoder_name = cfg.model.ENCODER, 
            encoder_weights = cfg.model.ENCODER_WEIGHTS, 
            classes = len(cfg.data.CLASSES), 
            activation = cfg.model.ACTIVATION,
            in_channels = INPUT_CHANNELS
        )

    
    if cfg.model.ARCH == 'FuseUNet':
        print(f"===> Network Architecture: {cfg.model.ARCH}")
        # create segmentation model with pretrained encoder

        input_channels = []
        for sat in cfg.data.satellites:
            if cfg.data.stacking: tmp = len(cfg.data.prepost) * INPUT_CHANNELS_DICT[sat]
            else: tmp = INPUT_CHANNELS_DICT[sat]
            input_channels.append(tmp)

        from models.FuseUNet import FuseUnet
        model = FuseUnet(
            encoder_name = cfg.model.ENCODER, 
            encoder_weights = cfg.model.ENCODER_WEIGHTS, 
            in_channels = input_channels,
            classes = len(cfg.data.CLASSES), 
            activation = cfg.model.ACTIVATION,
        )
    
    if cfg.model.ARCH == "Vanilla_unet":
        model = Vanilla_unet(2*INPUT_CHANNELS, len(cfg.data.CLASSES)) #'FC-EF'
    
    if cfg.model.ARCH == "SiamUnet_conc":
        model = SiamUnet_conc(INPUT_CHANNELS, len(cfg.data.CLASSES)) #'FC-Siam-conc'

    if cfg.model.ARCH == "SiamUnet_diff":
        model = SiamUnet_diff(INPUT_CHANNELS, len(cfg.data.CLASSES)) #'FC-Siam-diff'

    if cfg.model.ARCH == "SiamUnet_minDiff":
        model = SiamUnet_diff(INPUT_CHANNELS, len(cfg.data.CLASSES)) #'FC-Siam-diff'

    if cfg.model.ARCH == "Paddle_unet":
        model = Paddle_unet(2*INPUT_CHANNELS, len(cfg.data.CLASSES)) #'FC-EF'

    if cfg.model.ARCH == "cdc_unet":
        model = cdc_unet(2*INPUT_CHANNELS, len(cfg.data.CLASSES)) #'FC-EF'

    return model