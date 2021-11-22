
import smp


def init_model(cfg):

    INPUT_CHANNELS_DICT = {}
    for sat in cfg.data.satellites:
        INPUT_CHANNELS_DICT[sat] = len(list(cfg.data.INPUT_BANDS[sat]))

    # single sensor
    if cfg.data.stacking and (1==len(cfg.data.satellites)): 
        INPUT_CHANNELS = len(cfg.data.prepost) * INPUT_CHANNELS_DICT[cfg.data.satellites[0]]
    
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
        return model

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
        return model

    
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
        return model

        