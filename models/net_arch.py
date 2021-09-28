
import smp
def init_model(cfg):
    
    # UNet
    if cfg.model.ARCH == 'UNet':
        print(f"===> Network Architecture: {cfg.model.ARCH}")
        # create segmentation model with pretrained encoder

        if 'S1' == list(cfg.data.satellites)[0]:
            INPUT_CHANNELS = 2* len(list(cfg.data.S1_INPUT_BANDS))

        if 'ALOS' == list(cfg.data.satellites)[0]:
            INPUT_CHANNELS = 2* len(list(cfg.data.S1_INPUT_BANDS))

        if 'S2' == list(cfg.data.satellites)[0]:
            INPUT_CHANNELS = 2* len(list(cfg.data.S2_INPUT_BANDS))

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

        S1_INPUT_CHANNELS = len(list(cfg.data.S1_INPUT_BANDS))
        S2_INPUT_CHANNELS = len(list(cfg.data.S2_INPUT_BANDS))

        from models.FuseUNet import FuseUnet
        model = FuseUnet(
            encoder_name = cfg.model.ENCODER, 
            encoder_weights = cfg.model.ENCODER_WEIGHTS, 
            in_channels = (S1_INPUT_CHANNELS, S2_INPUT_CHANNELS),
            classes = len(cfg.data.CLASSES), 
            activation = cfg.model.ACTIVATION,
        )
        return model