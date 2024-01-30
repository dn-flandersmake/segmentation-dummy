from torch import nn
from segmentation_models_pytorch.losses import DiceLoss



def get_loss(name, loss_opts):
    if name == 'bce':
        return nn.BCEWithLogitsLoss(**loss_opts)
    elif name == 'crossentropy':
        return nn.CrossEntropyLoss(**loss_opts)
    elif name == "diceloss":
        return DiceLoss(**loss_opts)
    else:
        raise RuntimeError(f'Loss {name} is not available!')