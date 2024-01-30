import albumentations as A
from albumentations.pytorch import ToTensorV2
import datetime

def get_date_and_time():
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    return formatted_datetime

config = dict(
    save_dir = f'./experiments/{get_date_and_time()}',

    train_dataset = {
        'name': 'segmentation_dataset',
        'kwargs': {
            'data_dir': './data',
            'transf':                    
                A.Compose([
                    A.Resize(256, 256),
                    A.Normalize(),
                    ToTensorV2(),
                ]),
        },
        'batch_size': 4,
        'workers': 4
    }, 

    model = {
        'name': 'unet', 
        'kwargs': {
            'encoder_name': 'resnet18',
            'classes': 1
        }
    },

    loss = {
        'name': 'diceloss',
        'kwargs': {
            'mode': 'binary',
            'from_logits': True,
        }
    },

    lr=5e-4,
    n_epochs=100,
)