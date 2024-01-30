from torch.utils.data import Dataset
import glob
import os
from PIL import Image
import numpy as np

class SegmentationDataset(Dataset):

    def __init__(self, data_dir='./data', transf=None):

        # read images
        images = glob.glob(os.path.join(data_dir, 'images', '*.jpg'))
        images.sort()

        # read labels
        labels = glob.glob(os.path.join(data_dir, 'labels', '*.png'))
        labels.sort()

        self.images = images
        self.labels = labels
        self.transf = transf
        
        print(f'Segmentation dataset created, found {len(self)} images.')

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):

        image_p = self.images[index]
        image = Image.open(image_p)

        label_p = self.labels[index]
        label = Image.open(label_p)

        if self.transf:
            out = self.transf(image=np.array(image), mask=np.array(label))
            image, label = out['image'], out['mask']

        return {
            'image': image,
            'label': label
        }