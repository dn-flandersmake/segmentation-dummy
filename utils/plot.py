import torch 
import torchvision
import os

def denorm(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mean = torch.as_tensor(mean, dtype=img.dtype, device=img.device)
    std = torch.as_tensor(std, dtype=img.dtype, device=img.device)
    return img.mul(std[:,None,None]).add(mean[:,None,None])

def plot(im, pred, gt=None, path='./output.jpg'):
    im = denorm(im).cpu()
    pred = pred.expand_as(im).cpu()
    gt = gt.expand_as(im).cpu()
    image_grid = torchvision.utils.make_grid(
        [im, pred, gt], nrow=3, padding=5, pad_value=1
    )

    torchvision.utils.save_image(image_grid, path)