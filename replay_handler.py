# imports and installs
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
from torch.optim import SGD, Adam, ASGD

import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import monai
from monai.transforms import (ScaleIntensityRange, Compose, AddChannel, RandSpatialCrop, ToTensor, 
                            RandAxisFlip, Activations, AsDiscrete, Resize, RandRotate, RandFlip, EnsureType,
                             KeepLargestConnectedComponent, CenterSpatialCrop)
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss, FocalLoss, GeneralizedDiceLoss, DiceCELoss, DiceFocalLoss
from monai.networks.nets import UNet, VNet, UNETR, SwinUNETR, AttentionUnet
from monai.data import decollate_batch, ImageDataset
from monai.utils import set_determinism
import os, json
import wandb
from time import time
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
from random import sample
from torchvision.transforms import ToPILImage

seed = 2000
torch.manual_seed(seed)
set_determinism(seed=seed)



# Transforms for images & labels
class CropRoI(monai.transforms.Transform):
    def __init__(self, roi_size: tuple | list|int, center: tuple | list | int = None, random_center: bool = False):
        self.roi_size = roi_size
        self.center = center
        self.random_center = random_center
        
        if isinstance(self.roi_size, int):
            self.roi_size = (self.roi_size, self.roi_size, -1)
        if isinstance(self.center, int):
            self.center = (self.center, self.center, -1)
        
            
    def __call__(self, data):
        if self.random_center:
            cropped = monai.transforms.RandSpatialCrop(roi_size= self.roi_size, random_center = True, random_size=False)(data)
            return cropped
        else:
            cropped = data[:, self.center[0] - self.roi_size[0]//2 : self.center[0] + self.roi_size[0]//2,
                           self.center[1] - self.roi_size[1]//2 : self.center[1] + self.roi_size[1]//2,
                           :]
            return cropped
        
        

centers = {
    'prostate158': (128, 128),
    'isbi': (128, 128),
    'promise12': (128, 128),
    'decathlon': (128, 128),
    'harp' : (128, 128),
    'drayd' : (128, 128),
    'spleen': (128, 128),
}
roi_size = {
    'prostate158': 160,
    'isbi': 160,
    'promise12': 160,
    'decathlon': 160,
    'harp' : 128,
    'drayd' : 128,
    'spleen': 160
}

def crop_save(
    img_paths : list,
    label_paths : list,
    dataset_name : str,
):
    # Create a new directory for storing the images and labels sampled from the dataset
    if not os.path.exists('replay_buffer'):
        os.makedirs('replay_buffer')
        
    if not os.path.exists(f'replay_buffer/{dataset_name}/imagesTr'):
        os.makedirs(f'replay_buffer/{dataset_name}/imagesTr')
        os.makedirs(f'replay_buffer/{dataset_name}/labelsTr')
        
    transforms = [AddChannel(),
                  CropRoI(roi_size=roi_size[dataset_name], center=centers[dataset_name], random_center=False),
            ]
    
    print(f"Img paths: {img_paths}")
    
    dataset = ImageDataset(img_paths, label_paths,
                               transform=Compose(transforms),
                               seg_transform=Compose(transforms))
    
    for i, (img, label) in enumerate(dataset):
     
        print(f'img shape: {img.shape}')
        print(f'label shape: {label.shape}')
        
        # Save the image and label in .nii.gz format
        nib.save(nib.Nifti1Image(img.squeeze(), np.eye(4)), f'replay_buffer/{dataset_name}/imagesTr/img_{i}.nii.gz')
        nib.save(nib.Nifti1Image(label.squeeze(), np.eye(4)), f'replay_buffer/{dataset_name}/labelsTr/label_{i}.nii.gz')
        print(f'img_{i}.nii.gz and label_{i}.nii.gz saved in replay_buffer/{dataset_name}')
        
    
    
if __name__ == "__main__":
    idx2imgpath = json.load(open('idx2imgpath.json'))
    idx2labelpath = json.load(open('idx2labelpath.json'))
    
    dataset_name = 'spleen'
    img_paths = [idx2imgpath[dataset_name]['10']]
    label_paths = [idx2labelpath[dataset_name]['10']]
    
    crop_save(img_paths, label_paths, dataset_name)