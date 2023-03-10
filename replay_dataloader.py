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
import os
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


def get_dataloader(img_paths : list, label_paths : list, dname : str):
        
    ttset = "train"
        
    dataset = ImageDataset(img_paths, label_paths,
                            transform=Compose(transforms_map[f'{ttset}_img_transform']), 
                            seg_transform=Compose(transforms_map[f'{ttset}_label_transform']))
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return  dataloader

batch_size = 1
train_roi_size = 160
resize_dim = 256

# Transforms for images & labels
transforms_map = {
        "train_img_transform" : [
            AddChannel(),
            ToTensor()
            ],
        "train_label_transform" : [
            AddChannel(),
            AsDiscrete(threshold=0.5),
            ToTensor()
            ],
    }

    

# 1. Image & Label paths
dataset_map = {
        "prostate" : {
            "data_dir" : "replay_buffer/prostate/",
            'train' :  {'images' : [], 'labels' : []}
            },
        "spleen" : {
            "data_dir" : "replay_buffer/spleen/",
            'train' :  {'images' : [], 'labels' : []}
            },
        "hippo" : {
            "data_dir" : "replay_buffer/hippo/",
            'train' :  {'images' : [], 'labels' : []}
            }
        
    }

def get_replay_dataloaders():
    
    for dataset in dataset_map:
        print(f"------------{dataset}------------")
        data_dir = dataset_map[dataset]['data_dir']

        img_paths = glob(data_dir + "imagesTr/*.nii.gz")
        label_paths = glob(data_dir + "labelsTr/*.nii.gz")
        
        if len(img_paths) == 0:
            img_paths = glob(data_dir + "imagesTr/*.nii")
            label_paths = glob(data_dir + "labelsTr/*.nii")
        
        print("Number of images: {}".format(len(img_paths)))
        print("Number of labels: {}".format(len(label_paths)))
        
        if len(img_paths) == 0:
            print(f"Images not found in {data_dir}imagesTr/")
            continue
        
        img_paths.sort()
        label_paths.sort()
        
        dataset_map[dataset]['train']['images'] = img_paths
        dataset_map[dataset]['train']['labels'] = label_paths
        
    dataloaders_map = {}

    for dataset in dataset_map:
        # print(f"------------{dataset}------------")
        
        if len(dataset_map[dataset]['train']['images']) == 0:
            continue
        
        dataloaders_map[dataset] = {}
        ttset = "train"
            
        dataloaders_map[dataset][ttset] = get_dataloader(img_paths = dataset_map[dataset][ttset]['images'],
                                                    label_paths = dataset_map[dataset][ttset]['labels'],
                                                    dname=dataset)
    
            # print(f"""No of samples in {dataset}-{ttset} : {len(dataloaders_map[dataset][ttset])}""")

    # 7. That's it
    
    return dataloaders_map, dataset_map


if __name__ == "__main__":
    start = time()
    
    dataloaders_map, dataset_map = get_replay_dataloaders()
    print("Done")
    # print(f"Data loaders map: {dataloaders_map}")
    # print(f"Dataset map: {dataset_map}")
    
    abdomenct1k_train = dataloaders_map['spleen']['train']
    imgs, labels = next(iter(abdomenct1k_train))
    imgs = rearrange(imgs, 'b c h w d -> (b d) c h w')
    labels = rearrange(labels, 'b c h w d -> (b d) c h w')
    print(f"Image shape : {imgs.shape}")
    print(f"Label shape : {labels.shape}")
    
    print(f"Image min : {imgs.min()}")
    print(f"Image max : {imgs.max()}")

    img_no = int(labels.shape[0]//2)
    plt.figure(figsize=(6*3,6*1))
    plt.subplot(1,3,1)
    plt.imshow(imgs[img_no,0], cmap='gray')
    plt.axis('off')
    plt.title('Image')
    plt.subplot(1,3,2)
    plt.imshow(labels[img_no,0], cmap='gray')
    plt.axis('off')
    plt.title('Label')
    plt.subplot(1,3,3)
    plt.imshow(imgs[img_no,0], cmap='gray')
    plt.imshow(labels[img_no,0], 'copper', alpha=0.2)
    plt.axis('off')
    plt.title('Overlay')
    # plt.show()
    plt.savefig('spleen.png')
    
    
    # print(f"Completed in: {time() - start:.1f} seconds")
    
    # import json
    # idx2imgpath = {}
    # idx2labelpath = {}
    # for dataset in dataset_map:
    #     print(f"------------{dataset}------------")
    #     idx2imgpath[dataset] = {}
    #     idx2labelpath[dataset] = {}
    #     for idx, (img_path, label_path) in enumerate(zip(dataset_map[dataset]['train']['images'], dataset_map[dataset]['train']['labels'])):
    #         idx2imgpath[dataset][idx] = img_path
    #         idx2labelpath[dataset][idx] = label_path
            
    #     # Sort by idx
    #     idx2imgpath[dataset] = {k: v for k, v in sorted(idx2imgpath[dataset].items(), key=lambda item: item[0])}
    #     idx2labelpath[dataset] = {k: v for k, v in sorted(idx2labelpath[dataset].items(), key=lambda item: item[0])}
        
    # filename_img = 'idx2imgpath.json'
    # filename_label = 'idx2labelpath.json'
    # if not os.path.exists(filename_img):
    #     os.mknod(filename_img)
    #     os.mknod(filename_label)
    # with open(filename_img, 'w') as fp:
    #     json.dump(idx2imgpath, fp)
    # with open(filename_label, 'w') as fp:
    #     json.dump(idx2labelpath, fp)
        
    
    print(f"Completed in: {time() - start:.1f} seconds")