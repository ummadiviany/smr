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
from monai.data import decollate_batch
from monai.utils import set_determinism
import os
import wandb
from time import time
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
from random import sample
from torchvision.transforms import ToPILImage
from dataset import ImageDataset

seed = 2000
torch.manual_seed(seed)
set_determinism(seed=seed)

def get_img_label_folds(img_paths, label_paths):
    
    fold = list(range(0,len(img_paths)))
    fold = sample(fold, k=len(fold))
    fold_imgs = [img_paths[i] for i in fold]
    fold_labels = [label_paths[i] for i in fold]
    return fold_imgs, fold_labels

def get_dataloader(img_paths : list, label_paths : list, train : bool, train_roi_size : int = 160):
    
    if train:
        ttset = "train"
    else:
        ttset = "test"
        
    # Transforms for images & labels
    transforms_map = {
            "train_img_transform" : [
                AddChannel(),
                RandSpatialCrop(roi_size= train_roi_size, random_center = True, random_size=False),
                ToTensor()
                ],
            "train_label_transform" : [
                AddChannel(),
                RandSpatialCrop(roi_size= train_roi_size, random_center = True, random_size=False),
                AsDiscrete(threshold=0.5),
                ToTensor()
                ],
            "test_img_transform" : [
                AddChannel(),
                ToTensor()
                ],
            "test_label_transform" : [
                AddChannel(),
                AsDiscrete(threshold=0.5),
                ToTensor()
                ],
        }

                
    dataset = ImageDataset(img_paths, label_paths,
                            transform=Compose(transforms_map[f'{ttset}_img_transform']), 
                            seg_transform=Compose(transforms_map[f'{ttset}_label_transform']))
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return  dataloader

batch_size = 1
train_roi_size = 160
resize_dim = 256

# 1. Image & Label paths
dataset_map = {
        # "prostate" : {
        #     "data_dir" : "../../prostate/dec_resized/",
        #     "test_size" : 0.2,
        #     'test' :  {'images' : [], 'labels' : []},
        #     'train' :  {'images' : [], 'labels' : []}
        #     },
        # "spleen" : {
        #     "data_dir" : "../../spleen_hc/dec_hc_resized/",
        #     "test_size" : 0.2,
        #     'test' :  {'images' : [], 'labels' : []},
        #     'train' :  {'images' : [], 'labels' : []}
        #     },
        # "hippo" : {
        #     "data_dir" : "../../hippo/drayd_pax/",
        #     "test_size" : 0.2,
        #     'test' :  {'images' : [], 'labels' : []},
        #     'train' :  {'images' : [], 'labels' : []}
        #     },   
        "prostate158" : {
            "data_dir" : "../cl/datasets/prostate158aligned/",
            "test_size" : 0.2,
            'test' :  {'images' : [], 'labels' : []},
            'train' :  {'images' : [], 'labels' : []}
            },
        "promise12" : {
            "data_dir" : "../cl/datasets/promise12prostatealigned/",
            "test_size" : 0.2,
            'test' :  {'images' : [], 'labels' : []},
            'train' :  {'images' : [], 'labels' : []}
            },
        "isbi" : {
            "data_dir" : "../cl/datasets/isbiprostatealigned/",
            "test_size" : 0.2,
            'test' :  {'images' : [], 'labels' : []},
            'train' :  {'images' : [], 'labels' : []}
            },
        "decathlon" : {
            "data_dir" : "../cl/datasets/decathlonprostatealigned/",
            "test_size" : 0.2,
            'test' :  {'images' : [], 'labels' : []},
            'train' :  {'images' : [], 'labels' : []}
            },
        "harp" : {
            "data_dir" : "../cl/hippo/harp_pax/",
            "test_size" : 0.2,
            'test' :  {'images' : [], 'labels' : []},
            'train' :  {'images' : [], 'labels' : []}
            },
        "drayd" : {
            "data_dir" : "../cl/hippo/drayd_pax/",
            "test_size" : 0.2,
            'test' :  {'images' : [], 'labels' : []},
            'train' :  {'images' : [], 'labels' : []}
            },   
    }

def get_dataloaders(train_roi_size : int = 160):
    
    for dataset in dataset_map:
        print(f"------------{dataset}------------")
        data_dir = dataset_map[dataset]['data_dir']

        img_paths = glob(data_dir + "imagesTr/*.nii*")
        label_paths = glob(data_dir + "labelsTr/*.nii*")
        
        
        print("Number of images: {}".format(len(img_paths)))
        print("Number of labels: {}".format(len(label_paths)))
        
        img_paths.sort()
        label_paths.sort()
        
        # 2. Folds

        images_fold, labels_fold  = get_img_label_folds(img_paths, label_paths)
        
        # print("Number of images: {}".format(len(images_fold)))
        # print("Number of labels: {}".format(len(labels_fold)))
        
        # Get train and test sets
        # 3. Split into train - test
        train_idx = int(len(images_fold) * (1 - dataset_map[dataset]['test_size']))
        
        # Store train & test sets 
        
        dataset_map[dataset]['train']['images'] = images_fold[:train_idx]
        dataset_map[dataset]['train']['labels'] = labels_fold[:train_idx]
        
        dataset_map[dataset]['test']['images'] = images_fold[train_idx:]
        dataset_map[dataset]['test']['labels'] = labels_fold[train_idx:]
        
        dataloaders_map = {}

    for dataset in dataset_map:
        # print(f"------------{dataset}------------")
        dataloaders_map[dataset] = {}
        for ttset in ['train', 'test']:
            
            if ttset == 'train':
                train = True
            else:
                train = False
            
            dataloaders_map[dataset][ttset] = get_dataloader(img_paths = dataset_map[dataset][ttset]['images'],
                                                            label_paths = dataset_map[dataset][ttset]['labels'],
                                                            train = train,
                                                            train_roi_size = train_roi_size,
                                                            )
            
            # print(f"""No of samples in {dataset}-{ttset} : {len(dataloaders_map[dataset][ttset])}""")

    # 7. That's it
    
    return dataloaders_map, dataset_map


if __name__ == "__main__":
    start = time()
    
    dataloaders_map, dataset_map = get_dataloaders()
    print("Done")
    # print(f"Data loaders map: {dataloaders_map}")
    # print(f"Dataset map: {dataset_map}")
    
    # abdomenct1k_train = dataloaders_map['hippo']['test']
    # imgs, labels = next(iter(abdomenct1k_train))
    # imgs = rearrange(imgs, 'b c h w d -> (b d) c h w')
    # labels = rearrange(labels, 'b c h w d -> (b d) c h w')
    # print(f"Image shape : {imgs.shape}")
    # print(f"Label shape : {labels.shape}")
    
    # print(f"Image min : {imgs.min()}")
    # print(f"Image max : {imgs.max()}")

    # img_no = int(labels.shape[0]//2)
    # plt.figure(figsize=(6*3,6*1))
    # plt.subplot(1,3,1)
    # plt.imshow(imgs[img_no,0], cmap='gray')
    # plt.axis('off')
    # plt.title('Image')
    # plt.subplot(1,3,2)
    # plt.imshow(labels[img_no,0], cmap='gray')
    # plt.axis('off')
    # plt.title('Label')
    # plt.subplot(1,3,3)
    # plt.imshow(imgs[img_no,0], cmap='gray')
    # plt.imshow(labels[img_no,0], 'copper', alpha=0.2)
    # plt.axis('off')
    # plt.title('Overlay')
    # # plt.show()
    # plt.savefig('spleen.png')
    
    
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