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
import argparse
import pandas as pd




from dataloader import get_dataloader, get_img_label_folds, get_dataloaders
# --------------------------------Input Arguments to be passed----------------------------------------------------
parser = argparse.ArgumentParser(description='For training config')

parser.add_argument('--device', type=str, help='Specify the device to use', required=True)
parser.add_argument('--epochs', type=int, help='No of epochs', required=True)
parser.add_argument('--lr', type=float, help='Learning rate', required=True)
parser.add_argument('--wandb_log', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--seed', type=int, help='Seed', default='2000')
parser.add_argument('--filename', type=str, help='Name of the file to be saved', required=True)
parser.add_argument('--roi_size', type=int, help='Size of the roi to be cropped', default='160')

parsed_args = parser.parse_args()


device = parsed_args.device
epochs = parsed_args.epochs
initial_lr = parsed_args.lr
wandb_log = parsed_args.wandb_log
seed = parsed_args.seed
filename = parsed_args.filename
roi_size = parsed_args.roi_size

print(f"Training on {device}")
print(f"No of epochs : {epochs}")
print(f"Learning rate : {initial_lr}")
print(f"Wandb logging : {wandb_log}")
print(f"Seed : {seed}")

torch.manual_seed(seed)
set_determinism(seed=seed)
# ----------------------------Get Dataloaders--------------------------------------------

dataloaders_map, dataset_map = get_dataloaders()

# Create a joint train loader for all datasets
img_paths = []
label_paths = []
for dataset in dataset_map:
    if dataset in ['harp', 'drayd']:
        img_paths += dataset_map[dataset]['train']['images']
        label_paths += dataset_map[dataset]['train']['labels']
joint_train_loader = get_dataloader(
    img_paths = img_paths,
    label_paths = label_paths,
    train = True,
    train_roi_size = roi_size,
)

# ----------------------------Train Config-----------------------------------------------

model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=2,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=3,
    ).to(device)

optimizer = Adam(model.parameters(), lr=initial_lr, weight_decay=1e-5)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, verbose=True)
dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
hd_metric = HausdorffDistanceMetric(include_background=False, percentile = 95.)
post_pred = Compose([
    EnsureType(), AsDiscrete(argmax=True, to_onehot=2),
    # KeepLargestConnectedComponent(applied_labels=[1], is_onehot=True, connectivity=2)
])
post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2)])
argmax = AsDiscrete(argmax=True)
dice_ce_loss = DiceCELoss(to_onehot_y=True, softmax=True,)

# ------------------------------------WANDB Logging-------------------------------------

# List for datasets will be passed as an argument for this file


config = {
    "Model" : "UNet2D",
    # "Dataset Name" : f"{dataset_name.upper()}",
    "Train Input size" :(256, 256),
    "Train Mode" : f"Patch based (160, 160)",
    "Test Input size" : (256, 256),
    "Test mode" : f"Sliding Window(160, 160)",
    "Batch size" : "No of slices in original volume",
    "No of volumes per batch" : 1,
    "Epochs" : epochs,
    "Optimizer" : "Adam",
    "Scheduler" : "CosineAnnealingLR",
    "Initial LR" : initial_lr,
    "Loss" : "DiceCELoss", 
    "Train Data Augumentations" : "Resize(256,256), RandSpatialCrop(160, 160)",
    "Test Data Preprocess" : "Resize(256,256)",
    "Train samples" : {"Promise12" : 45, "ISBI" : 63, "Decathlon" : 25, "Prostate158" : 119, "Prostatex" : 56},
    "Test Samples" : {"Promise12" : 5, "ISBI" : 16, "Decathlon" : 7, "Prostate158" : 20, "Prostatex" : 10},
#     RandFlip, RandRotate90, RandGaussianNoise, RandGaussSmooth, RandBiasField, RandContrast
    "Pred Post Processing" : "KeepLargestConnectedComponent"
}

if wandb_log:
    wandb.login()
    wandb.init(project="HPF_Joint", entity="vinayu", config = config)
    
batch_size = 1
test_shuffle = True
val_interval = 5
batch_interval = 25
img_log_interval = 15
log_images = True
# iterations = 30

def train(train_loaders : dict, em_loader : DataLoader = None):
    """
    Inputs : No Inputs
    Outputs : No Outputs
    Function : Trains all datasets and logs metrics to WANDB
    """
    
    train_start = time()
    epoch_loss = 0
    model.train()
    print('\n')
    
    
    # Iterating over the dataset
    for i, (imgs, labels, _) in enumerate(joint_train_loader):
        
        imgs, labels = imgs.to(device), labels.to(device)
        
        
        imgs = rearrange(imgs, 'b c h w d -> (b d) c h w')
        labels = rearrange(labels, 'b c h w d -> (b d) c h w')
        
        # print(f"Rearranged : {imgs.shape}, {labels.shape}")
        
        optimizer.zero_grad()
        preds = model(imgs)
        loss = dice_ce_loss(preds, labels)

        preds = [post_pred(i) for i in decollate_batch(preds)]
        preds = torch.stack(preds)
        labels = [post_label(i) for i in decollate_batch(labels)]
        labels = torch.stack(labels)
        
        # Metric scores
        dice_metric(preds, labels)
        hd_metric(preds, labels)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        if i % batch_interval == 0:
            print(f"Epoch: [{epoch}/{epochs}], Batch: [{i}/{len(joint_train_loader)}], Loss: {loss.item() :.4f}, \
                  Dice: {dice_metric.aggregate().item() * 100 :.2f}, HD: {hd_metric.aggregate().item() :.2f}")
    
    # Print metrics, log data, reset metrics
    
    print(f"\nEpoch: [{epoch}/{epochs}], Avg Loss: {epoch_loss / len(joint_train_loader) :.3f}, \
              Train Dice: {dice_metric.aggregate().item() * 100 :.2f}, Train HD: {hd_metric.aggregate().item() :.2f}, Time : {int(time() - train_start)} sec")

    log_metrics = {f"Train Dice" : dice_metric.aggregate().item() * 100,
                   f"Train HD" : hd_metric.aggregate().item(),
                   f"Train Loss" : epoch_loss / len(joint_train_loader),
                   # "Learning Rate" : scheduler.get_last_lr()[0],
                   f"Epoch" : epoch }
    if wandb_log:
        wandb.log(log_metrics)
        print(f'Logged training metrics to wandb')


    dice_metric.reset()
    hd_metric.reset()
    scheduler.step()
    
    
def validate(test_loader : DataLoader, dataset_name : str = None):
    """
    Inputs : Testing dataloader
    Outputs : Returns Dice, HD
    Function : Validate on the given dataloader and return the mertics 
    """
    train_start = time()
    model.eval()
    with torch.no_grad():
        # Iterate over all samples in the dataset
        for i, (imgs, labels, index) in enumerate(test_loader, 1):
            imgs, labels = imgs.to(device), labels.to(device)
            imgs = rearrange(imgs, 'b c h w d -> (b d) c h w')
            labels = rearrange(labels, 'b c h w d -> (b d) c h w')
           
            troi_size = (roi_size, roi_size)
            preds = sliding_window_inference(inputs=imgs, roi_size=troi_size, sw_batch_size=4,
                                            predictor=model, overlap = 0.5, mode = 'gaussian', device=device)
            
            preds = [post_pred(i) for i in decollate_batch(preds)]
            preds = torch.stack(preds)
            labels = [post_label(i) for i in decollate_batch(labels)]
            labels = torch.stack(labels)

            dice_metric(preds, labels)
            hd_metric(preds, labels)

        val_dice = dice_metric.aggregate().item()
        val_hd = hd_metric.aggregate().item()
        
        dice_metric.reset()
        hd_metric.reset()
        
        print("-"*100)
        print(f"Epoch : [{epoch}/{epochs}], Dataset : {dataset_name.upper()}, Test Avg Dice : {val_dice*100 :.2f}, Test Avg HD : {val_hd :.2f}, Time : {int(time() - train_start)} sec")
        print("-"*100)
        
        if wandb_log and log_images and epoch % img_log_interval == 0:
                preds = torch.stack([argmax(c) for c in preds])
                labels = torch.stack([argmax(c) for c in labels])
                f = make_grid(torch.cat([imgs,labels,preds],dim=3), nrow =2, padding = 20, pad_value = 1)
                images = wandb.Image(ToPILImage()(f.cpu()), caption="Left: Input, Middle : Ground Truth, Right: Prediction")
                wandb.log({f"{dataset_name.upper()} Test Predictions": images, "Epoch" : epoch})
                print(f'Logged {dataset_name} segmentation predeictions to wandb')
                  
        return val_dice, val_hd
    
# -------------------------------Training Loop----------------------------------------

test_dataset_names = ['harp', 'drayd']

for epoch in range(1, epochs+1):   

    train(train_loaders = dataloaders_map, em_loader = None)
    
    if epoch % val_interval == 0:
        for dname in test_dataset_names:
            val_dice, val_hd = validate(test_loader = dataloaders_map[dname]['test'], dataset_name = dname)
            
            log_metrics = {}
            log_metrics[f'Epoch'] = epoch
            log_metrics[f'{dname}_curr_dice'] = val_dice*100 
            log_metrics[f'{dname}_curr_hd'] = val_hd

            if wandb_log:
                # Quantiative metrics
                wandb.log(log_metrics)
                print(f'Logged {dname} test metrics to wandb')
                
                
# Save the model
torch.save(model.state_dict(), f"{filename}.pth")