# imports and installs
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
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
from monai.networks.nets import UNet
from monai.data import decollate_batch, ImageDataset
from monai.utils import set_determinism
import os
import wandb
from time import time
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.optim.lr_scheduler import  CosineAnnealingLR
from random import sample
from torchvision.transforms import ToPILImage
import argparse, shutil
import pandas as pd

from dataloader import get_dataloader, get_dataloaders
from clmetrics import print_cl_metrics
# ------------------------------------------------------------------------------------

# python train.py --order decathlon,promise12,isbi,prostate158 --device cuda:0 --optimizer adam --initial_epochs 100 --lr 1e-3 --lr_decay 1 --epoch_decay 1 --replay --store_samples 5 --wandb_log True --seed 2000 --sampling_strategy random
# python train.py --order decathlon,promise12,isbi,prostate158 --device cuda:0 --optimizer adam --initial_epochs 100 --lr 1e-3 --lr_decay 1 --epoch_decay 1 --replay --store_samples 5 --wandb_log True --seed 2000 --sampling_strategy random --order_reverese

parser = argparse.ArgumentParser(description='For training config')

parser.add_argument('--order', type=str, help='order of the dataset domains')
parser.add_argument('--device', type=str, help='Specify the device to use')
parser.add_argument('--optimizer', type=str, help='Specify the optimizer to use')

parser.add_argument('--initial_epochs', type=int, help='No of epochs')
parser.add_argument('--lr', type=float, help='Learning rate')
parser.add_argument('--lr_decay', type=float, help='Learning rate decay factor for each dataset')
parser.add_argument('--epoch_decay', type=float, help='epochs will be decayed after training on each dataset')

parser.add_argument('--replay', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--store_samples', type=int, help='No of samples to store for replay')
parser.add_argument('--seed', type=int, help='Seed for the experiment')
parser.add_argument('--wandb_log', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--order_reverse', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--sampling_strategy', type=str, default='random', help='Sampling strategy for replay buffer')
parser.add_argument('--cropstore', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--fbr', type=float, default=1.0, help='Foreground/background ratio for replay buffer')

parser.add_argument('--filename', type=str, help='Name of the file to save the model')
parser.add_argument('--l2reg', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--alpha', type=float, default=0.1, help='Lambda for l2 regularization')
parser.add_argument('--ewc', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--ewc_weight', type=float, default=1.0, help='Lambda for EWC regularization')
parser.add_argument('--latent_replay', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--roi_size', type=int, help='Size of the roi to be cropped', default=160)

parsed_args = parser.parse_args()

domain_order = parsed_args.order.split(',')
device = parsed_args.device
optimizer_name = parsed_args.optimizer

initial_epochs = parsed_args.initial_epochs
initial_lr = parsed_args.lr
lr_decay = parsed_args.lr_decay
epoch_decay = parsed_args.epoch_decay
use_replay = parsed_args.replay
store_samples = parsed_args.store_samples
seed = parsed_args.seed
wandb_log = parsed_args.wandb_log
order_reverse = parsed_args.order_reverse

sampling_strategy = None
sampling_strategy = parsed_args.sampling_strategy
cropstore = parsed_args.cropstore
filename = parsed_args.filename
ewc = parsed_args.ewc
ewc_weight = parsed_args.ewc_weight
l2reg = parsed_args.l2reg
alpha = parsed_args.alpha
latent_replay = parsed_args.latent_replay
roi_size = parsed_args.roi_size

if order_reverse:
    domain_order = domain_order[::-1]

print('-'*100)
print(f"{'-->'.join(domain_order)}")
print(f"Using device : {device}")
print(f"Initially training for {initial_epochs} epochs")
print(f"Using optimizer : {optimizer_name}")

print(f"Inital learning rate : {initial_lr}")
print(f"Using replay : {use_replay}")
print(f"Replay Sample Size : {store_samples}")
print(f"Sampling strategy : {sampling_strategy}")
print(f"Using cropstore : {cropstore}")

print(f"LR decay  : {lr_decay}")
print(f"Epoch decay : {epoch_decay}")

print(f"Seed : {seed}")
print(f"Wandb logging : {wandb_log}")
print('-'*100)

# Set seed for reproducibility
torch.manual_seed(seed)
set_determinism(seed=seed)

# --------------------------------------------------------------------------------
dataloaders_map, dataset_map = get_dataloaders(train_roi_size=roi_size)
# Map idx to paths
from utils import idx2path
idx2path(dataset_map)
pos_class_map = {}
importance_map = {}
prev_grads_map = {}
# ----------------------------Train Config----------------------------------------

model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=2,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=3,
    ).to(device)


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


if wandb_log:
    wandb.login()
    wandb.init(project="HPF_Sequential", entity="vinayu", config = vars(parsed_args))
    
batch_size = 1
test_shuffle = True
val_interval = 5
batch_interval = 25
img_log_interval = 15
log_images = False

# --------------------------------------------------------------------------------
from importance import get_importance
from baselines import EWC, L2Reg
from copy import deepcopy

if l2reg:
    l2_regulizer = L2Reg(alpha)
if ewc:
    ewc_regulizer = ewc = EWC(model=model, weight=ewc_weight)

# -----------------------------------------------------------------------
import random
model_layer_map = {}
for name, module in model.named_modules():
    model_layer_map[name] = module
    
def get_model_layer(layer_name):
    return model_layer_map[layer_name]

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model_layer_name = 'model.1.submodule.1.submodule.1.submodule.1.submodule.conv.unit2.adn.A'
model_layer = get_model_layer(model_layer_name)

replay_memory = []
l1_loss = nn.L1Loss()

@torch.no_grad()
def accumulate_latent_replay_memory():
    print('-'*100)
    print("Accumulating replay memory...")
    print(f"Storing {store_samples} latent representations for {dataset_name.capitalize()} to replay buffer")
    
    for _ in range(store_samples):
        imgs, _, _ = next(iter(dataloaders_map[dataset_name]['train']))
        imgs = imgs.to(device)
        imgs = rearrange(imgs, 'b c h w d -> (b d) c h w')
        
        model_layer.register_forward_hook(get_activation(model_layer_name))
        _ = model(imgs)
        latent_representation = activation[model_layer_name]
        print(f"Latent representation shape : {latent_representation.shape}")
        replay_memory.append(latent_representation.cpu())

    print(f"Current replay memory size : {len(replay_memory)}")
    print('-'*100)
    
# --------------------------------------------------------------------

def train(train_loader : DataLoader, em_loader : DataLoader|dict = None):
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
    for i, (imgs, labels, index) in enumerate(train_loader, 1):
        
        index = index.item()
        if epoch == 1:
            # Positive class ranking
            pos_precentage = labels.sum() / labels.numel()
            pos_class_map[dataset_name][index] = pos_precentage.item()
            
            
        if sampling_strategy == 'gpcc' or cropstore == True:
            # Find the sample importance using gradients difference
            if index not in importance_map[dataset_name]:
                importance_map[dataset_name][index] = 0
            if i != 1:
                imgsi, labelsi = imgs.to(device), labels.to(device)
                imgsi = rearrange(imgsi, 'b c h w d -> (b d) c h w')
                labelsi = rearrange(labelsi, 'b c h w d -> (b d) c h w')
                loss = dice_ce_loss(model(imgsi), labelsi)
                loss.backward()
                curr_grads = {name:param.grad.data.detach().clone().cpu() for name, param in model.named_parameters() if param.grad is not None}
                if index not in prev_grads_map[dataset_name]:
                    prev_grads_map[dataset_name][index] = curr_grads
                importance_map[dataset_name][index] += get_importance(curr_grads, prev_grads_map[dataset_name][index])
                prev_grads_map[dataset_name][index] = curr_grads
                
                optimizer.zero_grad()
                torch.cuda.empty_cache()
        
        if em_loader is not None:
            if isinstance(em_loader, dict):
                # print(f'Using cropstore replay')
                replay_imgs, replay_labels = [], []
                for rdname in replay_dataloaders_map:
                    rimgs, rlabels = next(iter(replay_dataloaders_map[rdname]['train']))
                    # print(f"Replay dataset : {rdname}, Replay img shape : {rimgs.shape}, Replay label shape : {rlabels.shape}")
                    replay_imgs.append(rimgs), replay_labels.append(rlabels)
                imgs, labels = torch.cat([imgs, *replay_imgs], dim=-1), torch.cat([labels, *replay_labels], dim=-1)
                
            else:   
                em_imgs, em_labels, _ = next(iter(em_loader))
                # Stacking up batch from current dataset and episodic memeory 
                imgs, labels = torch.cat([imgs, em_imgs], dim=-1), torch.cat([labels, em_labels], dim=-1)
        
        # print(f"Img shape : {imgs.shape}, Label shape : {labels.shape}")
        imgs = rearrange(imgs, 'b c h w d -> (b d) c h w')
        labels = rearrange(labels, 'b c h w d -> (b d) c h w')
        
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        preds = model(imgs)

        loss = dice_ce_loss(preds, labels)
        
        if dataset_name != domain_order[0] and l2reg == True:
            # Adding L2 regularization to the loss
            # L2 regularization is only added to the loss after the first domain is trained
            prev_dataset_name = domain_order[domain_order.index(dataset_name) - 1]
            l2_loss = l2_regulizer(model, models_map[prev_dataset_name])
            # print(f"l2_loss : {l2_loss:.3f}")
            loss += l2_loss

        if ewc:
            # Adding EWC regularization loss to the main loss
            ewc_loss = ewc.compute_consolidation_loss()
            loss += ewc_loss
            
        if latent_replay and dataset_name != domain_order[0]:
            # Sample one latent representation from replay memory
            replay_sample = random.sample(replay_memory, 1)[0].to(device)
            
            # Compute L1 loss for representation replay samples and add to total loss
            model_layer.register_forward_hook(get_activation(model_layer_name))
            latent_representation = activation[model_layer_name]
            
            latent_loss = torch.abs(replay_sample.mean() - latent_representation.mean())
            loss += latent_loss
        
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
            print(f"Epoch: [{epoch}/{epochs}], Batch: [{i}/{len(train_loader)}], Loss: {loss.item() :.4f}, \
                  Dice: {dice_metric.aggregate().item() * 100 :.2f}, HD: {hd_metric.aggregate().item() :.2f}")
    
    # Print metrics, log data, reset metrics
    
    print(f"\nEpoch: [{epoch}/{epochs}], Avg Loss: {epoch_loss / len(train_loader) :.3f}, \
              Train Dice: {dice_metric.aggregate().item() * 100 :.2f}, Train HD: {hd_metric.aggregate().item() :.2f}, Time : {int(time() - train_start)} sec")

    log_metrics = {f"{dataset_name.upper()} Train Dice" : dice_metric.aggregate().item() * 100,
                   f"{dataset_name.upper()} Train HD" : hd_metric.aggregate().item(),
                   f"{dataset_name.upper()} Train Loss" : epoch_loss / len(train_loader),
                   # "Learning Rate" : scheduler.get_last_lr()[0],
                   f"Epoch" : epoch }
    if wandb_log:
        wandb.log(log_metrics)
        print(f'Logged training metrics to wandb')


    dice_metric.reset()
    hd_metric.reset()
    scheduler.step()
    torch.cuda.empty_cache()
    
    # Sort and normalize the importance_map and pos_class_map
    if sampling_strategy == 'importance':
        importance_map[dataset_name] = dict(sorted(importance_map[dataset_name].items(), key=lambda item: item[1], reverse=True))
        max_val = max(importance_map[dataset_name].values())
        for key in importance_map[dataset_name]:
            importance_map[dataset_name][key] = importance_map[dataset_name][key] / max_val
    pos_class_map[dataset_name] = dict(sorted(pos_class_map[dataset_name].items(), key=lambda item: item[1], reverse=True))
    
    
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

            # preds = model(imgs)
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
                wandb.log({f"{metric_prefix}_{dataset_name.upper()} Predictions": images, "Epoch" : epoch})
                print(f'Logged {dataset_name} segmentation predeictions to wandb')
            
        
        return val_dice, val_hd
    
dataloaders_map, dataset_map = get_dataloaders()
    
# -----------------------------------------------------------------------

# Empty replay buffer as a list
replay_buffer = {
    "train" : {
        'images' : [],
        'labels' : [],
    },
}

import json
from replay_handler import crop_save
from replay_dataloader import get_replay_dataloaders
# label_class_map = json.load(open('label_class_map.json'))
# merged_map = json.load(open('merged_map_idxs.json', 'r'))
idx2imgpath = json.load(open('idx2imgpath.json', 'r'))
idx2labelpath = json.load(open('idx2labelpath.json', 'r'))

def accumulate_replay_buffer(sampling_strategy : str = 'random'):
    print('-'*100)
    print(f"Accumulating replay buffer using {sampling_strategy} sampling strategy")
    print(f"Storing {store_samples} Samples from {dataset_name.capitalize()} to replay buffer")
    idxs = list(idx2imgpath[dataset_name].keys())
    
    if sampling_strategy == 'representative':
        idxs = list(map(int, idxs[::len(idxs)//store_samples]))
    elif sampling_strategy == 'class1_representative':
        idxs = list(map(int, idxs[:store_samples]))
    elif sampling_strategy == 'random':
        idxs = sample(idxs, store_samples)
    elif sampling_strategy == 'gpcc':
        print("Using GPCC")
        pcridxs = sample(pos_class_map[dataset_name].keys(), store_samples//2)
        gbridxs = sample(importance_map[dataset_name].keys(), store_samples//2)
        idxs = pcridxs + gbridxs
        
    idxs = list(map(int, idxs))
    img_paths = [idx2imgpath[dataset_name][str(idx)] for idx in idxs]
    label_paths = [idx2labelpath[dataset_name][str(idx)] for idx in idxs]
        
    # Save the cropped images and labels to the replay_buffer/{dataset_name} directory
    # Write a function which takes the following inputs
        # img_paths : list of image paths
        # label_paths : list of label paths
        # dataset_name : name of the dataset
        
    if cropstore:
        crop_save(img_paths, label_paths, dataset_name)
        print(f"Cropped & Saved {len(idxs)} samples to replay buffer/{dataset_name} directory")

    print(f"Selected indexes : {idxs}")
        
    # Add the img_paths and label_paths to the replay_buffer
    replay_buffer['train']['images'] += img_paths
    replay_buffer['train']['labels'] += label_paths
        
    print(f"Current replay buffer size : {len(replay_buffer['train']['images'])}")
    print('-'*100)

optimizer_map ={
    'sgd' : torch.optim.SGD,
    'rmsprop' : torch.optim.RMSprop,
    'adam' : torch.optim.Adam,
}

optimizer_params  = {
    'sgd' : {'momentum' : 0.9, 'weight_decay' : 1e-5, 'nesterov' : True},
    'rmsprop' : {'momentum' : 0.9, 'weight_decay' : 1e-5},
    'adam' : {'weight_decay': 1e-5,},   
}

models_map = {}
test_metrics = []
epochs_list = [80, 60, 40, 20]
epochs_list = [60, 30]
# epochs_list = [initial_epochs]*len(domain_order)
# epochs_list = [1, 1, 1, 1]

for i, dataset_name in enumerate(domain_order, 1):
    print(f"Training on {dataset_name} domain")
    
    # epochs = int(initial_epochs * (epoch_decay**(i-1)))
    epochs = epochs_list[i-1]
    lr = initial_lr * (lr_decay**(i-1))
    
    optimizer = optimizer_map[optimizer_name](model.parameters(), lr = lr, **optimizer_params[optimizer_name])    
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, verbose=True)

    train_loader = dataloaders_map[dataset_name]['train']
    if i != 1 and use_replay:
        if cropstore:
            # print("Using cropstore and getting replay dataloaders")
            replay_dataloaders_map, replay_dataset_map = get_replay_dataloaders()
        else:
            em_loader = get_dataloader(img_paths = replay_buffer['train']['images'],
                                        label_paths = replay_buffer['train']['labels'],
                                        train=True)
                                    
    # test_dataset_names = ['prostate158', 'isbi', 'promise12', 'decathlon']
    test_dataset_names = ['harp', 'drayd']

    metric_prefix  = i
    pos_class_map[dataset_name] = {}
    importance_map[dataset_name] = {}
    prev_grads_map[dataset_name] = {}
    
    for epoch in range(1, epochs + 1):   
        
            if i == 1 or not use_replay:
                train(train_loader = train_loader, em_loader = None)
            else:
                if use_replay and cropstore:
                    train(train_loader = train_loader, em_loader = replay_dataloaders_map)
                else:
                    train(train_loader = train_loader, em_loader = em_loader)
                
              
            if epoch % val_interval == 0:
                test_metric = []
                for dname in test_dataset_names:
                    val_dice, val_hd = validate(test_loader = dataloaders_map[dname]['test'], dataset_name = dname)
                    
                    log_metrics = {}
                    log_metrics[f'Epoch'] = epoch
                    log_metrics[f'{metric_prefix}_{dname}_curr_dice'] = val_dice*100 
                    log_metrics[f'{metric_prefix}_{dname}_curr_hd'] = val_hd

                    if wandb_log:
                        # Quantiative metrics
                        wandb.log(log_metrics)
                        print(f'Logged {dname} test metrics to wandb')
                        
                    test_metric.append(val_dice*100)
                    
    test_metrics.append(test_metric)
    
    # If using replay buffer, accumulate samples from current domain
    if use_replay:
        # Store samples to replay buffer using the sampling strategy
        accumulate_replay_buffer(sampling_strategy = sampling_strategy)    
        
    if l2reg:
        models_map[dataset_name] = deepcopy(model)
    if ewc:
        print("Updating model weights with EWC Constraint")
        ewc.register_ewc_params(train_loader)
    
    if latent_replay:
        accumulate_latent_replay_memory()
        
        
cl_metrics = print_cl_metrics(domain_order, test_dataset_names, test_metrics)
if wandb_log:
    wandb.log(cl_metrics)
    print(f'Logged CL metrics to wandb')
    
# Delete the replay buffer directory
if os.path.exists('replay_buffer'):
    shutil.rmtree('replay_buffer')

# Save the model
torch.save(model.state_dict(), f'{filename}.pth')