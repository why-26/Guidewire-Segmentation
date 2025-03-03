import argparse
import logging
import os
import random
import sys
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
import wandb
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from typing import Union,Tuple,List

# sys.path.append(os.path.realpath('/home/BI_UNet_331'))
from configs.config import setting_config
from evaluate import eval_dice
from models import FGANet
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from data_augmentation.limited_length_multithreaded_augmenter import LimitedLenWrapper
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, GammaTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from data_augmentation.masking import MaskTransform
from utils.compound_loss import DC_and_BCE_loss,Supervision_Loss
from utils.dice_score import SoftDiceLoss
from utils.dataloading.guidewire_Dataset import GUidewireDataset
from utils.dataloading.guidewire_dataloader import GuidewireDataLoader
from utils.utils1 import unpack_dataset
from utils.dataset_loading import Guidewire2024
from utils.deep_supervision import DeepSupervisionWrapper
from utils.polylr import PolyLRScheduler
os.environ["WANDB_DISABLE_CODE"] = "True"

wandb.login()

logging.shutdown() 
logging.basicConfig(level=logging.INFO) 

root = './dataset/'
dir_checkpoint = Path('')
predataset_path = ''
config = setting_config

def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 16,
        fold = 0, 
        num_workers = 0,
        learning_rate: float = 1e-6,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        inital_patch_size = [602,602],
        final_patch_size = [512,512],
        amp: bool = False,
        oversample_foreground_percent = 0.43, 
        weight_decay: float = 1e-8,
        num_iterations_per_epoch = 250, 
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):

    def load_json(file: str):
        with open(file, 'r') as f:
            a = json.load(f)
        return a
    

    def do_split():
        splits_file = './splits_final.json'
        print("Using split from existing split file:",splits_file)
        splits = load_json(splits_file)
        print(f"The split file contains {len(splits)}splits.")

        print("Desired fold for training:%d" % fold)
        if fold < len(splits):
            tr_keys = splits[fold]['train']
            val_keys = splits[fold]['val']
            print("This split has %d training and %d validation cases."
                  % (len(tr_keys),len(val_keys)))
        return tr_keys,val_keys
    
    def get_datasets():
        tr_keys,val_keys = do_split()
        train_dataset = GUidewireDataset(predataset_path,tr_keys)
        val_dataset = GUidewireDataset(predataset_path,val_keys)
        return train_dataset,val_dataset
    
    train_dataset,val_dataset = get_datasets()

    def get_train_transforms(patch_size:Tuple[int],
                             rotation_for_DA:dict,
                             mirror_axes:Tuple[int,...], 
                             order_resampling_data: int = 3,
                             order_resampling_seg: int = 1,
                             border_val_seg:int = -1,
                             use_mask_for_norm:List[bool]= None)->AbstractTransform:
        tr_transforms = []
        patch_size_spatial = patch_size
        ignore_axes = None
        

        tr_transforms.append(SpatialTransform(
            patch_size_spatial,patch_center_dist_from_border=None,
            do_elastic_deform=False, alpha=(0, 0), sigma=(0, 0),
            do_rotation=True, angle_x=rotation_for_DA['x'], angle_y=rotation_for_DA['y'], angle_z=rotation_for_DA['z'],
            p_rot_per_axis=1,
            do_scale=True,scale=(0.7,1.4),
            border_mode_data="constant", border_cval_data=0, order_data=order_resampling_data,
            border_mode_seg="constant", border_cval_seg=border_val_seg, order_seg=order_resampling_seg,
            random_crop=False,  
            p_el_per_sample=0, p_scale_per_sample=0.2, p_rot_per_sample=0.2,
            independent_scale_for_each_axis=False  
        ))
        
        tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1)) 
        tr_transforms.append(GaussianBlurTransform((0.5,1.),different_sigma_per_channel=False,p_per_sample=0.2)) # 高斯模糊
        tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75,1.25),p_per_sample=0.15)) # 亮度调整
        tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
        tr_transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5,1),per_channel=False,
                                                            order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                                            ignore_axes=ignore_axes)) 
        tr_transforms.append(GammaTransform((0.7, 1.5), True, True, retain_stats=True, p_per_sample=0.1)) # 两次伽马变换，调整亮度
        tr_transforms.append(GammaTransform((0.7, 1.5), False, True, retain_stats=True, p_per_sample=0.3))
        
        if mirror_axes is not None and len(mirror_axes) > 0:
            tr_transforms.append(MirrorTransform(mirror_axes)) 
        
        if use_mask_for_norm is not None and any(use_mask_for_norm):
            tr_transforms.append(MaskTransform([i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                                               mask_idx_in_seg=0,seg_outside_to=0))
        tr_transforms.append(RemoveLabelTransform(-1,0))    
        tr_transforms.append(RenameTransform('seg','target',True))
        tr_transforms.append(NumpyToTensor(['data','target'],'float'))
        tr_transforms = Compose(tr_transforms)
        return tr_transforms
    
    def get_val_transforms():
        val_transforms = []
        val_transforms.append(RemoveLabelTransform(-1,0))
        val_transforms.append(RenameTransform('seg','target',True))
        val_transforms.append(NumpyToTensor(['data','target'],'float'))
        val_transforms = Compose(val_transforms)
        return val_transforms
    
    def get_dataloader():
        patch_size = final_patch_size
        dim = len(patch_size)
        rotation_for_DA = {'x': (-3.141592653589793, 3.141592653589793), 'y': (0, 0), 'z': (0, 0)}
        mirror_axes = (0,1)
        use_mask_for_norm = [True,True,True]
        train_transforms = get_train_transforms(
            patch_size,rotation_for_DA,mirror_axes,order_resampling_data=3,order_resampling_seg=1,use_mask_for_norm=use_mask_for_norm)
        val_transforms = get_val_transforms()
        train_dataset,val_dataset = get_datasets()
        train_loader = GuidewireDataLoader(train_dataset,batch_size,inital_patch_size,final_patch_size,oversample_foreground_percent,sampling_proabilities=None,pad_sides=None)
        val_loader = GuidewireDataLoader(val_dataset,batch_size,final_patch_size,final_patch_size,oversample_foreground_percent,sampling_proabilities=None,pad_sides=None)    
        mt_gen_train = SingleThreadedAugmenter(train_loader, train_transforms)
        mt_gen_val = SingleThreadedAugmenter(val_loader, val_transforms)

        return mt_gen_train, mt_gen_val
    
    train_loader,val_loader = get_dataloader()
    
    # (Initialize logging)
    experiment = wandb.init(project='BI-UNET', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )
    
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Fold:            {fold}
        Learning rate:   {learning_rate}
        Training size:   {len(train_dataset)}
        Validation size: {len(val_dataset)}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
        weight_decay:    {weight_decay}
    ''')
    

    def build_loss():
        deep_loss = Supervision_Loss({},{})
        return deep_loss
        
    
    criterion = build_loss()
    # criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True) 
    # scheduler = PolyLRScheduler(optimizer,learning_rate,epochs)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    global_step = 0

    
    for epoch in range(1, epochs + 1): 
        epoch_start_time = time.time()
        # 开启训练模式
        model.train() 
        threshold = 0.5
        epoch_loss = 0
        loss_list = []
        one_epoch_total_num = num_iterations_per_epoch * batch_size 
        with tqdm(total=one_epoch_total_num, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch_id in range(num_iterations_per_epoch):
                batch = next(train_loader)
                images, true_masks = batch['data'], batch['target']  
                
                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'
                images = images.to(device=device, dtype=torch.float,non_blocking=True)
                true_masks = true_masks.to(device, dtype=torch.float,non_blocking=True)
    

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    gt_pre,masks_pred = model(images) 
                    # loss = criterion(masks_pred.squeeze(1), true_masks.float()) # (b,h,w) (b,h,w)
                    
                    loss = criterion(gt_pre,masks_pred,true_masks)
                    
                optimizer.zero_grad(set_to_none=True) 
                grad_scaler.scale(loss).backward() 
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping) 
                grad_scaler.step(optimizer) 
        
                loss_list.append(loss.item())
                
                pbar.update(images.shape[0]) 
                epoch_loss += loss.item() 
                
                experiment.log({
                    'train loss': np.mean(loss_list),
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': np.mean(loss_list)})
                

            val_score,auc,f1,miou = eval_dice(model,val_loader,criterion,epoch,device,amp)
            scheduler.step(val_score)

            logging.info('Validation Dice score: {}'.format(val_score))

             
            if epoch % 2 == 0:  
                
                histograms = {} 
                for tag, value in model.named_parameters():
                    tag = tag.replace('/', '.')
                    if  value is not None and not (torch.isinf(value) | torch.isnan(value)).any():
                                            histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                    if value.grad is not None and not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                            histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())
            try:
                pred_masks_binary = (masks_pred > threshold).float()   
                pred_mask_binary_vis = pred_masks_binary[0].cpu()

                experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice': val_score,
                                'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(true_masks[0].float().cpu()),
                                    'pred': wandb.Image(pred_mask_binary_vis), 
                                },
                                'validation AUC':auc,
                                'MIOU':miou,
                                'F1-Score':f1,
                                'epoch': epoch,
                                **histograms
                                        })
            except:
                pass
  
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        

        if save_checkpoint and epoch % 10 == 0:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict() 
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-3,help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=True, help='Use mixed precisisupsamplingon')
    parser.add_argument('--bilinear', action='store_true', default=True, help='Use bilinear ')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')
    return parser.parse_args()


if __name__ == '__main__':  
    args = get_args()
 
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    model = FGANet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear,gt_ds=True)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
    




