import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as TF
import numpy as np
import os
import math
import random
import logging
import logging.handlers
from batchgenerators.utilities.file_and_folder_operations import *
from matplotlib import pyplot as plt
import multiprocessing
from multiprocessing import Pool


def plot_img_and_mask(img, mask):
    classes = mask.max() + 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    for i in range(classes):
        ax[i + 1].set_title(f'Mask (class {i + 1})')
        ax[i + 1].imshow(mask == i)
    plt.xticks([]), plt.yticks([])
    plt.show()


class myToTensor:
    def __init__(self):
        pass
    def __call__(self, data):
        image, mask = data 
        return torch.tensor(image).permute(2,0,1), torch.tensor(mask).permute(2,0,1) # （H,W,C）--->(C,H,W)
       

class myResize:
    def __init__(self, size_h=256, size_w=256):
        self.size_h = size_h
        self.size_w = size_w
    def __call__(self, data):
        image, mask = data
        return TF.resize(image, [self.size_h, self.size_w]), TF.resize(mask, [self.size_h, self.size_w])
       

class myRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, data):
        image, mask = data
        if random.random() < self.p: return TF.hflip(image), TF.hflip(mask)
        else: return image, mask
            

class myRandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, data):
        image, mask = data
        if random.random() < self.p: return TF.vflip(image), TF.vflip(mask)
        else: return image, mask


class myRandomRotation:
    def __init__(self, p=0.5, degree=[0,360]):
        self.angle = random.uniform(degree[0], degree[1])
        self.p = p
    def __call__(self, data):
        image, mask = data
        if random.random() < self.p: return TF.rotate(image,self.angle), TF.rotate(mask,self.angle)
        else: return image, mask 


class myNormalize:
    def __init__(self, data_name, train=True):
        if data_name == 'guidewire2024':
            if train:
                self.mean = 69.1417
                self.std = 58.9369
            else:
                self.mean = 127.0681
                self.std = 41.2598
            
    def __call__(self, data):
        img, msk = data
        img_normalized = (img-self.mean)/self.std
        img_normalized = ((img_normalized - np.min(img_normalized)) 
                            / (np.max(img_normalized)-np.min(img_normalized))) * 255.
        return img_normalized, msk

def get_train_images_label(dataset_file:str,dataset_json:dict=None):
    if dataset_json is None:
        dataset_json = load_json(join(dataset_file,'dataset.json'))
    images = os.listdir(os.path.join(dataset_file,'train','images'))
    identifiers = [name.split('.')[0] for name in images]
    names = np.array(identifiers)
    images = [os.path.join(dataset_file,'train/images',i + '.png') for i in names]
    segs = [os.path.join(dataset_file,'train/masks',i + '.png') for i in names]
    dataset = {i:{'images':im,'label':se} for i,im,se in zip(names,images,segs)}
    return dataset

def _convet_to_npy(npz_file:str,unpack_segmentation:bool=True,overwrite_existing:bool=False)-> None:
    try:
        a = np.load(npz_file)  # inexpensive, no compression is done here. This just reads metadata
        if overwrite_existing or not isfile(npz_file[:-3] + "npy"):
            np.save(npz_file[:-3] + "npy", a['data'])
        if unpack_segmentation and (overwrite_existing or not isfile(npz_file[:-4] + "_seg.npy")):
            np.save(npz_file[:-4] + "_seg.npy", a['seg'])
    except KeyboardInterrupt:
        if isfile(npz_file[:-3] + "npy"):
            os.remove(npz_file[:-3] + "npy")
        if isfile(npz_file[:-4] + "_seg.npy"):
            os.remove(npz_file[:-4] + "_seg.npy")
        raise KeyboardInterrupt
def unpack_dataset(folder:str,unpack_segmentation:bool=True,overwrite_existing:bool=False,num_processes:int = 6):
    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        npz_files = subfiles(folder,True,None,".npz",True) 
        p.starmap(_convet_to_npy, zip(npz_files,
                                       [unpack_segmentation] * len(npz_files),
                                       [overwrite_existing] * len(npz_files))
                  )



        
        
    
    
    
    
    
    