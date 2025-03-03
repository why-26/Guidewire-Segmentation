# 数据集加载文件
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from configs.config import setting_config

class Guidewire2024(Dataset):
    def __init__(self,dataset,root,config,train=True):
        super(Guidewire2024,self).__init__()
        config = setting_config
        self.dataset= dataset
        if train:
            images = os.listdir(root+'train/images/')
            masks = os.listdir(root+'train/masks/')
            self.data = []
            for i in range(len(dataset)):
                img_path = root + 'train/images/' + dataset[i] + '.png'
                mask_path = root + 'train/masks/' + dataset[i] + '.png'
                self.data.append([img_path,mask_path])
            self.transformer = config.train_transformer
        else:
            images = os.listdir(root+'train/images/')
            masks = os.listdir(root+'train/masks/')
            self.data = []
            for i in range(len(dataset)):
                img_path = root + 'train/images/' + dataset[i] + '.png'
                mask_path = root + 'train/masks/' + dataset[i] + '.png'
                self.data.append([img_path,mask_path])
            self.transformer = config.test_transformer
    
    def __getitem__(self, index):
        img_path,mask_path = self.data[index]
        img = np.array(Image.open(img_path).convert('RGB'))
        msk = np.expand_dims(np.array(Image.open(mask_path).convert('L')), axis=2) / 255
        img,msk = self.transformer((img,msk))
        return {
            'image': img,
            'mask': msk
        }
   
    def __len__(self):
       return len(self.data)
   
