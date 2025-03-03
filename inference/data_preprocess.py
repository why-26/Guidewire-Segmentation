import multiprocessing
import queue
from torch.multiprocessing import Event, Process, Queue, Manager
import sys
import os
sys.path.append(os.path.abspath('/home/jiajun/baobao/BI_UNet'))

from time import sleep
from typing import Union, List

import numpy as np
import torch
from batchgenerators.dataloading.data_loader import DataLoader
from preprocess.preprocessing.processor import Guidewire_Preprocessor

class PreprocessFromNpy(DataLoader):
    def __init__(self,list_of_images:List[np.ndarray],truncated_names:Union[List[str],None],
                 num_threads_in_multithread:int = 1,verbose:bool = False):
        preprocessor = Guidewire_Preprocessor(verbose=verbose)
        self.preprocessor = preprocessor
        self.truncted_names= truncated_names
        
        super().__init__(
            list(zip(list_of_images,truncated_names)),
            1,num_threads_in_multithread,
            seed_for_shuffle=1, return_incomplete=True,
            shuffle=False, infinite=False, sampling_probabilities=None)
        
        self.indices = list(range(len(list_of_images)))
        
    def generate_train_batch(self):
        idx = self.get_indices()[0]
        image = self._data[idx][0]
        ofname = self._data[idx][1]
        data,seg = self.preprocessor.run_case_npy(image,None,None)
        data = torch.from_numpy(data)
        return {'data':data,'ofile':ofname}
    
        