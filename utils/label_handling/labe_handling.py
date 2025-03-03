from __future__ import annotations
from time import time
from typing import Union, List, Tuple, Type

import numpy as np
import torch

class labelManeger(object):
    def __init__(self,label_dict,regions_class_order:Union[List[int],None],force_use_labels:bool=False,inference_nonlin=None):
        self
        self.label_dict = label_dict
        self.regions_class_order = regions_class_order
        self._force_use_labels = force_use_labels

        if force_use_labels:
            self._has_regions = False
        else:
            self._has_regions: bool = any(
                [isinstance(i,(tuple,list)) and len(i) > 1 for i in self.label_dict.values()]
            )
        self._all_labels:List[int] = self._get_all_labels()               
      
        if inference_nonlin is None:
            self.inference_nonlin = torch.sigmoid if self._has_regions else torch.softmax
        else:
            self.inference_nonlin = inference_nonlin
    
    
        
        