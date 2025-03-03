from typing import List,Union, Tuple
import numpy as nps
from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.utilities.file_and_folder_operations import *
from utils.dataloading.guidewire_Dataset import GUidewireDataset

class GuidewireDataLoaderBase(DataLoader):
    def __init__(self,
                 data:GUidewireDataset, 
                 batch_size:int, 
                 patch_size:Union[List[int],Tuple[int,...],np.ndarray], 
                 final_patch_size:Union[List[int],Tuple[int,...],np.ndarray],
                 oversample_foreground_percent:float=0.0,
                 sampling_proabilities:Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 pad_sides:Union[List[int],Tuple[int,...],np.ndarray]=None):
        super().__init__(data,batch_size,1,None,True,False,True,sampling_proabilities)
        assert isinstance(data,GUidewireDataset),'we need dicyionaries as data'
        self.indices = list(data.keys()) 
        
        self.oversample_foreground_percent = oversample_foreground_percent 
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys()) 
        print(len(self.list_of_keys)) 
        self.need_to_pad = (np.array(patch_size)-np.array(final_patch_size)).astype(int) 
        if pad_sides is not None:
            if not isinstance(pad_sides,np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad +=pad_sides 
        self.num_channels = None
        self.pad_sides = pad_sides
        self.data_shape,self.seg_shape = self.determine_shape() 
        self.sampling_probabilities = sampling_proabilities
        self.annotated_classes_key = (0,1)
        self.get_do_oversamlpe = self.oversample_last_percent 

    def oversample_last_percent(self,sample_idx:int) -> bool:
        return not sample_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))

    def determine_shape(self):
        data,seg,properties = self._data.load_case(self.indices[0]) 
        data_channels = data.shape[0]  
        seg_channels = seg.shape[0]  
        data_shape = (self.batch_size,data_channels,*self.patch_size)
        seg_shape = (self.batch_size,seg_channels,*self.patch_size)
        return data_shape,seg_shape
    

    def get_bbox(self,data_shape:np.ndarray,force_fg:bool,class_locations:Union[dict,None],overwrite_class:Union[int,Tuple[int,...]]=None,verbose:bool=False):
        need_to_pad = self.need_to_pad.copy()
        dim = len(data_shape) 
        
        for d in range(dim):
            if need_to_pad[d] + data_shape[d] < self.patch_size[d]:
                need_to_pad[d] = self.patch_size[d] - data_shape[d]
        

        lbs = [-need_to_pad[i] // 2 for i in range(dim)] 
        ubs = [data_shape[i] + need_to_pad[i] // 2 + need_to_pad[i] % 2 - self.patch_size[i] for i in range(dim)] 


        if not force_fg: 
            bbox_lbs = [np.random.randint(lbs[i],ubs[i] + 1) for i in range(dim)] 
        else:
            assert class_locations is not None,'if force_fg is set class_locations cannot be None' 
            if overwrite_class is not None:
                assert overwrite_class in class_locations.keys(),'desired class ("overwrite_class") does not ' \
                                                                      'have class_locations (missing key)'
           
            eligible_class_or_regions = [i for i in class_locations.keys() if len(class_locations[i])>0]
            
            tmp = [i == self.annotated_classes_key if isinstance(i,tuple) else False for i in eligible_class_or_regions]
            if any(tmp):
                if len(eligible_class_or_regions) > 1:
                    eligible_class_or_regions.pop(np.where(tmp)[0][0]) 
            
            if len(eligible_class_or_regions) == 0:
                selected_class = None
                if verbose:
                     print('case does not contain any foreground classes')
            else:
                
                selected_class =  eligible_class_or_regions[np.random.choice(len(eligible_class_or_regions))] if \
                        (overwrite_class is None or (overwrite_class not in eligible_class_or_regions)) else overwrite_class
        
            voxels_if_that_class = class_locations[selected_class] if selected_class is not None else None
    
            if voxels_if_that_class is not None and len(voxels_if_that_class)>0:
                selected_voxel = voxels_if_that_class[np.random.choice(len(voxels_if_that_class))]
                bbox_lbs = [max(lbs[i],selected_voxel[i+1]-self.patch_size[i] // 2)for i in range(dim)] 
        bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(dim)]
        return bbox_lbs,bbox_ubs
            
        
        
    
            
            
            

            
            