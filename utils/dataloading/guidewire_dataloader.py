import numpy as np
import sys
import os
import torch
# sys.path.append(os.path.dirname(os.path.realpath('/home/jiajun/baobao/BI_UNet')))
# print(sys.path)
sys.path.append('')
from utils.dataloading.guidewire_Dataset import GUidewireDataset
from utils.dataloading.basedataloader import GuidewireDataLoaderBase

class GuidewireDataLoader(GuidewireDataLoaderBase):
    def generate_train_batch(self):
        selected_keys = self.get_indices()
        data_all = np.zeros(self.data_shape,dtype=np.float32)
        seg_all = np.zeros(self.seg_shape,dtype=np.int16)
        case_properties = []
        
        for j,current_key in enumerate(selected_keys):
            force_fg = self.get_do_oversamlpe(j) 
            data,seg,properties  =self._data.load_case(current_key) 
            case_properties.append(properties)
        
            if not force_fg:
                selected_class_or_region = None
            else:
                eligible_class_or_regions = [i for i in properties['class_locations'].keys() if len(properties['class_locations'][i]) > 0]

                tmp = [i == self.annotated_classes_key if isinstance(i,tuple) else False for i in eligible_class_or_regions]
                selected_class_or_region = eligible_class_or_regions[np.random.choice(len(eligible_class_or_regions))] if len(eligible_class_or_regions)>0 else None
        
            class_locations = {
                selected_class_or_region:properties['class_locations'][selected_class_or_region]
            }if(selected_class_or_region is not None) else None
                
            shape = data.shape[1:] 
            dim = len(shape) 
            bbox_lbs,bbox_ubs = self.get_bbox(shape,force_fg if selected_class_or_region is not None else None,
                                                  class_locations,overwrite_class=selected_class_or_region)
        
            valid_bbox_lbs = [max(0,bbox_lbs[i]) for i in range(dim)]
            valid_bbox_ubs = [min(shape[i],bbox_ubs[i]) for i in range(dim)]
            this_slice = tuple([slice(0,data.shape[0])] + [slice(i,j) for i,j in zip(valid_bbox_lbs,valid_bbox_ubs)])
            data = data[this_slice]
                
            this_slice = tuple([slice(0,seg.shape[0])] + [slice(i,j) for i,j in zip(valid_bbox_lbs,valid_bbox_ubs)])
            seg = seg[this_slice]
                
            padding = [(-min(0,bbox_lbs[i]),max(bbox_ubs[i] - shape[i],0)) for i in range(dim)]
            data_all[j] = np.pad(data, ((0, 0), *padding), 'constant', constant_values=0)
            seg_all[j] = np.pad(seg, ((0, 0), *padding), 'constant', constant_values=-1)
                
        return {'data': data_all, 'seg': seg_all, 'properties': case_properties, 'keys': selected_keys}

if __name__ == '__main__':
    folder = ''
    dataset = GUidewireDataset(folder,None)
    dataloader = GuidewireDataLoader(dataset,16,(527,527),(448,448),0.43,None,None)
    batch = next(dataloader)
    images, true_masks = batch['data'], batch['seg']
    num = images.shape[0]
    print("num",num)
    for i in range(num):
        image_tensor = torch.from_numpy(images[i])
        mask_tensor = torch.from_numpy(true_masks[i])
        
        non_zero_images = torch.count_nonzero(image_tensor)
        non_zero_true_masks = torch.count_nonzero(mask_tensor)

        print(f"{non_zero_images.item()}")
        print(f"{non_zero_true_masks.item()}")
    print('success')