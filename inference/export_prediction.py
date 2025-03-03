import os
from copy import deepcopy
from typing import Union, List
import imageio

import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice
from batchgenerators.utilities.file_and_folder_operations import load_json, isfile, save_pickle


def convert_probabilities_to_segmentation(predicted_probabilities:Union[np.ndarray,torch.Tensor]) -> Union[np.ndarray,torch.Tensor]:
    if not isinstance(predicted_probabilities,(np.ndarray,torch.Tensor)):
        raise RuntimeError(f"Unexpected input type. Expected np.ndarray or torch.Tensor,"
                               f" got {type(predicted_probabilities)}")
    threshold = 0.9 
    segmentation = (predicted_probabilities > threshold).float()
    return segmentation

def convert_predicted_logits_to_segmentation_with_correct_shape(predicted_logits:Union[torch.Tensor,np.array],properties_dict: dict,
                                                                return_probabilities:bool=False,num_threads_torch:int = 8):
    
    old_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads_torch)
    predicted_probabilities = predicted_logits
    segmentation = convert_probabilities_to_segmentation(predicted_probabilities)
    print("segmentation",segmentation.shape)
    
    if isinstance(segmentation,torch.Tensor):
        segmentation = segmentation.cpu().numpy()
    
    foreground_labels= [1]
    segmentation_reverted_cropping = np.zeros(properties_dict['shape_before_cropping'],
                                              dtype=np.uint8 if len(foreground_labels) < 255 else np.uint16)
    slicer = bounding_box_to_slice(properties_dict['bbox_used_for_cropping'])
    segmentation_reverted_cropping[slicer] = segmentation   

    nonzero_count = np.count_nonzero(segmentation)    
    if nonzero_count == 0:
        print("segmentation all 0")
    else:
        print(f"segmentation{nonzero_count}")
    print("nonzero_count",nonzero_count)
    
    torch.set_num_threads(old_threads)
    return segmentation_reverted_cropping

def export_prediction_from_logits(predicted_array_or_file:Union[np.ndarray,torch.Tensor],properties_dict: dict,output_file_truncated:str,save_probabilities:bool=False):
        ret = convert_predicted_logits_to_segmentation_with_correct_shape(
            predicted_array_or_file,properties_dict,return_probabilities=save_probabilities
        )
        del predicted_array_or_file
        
        # 保存结果
        segmentation_final = ret 
        del ret

        if isinstance(segmentation_final, torch.Tensor):
            segmentation_final = segmentation_final.numpy()

        if segmentation_final.dtype != np.uint8:
            segmentation_final = segmentation_final.astype(np.uint8)

        # segmentation_final = np.where(segmentation_final > 0, 255, 0).astype(np.uint8)
        
        print("output_file_truncated ",output_file_truncated )
        imageio.imwrite(output_file_truncated + '.png', segmentation_final)
        