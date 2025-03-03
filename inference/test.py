import sys
import os
import inspect
import multiprocessing
from copy import deepcopy
sys.path.append(os.path.abspath('/home/BI-UNET'))
import logging
import matplotlib.pyplot as plt
from PIL import Image
from typing import Tuple, Union, List, Optional
from multiprocessing import Pool
import re
from skimage import io
import torchvision.transforms as transforms
import time

import numpy as np
import torch
import json
from collections.abc import Iterable
# from acvl_utils.cropping_and_padding.padding import pad_nd_image
from padding import pad_nd_image
from batchgenerators.utilities.file_and_folder_operations import load_json, join
from torch import nn
from tqdm import tqdm
from models.model import UNet
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from sliding_window_prediction import compute_gaussian,compute_steps_for_sliding_window
from export_prediction import export_prediction_from_logits
from data_preprocess import PreprocessFromNpy
from data_iterators import preprocessing_iterator_fromnpy,preprocessing_iterator_fromfiles

default_num_processes = 8


def subfiles(folder: str, join: bool = True, prefix: str = None, suffix: str = None, sort: bool = True) -> List[str]:
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y
    res = [l(folder, i) for i in os.listdir(folder) if os.path.isfile(os.path.join(folder, i))
           and (prefix is None or i.startswith(prefix))
           and (suffix is None or i.endswith(suffix))]
    if sort:
        res.sort()
    return res


def get_identifiers_from_splitted_dataset_folder(folder: str, file_ending: str):
    files = subfiles(folder, suffix=file_ending, join=False)
    print("len_iles",len(files))
    crop = len(file_ending)
    files = [i[:-crop] for i in files]
    files = np.unique(files)
    return files


def create_lists_from_splitted_dataset_folder(folder: str, file_ending: str, identifiers: List[str] = None) -> List[
    List[str]]:
    """
    does not rely on dataset.json
    """
    print("folder",folder)
    print("file_ending",file_ending)
    if identifiers is None:
        identifiers = get_identifiers_from_splitted_dataset_folder(folder, file_ending)
    files = subfiles(folder, suffix=file_ending, join=False, sort=True)
    list_of_lists = []
    for f in identifiers:
        p = re.compile(re.escape(f) + re.escape(file_ending))
        list_of_lists.append([join(folder, i) for i in files if p.fullmatch(i)])

    return list_of_lists

def recursive_fix_for_json_export(my_dict: dict):
    keys = list(my_dict.keys()) 
    for k in keys:
        if isinstance(k, (np.int64, np.int32, np.int8, np.uint8)):
            tmp = my_dict[k]
            del my_dict[k]
            my_dict[int(k)] = tmp
            del tmp
            k = int(k)

        if isinstance(my_dict[k], dict):
            recursive_fix_for_json_export(my_dict[k])
        elif isinstance(my_dict[k], np.ndarray):
            assert my_dict[k].ndim == 1, 'only 1d arrays are supported'
            my_dict[k] = fix_types_iterable(my_dict[k], output_type=list)
        elif isinstance(my_dict[k], (np.bool_,)):
            my_dict[k] = bool(my_dict[k])
        elif isinstance(my_dict[k], (np.int64, np.int32, np.int8, np.uint8)):
            my_dict[k] = int(my_dict[k])
        elif isinstance(my_dict[k], (np.float32, np.float64, np.float16)):
            my_dict[k] = float(my_dict[k])
        elif isinstance(my_dict[k], list):
            my_dict[k] = fix_types_iterable(my_dict[k], output_type=type(my_dict[k]))
        elif isinstance(my_dict[k], tuple):
            my_dict[k] = fix_types_iterable(my_dict[k], output_type=tuple)
        elif isinstance(my_dict[k], torch.device):
            my_dict[k] = str(my_dict[k])
        else:
            pass  # pray it can be serialized

def save_json(obj, file: str, indent: int = 4, sort_keys: bool = True) -> None:
    with open(file, 'w') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)


class Predictor(object):
    def __init__(self,network,tile_step_size:float = 0.5,use_gaussian:bool=True,use_imrroring:bool=True,perform_everything_on_device:bool=True,
                 device:torch.device=torch.device('cuda'),verbose:bool=False,verbose_preprocessing:bool=False,allow_tqdm:bool=True):
        self.verbose = verbose
        self.verbose_preprocessing = verbose_preprocessing
        self.allow_tqdm =allow_tqdm
        self.network = network
        self.dataset_json = None
        
        self.tile_step_size = tile_step_size
        self.use_gaussian = use_gaussian
        self.device = device
        self.perform_everything_on_device = perform_everything_on_device
        self.final_patch_size = [512,512]
    
    def initialize_from_trained_model_folder(self,model_training_output_dir:str,use_folds:Union[Tuple[Union[int,str],None]],
                                             checkpoint_name:str='checkpoint_final.pth'):

        dataset_json = load_json(join(model_training_output_dir))
        self.dataset_json = dataset_json

   
    def _internal_get_sliding_window_slicers(self,image_size:Tuple[int, ...]):
        print("image_size",image_size)
        slicers = []
        steps = compute_steps_for_sliding_window(image_size,self.final_patch_size,self.tile_step_size)
        for sx in steps[0]:
            for sy in steps[1]:
                slicers.append(
                    tuple([slice(None),*[slice(si,si + ti) for si,ti in zip((sx,sy),self.final_patch_size)]])
                )
        return slicers

    def _manage_input_and_output_lists(self, list_of_lists_or_source_folder: Union[str, List[List[str]]],
                                       output_folder_or_list_of_truncated_output_files: Union[None, str, List[str]],
                                       overwrite: bool = True,
                                       part_id: int = 0,
                                       num_parts: int = 1,
                                       save_probabilities: bool = False):
            if isinstance(list_of_lists_or_source_folder, str):
                list_of_lists_or_source_folder = create_lists_from_splitted_dataset_folder(list_of_lists_or_source_folder,
                                                                                      self.dataset_json['file_ending'])
        
            print(f'There are {len(list_of_lists_or_source_folder)} cases in the source folder')
            list_of_lists_or_source_folder = list_of_lists_or_source_folder[part_id::num_parts]
            
            caseids = [os.path.basename(i[0])[:-(len(self.dataset_json['file_ending']))] for i in
                    list_of_lists_or_source_folder]
            print(
                f'I am process {part_id} out of {num_parts} (max process ID is {num_parts - 1}, we start counting with 0!)')
            print(f'There are {len(caseids)} cases that I would like to predict')

            if isinstance(output_folder_or_list_of_truncated_output_files, str):
                output_filename_truncated = [join(output_folder_or_list_of_truncated_output_files, i) for i in caseids]
            else:
                output_filename_truncated = output_folder_or_list_of_truncated_output_files
            # print("output_filename_truncated",output_filename_truncated)


            if not overwrite and output_filename_truncated is not None:
                tmp = [isfile(i + self.dataset_json['file_ending']) for i in output_filename_truncated]
                if save_probabilities:
                    tmp2 = [isfile(i + '.npz') for i in output_filename_truncated]
                    tmp = [i and j for i, j in zip(tmp, tmp2)]
                not_existing_indices = [i for i, j in enumerate(tmp) if not j]

                output_filename_truncated = [output_filename_truncated[i] for i in not_existing_indices]
                list_of_lists_or_source_folder = [list_of_lists_or_source_folder[i] for i in not_existing_indices]
                print(f'overwrite was set to {overwrite}, so I am only working on cases that haven\'t been predicted yet. '
                    f'That\'s {len(not_existing_indices)} cases.')

            return list_of_lists_or_source_folder, output_filename_truncated

    
    def predict_sliding_window_return_logits(self,input_image:torch.Tensor) -> Union[np.ndarray,torch.Tensor]:
        assert isinstance(input_image,torch.Tensor)
        
        self.network = self.network.to(self.device)
        self.network.eval()
        
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        else:
            pass
        
        with torch.no_grad():
            device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
            with torch.autocast(device_type, enabled=(device_type == 'cuda')):
                assert input_image.ndim == 3,'input_image must be a 3D np.ndarry or torch.Tensor (c,x,y)'
                
                if self.verbose: print(f'Input shape: {input_image.shape}')
                if self.verbose: print("step_size:", self.tile_step_size)
                
                data,slicer_revert_padding = pad_nd_image(input_image,self.final_patch_size,'constant',{'value':0},True,None)
                # print("data",data.shape)
                slicers = self._internal_get_sliding_window_slicers(data.shape[1:])
                # print("slicers",slicers)
                results_device = self.device if self.perform_everything_on_device else torch.device('cpu')
                if self.verbose:
                    print(f'move image to device {results_device}')
   
                data = data.to(results_device)
                predicted_logits = torch.zeros((1,*data.shape[1:]),dtype=torch.half,device=results_device)
                n_predictions = torch.zeros(data.shape[1:],dtype=torch.half,device=results_device)
                if self.use_gaussian:
                    gaussian = compute_gaussian(tuple(self.final_patch_size),sigma_scale=1. / 8,
                                                value_scaling_factor=10,device=results_device)
                print('running prediction')
                if not self.allow_tqdm and self.verbose:print(f'{len(slicers)} steps')
                a = 0
                for sl in tqdm(slicers,disable=not self.allow_tqdm):
                    a = a + 1
                    workon = data[sl][None]
                    workon = workon.to(self.device,non_blocking=False)
                    prediction = self.network(workon).to(results_device)
                    prediction = prediction.squeeze(0)
                    predicted_logits[sl] += (prediction * gaussian if self.use_gaussian else prediction)
                    n_predictions[sl[1:]] += (gaussian if self.use_gaussian else 1)
                    
                predicted_logits /= n_predictions
                print("predicted_logits",predicted_logits.shape)
                if torch.any(torch.isinf(predicted_logits)):
                    raise RuntimeError('Encountered inf in predicted array')
        print("predicted_logits_1",predicted_logits[tuple([slice(None), *slicer_revert_padding[1:]])].shape)
        cpu_tensor = predicted_logits[tuple([slice(None), *slicer_revert_padding[1:]])].cpu()
        cpu_array = cpu_tensor.numpy()
        nonzero_count_1 = np.count_nonzero(cpu_array) 
        if nonzero_count_1 == 0:
            print("predicted_logits_1中全为0")
        else:
            print(f"predicted_logits_1中有{nonzero_count_1}个非零元素")
        return predicted_logits[tuple([slice(None), *slicer_revert_padding[1:]])]
    
    # 
    def predict_from_files(self,
                           list_of_lists_or_source_folder:Union[str,List[List[str]]],
                           output_folder_list:Union[str,None,List[str]],
                           save_probabilities:bool=False,
                           overwrite:bool=True,
                           num_processes_preprocessing:int=default_num_processes,
                           num_processes_segmentation_export: int = default_num_processes,
                           num_parts:int=1,
                           part_id:int=0):
        if isinstance(output_folder_list,str):
            output_folder = output_folder_list
        elif isinstance(output_folder_list,list):
            output_folder = os.path.dirname(output_folder_list[0])
        else:
            output_folder = None
        if output_folder is not None:
            my_init_kwargs = {}
            for k in inspect.signature(self.predict_from_files).parameters.keys():
                my_init_kwargs[k] = locals()[k]
            my_init_kwargs = deepcopy(my_init_kwargs)
            recursive_fix_for_json_export(my_init_kwargs)
            os.makedirs(output_folder, exist_ok=True)
            save_json(my_init_kwargs, join(output_folder, 'predict_from_raw_data_args.json'))
            #
            save_json(self.dataset_json, join(output_folder, 'dataset.json'), sort_keys=False)
        
        list_of_lists_or_source_folder, output_filename_truncated = self._manage_input_and_output_lists(list_of_lists_or_source_folder,
                                                                                                        output_folder_list,
                                                                                                        overwrite,part_id,num_parts,
                                                                                                        save_probabilities)
        if len(list_of_lists_or_source_folder) == 0:
            return 

        data_iterator = self._internal_get_data_iterator_from_lists_of_filenames(list_of_lists_or_source_folder,output_filename_truncated,num_processes_preprocessing)
        return self.predict_from_data_iterator(data_iterator,save_probabilities,num_processes_segmentation_export)
    
    def _internal_get_data_iterator_from_lists_of_filenames(self,
                                                            input_list_of_lists: List[List[str]],
                                                            output_filenames_truncated: Union[List[str], None],
                                                            num_processes: int):

        
        return preprocessing_iterator_fromfiles(input_list_of_lists,output_filenames_truncated,
                                                self.dataset_json,num_processes, self.device.type == 'cuda',self.verbose_preprocessing)
    

    def predict_from_data_iterator(self,data_iterator,save_probabilities=False,num_processes: int=8):
        with multiprocessing.get_context("spawn").Pool(num_processes) as export_pool:
            worker_list = [i for i in export_pool._pool]
            r = []
            for preprocessed in data_iterator:
                data = preprocessed['data']
                if isinstance(data,str):            
                    delfile = data
                    data = torch.from_numpy(np.load(data))
                    os.remove(delfile)
                ofile = preprocessed['ofile']
                
                if ofile is not None:
                    print(f'\nPredicting {os.path.basename(ofile)}:')
                else:
                    print(f'\nPredicting image of shape {data.shape}:')
                properties = preprocessed['data_properties']
                proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)
                prediction = self.predict_logits_from_preprocessed_data(data).cpu()
                print("prediction",prediction.shape)

                if ofile is not None:
                    r.append(
                        export_pool.starmap_async(
                        export_prediction_from_logits,
                        ((prediction,properties,ofile,save_probabilities),)
                        )
                    )

                if ofile is not None:
                    print(f'done with {os.path.basename(ofile)}')
                else:
                    print(f'\nDone with image of shape {data.shape}:')

                ret = [i.get()[0] for i in r]

            if isinstance(data_iterator,MultiThreadedAugmenter):
                data_iterator._finish()

            compute_gaussian.cache_clear()

            empty_cache(self.device)

            return ret

    def predict_single_npy_array(self,filename,input_image:np.ndarray,output_file_truncated:str,save_or_return_probabilities:bool = False):
        image_process = PreprocessFromNpy([input_image],[output_file_truncated],num_threads_in_multithread=1,verbose=self.verbose)
        print("preprocessing!")
        dct = next(image_process) 
        print("predicting!")
        predicted_logits = self.predict_logits_from_preprocessed_data(dct['data']).cpu()
        threshold = 0.9 
        predicted_class = (predicted_logits > threshold).float() 
        print("save results!")
        export_prediction_from_logits(filename,predicted_class,output_file_truncated= output_file_truncated,save_probabilities=save_or_return_probabilities)
    
    
    def predict_logits_from_preprocessed_data(self,data:torch.Tensor) -> torch.Tensor:
        default_num_processes = 8
        n_threads = torch.get_num_threads()
        torch.set_num_threads(default_num_processes if default_num_processes < n_threads else n_threads)
        with torch.no_grad():
            prediction = None
            if prediction is None:
                prediction = self.predict_sliding_window_return_logits(data).to('cpu')
            else:
                prediction += self.predict_sliding_window_return_logits(data).to('cpu')
            
            print('Prediction Done')
            prediction = prediction.to('cpu')

        cpu_array = prediction.numpy()
        nonzero_count_1 = np.count_nonzero(cpu_array) 
        if nonzero_count_1 == 0:
            print("prediction中全为0")
        else:
            print(f"prediction中有{nonzero_count_1}个非零元素")
        
        print("prediction",prediction.shape)
        return prediction


def read_images(image_fnames: List[str]) ->np.ndarray:
        images = []

        img = Image.open(image_fnames)
        npy_image = np.array(img)
        npy_image = npy_image.transpose((2, 0, 1))
        images.append(npy_image)
        if not all(i.shape == images[0].shape for i in images):
            raise RuntimeError('Not all input images have the same shape!')

        stacked_images = np.vstack(images)

        return stacked_images.astype(np.float32)
    

def check_workers_alive_and_busy(export_pool: Pool, worker_list: List, results_list: List, allowed_num_queued: int = 0):

    alive = [i.is_alive() for i in worker_list]
    if not all(alive):
        raise RuntimeError('Some background workers are no longer alive')

    not_ready = [not i.ready() for i in results_list]
    if sum(not_ready) >= (len(export_pool._pool) + allowed_num_queued):
        return True
    return False

def evaluate_fps_on_dataset(predictor, test_loader, device, num_runs=100):
    predictor.network.eval()
    total_time = 0.0
    timings = []
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for data in test_loader:
            inputs = data.to(device)

            starter.record()
            for _ in range(num_runs):
                predictor.network(inputs)
            ender.record()
            torch.cuda.synchronize()

            elapsed_time = starter.elapsed_time(ender) / num_runs
            timings.append(elapsed_time)
            total_time += elapsed_time

    timings = np.array(timings)
    mean_syn = np.mean(timings)
    std_syn = np.std(timings)
    mean_fps = 1000. / mean_syn


    print(' * Mean@1 {mean_syn:.3f}ms Std@5 {std_syn:.3f}ms FPS@1 {mean_fps:.2f}'.format(mean_syn=mean_syn, std_syn=std_syn, mean_fps=mean_fps))
    return mean_fps


if __name__ == '__main__':
    
    in_files = './dataset/test/images'
    out_files = ',/infer_results'
    
    net = UNet(n_channels=3,n_classes=1,bilinear=True,gt_ds=False)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model')
    logging.info(f'Using device{device}')                                                                  
    net.to(device=device)

    model = './checkpoints/best.pth'
    state_dict = torch.load(model,map_location=device)
    mask_values = state_dict.pop('mask_values',[0,1])
    net.load_state_dict(state_dict,strict=False)
    predictor = Predictor(network=net,tile_step_size=0.5,use_gaussian=True,use_imrroring=False,perform_everything_on_device=True,
                          device=torch.device('cuda',0),verbose=False,verbose_preprocessing=False,allow_tqdm=True)
    predictor.initialize_from_trained_model_folder(
        './dataset/dataset.json',
        use_folds=(0, ),
    )
    logging.info('Model loaded!') 
    

    # test_image_paths = [os.path.join(in_files, filename) for filename in os.listdir(in_files) if filename.endswith('.png')]

    # fps = evaluate_fps_on_dataset(predictor.network, test_image_paths, device, num_runs=1)
    # print(f"FPS: {fps:.2f}")
    
    
    test_image_paths = [os.path.join(in_files, filename) for filename in os.listdir(in_files) if filename.endswith('.png')]

    class TestDataset(torch.utils.data.Dataset):
        def __init__(self, image_paths):
            self.image_paths = image_paths

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            image_path = self.image_paths[idx]
            image = io.imread(image_path)
            image = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((512, 512))
            ])(image)
            return image

    test_dataset = TestDataset(test_image_paths)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    mean_fps = evaluate_fps_on_dataset(predictor, test_loader, device, num_runs=1)
    print(f"FPS: {mean_fps:.2f}")
    
    
    # filename = '000458.png'
    # print("filename",filename)
    # logging.info(f'Predicting image {filename}...')
    # file_path = os.path.join(in_files, filename)
    #     # print("file_path",file_path)
    # img = read_images(file_path) # (512,512,3)
    # img = np.expand_dims(img, axis=0)
    #     # print("Image min and max:", np.min(img), np.max(img))
        
    # predictor.predict_from_files('/home/BI-UNET/guidewire2024/test/images',
    #                              '/home/BI-UNET/infer_results',
    #                              save_probabilities=False, overwrite=True,
    #                              num_processes_preprocessing=2, num_processes_segmentation_export=2,
    #                              num_parts=1, part_id=0)    
        


   

    
    