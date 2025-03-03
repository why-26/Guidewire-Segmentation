
import multiprocessing
import queue
from torch.multiprocessing import Event, Process, Queue, Manager

from time import sleep
from typing import Union, List

import numpy as np
import torch
from batchgenerators.dataloading.data_loader import DataLoader
from utils.dataloading.guidewire_dataloader import GuidewireDataLoader
from preprocess.preprocessing.processor import Guidewire_Preprocessor

def convert_labelmap_to_one_hot(segmentation: Union[np.ndarray, torch.Tensor],
                                all_labels: Union[List, torch.Tensor, np.ndarray, tuple],
                                output_dtype=None) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(segmentation, torch.Tensor):
        result = torch.zeros((len(all_labels), *segmentation.shape),
                             dtype=output_dtype if output_dtype is not None else torch.uint8,
                             device=segmentation.device)
        result.scatter_(0, segmentation[None].long(), 1) 
    else:
        result = np.zeros((len(all_labels), *segmentation.shape),
                          dtype=output_dtype if output_dtype is not None else np.uint8)
        for i, l in enumerate(all_labels):
            result[i] = segmentation == l
    return result

def preprocess_fromnpy_save_to_queue(list_of_images: List[np.ndarray],
                                     list_of_image_properties: List[dict],
                                     truncated_ofnames: Union[List[str], None],
                                     dataset_json: dict,
                                     target_queue: Queue,
                                     done_event: Event,
                                     abort_event: Event,
                                     verbose: bool = False):
    try:
        preprocessor = Guidewire_Preprocessor(verbose=verbose)
        for idx in range(len(list_of_images)):
            data, seg = preprocessor.run_case_npy(list_of_images[idx],
                                                  list_of_image_properties[idx],
                                                  dataset_json)
            data = torch.from_numpy(data).contiguous().float()
            item = {'data': data, 'data_properties': list_of_image_properties[idx],
                    'ofile': truncated_ofnames[idx] if truncated_ofnames is not None else None}
            success = False
            while not success:
                try:
                    if abort_event.is_set():
                        return
                    target_queue.put(item, timeout=0.01)
                    success = True
                except queue.Full:
                    pass
        done_event.set()
    except Exception as e:
        abort_event.set()
        raise e

def preprocess_fromfiles_save_to_queue(list_of_lists: List[List[str]],
                                       output_filenames_truncated: Union[None, List[str]],
                                       dataset_json: dict,
                                       target_queue: Queue,
                                       done_event: Event,
                                       abort_event: Event,
                                       verbose: bool = False):
    try:
        preprocessor = Guidewire_Preprocessor(verbose=verbose)
        for idx in range(len(list_of_lists)):
            print("list_of_lists",list_of_lists[idx])
            image_paths = ",".join(list_of_lists[idx]) 
            print("image_paths",image_paths)
            data, seg, data_properties = preprocessor.run_case(image_paths,
                                                               None,
                                                               dataset_json)

            data = torch.from_numpy(data).contiguous().float()
            print("2222222_data.shape",data.shape)

            item = {'data': data, 'data_properties': data_properties,
                    'ofile': output_filenames_truncated[idx] if output_filenames_truncated is not None else None}
            success = False
            while not success:
                try:
                    if abort_event.is_set():
                        return
                    target_queue.put(item, timeout=0.01)
                    success = True
                except queue.Full:
                    pass
        done_event.set()
    except Exception as e:
        abort_event.set()
        raise e

def preprocessing_iterator_fromfiles(list_of_lists: List[List[str]],
                                     output_filenames_truncated: Union[None, List[str]],
                                     dataset_json: dict,
                                     num_processes: int,
                                     pin_memory: bool = False,
                                     verbose: bool = False):
    context = multiprocessing.get_context('spawn')
    manager = Manager()
    num_processes = 3
    num_processes = min(len(list_of_lists), num_processes)
    assert num_processes >= 1
    processes = []
    done_events = []
    target_queues = []
    abort_event = manager.Event()
    for i in range(num_processes):
        print("num_processes",num_processes)
        event = manager.Event()
        queue = Manager().Queue(maxsize=1)
        pr = context.Process(target=preprocess_fromfiles_save_to_queue,
                     args=(
                         list_of_lists[i::num_processes],
                         output_filenames_truncated[
                         i::num_processes] if output_filenames_truncated is not None else None,
                         dataset_json,
                         queue,
                         event,
                         abort_event,
                         verbose
                     ), daemon=True)
        pr.start()
        target_queues.append(queue)
        done_events.append(event)
        processes.append(pr)
    worker_ctr = 0
    while (not done_events[worker_ctr].is_set()) or (not target_queues[worker_ctr].empty()):
        if not target_queues[worker_ctr].empty():
            item = target_queues[worker_ctr].get()
            worker_ctr = (worker_ctr + 1) % num_processes
        else:
            all_ok = all(
                [i.is_alive() or j.is_set() for i, j in zip(processes, done_events)]) and not abort_event.is_set()
            if not all_ok:
                raise RuntimeError('Background workers died. Look for the error message further up! If there is '
                                   'none then your RAM was full and the worker was killed by the OS. Use fewer '
                                   'workers or get more RAM in that case!')
            sleep(0.01)
            continue
        if pin_memory:
            [i.pin_memory() for i in item.values() if isinstance(i, torch.Tensor)]
        yield item
    [p.join() for p in processes]




def preprocessing_iterator_fromnpy(list_of_images: List[np.ndarray],
                                   list_of_image_properties: List[dict],
                                   truncated_ofnames: Union[List[str], None],
                                   dataset_json: dict,
                                   num_processes: int,
                                   pin_memory: bool = False,
                                   verbose: bool = False):
    context = multiprocessing.get_context('spawn')
    manager = Manager()
    num_processes = min(len(list_of_images), num_processes)
    assert num_processes >= 1
    target_queues = []
    processes = []
    done_events = []
    abort_event = manager.Event()
    for i in range(num_processes):
        
        event = manager.Event()
        queue = manager.Queue(maxsize=1)
        # 
        pr = context.Process(target=preprocess_fromnpy_save_to_queue,
                     args=(
                         list_of_images[i::num_processes],
                         list_of_image_properties[i::num_processes],
                         truncated_ofnames[i::num_processes] if truncated_ofnames is not None else None,
                         dataset_json,
                         queue,
                         event,
                         abort_event,
                         verbose
                     ), daemon=True)
        pr.start()
        done_events.append(event)
        processes.append(pr)
        target_queues.append(queue)
    worker_ctr = 0
    while (not done_events[worker_ctr].is_set()) or (not target_queues[worker_ctr].empty()):
        if not target_queues[worker_ctr].empty():
            item = target_queues[worker_ctr].get()
            worker_ctr = (worker_ctr + 1) % num_processes
        else:
            all_ok = all(
                [i.is_alive() or j.is_set() for i, j in zip(processes, done_events)]) and not abort_event.is_set()
            if not all_ok:
                raise RuntimeError('Background workers died. Look for the error message further up! If there is '
                                   'none then your RAM was full and the worker was killed by the OS. Use fewer '
                                   'workers or get more RAM in that case!')
            sleep(0.01)
            continue
        if pin_memory:
            [i.pin_memory() for i in item.values() if isinstance(i, torch.Tensor)]
        yield item
    [p.join() for p in processes]


