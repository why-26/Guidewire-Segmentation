import os
from typing import List

import numpy as np
import shutil

from batchgenerators.utilities.file_and_folder_operations import join, load_pickle, isfile

def get_case_identifiers(folder: str) -> List[str]:
    """
    finds all npz files in the given folder and reconstructs the training case names from them
    """
    case_identifiers = [i[:-4] for i in os.listdir(folder) if i.endswith("npz") and (i.find("segFromPrevStage") == -1)]
    return case_identifiers

class GUidewireDataset(object):
    def __init__(self,folder:str,case_indentifiers:List[str] = None):
        super().__init__()
        if case_indentifiers is None:
            case_indentifiers = get_case_identifiers(folder)
        case_indentifiers.sort()
        self.dataset = {}
        for c in case_indentifiers:
            self.dataset[c] = {}
            self.dataset[c]['data_file'] = join(folder,f"{c}.npz")
            self.dataset[c]['properties_file'] = join(folder,f"{c}.pkl")
        self.keep_files_open = False
        
    def __getitem__(self,key):
        ret = {**self.dataset[key]}
        if 'properties' not in ret.keys():
            ret['properties'] = load_pickle(ret['properties_file'])
        return ret
    
    def keys(self):
        return self.dataset.keys()
    
    def __len__(self):
        return self.dataset.__len__()
    
    def items(self):
        return self.dataset.items()
    
    def values(self):
        return self.dataset.values()
    
    def load_case(self,key):
        entry = self[key]
        if 'open_data_file' in entry.keys():
            data = entry['open_data_file']
        elif isfile(entry['data_file'][:-4] + ".npy"):
            data = np.load(entry['data_file'][:-4] + ".npy", 'r')
            if self.keep_files_open:
                self.dataset[key]['open_data_file'] = data
        else:
            data = np.load(entry['data_file'])['data']

        if 'open_seg_file' in entry.keys():
            seg = entry['open_seg_file']
        elif isfile(entry['data_file'][:-4] + "_seg.npy"):
            seg = np.load(entry['data_file'][:-4] + "_seg.npy", 'r')
            if self.keep_files_open:
                self.dataset[key]['open_seg_file'] = seg
        else:
            seg = np.load(entry['data_file'])['seg']
        
        return data,seg,entry['properties']


