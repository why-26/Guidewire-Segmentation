from torchvision import transforms
from utils.utils1 import *

from datetime import datetime

class setting_config:


    network = 'egeunet'
    model_config = {
        'num_classes': 1, 
        'input_channels': 3, 
        'c_list': [8,16,24,32,48,64], 
        'bridge': True,
        'gt_ds': True,
    }

    datasets = 'guidewire2024' 
    if datasets == 'guidewire2024':
        data_path = './data/guidewire2024/'
    else:
        raise Exception('datasets in not right!')

    pretrained_path = './pre_trained/'
    num_classes = 1
    input_size_h = 512
    input_size_w = 512
    input_channels = 3
    distributed = False
    local_rank = -1
    num_workers = 0
    seed = 42
    world_size = None
    rank = None
    amp = False
    gpu_id = '0'
    batch_size = 8
    epochs = 250 

    work_dir = 'results/' + network + '_' + datasets + '_' + datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss') + '/'

    print_interval = 20
    val_interval = 10  
    save_interval = 100
    threshold = 0.5

    train_transformer = transforms.Compose([
        myNormalize(datasets, train=True),
        myToTensor(),
        myRandomHorizontalFlip(p=0.5),
        myRandomVerticalFlip(p=0.5),
        myRandomRotation(p=0.5, degree=[0, 360]),
        myResize(input_size_h, input_size_w)
    ])
    test_transformer = transforms.Compose([
        myNormalize(datasets, train=False),
        myToTensor(),
        myResize(input_size_h, input_size_w)
    ])

    opt = 'AdamW'