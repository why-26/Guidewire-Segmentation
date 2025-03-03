# FGA-Net: Lightweight Attention Network for Guidewire Segmentation in Fluoroscopic Images

![Guidewire Segmentation Demo](./segmentation_demo/) 

A lightweight attention-based CCN Network for guidewire segmentation in clinical fluoroscopic images. Designed for vascular interventional surgery applications.

## Key Features
- 🧩 5-fold cross-validation training strategy
- ⚡ Mixed-precision training (AMP)
- 🪟 Sliding window inference with Gaussian-weighted fusion
- 🏥 Clinical-optimized preprocessing pipeline


## Quick Start

### Prerequisites
- NVIDIA GPU with CUDA 11.8+
- Conda package manager

### Installation
```bash
conda env create -f environment.yml
pip install -r requirements.txt
conda activate fga-net
```
## Description
This model was trained from scratch with 1936 images and scored a [Dice coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient) of 0.988423 on over 903 test images.


It can be easily used for guidewire segmentation,  medical segmentation, ...


## Usage
**Note : Use Python 3.11 or newer**

### Training
python train.py \
    --epochs 180 \
    --batch-size 8 \
    --learning-rate 1e-6 \
    --amp  # Enable automatic mixed precision

### Inference
python inference/test.py \
    --input data/test_images \
    --output results \
    --model checkpoints/model_best.pth

## Project Structure
├── configs/                  # Training configurations
│   └── config.py             # Main configuration file
├── data_augmentation/        # Augmentation strategies
├── inference/                # Prediction modules
│   ├── sliding_window_prediction.py  # Sliding window 
│   └── test.py               # Prediction Result
├── preprocess/               # Data preprocessing
│   ├── cropping/             # Non-zero region cropping
│   └── normalization/        # Intensity normalization
├── models/                   # Modified U-Net implementation
│   ├── model.py/             # Model
│   └── model_parts.py        # Network components
├── train.py                  # Main training script
└── environment.yml           # Conda environment specification


## Data
├── train/               
│   ├── images/  
│   └── masks/              
├── test/              
│   ├── images/             
│   └── masks/        
├── dataset.json                     


## Weights & Biases
We use [Weights & Biases](https://wandb.ai/). to track and visualize the training process, including metrics, model performance, and system resources. W&B provides a powerful dashboard for monitoring experiments, comparing runs, and sharing results.

