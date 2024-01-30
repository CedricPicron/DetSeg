#!/bin/bash

# Set environment variables
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5 8.0 8.6 8.9 9.0+PTX"

# Create and activate new conda environment
conda create --name pytorch1.13
conda activate pytorch1.13

# Intall using conda
conda install pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pytorch-scatter -c pyg
conda install -c conda-forge pyparsing
conda install -c conda-forge shapely
conda install -c conda-forge tqdm
conda install scipy

# Deactivate and reactivate environment such that correct python path is used
conda deactivate
conda activate pytorch1.13

# Install MMDetection
pip install -U openmim
mim install mmengine
mim install mmcv
python -m pip install git+https://github.com/open-mmlab/mmdetection.git

# Install using pip
python -m pip install tidecv
python -m pip install timm

python -m pip install git+https://github.com/CedricPicron/boundary-iou-api.git
python -m pip install git+https://github.com/CedricPicron/Deformable-DETR.git
python -m pip install git+https://github.com/facebookresearch/detectron2.git

pip uninstall panopticapi
python -m pip install git+https://github.com/CedricPicron/panopticapi.git

# Install local extensions
cd models/extensions/deformable
./make.sh
cd -
