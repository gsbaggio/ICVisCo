#!/bin/bash

# Script para treinar HiFiC em RTX 3090
# Configurações especiais para compatibilidade com GPUs Ampere

# Configurar CUDA
export CUDA_ROOT=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CUDA_HOME=$CONDA_PREFIX
export TF_FORCE_GPU_ALLOW_GROWTH=true

cd /home/gabrielbaggio/Documentos/Trabalhos/ICVisCo/HiFiC360/models

python -m hific.train \
  --config mselpips \
  --ckpt_dir ckpts/mse_lpips_teste \
  --num_steps 1M \
  --local_image_dir ../../../../../tensorflow_datasets/SUN360 \
  --batch_size 4 \
  --crop_size 256 \
  "$@"
