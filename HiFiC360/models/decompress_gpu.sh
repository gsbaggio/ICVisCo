#!/bin/bash

# Comando simplificado para descompressão com GPU
# Uso: ./decompress_gpu.sh [pasta_base] [pasta_modelo] [arquivo_ou_none]
# Exemplo: ./decompress_gpu.sh files/hific hific-mi none

BASE_FOLDER=${1:-files/hific}
MODEL_FOLDER=${2:-hific-mi}
INPUT_FILE=${3:-none}

# Ativar ambiente e configurar GPU
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hific

# Configurar CUDA
export CUDA_ROOT=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CUDA_HOME=$CONDA_PREFIX
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Navegar para o diretório correto
cd /home/gabrielbaggio/Documentos/Trabalhos/ICVisCo/HiFiC360/models

# Executar descompressão
echo "Descomprimindo com GPU: Pasta=$BASE_FOLDER, Modelo=$MODEL_FOLDER, Entrada=$INPUT_FILE"
python tfci.py decompress $INPUT_FILE --base_folder $BASE_FOLDER --model_folder $MODEL_FOLDER