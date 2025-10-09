#!/bin/bash

# Comando simplificado para compressão com GPU
# Uso: ./compress_gpu.sh [modelo] [pasta_base] [imagem_ou_none]
# Exemplo: ./compress_gpu.sh hific-mi files/hific none

MODEL=${1:-hific-mi}
BASE_FOLDER=${2:-files/hific}
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

# Executar compressão
echo "Comprimindo com GPU: Modelo=$MODEL, Pasta=$BASE_FOLDER, Entrada=$INPUT_FILE"
python tfci.py compress $MODEL $INPUT_FILE --base_folder $BASE_FOLDER