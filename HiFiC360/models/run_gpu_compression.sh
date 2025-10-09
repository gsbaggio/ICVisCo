#!/bin/bash

# Script para executar compressão com GPU habilitada
# Configura variáveis de ambiente necessárias para TensorFlow usar GPU

# Ativar ambiente conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hific

# Configurar variáveis de ambiente para CUDA
export CUDA_ROOT=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH

# Configurações TensorFlow para melhor uso da GPU
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR=cuda_malloc_async

# Mostrar informações da GPU antes de iniciar
echo "=== Informações da GPU ==="
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv

echo ""
echo "=== Verificando TensorFlow GPU ==="
python -c "import tensorflow as tf; print('GPU Available:', tf.config.list_physical_devices('GPU')); print('TensorFlow version:', tf.__version__)"

echo ""
echo "=== Iniciando compressão ==="
echo "Monitorando GPU durante a execução..."
echo "Para monitorar em tempo real, execute em outro terminal: watch -n 1 nvidia-smi"

# Navegar para o diretório correto
cd /home/gabrielbaggio/Documentos/Trabalhos/ICVisCo/HiFiC360/models

# Executar o comando de compressão
python tfci.py compress hific-mi none --base_folder files/hific

echo ""
echo "=== Compressão finalizada ==="
echo "Estado final da GPU:"
nvidia-smi