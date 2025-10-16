Alguns programas que quero testar para compressão de imagens

## COMPRESSÃO 

# Comprimir todas as imagens em files/hific/hific-mi/original/ (GPU)

export CUDA_ROOT=$CONDA_PREFIX && export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH && export CUDA_HOME=$CONDA_PREFIX && export TF_FORCE_GPU_ALLOW_GROWTH=true && cd /home/gabrielbaggio/Documentos/Trabalhos/ICVisCo/HiFiC360/models && python tfci.py compress hific-mi none --base_folder files/hific

ou

./compress_gpu.sh hific-mi files/hific none

# Comprimir todas as imagens em files/hific/hific-mi/original/ (CPU)
python tfci.py compress hific-mi none --base_folder files/hific

# Comprimir arquivo específico (CPU)
python tfci.py compress hific-mi imagem.png --base_folder files/hific



## DECOMPRESSÃO 

./decompress_gpu.sh files/hific hific-mi none

# Descomprimir especificando modelo
python tfci.py decompress none --base_folder files/hific --model_folder hific-mi

# Descompressão com auto-detecção de modelo
python tfci.py decompress none --base_folder files/hific

# Descomprimir arquivo específico
python tfci.py decompress arquivo.tfci --base_folder files/hific --model_folder hific-mi



## ANÁLISE COMPRESSÃO



# LPIPS 360 com peso coseno
python compression_analysis.py --base_dir files --methods hific --metrics psnr lpips lpips360 --force_cpu --lpips360_weight_type cosine --lpips360_pole_weight 0.3

# Comparação LPIPS vs LPIPS 360 peso linear
python compression_analysis.py --base_dir files --methods hific --metrics psnr lpips lpips360 --force_cpu --lpips360_weight_type linear --lpips360_pole_weight 0.5

# LPIPS 360 com peso quadrático
python compression_analysis.py --base_dir files --methods hific --metrics lpips360 --force_cpu --lpips360_weight_type quadratic --lpips360_pole_weight 0.8