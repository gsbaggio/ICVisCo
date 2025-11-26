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


primeiro passo:

python -m hific.train --config mselpips_lo --ckpt_dir ckpts/mse_lpips_lo_200k --num_steps 200k --local_image_dir ../SUN360/train
            
segundo passo:            

python -m hific.train --config hific_lo --ckpt_dir ckpts/hific_mse_lpips_lo_200k --init_autoencoder_from_ckpt_dir ckpts/mse_lpips_lo_200k --num_steps 200k --local_image_dir ../SUN360/train
            

avaliar o modelo:

python -m hific.evaluate   --config hific   --ckpt_dir ckpts/hific_test   --out_dir evaluation_results/   --local_image_dir ../SUN360/test


DOCKER:

Buildar
SÓ USAR ESSE SE QUISER CRIAR NOVA IMAGEM -> docker build -t hific-360-env .

Rodar
docker run --gpus all -it --rm -v "$(pwd)":/app hific-360-env


target low = 0.14
target mid = 0.3
target high = 0.45

./train_all_models.sh