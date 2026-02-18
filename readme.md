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

python -m hific.evaluate   --config hific   --ckpt_dir ckpts/hific_test   --out_dir evaluation_results/   --local_image_dir ../SUN360/test-10

python -m hific.evaluate --config mselpips_lo mselpips_mi mselpips_hi hific_lo hific_mi hific_hi mselpips_lo mselpips_mi mselpips_hi hific_lo hific_mi hific_hi mselpips_lo mselpips_mi mselpips_hi hific_lo hific_mi hific_hi mselpips_lo mselpips_mi mselpips_hi hific_lo hific_mi hific_hi --ckpt_dir ckpts/mse_lpips_lo_200k ckpts/mse_lpips_mi_200k ckpts/mse_lpips_hi_200k ckpts/hific_mse_lpips_lo_200k ckpts/hific_mse_lpips_mi_200k ckpts/hific_mse_lpips_hi_200k ckpts/mse_ssim_lo_200k ckpts/mse_ssim_mi_200k ckpts/mse_ssim_hi_200k ckpts/hific_mse_ssim_lo_200k ckpts/hific_mse_ssim_mi_200k ckpts/hific_mse_ssim_hi_200k ckpts/WSmse_WSssim_lo_200k ckpts/WSmse_WSssim_mi_200k ckpts/WSmse_WSssim_hi_200k ckpts/hific_WSmse_WSssim_lo_200k ckpts/hific_WSmse_WSssim_mi_200k ckpts/hific_WSmse_WSssim_hi_200k ckpts/gauss_WSmse_WSssim_lo_200k ckpts/gauss_WSmse_WSssim_mi_200k ckpts/gauss_WSmse_WSssim_hi_200k ckpts/gauss_hific_WSmse_WSssim_lo_200k ckpts/gauss_hific_WSmse_WSssim_mi_200k ckpts/gauss_hific_WSmse_WSssim_hi_200k --out_dir results/mselpips_lo results/mselpips_mi results/mselpips_hi results/hificlpips_lo results/hificlpips_mi results/hificlpips_hi results/msessim_lo results/msessim_mi results/msessim_hi results/hificssim_lo results/hificssim_mi results/hificssim_hi results/WSmsessim_lo results/WSmsessim_mi results/WSmsessim_hi results/WShificssim_lo results/WShificssim_mi results/WShificssim_hi results/gaussssim_lo results/gaussssim_mi results/gaussssim_hi results/gausshific_lo results/gausshific_mi results/gausshific_hi --group LPIPS LPIPS LPIPS HiFiCLPIPS HiFiCLPIPS HiFiCLPIPS SSIM SSIM SSIM HiFiCSSIM HiFiCSSIM HiFiCSSIM WSSSIM WSSSIM WSSSIM WSHiFiCSSIM WSHiFiCSSIM WSHiFiCSSIM GAUSS GAUSS GAUSS GAUSSHIFIC GAUSSHIFIC GAUSSHIFIC --local_image_dir ../CTC-360-resized --results_csv results/resultados_finais.csv

python -m hific.evaluate --config mselpips_lo mselpips_mi mselpips_hi --ckpt_dir ckpts/SWHDC_WSmse_WSssim_lo_200k ckpts/SWHDC_WSmse_WSssim_mi_200k ckpts/SWHDC_WSmse_WSssim_hi_200k --out_dir results/SWHDC_lo results/SWHDC_mi results/SWHDC_hi --group SWHDC SWHDC SWHDC --local_image_dir ../CTC-360-resized --results_csv results/resultados_finais_new.csv

python -m hific.evaluate --config mselpips_lo mselpips_mi --ckpt_dir ckpts/WSmse_WSssim_256x512_lo_200k ckpts/WSmse_WSssim_256x512_mi_200k --out_dir results/WSSSSIM_256x512_lo results/WSSSIM_256x512_mi --group 256x512teste 256x512teste --local_image_dir ../CTC-360-resized --results_csv results/resultados_teste_256x512.csv

python plot_results.py results/resultados_finais.csv --output results/graficos_comparativos.png

DOCKER:

Buildar
SÓ USAR ESSE SE QUISER CRIAR NOVA IMAGEM -> docker build -t hific-360-env .

Rodar
docker run --gpus all -it --rm -v "$(pwd)":/app hific-360-env


target low = 0.14
target mid = 0.3
target high = 0.45

./train_all_models.sh

SWHDC2 - WS SSIM, learn = True, crop 256x512

SWHDC3 - WS SSIM, learn = False, crop 256x512

SWHDC4 - WS SSIM, learn = True, crop 256x256


python -m hific.evaluate --config mselpips_mi --ckpt_dir ckpts/SWHDC6_teste_mi_100k --out_dir results/SWHDC6_teste --group SWHDC6 --local_image_dir .
./CTC-360-resized --results_csv results/resultados_swhdc6_teste.csv