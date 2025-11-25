#!/bin/bash
# filepath: /home/gabrielbaggio/Documentos/Trabalhos/ICVisCo/train_all_models.sh

# # MSE+LPIPS Low
# echo "Training mselpips_lo..."
# python -m hific.train --config mselpips_lo --ckpt_dir ckpts/mse_lpips_lo_200k --num_steps 200000 --local_image_dir ../SUN360/train

# # HiFiC Low
# echo "Training hific_lo..."
# python -m hific.train --config hific_lo --ckpt_dir ckpts/hific_mse_lpips_lo_200k --init_autoencoder_from_ckpt_dir ckpts/mse_lpips_lo_200k --num_steps 200000 --local_image_dir ../SUN360/train

# # MSE+LPIPS Mid
# echo "Training mselpips_mi..."
# python -m hific.train --config mselpips_mi --ckpt_dir ckpts/mse_lpips_mi_200k --num_steps 200000 --local_image_dir ../SUN360/train

# # HiFiC Mid
# echo "Training hific_mi..."
# python -m hific.train --config hific_mi --ckpt_dir ckpts/hific_mse_lpips_mi_200k --init_autoencoder_from_ckpt_dir ckpts/mse_lpips_mi_200k --num_steps 200000 --local_image_dir ../SUN360/train

# MSE+LPIPS High
echo "Training mselpips_hi..."
python -m hific.train --config mselpips_hi --ckpt_dir ckpts/mse_lpips_hi_200k --num_steps 200000 --local_image_dir ../SUN360/train

# HiFiC High
echo "Training hific_hi..."
python -m hific.train --config hific_hi --ckpt_dir ckpts/hific_mse_lpips_hi_200k --init_autoencoder_from_ckpt_dir ckpts/mse_lpips_hi_200k --num_steps 200000 --local_image_dir ../SUN360/train

echo "All models trained successfully!"