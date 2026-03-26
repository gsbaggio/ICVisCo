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

python -m hific.evaluate --config mselpips_lo mselpips_mi mselpips_hi --ckpt_dir ckpts/SWHDC_WSmse_WSssim_256x512_lo_200k ckpts/SWHDC_WSmse_WSssim_256x512_mi_200k ckpts/SWHDC_WSmse_WSssim_256x512_hi_200k --out_dir results/SWHDC_256x512_lo results/SWHDC_256x512_mi results/SWHDC_256x512_hi --group SWHDC256x512 SWHDC256x512 SWHDC256x512 --local_image_dir ../CTC-360-resized --results_csv results/resultados_SWHDC_256x512.csv

python -m hific.evaluate --config mselpips_lo mselpips_mi mselpips_hi --ckpt_dir ckpts/mse_ssim_256x512_lo_200k ckpts/mse_ssim_256x512_mi_200k ckpts/mse_ssim_256x512_hi_200k --out_dir results/SSIM_256x512_lo results/SSIM_256x512_mi results/SSIM_256x512_hi --group SSIM256x512 SSIM256x512 SSIM256x512 --local_image_dir ../CTC-360-resized --results_csv results/resultados_finais_256x512.csv

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

<CONTEXT>

I am working on a project where I utilized a pre-existing model for image compression called HiFIC (High-Fidelity Generative Image Compression). I performed several modifications and tests to adapt it for equirectangular image compression within the context of omnidirectional (360-degree) images.

Specifically, I made the following changes:

* Loss Function: I transitioned from MSE + LPIPS to MSE + SSIM and WS-MSE + WS-SSIM. I replaced LPIPS because it is trained on perspective images rather than equirectangular images, which contain specific distortions.

* Ablation Study: I first tested MSE + SSIM to evaluate changes incrementally. Subsequently, since our focus was maintaining higher quality at the center (the area of greatest importance), I experimented with latitude-weighted loss functions using WS-MSE + WS-SSIM. Then, I realized other changes, like Crop Extraction and  Convolutions, described below:

* Crop Extraction: I modified the training crop extraction (originally 256x256). I implemented a Normal (Gaussian) Distribution to sample more frequently from the center of the image. However, this specific experiment did not yield significant results. I also tested with 256x512 crop sizes.

* Convolutions: I replaced the standard convolutions described in the HiFIC paper with SWHDC (Sphere-Wrapped Horizontal Deformable Convolution) and SWHDC with Learnable Weights. While using these different convolutions, I used the WS-MSE + WS-SSIM loss functions.

* Input/Output Dimensions: When using SWHDC and its learnable variant, the results improved when using 256x512 crops. Since the training images are 1024x512, this size covers the full height of the image. Performance remained largely unchanged with 256x256 crops.

Finally, I analyzed the results by comparing input and reconstructed images across six metrics: PSNR, SSIM, and MSE, as well as the more critical metrics for equirectangular content: WS-PSNR, WS-SSIM, and WS-MSE (focusing on the center of the images, since its more important on equiretangular images).

*Note: I also have a graph of all the results, I will include it on the paper.

</CONTEXT>

<CODE>

<archs_original.py>

# Copyright 2020 Google LLC. All Rights Reserved.

#

# Licensed under the Apache License, Version 2.0 (the "License");

# you may not use this file except in compliance with the License.

# You may obtain a copy of the License at

#

#     http://www.apache.org/licenses/LICENSE-2.0

#

# Unless required by applicable law or agreed to in writing, software

# distributed under the License is distributed on an "AS IS" BASIS,

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# See the License for the specific language governing permissions and

# limitations under the License.

# ==============================================================================

"""Training code for HiFiC."""

import argparse

import sys

import os

import tensorflow.compat.v1 as tf

from hific import configs

from hific import helpers

from hific import model

# Show custom tf.logging calls.

tf.logging.set_verbosity(tf.logging.INFO)

# Configurações para compatibilidade com RTX 3090

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

os.environ['TF_ENABLE_CUBLAS_TENSOR_OP_MATH_FP32'] = '1'

SAVE_CHECKPOINT_STEPS = 1000

def train(config_name, ckpt_dir, num_steps: int, auto_encoder_ckpt_dir,

          batch_size, crop_size, lpips_weight_path, create_image_summaries,

          tfds_arguments: model.TFDSArguments, local_image_dir=None):

  """Train the model."""

  config = configs.get_config(config_name)

  hific = model.HiFiC(config, helpers.ModelMode.TRAINING, lpips_weight_path,

                      auto_encoder_ckpt_dir, create_image_summaries)import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''

Se local_image_dir for especificado, usar imagens locais ao invés de TFDS

if local_image_dir:

# Suporta PNG, JPG e JPEG

import glob as glob_module

png_images = glob_module.glob(f"{local_image_dir}/.png")

jpg_images = glob_module.glob(f"{local_image_dir}/.jpg")

jpeg_images = glob_module.glob(f"{local_image_dir}/*.jpeg")

all_images = png_images + jpg_images + jpeg_images


if not all_images:

  raise ValueError(f'No images found in {local_image_dir}. '

                  'Make sure there are *.png, *.jpg, or *.jpeg files.')


tf.logging.info(f'Found {len(all_images)} images in {local_image_dir}')

tf.logging.info(f'  PNG: {len(png_images)}, JPG: {len(jpg_images)}, '

               f'JPEG: {len(jpeg_images)}')


# Usar padrão que pega todos os formatos

images_glob = f"{local_image_dir}/*.[pjpJ][npNP][gGgE]*"

dataset = hific.build_input(batch_size, crop_size,

                            images_glob=images_glob)

else:

dataset = hific.build_input(batch_size, crop_size,

tfds_arguments=tfds_arguments)

iterator = tf.data.make_one_shot_iterator(dataset)

get_next = iterator.get_next()

hific.build_model(**get_next)

train_op = hific.train_op

hooks = hific.hooks + [tf.train.StopAtStepHook(last_step=num_steps)]

global_step = tf.train.get_or_create_global_step()

tf.logging.info(f'\nStarting MonitoredTrainingSession at {ckpt_dir}\n')


Configuração de sessão para RTX 3090

session_config = tf.ConfigProto()

session_config.gpu_options.allow_growth = True

session_config.gpu_options.per_process_gpu_memory_fraction = 0.95


Força o uso de operações mais compatíveis

session_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.OFF

with tf.train.MonitoredTrainingSession(

checkpoint_dir=ckpt_dir,

save_checkpoint_steps=SAVE_CHECKPOINT_STEPS,

hooks=hooks,

config=session_config) as sess:

if auto_encoder_ckpt_dir:

hific.restore_autoencoder(sess)

tf.logging.info('Session setup, starting training...')

while True:

if sess.should_stop():

break

global_step_np, _ = sess.run([global_step, train_op])

if global_step_np == 0:

tf.logging.info('First iteration passed.')

if global_step_np > 1 and global_step_np % 100 == 1:

tf.logging.info(f'Iteration {global_step_np}')

tf.logging.info('Training session closed.')

def parse_args(argv):

"""Parses command line arguments."""

parser = argparse.ArgumentParser(

formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--config', required=True,

choices=configs.valid_configs(),

help='The config to use.')

parser.add_argument('--ckpt_dir', required=True,

help=('Path to the folder where checkpoints should be '

'stored. Passing the same folder twice will resume '

'training.'))

parser.add_argument('--num_steps', default='1M',

help=('Number of steps to train for. Supports M and k '

'postfix for "million" and "thousand", resp.'))

parser.add_argument('--init_autoencoder_from_ckpt_dir',

metavar='AUTOENC_CKPT_DIR',

help=('If given, restore encoder, decoder, and '

'probability model from the latest checkpoint in '

'AUTOENC_CKPT_DIR. See README.md.'))

parser.add_argument('--batch_size', type=int, default=8,

help='Batch size for training.')

parser.add_argument('--crop_size', type=int, default=256,

help='Crop size for input pipeline (square crop).')

parser.add_argument('--crop_height', type=int, default=None,

help='Crop height (overrides crop_size if used with crop_width).')

parser.add_argument('--crop_width', type=int, default=None,

help='Crop width (overrides crop_size if used with crop_height).')

parser.add_argument('--lpips_weight_path',

help=('Where to store the LPIPS weights. Defaults to '

'current directory'))

parser.add_argument('--local_image_dir',

help=('Path to local directory containing training images '

'(*.png, *.jpg, *.jpeg). If provided, this will be used instead of '

'TFDS dataset. Example: tensorflow_datasets/my_images'))

helpers.add_tfds_arguments(parser)

parser.add_argument(

'--no-image-summaries',

dest='image_summaries',

action='store_false',

help='Disable image summaries.')

parser.set_defaults(image_summaries=True)

args = parser.parse_args(argv[1:])

if args.ckpt_dir == args.init_autoencoder_from_ckpt_dir:

raise ValueError('--init_autoencoder_from_ckpt_dir should not point to '

'the same folder as --ckpt_dir. If you simply want to '

'continue training the model in --ckpt_dir, you do not '

'have to pass --init_autoencoder_from_ckpt_dir, as '

'continuing training is the default.')

args.num_steps = _parse_num_steps(args.num_steps)

return args

def _parse_num_steps(steps):

try:

return int(steps)

except ValueError:

pass

if steps.endswith('M'):

return int(steps[:-1]) * 1000000

if steps.endswith('k'):

return int(steps[:-1]) * 1000

raise ValueError('Invalid num_steps value: {steps}')

def main(args):

crop_size = args.crop_size

if args.crop_height and args.crop_width:

crop_size = (args.crop_height, args.crop_width)

train(args.config, args.ckpt_dir, args.num_steps,

args.init_autoencoder_from_ckpt_dir, args.batch_size, crop_size,

args.lpips_weight_path, args.image_summaries,

helpers.parse_tfds_arguments(args), args.local_image_dir)

if name == 'main':

main(parse_args(sys.argv))

<\archs_original.py>

<archs_SWHDC.py>


Copyright 2020 Google LLC. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");

you may not use this file except in compliance with the License.

You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software

distributed under the License is distributed on an "AS IS" BASIS,

WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

See the License for the specific language governing permissions and

limitations under the License.

==============================================================================

"""Implement the components needed for HiFiC.

For more details, see the paper: https://arxiv.org/abs/2006.09965

The default values for all constructors reflect what was used in the paper.

"""

import collections

import math

from compare_gan.architectures import abstract_arch

from compare_gan.architectures import arch_ops

import numpy as np

import tensorflow.compat.v1 as tf

import tensorflow_compression as tfc

from hific.helpers import ModelMode

class SWHDCtf(tf.keras.layers.Layer):

"""

Spherical Weighted Hybrid Dilated Convolution

TensorFlow / Keras implementation

"""

def init(self, filters, kernel_size, dilations=[1, 2, 3], padding="same", strides=1, learn_weights=True, **kwargs):

super().init(**kwargs)

self.filters = filters

self.kernel_size = kernel_size

self.dilations = dilations

self.N = len(dilations)

self.learn_weights = learn_weights


    # Se 1, os pesos sao aprendidos. Se 0, usa o calculo geometrico SWHDC.

    # Voce pode mudar isso para True por padrao se quiser sempre aprender.

    

def build(self, input_shape):

    in_channels = input_shape[-1]

    kh, kw = self.kernel_size, self.kernel_size

    self.kernel = self.add_weight(

        name="kernel",

        shape=(kh, kw, in_channels, self.filters),

        trainable=True,

    )

    self.bias = self.add_weight(

        name="bias",

        shape=(self.filters,),

        trainable=True,

    )

    if self.learn_weights:

        # Perfil de pesos latentes com resolucao fixa (ex: 128 pontos latitudinais)

        # Shape: [1, 128, N] -> Onde N é o número de dilatações

        self.learned_profile = self.add_weight(

            name="learned_weights_profile",

            shape=(1, 128, self.N),

            initializer=tf.initializers.random_normal(mean=0.0, stddev=0.1),

            trainable=True

        )

    super().build(input_shape)

def circular_pad_width(self, x, pad):

    """

    Circular padding along width (axis=2 for NHWC).

    """

    if pad == 0:

        return x

    left = x[:, :, -pad:, :]

    right = x[:, :, :pad, :]

    return tf.concat([left, x, right], axis=2)   

def call(self, x):

    def reflect_or_symmetric_pad_height(x, v_pad, H):

        # Usar SYMMETRIC padding (repetir a borda) é mais estável nos polos 

        # do que REFLECT ou Zeros para imagens 360, evitando descontinuidades.

        return tf.pad(

              x,

              paddings=[[0, 0], [v_pad, v_pad], [0, 0], [0, 0]],

              mode="SYMMETRIC",

        )

    """

    x: [B, H, W, C] (NHWC)

    """

    B, H, W, C = tf.unstack(tf.shape(x))

    

    row_wise_weights = None

    

    # Calcular H float para usos matemáticos

    H_float = tf.cast(H, tf.float32)

    if self.learn_weights:

        # === Modo Aprendido ===

        # Interpolar o perfil aprendido (128) para a altura atual (H)

        # learned_profile: [1, 128, N]

        

        # Precisamos expandir dimensão para usar resize image: [Batch, H, W, Ch]

        # Vamos tratar: Batch=1, H=128, W=N, C=1 para redimensionar a altura

        # Mas tf.image.resize redimensiona H e W.

        

        # Estratégia: [1, 128, N] -> expand -> [1, 128, N, 1]

        profile_expanded = tf.expand_dims(self.learned_profile, axis=-1)

        

        # Resize para [1, H, N, 1]

        profile_resized = tf.image.resize(

            profile_expanded, 

            size=[H, self.N],

            method=tf.image.ResizeMethod.BILINEAR

        )

        

        # Remove dimensões extras -> [H, N]

        # O profile_resized terá valores arbitrarios. Aplicamos Softmax

        # ao longo da dimensão das dilatações (axis=1 agora após reshape ou axis=2 antes)

        

        w_interpolated = tf.reshape(profile_resized, [H, self.N]) # [H, N]

        

        # Softmax garante que a soma dos pesos das dilatações seja 1

        row_wise_weights = tf.nn.softmax(w_interpolated, axis=-1) # [H, N]

        

        # Transpor para ficar [N, H] como o código original espera

        row_wise_weights = tf.transpose(row_wise_weights, [1, 0]) # [N, H]

    else:

        # === Modo Calculado (SWHDC Original) ===

        # ===== Compute Rs =====

        # Usar centro dos pixels evita singularidade exata em 0 e pi

        # phi vai de (0.5/H)*pi ate (1 - 0.5/H)*pi

        step = math.pi / H_float

        start = 0.5 * step

        phi = tf.linspace(start, math.pi - start, H)

        

        eps = tf.keras.backend.epsilon()

        # Com pixel centers, sin(phi) nunca é 0, mas mantemos eps por segurança

        Rs = tf.minimum(

            tf.cast(self.N, tf.float32),

            tf.abs(1.0 / (tf.sin(phi) + eps))

        )  # [H]

        Rs = tf.expand_dims(Rs, axis=0)  # [1, H]

        dilations_tensor = tf.cast(

            tf.reshape(self.dilations, [self.N, 1]), tf.float32

        )  # [N, 1]

        cR = tf.math.ceil(Rs)

        fR = tf.math.floor(Rs)

        mask_exact = tf.equal(dilations_tensor, Rs)

        mask_floor = tf.logical_and(tf.equal(dilations_tensor, fR), tf.logical_not(mask_exact))

        mask_ceil = tf.logical_and(

            tf.equal(dilations_tensor, cR),

            tf.logical_not(tf.logical_or(mask_exact, mask_floor))

        )

        row_wise_weights = tf.zeros((self.N, H), dtype=tf.float32)

        

        # Broadcast values to match [N, H] for tf.where

        diff_cR_Rs = tf.tile(cR - Rs, [self.N, 1])

        diff_Rs_fR = tf.tile(Rs - fR, [self.N, 1])

        row_wise_weights = tf.where(mask_exact, tf.ones_like(row_wise_weights), row_wise_weights)

        row_wise_weights = tf.where(mask_floor, diff_cR_Rs, row_wise_weights)

        row_wise_weights = tf.where(mask_ceil, diff_Rs_fR, row_wise_weights)

    outputs = []

    # ===== Loop over dilations =====

    for idx, dilation_rate in enumerate(self.dilations):

        # Padding sizes

        v_pad = (self.kernel_size - 1) // 2

        h_pad = dilation_rate * (self.kernel_size - 1) // 2

        x2 = self.circular_pad_width(x, h_pad)

        x2 = reflect_or_symmetric_pad_height(x2, v_pad, H)

        # Dilated convolution (vertical dilation = 1, horizontal dilation = dilation_rate)

        out = tf.nn.conv2d(

            x2,

            self.kernel,

            strides=[1, 1, 1, 1],

            padding="VALID",

            dilations=[1, 1, dilation_rate, 1],

        )

        out = tf.nn.bias_add(out, self.bias)

        # Apply row-wise weights

        weights = tf.reshape(row_wise_weights[idx], [1, H, 1, 1])

        out = out * weights

        outputs.append(out)

    # Sum over dilations

    outputs = tf.stack(outputs, axis=0)  # [N, B, H, W, C]

    return tf.reduce_sum(outputs, axis=0)  # [B, H, W, C]

SCALES_MIN = 0.11

SCALES_MAX = 256

SCALES_LEVELS = 64


Output of discriminator, where real and fake are merged into single tensors.

DiscOutAll = collections.namedtuple(

"DiscOutAll",

["d_all", "d_all_logits"])


Split each tensor in a  DiscOutAll into 2.

DiscOutSplit = collections.namedtuple(

"DiscOutSplit",

["d_real", "d_fake",

"d_real_logits", "d_fake_logits"])

EntropyInfo = collections.namedtuple(

"EntropyInfo",

"noisy quantized nbits nbpp qbits qbpp",

)

FactorizedPriorInfo = collections.namedtuple(

"FactorizedPriorInfo",

"decoded latent_shape total_nbpp total_qbpp bitstring",

)

HyperInfo = collections.namedtuple(

"HyperInfo",

"decoded latent_shape hyper_latent_shape "

"nbpp side_nbpp total_nbpp qbpp side_qbpp total_qbpp "

"bitstream_tensors",

)

class Encoder(tf.keras.Sequential):

"""Encoder architecture."""

def init(self,

name="Encoder",

num_down=4,

num_filters_base=60,

num_filters_bottleneck=220):

"""Instantiate model.

Args:

name: Name of the layer.

num_down: How many downsampling layers to use.

num_filters_base: Num filters to base multiplier on.

num_filters_bottleneck: Num filters to output for bottleneck (latent).

"""

self._num_down = num_down

model = [

SWHDCtf(

filters=num_filters_base, kernel_size=7, dilations=[1, 2, 3]),

ChannelNorm(),

tf.keras.layers.ReLU()

]

for i in range(num_down):

model.extend([

tf.keras.layers.Conv2D(

filters=num_filters_base * 2 ** (i + 1),

kernel_size=3, padding="same", strides=2),

ChannelNorm(),

tf.keras.layers.ReLU()])

model.append(

SWHDCtf(

filters=num_filters_bottleneck,

kernel_size=3, dilations=[1, 2, 3]))

super(Encoder, self).init(layers=model, name=name)

@property

def num_downsampling_layers(self):

return self.num_down

class Decoder(tf.keras.layers.Layer):

"""Decoder/generator architecture."""

def init(self,

name="Decoder",

num_up=4,

num_filters_base=60,

num_residual_blocks=9,

):

"""Instantiate layer.

Args:

name: name of the layer.

num_up: how many upsampling layers.

num_filters_base: base number of filters.

num_residual_blocks: number of residual blocks.

"""

head = [ChannelNorm(),

SWHDCtf(

filters=num_filters_base * (2 ** num_up),

kernel_size=3, dilations=[1, 2, 3]),

ChannelNorm()]

residual_blocks = []

for block_idx in range(num_residual_blocks):

residual_blocks.append(

ResidualBlock(

filters=num_filters_base * (2 ** num_up),

kernel_size=3,

name="block{}".format(block_idx),

activation="relu",

padding="same"))

tail = []

for scale in reversed(range(num_up)):

filters = num_filters_base * (2 ** scale)

tail += [

tf.keras.layers.Conv2DTranspose(

filters=filters,

kernel_size=3, padding="same",

strides=2),

ChannelNorm(),

tf.keras.layers.ReLU()]

# Final conv layer.

tail.append(

SWHDCtf(

filters=3,

kernel_size=7,

dilations=[1, 2, 3]))

self._head = tf.keras.Sequential(head)

self._residual_blocks = tf.keras.Sequential(residual_blocks)

self._tail = tf.keras.Sequential(tail)

super(Decoder, self).init(name=name)

def call(self, inputs):

after_head = self._head(inputs)

after_res = self._residual_blocks(after_head)

after_res += after_head  # Skip connection

return self._tail(after_res)

class ResidualBlock(tf.keras.layers.Layer):

"""Implement a residual block."""

def init(

self,

filters,

kernel_size,

name=None,

activation="relu",

**kwargs_conv2d):

"""Instantiate layer.

Args:

filters: int, number of filters, passed to the conv layers.

kernel_size: int, kernel_size, passed to the conv layers.

name: str, name of the layer.

activation: function or string, resolved with keras.

**kwargs_conv2d: Additional arguments to be passed directly to Conv2D.

E.g. 'padding'.

"""

super(ResidualBlock, self).init()

kwargs_conv2d["filters"] = filters

kwargs_conv2d["kernel_size"] = kernel_size

block = [

SWHDCtf(dilations=[1, 2, 3], **kwargs_conv2d),

ChannelNorm(),

tf.keras.layers.Activation(activation),

SWHDCtf(dilations=[1, 2, 3], **kwargs_conv2d),

ChannelNorm()]

self.block = tf.keras.Sequential(name=name, layers=block)

def call(self, inputs, **kwargs):

return inputs + self.block(inputs, **kwargs)

class ChannelNorm(tf.keras.layers.Layer):

"""Implement ChannelNorm.

Based on this paper and keras' InstanceNorm layer:

Ba, Jimmy Lei, Jamie Ryan Kiros, and Geoffrey E. Hinton.

"Layer normalization."

arXiv preprint arXiv:1607.06450 (2016).

"""

def init(self,

epsilon: float = 1e-3,

center: bool = True,

scale: bool = True,

beta_initializer="zeros",

gamma_initializer="ones",

**kwargs):

"""Instantiate layer.

Args:

epsilon: For stability when normalizing.

center: Whether to create and use a {beta}.

scale: Whether to create and use a {gamma}.

beta_initializer: Initializer for beta.

gamma_initializer: Initializer for gamma.

**kwargs: Passed to keras.

"""

super(ChannelNorm, self).init(**kwargs)

self.axis = -1

self.epsilon = epsilon

self.center = center

self.scale = scale

self.beta_initializer = tf.keras.initializers.get(beta_initializer)

self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)

def build(self, input_shape):

self._add_gamma_weight(input_shape)

self._add_beta_weight(input_shape)

self.built = True

super().build(input_shape)

def call(self, inputs, modulation=None):

mean, variance = self._get_moments(inputs)

# inputs = tf.Print(inputs, [mean, variance, self.beta, self.gamma], "NORM")

return tf.nn.batch_normalization(

inputs, mean, variance, self.beta, self.gamma, self.epsilon,

name="normalize")

def _get_moments(self, inputs):

# Like tf.nn.moments but unbiased sample std. deviation.

# Reduce over channels only.

mean = tf.reduce_mean(inputs, [self.axis], keepdims=True, name="mean")

variance = tf.reduce_sum(

tf.squared_difference(inputs, tf.stop_gradient(mean)),

[self.axis], keepdims=True, name="variance_sum")

# Divide by N-1

inputs_shape = tf.shape(inputs)

counts = tf.reduce_prod([inputs_shape[ax] for ax in [self.axis]])

variance /= (tf.cast(counts, tf.float32) - 1)

return mean, variance

def _add_gamma_weight(self, input_shape):

dim = input_shape[self.axis]

shape = (dim,)

if self.scale:

self.gamma = self.add_weight(

shape=shape,

name="gamma",

initializer=self.gamma_initializer)

else:

self.gamma = None

def _add_beta_weight(self, input_shape):

dim = input_shape[self.axis]

shape = (dim,)

if self.center:

self.beta = self.add_weight(

shape=shape,

name="beta",

initializer=self.beta_initializer)

else:

self.beta = None

class _PatchDiscriminatorCompareGANImpl(abstract_arch.AbstractDiscriminator):

"""PatchDiscriminator architecture.

Implemented as a compare_gan layer. This has the benefit that we can use

spectral_norm from that framework.

"""

def init(self,

name,

num_filters_base=64,

num_layers=3,

):

"""Instantiate discriminator.

Args:

name: Name of the layer.

num_filters_base: Number of base filters. will be multiplied as we

go down in resolution.

num_layers: Number of downscaling convolutions.

"""

super(_PatchDiscriminatorCompareGANImpl, self).init(

name, batch_norm_fn=None, layer_norm=False, spectral_norm=True)

self._num_layers = num_layers

self._num_filters_base = num_filters_base

def call(self, x):

"""Overwriting compare_gan's call as we only need x."""

with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):

return self.apply(x)

def apply(self, x):

"""Overwriting compare_gan's apply as we only need x."""

if not isinstance(x, tuple) or len(x) != 2:

raise ValueError("Expected 2-tuple, got {}".format(x))

x, latent = x

x_shape = tf.shape(x)

# Upscale and fuse latent.

latent = arch_ops.conv2d(latent, 12, 3, 3, 1, 1,

name="latent", use_sn=self._spectral_norm)

latent = arch_ops.lrelu(latent, leak=0.2)

latent = tf.image.resize(latent, [x_shape[1], x_shape[2]],

tf.image.ResizeMethod.NEAREST_NEIGHBOR)

x = tf.concat([x, latent], axis=-1)

# The discriminator:

k = 4

net = arch_ops.conv2d(x, self._num_filters_base, k, k, 2, 2,

name="d_conv_head", use_sn=self._spectral_norm)

net = arch_ops.lrelu(net, leak=0.2)

num_filters = self._num_filters_base

for i in range(self.num_layers - 1):

num_filters = min(num_filters * 2, 512)

net = arch_ops.conv2d(net, num_filters, k, k, 2, 2,

name=f"d_conv{i}", use_sn=self._spectral_norm)

net = arch_ops.lrelu(net, leak=0.2)

num_filters = min(num_filters * 2, 512)

net = arch_ops.conv2d(net, num_filters, k, k, 1, 1,

name="d_conv_a", use_sn=self._spectral_norm)

net = arch_ops.lrelu(net, leak=0.2)

# Final 1x1 conv that maps to 1 Channel

net = arch_ops.conv2d(net, 1, k, k, 1, 1,

name="d_conv_b", use_sn=self._spectral_norm)

out_logits = tf.reshape(net, [-1, 1])  # Reshape all into batch dimension.

out = tf.nn.sigmoid(out_logits)

return DiscOutAll(out, out_logits)

class _CompareGANLayer(tf.keras.layers.Layer):

"""Base class for wrapping compare_gan classes as keras layers.

The main task of this class is to provide a keras-like interface, which

includes a trainable_variables. This is non-trivial however, as

compare_gan uses tf.get_variable. So we try to use the name scope to find

these variables.

"""

def init(self,

name,

compare_gan_cls,

**compare_gan_kwargs):

"""Constructor.

Args:

name: Name of the layer. IMPORTANT: Setting this to the same string

for two different layers will cause unexpected behavior since variables

are found using this name.

compare_gan_cls: A class from compare_gan, which should inherit from

either AbstractGenerator or AbstractDiscriminator.

**compare_gan_kwargs: keyword arguments passed to compare_gan_cls to

construct it.

"""

super(_CompareGANLayer, self).init(name=name)

compare_gan_kwargs["name"] = name

self._name = name

self._model = compare_gan_cls(**compare_gan_kwargs)

def call(self, x):

return self._model(x)

@property

def trainable_variables(self):

"""Get trainable variables."""

# Note: keras only returns something if self.training is true, but we

# don't have training as a flag to the constructor, so we always return.

# However, we only call trainable_variables when we are training.

return tf.get_collection(

tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._model.name)

class Discriminator(_CompareGANLayer):

def init(self):

super(Discriminator, self).init(

name="Discriminator",

compare_gan_cls=_PatchDiscriminatorCompareGANImpl)

class Hyperprior(tf.keras.layers.Layer):

"""Hyperprior architecture (probability model)."""

def init(self,

num_chan_bottleneck=220,

num_filters=320,

name="Hyperprior"):

super(Hyperprior, self).init(name=name)

self._num_chan_bottleneck = num_chan_bottleneck

self._num_filters = num_filters

self.analysis = tf.keras.Sequential([

tfc.SignalConv2D(

num_filters, (3, 3), name=f"layer{name}0",

corr=True,

padding="same_zeros", use_bias=True,

activation=tf.nn.relu),

tfc.SignalConv2D(

num_filters, (5, 5), name=f"layer{name}1",

corr=True, strides_down=2,

padding="same_zeros", use_bias=True,

activation=tf.nn.relu),

tfc.SignalConv2D(

num_filters, (5, 5), name=f"layer{name}_2",

corr=True, strides_down=2,

padding="same_zeros", use_bias=True,

activation=None)], name="HyperAnalysis")

def make_synthesis(syn_name):

return tf.keras.Sequential([

tfc.SignalConv2D(

num_filters, (5, 5), name=f"layer{syn_name}0",

corr=False, strides_up=2,

padding="same_zeros", use_bias=True,

kernel_parameterizer=None,

activation=tf.nn.relu),

tfc.SignalConv2D(

num_filters, (5, 5), name=f"layer{syn_name}1",

corr=False, strides_up=2,

padding="same_zeros", use_bias=True,

kernel_parameterizer=None,

activation=tf.nn.relu),

tfc.SignalConv2D(

num_chan_bottleneck, (3, 3), name=f"layer{syn_name}_2",

corr=False,

padding="same_zeros", use_bias=True,

kernel_parameterizer=None,

activation=None),

], name="HyperSynthesis")

self._synthesis_scale = _make_synthesis("scale")

self._synthesis_mean = _make_synthesis("mean")

self._side_entropy_model = FactorizedPriorLayer()

@property

def losses(self):

return self._side_entropy_model.losses

@property

def updates(self):

return self._side_entropy_model.updates

@property

def transform_layers(self):

return [self._analysis, self._synthesis_scale, self._synthesis_mean]

@property

def entropy_layers(self):

return [self._side_entropy_model]

def call(self, latents, image_shape, mode: ModelMode) -> HyperInfo:

"""Apply this layer to code latents.

Args:

latents: Tensor of latent values to code.

image_shape: The [height, width] of a reference frame.

mode: The training, evaluation or validation mode of the model.

Returns:

A HyperInfo tuple.

"""

training = (mode == ModelMode.TRAINING)

validation = (mode == ModelMode.VALIDATION)

latent_shape = tf.shape(latents)[1:-1]

hyper_latents = self._analysis(latents, training=training)

# Model hyperprior distributions and entropy encode/decode hyper-latents.

side_info = self._side_entropy_model(

hyper_latents, image_shape=image_shape, mode=mode, training=training)

hyper_decoded = side_info.decoded

scale_table = np.exp(np.linspace(

np.log(SCALES_MIN), np.log(SCALES_MAX), SCALES_LEVELS))

latent_scales = self._synthesis_scale(

hyper_decoded, training=training)

latent_means = self._synthesis_mean(

tf.cast(hyper_decoded, tf.float32), training=training)

if not (training or validation):

latent_scales = latent_scales[:, :latent_shape[0], :latent_shape[1], :]

latent_means = latent_means[:, :latent_shape[0], :latent_shape[1], :]

conditional_entropy_model = tfc.GaussianConditional(

latent_scales, scale_table, mean=latent_means,

name="conditional_entropy_model")

entropy_info = estimate_entropy(

conditional_entropy_model, latents, spatial_shape=image_shape)

compressed = None

if training:

latents_decoded = _ste_quantize(latents, latent_means)

elif validation:

latents_decoded = entropy_info.quantized

else:

compressed = conditional_entropy_model.compress(latents)

latents_decoded = conditional_entropy_model.decompress(compressed)

info = HyperInfo(

decoded=latents_decoded,

latent_shape=latent_shape,

hyper_latent_shape=side_info.latent_shape,

nbpp=entropy_info.nbpp,

side_nbpp=side_info.total_nbpp,

total_nbpp=entropy_info.nbpp + side_info.total_nbpp,

qbpp=entropy_info.qbpp,

side_qbpp=side_info.total_qbpp,

total_qbpp=entropy_info.qbpp + side_info.total_qbpp,

# We put everything that's needed for real arithmetic coding into

# the bistream_tensors tuple.

bitstream_tensors=(compressed, side_info.bitstring,

image_shape, latent_shape, side_info.latent_shape))

tf.summary.scalar("bpp/total/noisy", info.total_nbpp)

tf.summary.scalar("bpp/total/quantized", info.total_qbpp)

tf.summary.scalar("bpp/latent/noisy", entropy_info.nbpp)

tf.summary.scalar("bpp/latent/quantized", entropy_info.qbpp)

tf.summary.scalar("bpp/side/noisy", side_info.total_nbpp)

tf.summary.scalar("bpp/side/quantized", side_info.total_qbpp)

return info

def _ste_quantize(inputs, mean):

"""Calculates quantize(inputs - mean) + mean, sets straight-through grads."""

half = tf.constant(.5, dtype=tf.float32)

outputs = inputs

outputs -= mean

Rounding latents for the forward pass (straight-through).

outputs = outputs + tf.stop_gradient(tf.math.floor(outputs + half) - outputs)

outputs += mean

return outputs

class FactorizedPriorLayer(tf.keras.layers.Layer):

"""Factorized prior to code a discrete tensor."""

def init(self):

"""Instantiate layer."""

super(FactorizedPriorLayer, self).init(name="FactorizedPrior")

self._entropy_model = tfc.EntropyBottleneck(

name="entropy_model")

def compute_output_shape(self, input_shape):

batch_size = input_shape[0]

shapes = (

input_shape,  # decoded

[2],  # latent_shape = [height, width]

[],  # total_nbpp

[],  # total_qbpp

[batch_size],  # bitstring

)

return tuple(tf.TensorShape(x) for x in shapes)

@property

def losses(self):

return self._entropy_model.losses

@property

def updates(self):

return self._entropy_model.updates

def call(self, latents, image_shape, mode: ModelMode) -> FactorizedPriorInfo:

"""Apply this layer to code latents.

Args:

latents: Tensor of latent values to code.

image_shape: The [height, width] of a reference frame.

mode: The training, evaluation or validation mode of the model.

Returns:

A FactorizedPriorInfo tuple

"""

training = (mode == ModelMode.TRAINING)

validation = (mode == ModelMode.VALIDATION)

latent_shape = tf.shape(latents)[1:-1]

with tf.name_scope("factorized_entropy_model"):

noisy, quantized, _, nbpp, _, qbpp = estimate_entropy(

self._entropy_model, latents, spatial_shape=image_shape)

compressed = None

if training:

latents_decoded = noisy

elif validation:

latents_decoded = quantized

else:

compressed = self._entropy_model.compress(latents)

# Decompress using the spatial shape tensor and get tensor coming out of

# range decoder.

num_channels = latents.shape[-1].value

latents_decoded = self._entropy_model.decompress(

compressed, shape=tf.concat([latent_shape, [num_channels]], 0))

return FactorizedPriorInfo(

decoded=latents_decoded,

latent_shape=latent_shape,

total_nbpp=nbpp,

total_qbpp=qbpp,

bitstring=compressed)

def estimate_entropy(entropy_model, inputs, spatial_shape=None) -> EntropyInfo:

"""Compresses inputs with the given entropy model and estimates entropy.

Args:

entropy_model: An EntropyModel instance.

inputs: The input tensor to be fed to the entropy model.

spatial_shape: Shape of the input image (HxW). Must be provided forvalid == False.

Returns:

The 'noisy' and quantized inputs, as well as differential and discrete

entropy estimates, as an EntropyInfo named tuple.

"""


We are summing over the log likelihood tensor, so we need to explicitly

divide by the batch size.

batch = tf.cast(tf.shape(inputs)[0], tf.float32)


Divide by this to flip sign and convert from nats to bits.

quotient = tf.constant(-np.log(2), dtype=tf.float32)

num_pixels = tf.cast(tf.reduce_prod(spatial_shape), tf.float32)


Compute noisy outputs and estimate differential entropy.

noisy, likelihood = entropy_model(inputs, training=True)

log_likelihood = tf.log(likelihood)

nbits = tf.reduce_sum(log_likelihood) / (quotient * batch)

nbpp = nbits / num_pixels


Compute quantized outputs and estimate discrete entropy.

quantized, likelihood = entropy_model(inputs, training=False)

log_likelihood = tf.log(likelihood)

qbits = tf.reduce_sum(log_likelihood) / (quotient * batch)

qbpp = qbits / num_pixels

return EntropyInfo(noisy, quantized, nbits, nbpp, qbits, qbpp)

<\archs_SWHDC.py>

<model_ssim_base.py>


Copyright 2020 Google LLC. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");

you may not use this file except in compliance with the License.

You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software

distributed under the License is distributed on an "AS IS" BASIS,

WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

See the License for the specific language governing permissions and

limitations under the License.

==============================================================================

"""HiFiC model code."""

import collections

import glob

import itertools

from compare_gan.gans import loss_lib as compare_gan_loss_lib

import tensorflow.compat.v1 as tf

import tensorflow_datasets as tfds

from hific import archs

from hific import helpers

from hific.helpers import ModelMode

from hific.helpers import ModelType

from hific.helpers import TFDSArguments


How many dataset preprocessing processes to use.

DATASET_NUM_PARALLEL = 8


How many batches to prefetch.

DATASET_PREFETCH_BUFFER = 20


How many batches to fetch for shuffling.

DATASET_SHUFFLE_BUFFER = 10

BppPair = collections.namedtuple(

"BppPair", ["total_nbpp", "total_qbpp"])

Nodes = collections.namedtuple(

"Nodes",                    # Expected ranges for RGB:

["input_image",             # [0, 255]

"input_image_scaled",      # [0, 1]

"reconstruction",          # [0, 255]

"reconstruction_scaled",   # [0, 1]

"latent_quantized"])       # Latent post-quantization.

class _LossScaler(object):

"""Helper class to manage losses."""

def init(self, hific_config, ignore_schedules: bool):

# Set to true by model if training or validation.

self._ignore_schedules = ignore_schedules

self._config = hific_config

def get_rd_loss(self, distortion_loss, bpp_pair: BppPair, step):

"""Get R, D part of loss."""

loss_config = self._config.loss_config

weighted_distortion_loss = self._get_weighted_distortion_loss(

loss_config, distortion_loss)

weighted_rate = self._get_weighted_rate_loss(

loss_config, bpp_pair, step)

tf.summary.scalar("components/weighted_R", weighted_rate)

tf.summary.scalar("components/weighted_D", weighted_distortion_loss)

return weighted_rate + weighted_distortion_loss

def _get_weighted_distortion_loss(self, loss_config, distortion_loss):

"""Get weighted D."""

return distortion_loss * loss_config.CD * loss_config.C

def _get_weighted_rate_loss(self, loss_config, bpp_pair, step):

"""Get weighted R."""

total_nbpp, total_qbpp = bpp_pair

lmbda_a = self._get_scheduled_param(

loss_config.lmbda_a, self._config.lambda_schedule, step, "lmbda_a")

# For a target rate R_target, implement constrained optimization:

# { 1/lambda_a * R    if R > R_target

#   1/lambda_b * R    else

# We assume lambda_a < lambda_b, and thus 1/lambda_a > 1/lambda_b,

# i.e., if the rate R is too large, we want a larger factor on it.

if loss_config.lmbda_a >= loss_config.lmbda_b:

raise ValueError("Expected lmbda_a < lmbda_b, got {} >= {}".format(

loss_config.lmbda_a, loss_config.lmbda_b))

target_bpp = self._get_scheduled_param(

loss_config.target, loss_config.target_schedule, step, "target_bpp")

lmbda_b = self._get_scheduled_param(

loss_config.lmbda_b, self._config.lambda_schedule, step, "lmbda_b")

lmbda_inv = tf.where(total_qbpp > target_bpp,

1 / lmbda_a,

1 / lmbda_b)

tf.summary.scalar("lmbda_inv", lmbda_inv)

return lmbda_inv * total_nbpp * loss_config.C

def get_scaled_g_loss(self, g_loss):

"""Get scaled version of GAN loss."""

with tf.name_scope("scaled_g_loss"):

return g_loss * self._config.loss_config.CP

def _get_scheduled_param(self, param, param_schedule, global_step, name):

if (not self._ignore_schedules and param_schedule.vals

and any(step > 0 for step in param_schedule.steps)):

param = _scheduled_value(param, param_schedule, global_step, name)

tf.summary.scalar(name, param)

return param

def _pad(input_image, image_shape, factor):

"""Pad input_image such that H and W are divisible by factor."""

with tf.name_scope("pad"):

height, width = image_shape[0], image_shape[1]

pad_height = (factor - (height % factor)) % factor

pad_width = (factor - (width % factor)) % factor

return tf.pad(input_image,

[[0, 0], [0, pad_height], [0, pad_width], [0, 0]],

"REFLECT")

class HiFiC(object):

"""HiFiC Model class."""

def init(self,

config,

mode: ModelMode,

lpips_weight_path=None,

auto_encoder_ckpt_dir=None,

create_image_summaries=True):

"""Instantiate model.

Args:

config: A config, see configs.py

mode: Model mode.

lpips_weight_path: path to where LPIPS weights are stored or should be

stored. See helpers.ensure_lpips_weights_exist.

auto_encoder_ckpt_dir: If given, instantiate auto-encoder and probability

model from latest checkpoint in this folder.

create_image_summaries: Whether to create image summaries. Turn off to

save disk space.

"""

self._mode = mode

self._config = config

self._model_type = config.model_type

self._create_image_summaries = create_image_summaries

if not isinstance(self._model_type, ModelType):

raise ValueError("Invalid model_type: [{}]".format(

self._config.model_type))

self._auto_encoder_ckpt_path = None

self._auto_encoder_savers = None

if auto_encoder_ckpt_dir:

latest_ckpt = tf.train.latest_checkpoint(auto_encoder_ckpt_dir)

if not latest_ckpt:

raise ValueError(f"Did not find checkpoint in {auto_encoder_ckpt_dir}!")

self._auto_encoder_ckpt_path = latest_ckpt

if self.training and not lpips_weight_path:

lpips_weight_path = "lpips_weight__net-lin_alex_v0.1.pb"

self._lpips_weight_path = lpips_weight_path

self._transform_layers = []

self._entropy_layers = []

self._layers = None

self._encoder = None

self._decoder = None

self._discriminator = None

self._gan_loss_function = None

self._lpips_loss_weight = None

self._lpips_loss = None

self._entropy_model = None

self._optimize_entropy_vars = True

self._global_step_disc = None  # global_step used for D training

self._setup_discriminator = (

self._model_type == ModelType.COMPRESSION_GAN

and (self.training or self.validation))  # No disc for evaluation.

if self._setup_discriminator:

self._num_steps_disc = self._config.num_steps_disc

if self._num_steps_disc == 0:

raise ValueError("model_type=={} but num_steps_disc == 0.".format(

self._model_type))

tf.logging.info(

"GAN Training enabled. Training discriminator for {} steps.".format(

self._num_steps_disc))

else:

self._num_steps_disc = 0

self.input_spec = {

"input_image":

tf.keras.layers.InputSpec(

dtype=tf.uint8,

shape=(None, None, None, 3))}

if self._setup_discriminator:

# This is an optional argument to build_model. If training a

# discriminator, this is expected to contain multiple sub-batches.

# See build_input for details.

self.input_spec["input_images_d_steps"] = tf.keras.layers.InputSpec(

dtype=tf.uint8,

shape=(None, None, None, 3))

self._gan_loss_function = compare_gan_loss_lib.non_saturating

self._loss_scaler = _LossScaler(

self._config,

ignore_schedules=not self.training and not self.validation)

self._train_op = None

self._hooks = []

@property

def training(self):

"""True if in training mode."""

return self._mode == ModelMode.TRAINING

@property

def validation(self):

"""True if in validation mode."""

return self._mode == ModelMode.VALIDATION

@property

def evaluation(self):

"""True if in evaluation mode."""

return self._mode == ModelMode.EVALUATION

@property

def train_op(self):

return self._train_op

@property

def hooks(self):

return self._hooks

def _add_hook(self, hook):

self._hooks.append(hook)

@property

def num_steps_disc(self):

return self._num_steps_disc

def build_input(self,

batch_size,

crop_size,

images_glob=None,

tfds_arguments: TFDSArguments = None):

"""Build input dataset."""

if not (images_glob or tfds_arguments):

raise ValueError("Need images_glob or tfds_arguments!")

if self._setup_discriminator:

# Unroll dataset for GAN training. If we unroll for N steps,

# we want to fetch (N+1) batches for every step, where 1 batch

# will be used for G training, and the remaining N batches for D training.

batch_size *= (self._num_steps_disc + 1)

if self._setup_discriminator:

# Split the (N+1) batches into two arguments for build_model.

def _batch_to_dict(batch):

num_sub_batches = self._num_steps_disc + 1

sub_batch_size = batch_size // num_sub_batches

splits = [sub_batch_size, sub_batch_size * self._num_steps_disc]

input_image, input_images_d_steps = tf.split(batch, splits)

return dict(input_image=input_image,

input_images_d_steps=input_images_d_steps)

else:

def _batch_to_dict(batch):

return dict(input_image=batch)

dataset = self._get_dataset(batch_size, crop_size,

images_glob, tfds_arguments)

return dataset.map(_batch_to_dict)

def _get_dataset(self, batch_size, crop_size,

images_glob, tfds_arguments: TFDSArguments):

"""Build TFDS dataset.

Args:

batch_size: int, batch_size.

crop_size: int, will random crop to this (crop_size, crop_size)

images_glob:

tfds_arguments: argument for TFDS.

Returns:

Instance of tf.data.Dataset.

"""

if isinstance(crop_size, (list, tuple)):

crop_h, crop_w = crop_size

else:

crop_h = crop_w = crop_size

crop_h_float = tf.constant(crop_h, tf.float32) if crop_h else None

crop_w_float = tf.constant(crop_w, tf.float32) if crop_w else None


smallest_fac = tf.constant(0.75, tf.float32)

biggest_fac = tf.constant(0.95, tf.float32)

with tf.name_scope("tfds"):

  if images_glob:

    images = sorted(glob.glob(images_glob))

    tf.logging.info(

        f"Using images_glob={images_glob} ({len(images)} images)")

    filenames = tf.data.Dataset.from_tensor_slices(images)

    

    def _decode_image(filename):

      """Decodifica imagem (suporta PNG, JPG, JPEG)."""

      image_string = tf.read_file(filename)

      # tf.image.decode_image detecta automaticamente o formato

      image = tf.image.decode_image(image_string, channels=3)

      image.set_shape([None, None, 3])

      return image

    

    dataset = filenames.map(_decode_image)

  else:

    tf.logging.info(f"Using TFDS={tfds_arguments}")

    builder = tfds.builder(

        tfds_arguments.dataset_name, data_dir=tfds_arguments.downloads_dir)

    builder.download_and_prepare()

    split = "train" if self.training else "validation"

    dataset = builder.as_dataset(split=split)

  def _preprocess(features):

    # Capture variables from outer scope

    c_h, c_w = crop_h, crop_w

    

    # Create float tensors inside the function scope for safety/consistency with model.py

    c_h_f = tf.cast(c_h, tf.float32) if c_h else None

    c_w_f = tf.cast(c_w, tf.float32) if c_w else None

    if images_glob:

      image = features

    else:

      image = features[tfds_arguments.features_key]

    if not c_h or not c_w:

      return image

    tf.logging.info("Scaling down %s and cropping to %d x %d", image,

                    c_h, c_w)

    with tf.name_scope("random_scale"):

      # Scale down by at least `biggest_fac` and at most `smallest_fac` to

      # remove JPG artifacts. This code also handles images that have one

      # side  shorter than crop_size. In this case, we always upscale such

      # that this side becomes the same as `crop_size`. Overall, images

      # returned will never be smaller than `crop_size`.

      image_shape = tf.cast(tf.shape(image), tf.float32)

      height, width = image_shape[0], image_shape[1]

      

      # The smallest factor such that the downscaled image is still bigger

      # than the required crop dimensions.

      image_smallest_fac = tf.math.maximum(c_h_f / height, c_w_f / width)

      

      min_fac = tf.math.maximum(smallest_fac, image_smallest_fac)

      max_fac = tf.math.maximum(min_fac, biggest_fac)

      

      scale = tf.random_uniform([],

                                minval=min_fac,

                                maxval=max_fac,

                                dtype=tf.float32,

                                seed=42,

                                name=None)

                                

      # Ensure new dimensions are integers

      new_height = tf.cast(tf.ceil(scale * height), tf.int32)

      new_width = tf.cast(tf.ceil(scale * width), tf.int32)

      

      # Stack to create a 1-D Tensor (vector) of shape (2,)

      new_size = tf.stack([new_height, new_width])

      image = tf.image.resize_images(image, new_size)

    with tf.name_scope("random_crop"):

      image = tf.image.random_crop(image, [c_h, c_w, 3])

    return image

  dataset = dataset.map(

      _preprocess, num_parallel_calls=DATASET_NUM_PARALLEL)

  dataset = dataset.batch(batch_size, drop_remainder=True)

  if not self.evaluation:

    # Make sure we don't run out of data

    dataset = dataset.repeat()

    dataset = dataset.shuffle(buffer_size=DATASET_SHUFFLE_BUFFER)

  dataset = dataset.prefetch(buffer_size=DATASET_PREFETCH_BUFFER)

  return dataset

def build_model(self, input_image, input_images_d_steps=None):

"""Build model and losses and train_ops.

Args:

input_image: A single (B, H, W, C) image, in [0, 255]

input_images_d_steps: If training a discriminator, this is expected to

be a (B*N, H, W, C) stack of images, where N=number of sub batches.

See build_input.

Returns:

output_image and bitstrings if self.evaluation else None.

"""

if input_images_d_steps is None:

input_images_d_steps = []

else:

input_images_d_steps.set_shape(

self.input_spec["input_images_d_steps"].shape)

input_images_d_steps = tf.split(input_images_d_steps, self.num_steps_disc)

if self.evaluation and input_images_d_steps:

raise ValueError("Only need input_image for eval! {}".format(

input_images_d_steps))

input_image.set_shape(self.input_spec["input_image"].shape)

self.build_transforms()

if self.training:

# self._lpips_loss = LPIPSLoss(self._lpips_weight_path)

self._lpips_loss = True # Placeholder to enable perceptual loss logic

self._lpips_loss_weight = self._config.loss_config.lpips_weight

if self._setup_discriminator:

self.build_discriminator()

# Global step needs to be created for train, val and eval.

global_step = tf.train.get_or_create_global_step()

# Compute output graph.

nodes_gen, bpp_pair, bitstrings = 


self.compute_compression_graph(input_image)

if self.evaluation:

tf.logging.info("Evaluation mode: build_model done.")

reconstruction = tf.clip_by_value(nodes_gen.reconstruction, 0, 255.)

return reconstruction, bitstrings

nodes_disc = []  # list of Nodes, one for every sub-batch of disc

for i, sub_batch in enumerate(input_images_d_steps):

with tf.name_scope("sub_batch_disc{}".format(i)):

nodes, _, _ = self._compute_compression_graph(

sub_batch, create_summaries=False)

nodes_disc.append(nodes)

if self._auto_encoder_ckpt_path:

self._prepare_auto_encoder_restore()

# The following is inspired by compare_gan/gans/modular_gan.py:

# Let's say we want to train the discriminator for D steps for every 1 step

# of generator training. We do the unroll_graph=True options:

# The features given to the model_fn are split into

# D + 1 sub-batches. The code then creates D train_ops for the

# discriminator, each feeding a different sub-batch of features

# into the discriminator.

# The train_op for the generator then depends on all these D train_ops

# and uses the last (D+1 th) sub-batch.

# Note that the graph is only created once.

d_train_ops = []

if self._setup_discriminator:

tf.logging.info("Unrolling graph for discriminator")

self._global_step_disc = tf.get_variable(

"global_step_disc", [], dtype=global_step.dtype, trainable=False)

with tf.name_scope("steps"):

tf.summary.scalar("global_step", global_step)

tf.summary.scalar("global_step_disc", self._global_step_disc)

# Create optimizer once, and then call minimize on it multiple times

# within self._train_discriminator.

disc_optimizer = self._make_discriminator_optimizer(

self.global_step_disc)

for i, nodes in enumerate(nodes_disc):

with tf.name_scope("train_disc{}".format(i + 1)):

with tf.control_dependencies(d_train_ops):

d_train_ops.append(

self._train_discriminator(

nodes, disc_optimizer, create_summaries=(i == 0)))

# Depend on d_train_ops, which ensures all self._num_steps_disc steps of

# the discriminator will run before the generator training op.

with tf.control_dependencies(d_train_ops):

train_op = self._train_generator(nodes_gen, bpp_pair, global_step)

if self.training:

self._train_op = train_op

def prepare_for_arithmetic_coding(self, sess):

"""Run's the update op of the EntropyBottleneck."""

update_op = self._entropy_model.updates[0]

sess.run(update_op)

def restore_trained_model(self, sess, ckpt_dir):

"""Restore a trained model for evaluation."""

saver = tf.train.Saver()

latest_ckpt = tf.train.latest_checkpoint(ckpt_dir)

tf.logging.info("Restoring %s...", latest_ckpt)

saver.restore(sess, latest_ckpt)

def restore_autoencoder(self, sess):

"""Restore encoder, decoder and probability model from checkpoint."""

assert self._auto_encoder_savers

for saver in self._auto_encoder_savers:

tf.logging.info("Restoring %s...", saver)

saver.restore(sess, self._auto_encoder_ckpt_path)

def _prepare_auto_encoder_restore(self):

"""Prepare the savers needed to restore encoder, decoder, entropy_model."""

assert self._auto_encoder_savers is None

self._auto_encoder_savers = []

for name, layer in [

("entropy_model", self._entropy_model),

("encoder", self._encoder),

("decoder", self._decoder)]:

self.auto_encoder_savers.append(

tf.train.Saver(layer.variables, name=f"restore{name}"))

def build_transforms(self):

"""Instantiates all transforms used by this model."""

self._encoder = archs.Encoder()

self._decoder = archs.Decoder()

self._transform_layers.append(self._encoder)

self._transform_layers.append(self._decoder)

self._entropy_model = archs.Hyperprior()

self._transform_layers.extend(self._entropy_model.transform_layers)

self._entropy_layers.extend(self._entropy_model.entropy_layers)

self._layers = self._transform_layers + self._entropy_layers

def build_discriminator(self):

"""Instantiates discriminator."""

self._discriminator = archs.Discriminator()

def _compute_compression_graph(self, input_image, create_summaries=True):

"""Compute a forward pass through encoder and decoder.

Args:

input_image: Input image, range [0, 255]

create_summaries: Whether to create summaries

Returns:

tuple Nodes, BppPair

"""

with tf.name_scope("image_shape"):

image_shape = tf.shape(input_image)[1:-1]  # Get H, W.

if self.evaluation:

num_downscaling = self._encoder.num_downsampling_layers

factor = 2 ** num_downscaling

tf.logging.info("Padding to {}".format(factor))

input_image = _pad(input_image, image_shape, factor)

with tf.name_scope("scale_down"):

input_image_scaled = 


tf.cast(input_image, tf.float32) / 255.

info = self._get_encoder_out(input_image_scaled, image_shape)

decoder_in = info.decoded

total_nbpp = info.total_nbpp

total_qbpp = info.total_qbpp

bitstream_tensors = info.bitstream_tensors

reconstruction, reconstruction_scaled = 


self._compute_reconstruction(

decoder_in, image_shape, input_image_scaled.shape)

if create_summaries and self._create_image_summaries:

tf.summary.image(

"input_image", tf.saturate_cast(input_image, tf.uint8), max_outputs=1)

tf.summary.image(

"reconstruction",

tf.saturate_cast(reconstruction, tf.uint8),

max_outputs=1)

nodes = Nodes(input_image, input_image_scaled,

reconstruction, reconstruction_scaled,

latent_quantized=decoder_in)

return nodes, BppPair(total_nbpp, total_qbpp), bitstream_tensors

def _get_encoder_out(self,

input_image_scaled,

image_shape) -> archs.HyperInfo:

"""Compute encoder transform."""

encoder_out = self._encoder(input_image_scaled, training=self.training)

return self._entropy_model(encoder_out,

image_shape=image_shape,

mode=self._mode)

def _compute_reconstruction(self, decoder_in, image_shape, output_shape):

"""Compute pass through decoder.

Args:

decoder_in: Input to decoder transform.

image_shape: Tuple (height, width) of the image_shape

output_shape: Desired output shape.

Returns:

Tuple (reconstruction (in [0, 255],

reconstruction_scaled (in [0, 1]),

residual_scaled (in [-1, 1]) if it exists else None).

"""

reconstruction_scaled = self._decoder(

decoder_in, training=self.training)

with tf.name_scope("undo_padding"):

height, width = image_shape[0], image_shape[1]

reconstruction_scaled = reconstruction_scaled[:, :height, :width, :]

reconstruction_scaled.set_shape(output_shape)

with tf.name_scope("re_scale"):

reconstruction = reconstruction_scaled * 255.

return reconstruction, reconstruction_scaled

def _create_rd_loss(self, nodes: Nodes, bpp_pair: BppPair, step):

"""Computes noisy/quantized rd-loss and creates summaries."""

with tf.name_scope("loss"):

distortion_loss = self._compute_distortion_loss(nodes)

rd_loss = self._loss_scaler.get_rd_loss(distortion_loss, bpp_pair, step)

tf.summary.scalar("distortion_loss", distortion_loss)

tf.summary.scalar("rd_loss", rd_loss)

return rd_loss

def _compute_distortion_loss(self, nodes: Nodes):

input_image, reconstruction = nodes.input_image, nodes.reconstruction

with tf.name_scope("distortion"):

input_image = tf.cast(input_image, tf.float32)

reconstruction = tf.cast(reconstruction, tf.float32)

sq_err = tf.math.squared_difference(input_image, reconstruction)

distortion_loss = tf.reduce_mean(sq_err)

return distortion_loss

def _compute_perceptual_loss(self, nodes: Nodes):

input_image_scaled = nodes.input_image_scaled

reconstruction_scaled = nodes.reconstruction_scaled

# First the fake images, then the real! Otherwise no gradients.

# SSIM calculation

# tf.image.ssim expects images in range [0, max_val]

# nodes.input_image_scaled is [0, 1]

# nodes.reconstruction_scaled is [0, 1]


# User specified parameters: K1 = 0.01, K2 = 0.03, k = 11, sigma = 1.5

# These are defaults for tf.image.ssim


ssim_value = tf.image.ssim(input_image_scaled, reconstruction_scaled, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)


# Loss should be minimized. SSIM is maximized (1 is best).

# So loss = 1 - SSIM

ssim_loss = 1.0 - tf.reduce_mean(ssim_value)


return ssim_loss

def _create_gan_loss(self,

d_out: archs.DiscOutSplit,

create_summaries=True,

mode="g_loss"):

"""Create GAN loss using compare_gan."""

if mode not in ("g_loss", "d_loss"):

raise ValueError("Invalid mode: {}".format(mode))

assert self._gan_loss_function is not None

# Called within either train_disc or train_gen namescope.

with tf.name_scope("gan_loss"):

d_loss, _, _, g_loss = compare_gan_loss_lib.get_losses(

# Note: some fn's need other args.

fn=self._gan_loss_function,

d_real=d_out.d_real,

d_fake=d_out.d_fake,

d_real_logits=d_out.d_real_logits,

d_fake_logits=d_out.d_fake_logits)

loss = d_loss if mode == "d_loss" else g_loss

if create_summaries:

tf.summary.scalar("d_loss", d_loss)

tf.summary.scalar("g_loss", g_loss)

return loss

def _train_discriminator(self, nodes: Nodes, optimizer, create_summaries):

"""Creates a train_op for the discriminator.

Args:

nodes: Instance of Nodes, the nodes of the model to feed to D.

optimizer: Discriminator optimizer. Passed in because it will be re-used

in the different discriminator steps.

create_summaries: If True, create summaries.

Returns:

A training op if training, else no_op.

"""

d_out = self._compute_discriminator_out(

nodes,

create_summaries,

gradients_to_generator=False)  # Only train discriminator!

d_loss = self._create_gan_loss(d_out, create_summaries, mode="d_loss")

if not self.training:

return tf.no_op()

self._add_hook(tf.train.NanTensorHook(d_loss))

# Getting the variables here because they don't exist before calling

# _compute_discriminator_out for the first time!

disc_vars = self._discriminator.trainable_variables

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

with tf.control_dependencies(update_ops):

with tf.name_scope("min_d"):

train_op_d = optimizer.minimize(

d_loss, self._global_step_disc, disc_vars)

return train_op_d

def _train_generator(self, nodes: Nodes, bpp_pair: BppPair, step):

"""Create training op for generator.

This also create the optimizers for the encoder/decoder and entropy

layers.

Args:

nodes: The output of the model to create a R-D loss and feed to D.

bpp_pair: Instance of BppPair.

step: the global step of G.

Returns:

A training op if training, else None

"""

rd_loss = self._create_rd_loss(nodes, bpp_pair, step)

with tf.name_scope("train_gen"):

if self._setup_discriminator:

d_outs = self._compute_discriminator_out(nodes,

create_summaries=False,

gradients_to_generator=True)

g_loss = self._create_gan_loss(d_outs, create_summaries=True,

mode="g_loss")

scaled_g_loss = self._loss_scaler.get_scaled_g_loss(g_loss)

tf.summary.scalar("scaled_g_loss", scaled_g_loss)

loss_enc_dec_entropy = rd_loss + scaled_g_loss

else:

loss_enc_dec_entropy = rd_loss

if self._lpips_loss is not None:

tf.logging.info("Using SSIM...")

perceptual_loss = self._compute_perceptual_loss(nodes)

weighted_perceptual_loss = 


self._lpips_loss_weight * perceptual_loss

tf.summary.scalar("weighted_ssim",

weighted_perceptual_loss)

loss_enc_dec_entropy += weighted_perceptual_loss

tf.summary.scalar("loss_enc_dec_entropy", loss_enc_dec_entropy)

if self.training:

self._add_hook(tf.train.NanTensorHook(loss_enc_dec_entropy))

if self.validation:

return None

entropy_vars, transform_vars, _ = self._get_and_check_variables()

# Train G.

with tf.name_scope("min_g"):

train_op = self._make_enc_dec_entropy_train_op(

step, loss_enc_dec_entropy, entropy_vars, transform_vars)

return train_op

def _compute_discriminator_out(self,

nodes: Nodes,

create_summaries,

gradients_to_generator=True

) -> archs.DiscOutSplit:

"""Get discriminator outputs."""

with tf.name_scope("disc"):

input_image = nodes.input_image_scaled

reconstruction = nodes.reconstruction_scaled

if not gradients_to_generator:

reconstruction = tf.stop_gradient(reconstruction)

discriminator_in = tf.concat([input_image, reconstruction], axis=0)

# Condition D.

latent = tf.stop_gradient(nodes.latent_quantized)

latent = tf.concat([latent, latent], axis=0)

discriminator_in = (discriminator_in, latent)

disc_out_all = self._discriminator(discriminator_in,

training=self.training)

d_real, d_fake = tf.split(disc_out_all.d_all, 2)

d_real_logits, d_fake_logits = tf.split(disc_out_all.d_all_logits, 2)

disc_out_split = archs.DiscOutSplit(d_real, d_fake,

d_real_logits, d_fake_logits)

if create_summaries:

tf.summary.scalar("d_real", tf.reduce_mean(disc_out_split.d_real))

tf.summary.scalar("d_fake", tf.reduce_mean(disc_out_split.d_fake))

return disc_out_split

def _get_and_check_variables(self):

"""Make sure we train the right variables."""

entropy_vars = list(

itertools.chain.from_iterable(

x.trainable_variables for x in self._entropy_layers))

transform_vars = list(

itertools.chain.from_iterable(x.trainable_variables

for x in self._transform_layers))

# Just getting these for book-keeping

transform_vars_non_trainable = list(

itertools.chain.from_iterable(x.variables

for x in self._transform_layers))

disc_vars = (self._discriminator.trainable_variables

if self._setup_discriminator

else [])

# Check that we didn't miss any variables.

all_trainable = set(tf.trainable_variables())

all_known = set(transform_vars + entropy_vars + disc_vars)

if ((all_trainable != all_known) and

all_trainable != set(transform_vars_non_trainable) | all_known):

all_known |= set(transform_vars_non_trainable)

missing_in_trainable = all_known - all_trainable

missing_in_known = all_trainable - all_known

non_trainable_vars_str = 


"\n".join(sorted(v.name for v in transform_vars_non_trainable))

raise ValueError("Did not capture all variables! " +

" Missing in trainable: " + str(missing_in_trainable) +

" Missing in known: " + str(missing_in_known) +

" \n\nNon trainable transform vars: " +

non_trainable_vars_str)

return entropy_vars, transform_vars, disc_vars

def _make_enc_dec_entropy_train_op(self,

step,

loss,

entropy_vars,

transform_vars):

"""Create optimizers for encoder/decoder and entropy model."""

minimize_ops = []

assert len(self._entropy_model.losses) == 1

for i, (name, vs, l) in enumerate(

[("transform", transform_vars, loss),

("entropy", entropy_vars, loss),

("aux", entropy_vars, self._entropy_model.losses[0])

]):

optimizer = tf.train.AdamOptimizer(

learning_rate=_scheduled_value(

self.config.lr,

self.config.lr_schedule,

step,

"lr" + name,

summary=True),

name="adam" + name)

minimize = optimizer.minimize(

l, var_list=vs,

global_step=step if i == 0 else None)  # Only update step once.

minimize_ops.append(minimize)

return tf.group(minimize_ops, name="enc_dec_ent_train_op")

def _make_discriminator_optimizer(self, step):

if not self.training:

return None

return tf.train.AdamOptimizer(

learning_rate=_scheduled_value(

self._config.lr,

self._config.lr_schedule,

step,

"lr_disc",

summary=True),

name="adam_disc")

class LPIPSLoss(object):

"""Calcualte LPIPS loss."""

def init(self, weight_path):

helpers.ensure_lpips_weights_exist(weight_path)

def wrap_frozen_graph(graph_def, inputs, outputs):

def _imports_graph_def():

tf.graph_util.import_graph_def(graph_def, name="")

wrapped_import = tf.wrap_function(_imports_graph_def, [])

import_graph = wrapped_import.graph

return wrapped_import.prune(

tf.nest.map_structure(import_graph.as_graph_element, inputs),

tf.nest.map_structure(import_graph.as_graph_element, outputs))

# Pack LPIPS network into a tf function

graph_def = tf.GraphDef()

with open(weight_path, "rb") as f:

graph_def.ParseFromString(f.read())

self._lpips_func = tf.function(

wrap_frozen_graph(

graph_def, inputs=("0:0", "1:0"), outputs="Reshape_10:0"))

def call(self, fake_image, real_image):

"""Assuming inputs are in [0, 1]."""

# Move inputs to [-1, 1] and NCHW format.

def _transpose_to_nchw(x):

return tf.transpose(x, (0, 3, 1, 2))

fake_image = _transpose_to_nchw(fake_image * 2 - 1.0)

real_image = _transpose_to_nchw(real_image * 2 - 1.0)

loss = self._lpips_func(fake_image, real_image)

return tf.reduce_mean(loss)  # Loss is N111, take mean to get scalar.

def scheduled_value(value, schedule, step, name, summary=False):

"""Create a tensor whose value depends on global step.

Args:

value: The value to adapt.

schedule: Dictionary. Expects 'steps' and 'vals'.

step: The global_step to find to.

name: Name of the value.

summary: Boolean, whether to add a summary for the scheduled value.

Returns:

tf.Tensor.

"""

with tf.name_scope("schedule" + name):

if len(schedule["steps"]) + 1 != len(schedule["vals"]):

raise ValueError("Schedule expects one more value than steps.")

steps = [int(s) for s in schedule["steps"]]

steps = tf.stack(steps + [step + 1])

idx = tf.where(step < steps)[0, 0]

value = value * tf.convert_to_tensor(schedule["vals"])[idx]

if summary:

tf.summary.scalar(name, value)

return value

<\model_ssim_base.py>

<model_wsssim_base.py>


Copyright 2020 Google LLC. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");

you may not use this file except in compliance with the License.

You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software

distributed under the License is distributed on an "AS IS" BASIS,

WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

See the License for the specific language governing permissions and

limitations under the License.

==============================================================================

"""HiFiC model code."""

import collections

import glob

import itertools

import numpy as np

from compare_gan.gans import loss_lib as compare_gan_loss_lib

import tensorflow.compat.v1 as tf

import tensorflow_datasets as tfds

from hific import archs

from hific import helpers

from hific.helpers import ModelMode

from hific.helpers import ModelType

from hific.helpers import TFDSArguments


How many dataset preprocessing processes to use.

DATASET_NUM_PARALLEL = 8


How many batches to prefetch.

DATASET_PREFETCH_BUFFER = 20


How many batches to fetch for shuffling.

DATASET_SHUFFLE_BUFFER = 10

BppPair = collections.namedtuple(

"BppPair", ["total_nbpp", "total_qbpp"])

Nodes = collections.namedtuple(

"Nodes",                    # Expected ranges for RGB:

["input_image",             # [0, 255]

"input_image_scaled",      # [0, 1]

"reconstruction",          # [0, 255]

"reconstruction_scaled",   # [0, 1]

"latent_quantized",        # Latent post-quantization.

"offset_h",

"img_h",

"img_w"])

def _tf_weights(height, width, offset_h, crop_height, crop_width):

"""Calculates weights for equirectangular crop."""


height: (Batch,) or scalar

width: (Batch,) or scalar

offset_h: (Batch,) or scalar

crop_height: scalar

crop_width: scalar

Ensure inputs are float32

height = tf.cast(height, tf.float32)

width = tf.cast(width, tf.float32)

offset_h = tf.cast(offset_h, tf.float32)

crop_height_f = tf.cast(crop_height, tf.float32)

crop_width_f = tf.cast(crop_width, tf.float32)

def compute_single_weight(args):

h, w, off_h = args


  # phis calculation

  # indices 0 to crop_height

  i = tf.range(crop_height_f + 1, dtype=tf.float32)

  global_i = i + off_h

  phis = global_i * np.pi / h

  

  deltaTheta = 2 * np.pi / w

  

  cos_phis = tf.cos(phis)

  column = deltaTheta * (-cos_phis[1:] + cos_phis[:-1])

  

  # Expand to width

  # column shape: (crop_height,)

  # result shape: (crop_height, crop_width, 1)

  weights = tf.tile(column[:, tf.newaxis, tf.newaxis], [1, tf.cast(crop_width, tf.int32), 1])

  return weights

elems = (height, width, offset_h)

weights = tf.map_fn(compute_single_weight, elems, dtype=tf.float32)

return weights

class _LossScaler(object):

"""Helper class to manage losses."""

def init(self, hific_config, ignore_schedules: bool):

# Set to true by model if training or validation.

self._ignore_schedules = ignore_schedules

self._config = hific_config

def get_rd_loss(self, distortion_loss, bpp_pair: BppPair, step):

"""Get R, D part of loss."""

loss_config = self._config.loss_config

weighted_distortion_loss = self._get_weighted_distortion_loss(

loss_config, distortion_loss)

weighted_rate = self._get_weighted_rate_loss(

loss_config, bpp_pair, step)

tf.summary.scalar("components/weighted_R", weighted_rate)

tf.summary.scalar("components/weighted_D", weighted_distortion_loss)

return weighted_rate + weighted_distortion_loss

def _get_weighted_distortion_loss(self, loss_config, distortion_loss):

"""Get weighted D."""

return distortion_loss * loss_config.CD * loss_config.C

def _get_weighted_rate_loss(self, loss_config, bpp_pair, step):

"""Get weighted R."""

total_nbpp, total_qbpp = bpp_pair

lmbda_a = self._get_scheduled_param(

loss_config.lmbda_a, self._config.lambda_schedule, step, "lmbda_a")

# For a target rate R_target, implement constrained optimization:

# { 1/lambda_a * R    if R > R_target

#   1/lambda_b * R    else

# We assume lambda_a < lambda_b, and thus 1/lambda_a > 1/lambda_b,

# i.e., if the rate R is too large, we want a larger factor on it.

if loss_config.lmbda_a >= loss_config.lmbda_b:

raise ValueError("Expected lmbda_a < lmbda_b, got {} >= {}".format(

loss_config.lmbda_a, loss_config.lmbda_b))

target_bpp = self._get_scheduled_param(

loss_config.target, loss_config.target_schedule, step, "target_bpp")

lmbda_b = self._get_scheduled_param(

loss_config.lmbda_b, self._config.lambda_schedule, step, "lmbda_b")

lmbda_inv = tf.where(total_qbpp > target_bpp,

1 / lmbda_a,

1 / lmbda_b)

tf.summary.scalar("lmbda_inv", lmbda_inv)

return lmbda_inv * total_nbpp * loss_config.C

def get_scaled_g_loss(self, g_loss):

"""Get scaled version of GAN loss."""

with tf.name_scope("scaled_g_loss"):

return g_loss * self._config.loss_config.CP

def _get_scheduled_param(self, param, param_schedule, global_step, name):

if (not self._ignore_schedules and param_schedule.vals

and any(step > 0 for step in param_schedule.steps)):

param = _scheduled_value(param, param_schedule, global_step, name)

tf.summary.scalar(name, param)

return param

def _pad(input_image, image_shape, factor):

"""Pad input_image such that H and W are divisible by factor."""

with tf.name_scope("pad"):

height, width = image_shape[0], image_shape[1]

pad_height = (factor - (height % factor)) % factor

pad_width = (factor - (width % factor)) % factor

return tf.pad(input_image,

[[0, 0], [0, pad_height], [0, pad_width], [0, 0]],

"REFLECT")

class HiFiC(object):

"""HiFiC Model class."""

def init(self,

config,

mode: ModelMode,

lpips_weight_path=None,

auto_encoder_ckpt_dir=None,

create_image_summaries=True):

"""Instantiate model.

Args:

config: A config, see configs.py

mode: Model mode.

lpips_weight_path: path to where LPIPS weights are stored or should be

stored. See helpers.ensure_lpips_weights_exist.

auto_encoder_ckpt_dir: If given, instantiate auto-encoder and probability

model from latest checkpoint in this folder.

create_image_summaries: Whether to create image summaries. Turn off to

save disk space.

"""

self._mode = mode

self._config = config

self._model_type = config.model_type

self._create_image_summaries = create_image_summaries

if not isinstance(self._model_type, ModelType):

raise ValueError("Invalid model_type: [{}]".format(

self._config.model_type))

self._auto_encoder_ckpt_path = None

self._auto_encoder_savers = None

if auto_encoder_ckpt_dir:

latest_ckpt = tf.train.latest_checkpoint(auto_encoder_ckpt_dir)

if not latest_ckpt:

raise ValueError(f"Did not find checkpoint in {auto_encoder_ckpt_dir}!")

self._auto_encoder_ckpt_path = latest_ckpt

if self.training and not lpips_weight_path:

lpips_weight_path = "lpips_weight__net-lin_alex_v0.1.pb"

self._lpips_weight_path = lpips_weight_path

self._transform_layers = []

self._entropy_layers = []

self._layers = None

self._encoder = None

self._decoder = None

self._discriminator = None

self._gan_loss_function = None

self._lpips_loss_weight = None

self._lpips_loss = None

self._entropy_model = None

self._optimize_entropy_vars = True

self._global_step_disc = None  # global_step used for D training

self._setup_discriminator = (

self._model_type == ModelType.COMPRESSION_GAN

and (self.training or self.validation))  # No disc for evaluation.

if self._setup_discriminator:

self._num_steps_disc = self._config.num_steps_disc

if self._num_steps_disc == 0:

raise ValueError("model_type=={} but num_steps_disc == 0.".format(

self._model_type))

tf.logging.info(

"GAN Training enabled. Training discriminator for {} steps.".format(

self._num_steps_disc))

else:

self._num_steps_disc = 0

self.input_spec = {

"input_image":

tf.keras.layers.InputSpec(

dtype=tf.uint8,

shape=(None, None, None, 3))}

if self._setup_discriminator:

# This is an optional argument to build_model. If training a

# discriminator, this is expected to contain multiple sub-batches.

# See build_input for details.

self.input_spec["input_images_d_steps"] = tf.keras.layers.InputSpec(

dtype=tf.uint8,

shape=(None, None, None, 3))

self._gan_loss_function = compare_gan_loss_lib.non_saturating

self._loss_scaler = _LossScaler(

self._config,

ignore_schedules=not self.training and not self.validation)

self._train_op = None

self._hooks = []

@property

def training(self):

"""True if in training mode."""

return self._mode == ModelMode.TRAINING

@property

def validation(self):

"""True if in validation mode."""

return self._mode == ModelMode.VALIDATION

@property

def evaluation(self):

"""True if in evaluation mode."""

return self._mode == ModelMode.EVALUATION

@property

def train_op(self):

return self._train_op

@property

def hooks(self):

return self._hooks

def _add_hook(self, hook):

self._hooks.append(hook)

@property

def num_steps_disc(self):

return self._num_steps_disc

def build_input(self,

batch_size,

crop_size,

images_glob=None,

tfds_arguments: TFDSArguments = None):

"""Build input dataset."""

if not (images_glob or tfds_arguments):

raise ValueError("Need images_glob or tfds_arguments!")

if self._setup_discriminator:

# Unroll dataset for GAN training. If we unroll for N steps,

# we want to fetch (N+1) batches for every step, where 1 batch

# will be used for G training, and the remaining N batches for D training.

batch_size *= (self._num_steps_disc + 1)

if self._setup_discriminator:

# Split the (N+1) batches into two arguments for build_model.

def _batch_to_dict(batch):

images = batch["image"]

offsets = batch["offset_h"]

heights = batch["img_h"]

widths = batch["img_w"]

num_sub_batches = self._num_steps_disc + 1

sub_batch_size = batch_size // num_sub_batches

splits = [sub_batch_size, sub_batch_size * self._num_steps_disc]


    input_image, input_images_d_steps = tf.split(images, splits)

    input_offset, input_offsets_d_steps = tf.split(offsets, splits)

    input_height, input_heights_d_steps = tf.split(heights, splits)

    input_width, input_widths_d_steps = tf.split(widths, splits)

    return dict(input_image=input_image,

                input_images_d_steps=input_images_d_steps,

                input_offset=input_offset,

                input_offsets_d_steps=input_offsets_d_steps,

                input_height=input_height,

                input_heights_d_steps=input_heights_d_steps,

                input_width=input_width,

                input_widths_d_steps=input_widths_d_steps)

else:

  def _batch_to_dict(batch):

    return dict(input_image=batch["image"],

                input_offset=batch["offset_h"],

                input_height=batch["img_h"],

                input_width=batch["img_w"])

dataset = self._get_dataset(batch_size, crop_size,

                            images_glob, tfds_arguments)

return dataset.map(_batch_to_dict)

def _get_dataset(self, batch_size, crop_size,

images_glob, tfds_arguments: TFDSArguments):

"""Build TFDS dataset.

Args:

batch_size: int, batch_size.

crop_size: int, will random crop to this (crop_size, crop_size)

images_glob:

tfds_arguments: argument for TFDS.

Returns:

Instance of tf.data.Dataset.

"""

if isinstance(crop_size, (list, tuple)):

crop_h, crop_w = crop_size

else:

crop_h = crop_w = crop_size

crop_size_float = tf.constant(crop_h, tf.float32) if crop_h and isinstance(crop_h, int) else None

smallest_fac = tf.constant(0.75, tf.float32)

biggest_fac = tf.constant(0.95, tf.float32)

with tf.name_scope("tfds"):

if images_glob:

images = sorted(glob.glob(images_glob))

tf.logging.info(

f"Using images_glob={images_glob} ({len(images)} images)")

filenames = tf.data.Dataset.from_tensor_slices(images)


    def _decode_image(filename):

      """Decodifica imagem (suporta PNG, JPG, JPEG)."""

      image_string = tf.read_file(filename)

      # tf.image.decode_image detecta automaticamente o formato

      image = tf.image.decode_image(image_string, channels=3)

      image.set_shape([None, None, 3])

      return image

    

    dataset = filenames.map(_decode_image)

  else:

    tf.logging.info(f"Using TFDS={tfds_arguments}")

    builder = tfds.builder(

        tfds_arguments.dataset_name, data_dir=tfds_arguments.downloads_dir)

    builder.download_and_prepare()

    split = "train" if self.training else "validation"

    dataset = builder.as_dataset(split=split)

  def _preprocess(features):

    # Capture crop_h, crop_w from closure

    c_h, c_w = crop_h, crop_w

  

    if images_glob:

      image = features

    else:

      image = features[tfds_arguments.features_key]

    if not c_h or not c_w:

      image_shape = tf.shape(image)

      return {"image": image, "offset_h": 0, "img_h": image_shape[0], "img_w": image_shape[1]}

    tf.logging.info("Scaling down %s and cropping to %d x %d", image,

                    c_h, c_w)

    with tf.name_scope("random_scale"):

      # Scale down by at least `biggest_fac` and at most `smallest_fac` to

      # remove JPG artifacts. This code also handles images that have one

      # side  shorter than crop_size. In this case, we always upscale such

      # that this side becomes the same as `crop_size`. Overall, images

      # returned will never be smaller than `crop_size`.

      image_shape = tf.cast(tf.shape(image), tf.float32)

      height, width = image_shape[0], image_shape[1]

      smallest_side = tf.math.minimum(height, width)

      # The smallest factor such that the downscaled image is still bigger

      # than `crop_size`. Will be bigger than 1 for images smaller than

      # `crop_size`.

      

      # Use crop_h_float/crop_w_float defined in outer scope, or constant

      # We need to make sure we use crop_h and crop_w from closure

      c_h_f = tf.cast(c_h, tf.float32)

      c_w_f = tf.cast(c_w, tf.float32)

      

      image_smallest_fac = tf.math.maximum(c_h_f / height, c_w_f / width)

      min_fac = tf.math.maximum(smallest_fac, image_smallest_fac)

      # Ensure max_fac is at least min_fac to avoid range error

      max_fac = tf.math.maximum(min_fac, biggest_fac)

      scale = tf.random_uniform([],

                                minval=min_fac,

                                maxval=max_fac,

                                dtype=tf.float32,

                                seed=42,

                                name=None)

      new_height = tf.cast(tf.ceil(scale * height), tf.int32)

      new_width = tf.cast(tf.ceil(scale * width), tf.int32)

      

      # Stack to create a 1-D Tensor (vector) of shape (2,)

      new_size = tf.stack([new_height, new_width])

      image = tf.image.resize_images(image, new_size)

      

      scaled_shape = tf.shape(image)

      img_h = scaled_shape[0]

      img_w = scaled_shape[1]

    with tf.name_scope("random_crop"):

      max_offset_h = img_h - crop_h

      max_offset_w = img_w - crop_w

      

      offset_h = tf.random_uniform([], minval=0, maxval=max_offset_h + 1, dtype=tf.int32)

      offset_w = tf.random_uniform([], minval=0, maxval=max_offset_w + 1, dtype=tf.int32)

      

      image = tf.image.crop_to_bounding_box(image, offset_h, offset_w, crop_h, crop_w)

      

    return {"image": image, "offset_h": offset_h, "img_h": img_h, "img_w": img_w}

  dataset = dataset.map(

      _preprocess, num_parallel_calls=DATASET_NUM_PARALLEL)

  dataset = dataset.batch(batch_size, drop_remainder=True)

  if not self.evaluation:

    # Make sure we don't run out of data

    dataset = dataset.repeat()

    dataset = dataset.shuffle(buffer_size=DATASET_SHUFFLE_BUFFER)

  dataset = dataset.prefetch(buffer_size=DATASET_PREFETCH_BUFFER)

  return dataset

def build_model(self, input_image, input_images_d_steps=None,

input_offset=None, input_height=None, input_width=None,

input_offsets_d_steps=None, input_heights_d_steps=None, input_widths_d_steps=None):

"""Build model and losses and train_ops.

Args:

input_image: A single (B, H, W, C) image, in [0, 255]

input_images_d_steps: If training a discriminator, this is expected to

be a (B*N, H, W, C) stack of images, where N=number of sub batches.

See build_input.

Returns:

output_image and bitstrings if self.evaluation else None.

"""

if input_images_d_steps is None:

input_images_d_steps = []

input_offsets_d_steps = []

input_heights_d_steps = []

input_widths_d_steps = []

else:

input_images_d_steps.set_shape(

self.input_spec["input_images_d_steps"].shape)

input_images_d_steps = tf.split(input_images_d_steps, self.num_steps_disc)


  input_offsets_d_steps = tf.split(input_offsets_d_steps, self.num_steps_disc)

  input_heights_d_steps = tf.split(input_heights_d_steps, self.num_steps_disc)

  input_widths_d_steps = tf.split(input_widths_d_steps, self.num_steps_disc)

if self.evaluation and input_images_d_steps:

  raise ValueError("Only need input_image for eval! {}".format(

      input_images_d_steps))

input_image.set_shape(self.input_spec["input_image"].shape)

self.build_transforms()

if self.training:

  # self._lpips_loss = LPIPSLoss(self._lpips_weight_path)

  self._lpips_loss = True # Placeholder to enable perceptual loss logic

  self._lpips_loss_weight = self._config.loss_config.lpips_weight

if self._setup_discriminator:

  self.build_discriminator()

# Global step needs to be created for train, val and eval.

global_step = tf.train.get_or_create_global_step()

# Compute output graph.

nodes_gen, bpp_pair, bitstrings = \

  self._compute_compression_graph(input_image, input_offset, input_height, input_width)

if self.evaluation:

  tf.logging.info("Evaluation mode: build_model done.")

  reconstruction = tf.clip_by_value(nodes_gen.reconstruction, 0, 255.)

  return reconstruction, bitstrings

nodes_disc = []  # list of Nodes, one for every sub-batch of disc

for i, (sub_batch, sub_offset, sub_height, sub_width) in enumerate(zip(input_images_d_steps, input_offsets_d_steps, input_heights_d_steps, input_widths_d_steps)):

  with tf.name_scope("sub_batch_disc_{}".format(i)):

    nodes, _, _ = self._compute_compression_graph(

        sub_batch, sub_offset, sub_height, sub_width, create_summaries=False)

    nodes_disc.append(nodes)

if self._auto_encoder_ckpt_path:

  self._prepare_auto_encoder_restore()

# The following is inspired by compare_gan/gans/modular_gan.py:

# Let's say we want to train the discriminator for D steps for every 1 step

# of generator training. We do the unroll_graph=True options:

# The features given to the model_fn are split into

# D + 1 sub-batches. The code then creates D train_ops for the

# discriminator, each feeding a different sub-batch of features

# into the discriminator.

# The train_op for the generator then depends on all these D train_ops

# and uses the last (D+1 th) sub-batch.

# Note that the graph is only created once.

d_train_ops = []

if self._setup_discriminator:

  tf.logging.info("Unrolling graph for discriminator")

  self._global_step_disc = tf.get_variable(

      "global_step_disc", [], dtype=global_step.dtype, trainable=False)

  with tf.name_scope("steps"):

    tf.summary.scalar("global_step", global_step)

    tf.summary.scalar("global_step_disc", self._global_step_disc)

  # Create optimizer once, and then call minimize on it multiple times

  # within self._train_discriminator.

  disc_optimizer = self._make_discriminator_optimizer(

      self._global_step_disc)

  for i, nodes in enumerate(nodes_disc):

    with tf.name_scope("train_disc_{}".format(i + 1)):

      with tf.control_dependencies(d_train_ops):

        d_train_ops.append(

            self._train_discriminator(

                nodes, disc_optimizer, create_summaries=(i == 0)))

# Depend on `d_train_ops`, which ensures all `self._num_steps_disc` steps of

# the discriminator will run before the generator training op.

with tf.control_dependencies(d_train_ops):

  train_op = self._train_generator(nodes_gen, bpp_pair, global_step)

if self.training:

  self._train_op = train_op

def prepare_for_arithmetic_coding(self, sess):

"""Run's the update op of the EntropyBottleneck."""

update_op = self._entropy_model.updates[0]

sess.run(update_op)

def restore_trained_model(self, sess, ckpt_dir):

"""Restore a trained model for evaluation."""

saver = tf.train.Saver()

latest_ckpt = tf.train.latest_checkpoint(ckpt_dir)

tf.logging.info("Restoring %s...", latest_ckpt)

saver.restore(sess, latest_ckpt)

def restore_autoencoder(self, sess):

"""Restore encoder, decoder and probability model from checkpoint."""

assert self._auto_encoder_savers

for saver in self._auto_encoder_savers:

tf.logging.info("Restoring %s...", saver)

saver.restore(sess, self._auto_encoder_ckpt_path)

def _prepare_auto_encoder_restore(self):

"""Prepare the savers needed to restore encoder, decoder, entropy_model."""

assert self._auto_encoder_savers is None

self._auto_encoder_savers = []

for name, layer in [

("entropy_model", self._entropy_model),

("encoder", self._encoder),

("decoder", self._decoder)]:

self.auto_encoder_savers.append(

tf.train.Saver(layer.variables, name=f"restore{name}"))

def build_transforms(self):

"""Instantiates all transforms used by this model."""

self._encoder = archs.Encoder()

self._decoder = archs.Decoder()

self._transform_layers.append(self._encoder)

self._transform_layers.append(self._decoder)

self._entropy_model = archs.Hyperprior()

self._transform_layers.extend(self._entropy_model.transform_layers)

self._entropy_layers.extend(self._entropy_model.entropy_layers)

self._layers = self._transform_layers + self._entropy_layers

def build_discriminator(self):

"""Instantiates discriminator."""

self._discriminator = archs.Discriminator()

def _compute_compression_graph(self, input_image, offset_h=None, img_h=None, img_w=None, create_summaries=True):

"""Compute a forward pass through encoder and decoder.

Args:

input_image: Input image, range [0, 255]

create_summaries: Whether to create summaries

Returns:

tuple Nodes, BppPair

"""

with tf.name_scope("image_shape"):

image_shape = tf.shape(input_image)[1:-1]  # Get H, W.

if self.evaluation:

num_downscaling = self._encoder.num_downsampling_layers

factor = 2 ** num_downscaling

tf.logging.info("Padding to {}".format(factor))

input_image = _pad(input_image, image_shape, factor)

with tf.name_scope("scale_down"):

input_image_scaled = 


tf.cast(input_image, tf.float32) / 255.

info = self._get_encoder_out(input_image_scaled, image_shape)

decoder_in = info.decoded

total_nbpp = info.total_nbpp

total_qbpp = info.total_qbpp

bitstream_tensors = info.bitstream_tensors

reconstruction, reconstruction_scaled = 


self._compute_reconstruction(

decoder_in, image_shape, input_image_scaled.shape)

if create_summaries and self._create_image_summaries:

tf.summary.image(

"input_image", tf.saturate_cast(input_image, tf.uint8), max_outputs=1)

tf.summary.image(

"reconstruction",

tf.saturate_cast(reconstruction, tf.uint8),

max_outputs=1)

nodes = Nodes(input_image, input_image_scaled,

reconstruction, reconstruction_scaled,

latent_quantized=decoder_in,

offset_h=offset_h,

img_h=img_h,

img_w=img_w)

return nodes, BppPair(total_nbpp, total_qbpp), bitstream_tensors

def _get_encoder_out(self,

input_image_scaled,

image_shape) -> archs.HyperInfo:

"""Compute encoder transform."""

encoder_out = self._encoder(input_image_scaled, training=self.training)

return self._entropy_model(encoder_out,

image_shape=image_shape,

mode=self._mode)

def _compute_reconstruction(self, decoder_in, image_shape, output_shape):

"""Compute pass through decoder.

Args:

decoder_in: Input to decoder transform.

image_shape: Tuple (height, width) of the image_shape

output_shape: Desired output shape.

Returns:

Tuple (reconstruction (in [0, 255],

reconstruction_scaled (in [0, 1]),

residual_scaled (in [-1, 1]) if it exists else None).

"""

reconstruction_scaled = self._decoder(

decoder_in, training=self.training)

with tf.name_scope("undo_padding"):

height, width = image_shape[0], image_shape[1]

reconstruction_scaled = reconstruction_scaled[:, :height, :width, :]

reconstruction_scaled.set_shape(output_shape)

with tf.name_scope("re_scale"):

reconstruction = reconstruction_scaled * 255.

return reconstruction, reconstruction_scaled

def _create_rd_loss(self, nodes: Nodes, bpp_pair: BppPair, step):

"""Computes noisy/quantized rd-loss and creates summaries."""

with tf.name_scope("loss"):

distortion_loss = self._compute_distortion_loss(nodes)

rd_loss = self._loss_scaler.get_rd_loss(distortion_loss, bpp_pair, step)

tf.summary.scalar("distortion_loss", distortion_loss)

tf.summary.scalar("rd_loss", rd_loss)

return rd_loss

def _compute_distortion_loss(self, nodes: Nodes):

input_image, reconstruction = nodes.input_image, nodes.reconstruction

with tf.name_scope("distortion"):

input_image = tf.cast(input_image, tf.float32)

reconstruction = tf.cast(reconstruction, tf.float32)

sq_err = tf.math.squared_difference(input_image, reconstruction)

  if nodes.offset_h is not None and nodes.img_h is not None:

    crop_shape = tf.shape(input_image)

    crop_height = crop_shape[1]

    crop_width = crop_shape[2]

    

    weights = _tf_weights(nodes.img_h, nodes.img_w, nodes.offset_h, crop_height, crop_width)

    

    weighted_sq_err = sq_err * weights

    distortion_loss = tf.reduce_sum(weighted_sq_err) / (tf.reduce_sum(weights) * 3.0)

  else:

    distortion_loss = tf.reduce_mean(sq_err)

  

  return distortion_loss

def _compute_perceptual_loss(self, nodes: Nodes):

input_image_scaled = nodes.input_image_scaled

reconstruction_scaled = nodes.reconstruction_scaled

# First the fake images, then the real! Otherwise no gradients.


# SSIM calculation

# tf.image.ssim expects images in range [0, max_val]

# nodes.input_image_scaled is [0, 1]

# nodes.reconstruction_scaled is [0, 1]


# User specified parameters: K1 = 0.01, K2 = 0.03, k = 11, sigma = 1.5

# These are defaults for tf.image.ssim


if nodes.offset_h is not None and nodes.img_h is not None:

    # Implementação manual do SSIM para obter o mapa, já que tf.image.ssim não suporta return_index_map no TF 1.15

    

    def _ssim_map(img1, img2, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03):

        img1 = tf.convert_to_tensor(img1)

        img2 = tf.convert_to_tensor(img2)

        

        # Constantes

        c1 = (k1 * max_val) ** 2

        c2 = (k2 * max_val) ** 2

        

        # Filtro Gaussiano

        # Nota: tf.image.ssim usa um filtro gaussiano 1D aplicado separadamente em H e W

        # Aqui vamos simplificar usando tf.nn.depthwise_conv2d com um kernel gaussiano 2D ou similar

        # Para reproduzir exatamente o tf.image.ssim, precisaríamos criar o kernel gaussiano

        

        # Criando kernel gaussiano

        x = tf.range(filter_size, dtype=tf.float32)

        x = x - tf.cast(filter_size // 2, tf.float32)

        gauss = tf.exp(-(x**2) / (2 * filter_sigma**2))

        gauss = gauss / tf.reduce_sum(gauss)

        

        # Kernel 2D separável (H, 1, 1, 1) e (1, W, 1, 1) seria o ideal, mas depthwise espera (H, W, In, Multiplier)

        # Vamos criar um kernel (filter_size, filter_size, 1, 1)

        gauss_kernel_1d = gauss[:, tf.newaxis] # (11, 1)

        gauss_kernel_2d = tf.matmul(gauss_kernel_1d, tf.transpose(gauss_kernel_1d)) # (11, 11)

        gauss_kernel = gauss_kernel_2d[:, :, tf.newaxis, tf.newaxis] # (11, 11, 1, 1)

        

        # Replicar para 3 canais (depthwise conv)

        kernel = tf.tile(gauss_kernel, [1, 1, 3, 1]) # (11, 11, 3, 1)

        

        def _conv(img):

            return tf.nn.depthwise_conv2d(img, kernel, strides=[1, 1, 1, 1], padding='VALID')

        

        # Padding para manter o tamanho (VALID reduz o tamanho, SAME introduz bordas artificiais)

        # tf.image.ssim usa 'VALID' internamente mas computa a média apenas na área válida?

        # Na verdade, tf.image.ssim reduz as dimensões da imagem resultante.

        # Vamos usar SAME para manter o tamanho e alinhar com os pesos, ou VALID e cortar os pesos.

        # O código original do evaluate.py usa 'valid'.

        # Se usarmos 'VALID', o mapa de saída será menor que a entrada.

        # Precisamos ajustar os pesos também.

        

        # Vamos usar SAME para simplificar o alinhamento com os pesos globais

        # Mas cuidado com bordas.

        

        # Melhor abordagem: Usar VALID como no paper/implementação padrão e cortar os pesos.

        pad = filter_size // 2

        

        # Médias

        mu1 = _conv(img1)

        mu2 = _conv(img2)

        

        mu1_sq = mu1 * mu1

        mu2_sq = mu2 * mu2

        mu1_mu2 = mu1 * mu2

        

        sigma1_sq = _conv(img1 * img1) - mu1_sq

        sigma2_sq = _conv(img2 * img2) - mu2_sq

        sigma12 = _conv(img1 * img2) - mu1_mu2

        

        # SSIM map

        numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)

        denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)

        ssim_map = numerator / denominator

        

        return ssim_map

    ssim_map = _ssim_map(input_image_scaled, reconstruction_scaled)

    

    # Ajustar pesos para o tamanho do mapa SSIM (devido à convolução VALID)

    # O mapa SSIM é menor por (filter_size - 1) pixels em cada dimensão

    filter_size = 11

    pad = filter_size // 2

    

    crop_shape = tf.shape(input_image_scaled)

    crop_height = crop_shape[1]

    crop_width = crop_shape[2]

    

    # Pesos originais para o crop inteiro

    weights_full = _tf_weights(nodes.img_h, nodes.img_w, nodes.offset_h, crop_height, crop_width)

    

    # Cortar os pesos para corresponder à área válida do SSIM

    # weights_full shape: (Batch, H, W, 1)

    weights_valid = weights_full[:, pad : crop_height - pad, pad : crop_width - pad, :]

    

    weighted_ssim = ssim_map * weights_valid

    mean_ssim = tf.reduce_sum(weighted_ssim) / (tf.reduce_sum(weights_valid) * 3.0)

    ssim_loss = 1.0 - mean_ssim

else:

    ssim_value = tf.image.ssim(input_image_scaled, reconstruction_scaled, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)

    ssim_loss = 1.0 - tf.reduce_mean(ssim_value)


return ssim_loss

def _create_gan_loss(self,

d_out: archs.DiscOutSplit,

create_summaries=True,

mode="g_loss"):

"""Create GAN loss using compare_gan."""

if mode not in ("g_loss", "d_loss"):

raise ValueError("Invalid mode: {}".format(mode))

assert self._gan_loss_function is not None

# Called within either train_disc or train_gen namescope.

with tf.name_scope("gan_loss"):

d_loss, _, _, g_loss = compare_gan_loss_lib.get_losses(

# Note: some fn's need other args.

fn=self._gan_loss_function,

d_real=d_out.d_real,

d_fake=d_out.d_fake,

d_real_logits=d_out.d_real_logits,

d_fake_logits=d_out.d_fake_logits)

loss = d_loss if mode == "d_loss" else g_loss

if create_summaries:

tf.summary.scalar("d_loss", d_loss)

tf.summary.scalar("g_loss", g_loss)

return loss

def _train_discriminator(self, nodes: Nodes, optimizer, create_summaries):

"""Creates a train_op for the discriminator.

Args:

nodes: Instance of Nodes, the nodes of the model to feed to D.

optimizer: Discriminator optimizer. Passed in because it will be re-used

in the different discriminator steps.

create_summaries: If True, create summaries.

Returns:

A training op if training, else no_op.

"""

d_out = self._compute_discriminator_out(

nodes,

create_summaries,

gradients_to_generator=False)  # Only train discriminator!

d_loss = self._create_gan_loss(d_out, create_summaries, mode="d_loss")

if not self.training:

return tf.no_op()

self._add_hook(tf.train.NanTensorHook(d_loss))

# Getting the variables here because they don't exist before calling

# _compute_discriminator_out for the first time!

disc_vars = self._discriminator.trainable_variables

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

with tf.control_dependencies(update_ops):

with tf.name_scope("min_d"):

train_op_d = optimizer.minimize(

d_loss, self._global_step_disc, disc_vars)

return train_op_d

def _train_generator(self, nodes: Nodes, bpp_pair: BppPair, step):

"""Create training op for generator.

This also create the optimizers for the encoder/decoder and entropy

layers.

Args:

nodes: The output of the model to create a R-D loss and feed to D.

bpp_pair: Instance of BppPair.

step: the global step of G.

Returns:

A training op if training, else None

"""

rd_loss = self._create_rd_loss(nodes, bpp_pair, step)

with tf.name_scope("train_gen"):

if self._setup_discriminator:

d_outs = self._compute_discriminator_out(nodes,

create_summaries=False,

gradients_to_generator=True)

g_loss = self._create_gan_loss(d_outs, create_summaries=True,

mode="g_loss")

scaled_g_loss = self._loss_scaler.get_scaled_g_loss(g_loss)

tf.summary.scalar("scaled_g_loss", scaled_g_loss)

loss_enc_dec_entropy = rd_loss + scaled_g_loss

else:

loss_enc_dec_entropy = rd_loss

if self._lpips_loss is not None:

tf.logging.info("Using SSIM...")

perceptual_loss = self._compute_perceptual_loss(nodes)

weighted_perceptual_loss = 


self._lpips_loss_weight * perceptual_loss

tf.summary.scalar("weighted_ssim",

weighted_perceptual_loss)

loss_enc_dec_entropy += weighted_perceptual_loss

tf.summary.scalar("loss_enc_dec_entropy", loss_enc_dec_entropy)

if self.training:

self._add_hook(tf.train.NanTensorHook(loss_enc_dec_entropy))

if self.validation:

return None

entropy_vars, transform_vars, _ = self._get_and_check_variables()

# Train G.

with tf.name_scope("min_g"):

train_op = self._make_enc_dec_entropy_train_op(

step, loss_enc_dec_entropy, entropy_vars, transform_vars)

return train_op

def _compute_discriminator_out(self,

nodes: Nodes,

create_summaries,

gradients_to_generator=True

) -> archs.DiscOutSplit:

"""Get discriminator outputs."""

with tf.name_scope("disc"):

input_image = nodes.input_image_scaled

reconstruction = nodes.reconstruction_scaled

if not gradients_to_generator:

reconstruction = tf.stop_gradient(reconstruction)

discriminator_in = tf.concat([input_image, reconstruction], axis=0)

# Condition D.

latent = tf.stop_gradient(nodes.latent_quantized)

latent = tf.concat([latent, latent], axis=0)

discriminator_in = (discriminator_in, latent)

disc_out_all = self._discriminator(discriminator_in,

training=self.training)

d_real, d_fake = tf.split(disc_out_all.d_all, 2)

d_real_logits, d_fake_logits = tf.split(disc_out_all.d_all_logits, 2)

disc_out_split = archs.DiscOutSplit(d_real, d_fake,

d_real_logits, d_fake_logits)

if create_summaries:

tf.summary.scalar("d_real", tf.reduce_mean(disc_out_split.d_real))

tf.summary.scalar("d_fake", tf.reduce_mean(disc_out_split.d_fake))

return disc_out_split

def _get_and_check_variables(self):

"""Make sure we train the right variables."""

entropy_vars = list(

itertools.chain.from_iterable(

x.trainable_variables for x in self._entropy_layers))

transform_vars = list(

itertools.chain.from_iterable(x.trainable_variables

for x in self._transform_layers))

# Just getting these for book-keeping

transform_vars_non_trainable = list(

itertools.chain.from_iterable(x.variables

for x in self._transform_layers))

disc_vars = (self._discriminator.trainable_variables

if self._setup_discriminator

else [])

# Check that we didn't miss any variables.

all_trainable = set(tf.trainable_variables())

all_known = set(transform_vars + entropy_vars + disc_vars)

if ((all_trainable != all_known) and

all_trainable != set(transform_vars_non_trainable) | all_known):

all_known |= set(transform_vars_non_trainable)

missing_in_trainable = all_known - all_trainable

missing_in_known = all_trainable - all_known

non_trainable_vars_str = 


"\n".join(sorted(v.name for v in transform_vars_non_trainable))

raise ValueError("Did not capture all variables! " +

" Missing in trainable: " + str(missing_in_trainable) +

" Missing in known: " + str(missing_in_known) +

" \n\nNon trainable transform vars: " +

non_trainable_vars_str)

return entropy_vars, transform_vars, disc_vars

def _make_enc_dec_entropy_train_op(self,

step,

loss,

entropy_vars,

transform_vars):

"""Create optimizers for encoder/decoder and entropy model."""

minimize_ops = []

assert len(self._entropy_model.losses) == 1

for i, (name, vs, l) in enumerate(

[("transform", transform_vars, loss),

("entropy", entropy_vars, loss),

("aux", entropy_vars, self._entropy_model.losses[0])

]):

optimizer = tf.train.AdamOptimizer(

learning_rate=_scheduled_value(

self.config.lr,

self.config.lr_schedule,

step,

"lr" + name,

summary=True),

name="adam" + name)

minimize = optimizer.minimize(

l, var_list=vs,

global_step=step if i == 0 else None)  # Only update step once.

minimize_ops.append(minimize)

return tf.group(minimize_ops, name="enc_dec_ent_train_op")

def _make_discriminator_optimizer(self, step):

if not self.training:

return None

return tf.train.AdamOptimizer(

learning_rate=_scheduled_value(

self._config.lr,

self._config.lr_schedule,

step,

"lr_disc",

summary=True),

name="adam_disc")

class LPIPSLoss(object):

"""Calcualte LPIPS loss."""

def init(self, weight_path):

helpers.ensure_lpips_weights_exist(weight_path)

def wrap_frozen_graph(graph_def, inputs, outputs):

def _imports_graph_def():

tf.graph_util.import_graph_def(graph_def, name="")

wrapped_import = tf.wrap_function(_imports_graph_def, [])

import_graph = wrapped_import.graph

return wrapped_import.prune(

tf.nest.map_structure(import_graph.as_graph_element, inputs),

tf.nest.map_structure(import_graph.as_graph_element, outputs))

# Pack LPIPS network into a tf function

graph_def = tf.GraphDef()

with open(weight_path, "rb") as f:

graph_def.ParseFromString(f.read())

self._lpips_func = tf.function(

wrap_frozen_graph(

graph_def, inputs=("0:0", "1:0"), outputs="Reshape_10:0"))

def call(self, fake_image, real_image):

"""Assuming inputs are in [0, 1]."""

# Move inputs to [-1, 1] and NCHW format.

def _transpose_to_nchw(x):

return tf.transpose(x, (0, 3, 1, 2))

fake_image = _transpose_to_nchw(fake_image * 2 - 1.0)

real_image = _transpose_to_nchw(real_image * 2 - 1.0)

loss = self._lpips_func(fake_image, real_image)

return tf.reduce_mean(loss)  # Loss is N111, take mean to get scalar.

def scheduled_value(value, schedule, step, name, summary=False):

"""Create a tensor whose value depends on global step.

Args:

value: The value to adapt.

schedule: Dictionary. Expects 'steps' and 'vals'.

step: The global_step to find to.

name: Name of the value.

summary: Boolean, whether to add a summary for the scheduled value.

Returns:

tf.Tensor.

"""

with tf.name_scope("schedule" + name):

if len(schedule["steps"]) + 1 != len(schedule["vals"]):

raise ValueError("Schedule expects one more value than steps.")

steps = [int(s) for s in schedule["steps"]]

steps = tf.stack(steps + [step + 1])

idx = tf.where(step < steps)[0, 0]

value = value * tf.convert_to_tensor(schedule["vals"])[idx]

if summary:

tf.summary.scalar(name, value)

return value

<\model_wsssim_base.py>

<model_gauss_ssim.py>


Copyright 2020 Google LLC. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");

you may not use this file except in compliance with the License.

You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software

distributed under the License is distributed on an "AS IS" BASIS,

WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

See the License for the specific language governing permissions and

limitations under the License.

==============================================================================

"""HiFiC model code."""

import collections

import glob

import itertools

import numpy as np

from compare_gan.gans import loss_lib as compare_gan_loss_lib

import tensorflow.compat.v1 as tf

import tensorflow_datasets as tfds

from hific import archs

from hific import helpers

from hific.helpers import ModelMode

from hific.helpers import ModelType

from hific.helpers import TFDSArguments


How many dataset preprocessing processes to use.

DATASET_NUM_PARALLEL = 8


How many batches to prefetch.

DATASET_PREFETCH_BUFFER = 20


How many batches to fetch for shuffling.

DATASET_SHUFFLE_BUFFER = 10

BppPair = collections.namedtuple(

"BppPair", ["total_nbpp", "total_qbpp"])

Nodes = collections.namedtuple(

"Nodes",                    # Expected ranges for RGB:

["input_image",             # [0, 255]

"input_image_scaled",      # [0, 1]

"reconstruction",          # [0, 255]

"reconstruction_scaled",   # [0, 1]

"latent_quantized",        # Latent post-quantization.

"offset_h",

"img_h",

"img_w"])

def _tf_weights(height, width, offset_h, crop_height, crop_width):

"""Calculates weights for equirectangular crop."""


height: (Batch,) or scalar

width: (Batch,) or scalar

offset_h: (Batch,) or scalar

crop_height: scalar

crop_width: scalar

Ensure inputs are float32

height = tf.cast(height, tf.float32)

width = tf.cast(width, tf.float32)

offset_h = tf.cast(offset_h, tf.float32)

crop_height_f = tf.cast(crop_height, tf.float32)

crop_width_f = tf.cast(crop_width, tf.float32)

def compute_single_weight(args):

h, w, off_h = args


  # phis calculation

  # indices 0 to crop_height

  i = tf.range(crop_height_f + 1, dtype=tf.float32)

  global_i = i + off_h

  phis = global_i * np.pi / h

  

  deltaTheta = 2 * np.pi / w

  

  cos_phis = tf.cos(phis)

  column = deltaTheta * (-cos_phis[1:] + cos_phis[:-1])

  

  # Expand to width

  # column shape: (crop_height,)

  # result shape: (crop_height, crop_width, 1)

  weights = tf.tile(column[:, tf.newaxis, tf.newaxis], [1, tf.cast(crop_width, tf.int32), 1])

  return weights

elems = (height, width, offset_h)

weights = tf.map_fn(compute_single_weight, elems, dtype=tf.float32)

return weights

class _LossScaler(object):

"""Helper class to manage losses."""

def init(self, hific_config, ignore_schedules: bool):

# Set to true by model if training or validation.

self._ignore_schedules = ignore_schedules

self._config = hific_config

def get_rd_loss(self, distortion_loss, bpp_pair: BppPair, step):

"""Get R, D part of loss."""

loss_config = self._config.loss_config

weighted_distortion_loss = self._get_weighted_distortion_loss(

loss_config, distortion_loss)

weighted_rate = self._get_weighted_rate_loss(

loss_config, bpp_pair, step)

tf.summary.scalar("components/weighted_R", weighted_rate)

tf.summary.scalar("components/weighted_D", weighted_distortion_loss)

return weighted_rate + weighted_distortion_loss

def _get_weighted_distortion_loss(self, loss_config, distortion_loss):

"""Get weighted D."""

return distortion_loss * loss_config.CD * loss_config.C

def _get_weighted_rate_loss(self, loss_config, bpp_pair, step):

"""Get weighted R."""

total_nbpp, total_qbpp = bpp_pair

lmbda_a = self._get_scheduled_param(

loss_config.lmbda_a, self._config.lambda_schedule, step, "lmbda_a")

# For a target rate R_target, implement constrained optimization:

# { 1/lambda_a * R    if R > R_target

#   1/lambda_b * R    else

# We assume lambda_a < lambda_b, and thus 1/lambda_a > 1/lambda_b,

# i.e., if the rate R is too large, we want a larger factor on it.

if loss_config.lmbda_a >= loss_config.lmbda_b:

raise ValueError("Expected lmbda_a < lmbda_b, got {} >= {}".format(

loss_config.lmbda_a, loss_config.lmbda_b))

target_bpp = self._get_scheduled_param(

loss_config.target, loss_config.target_schedule, step, "target_bpp")

lmbda_b = self._get_scheduled_param(

loss_config.lmbda_b, self._config.lambda_schedule, step, "lmbda_b")

lmbda_inv = tf.where(total_qbpp > target_bpp,

1 / lmbda_a,

1 / lmbda_b)

tf.summary.scalar("lmbda_inv", lmbda_inv)

return lmbda_inv * total_nbpp * loss_config.C

def get_scaled_g_loss(self, g_loss):

"""Get scaled version of GAN loss."""

with tf.name_scope("scaled_g_loss"):

return g_loss * self._config.loss_config.CP

def _get_scheduled_param(self, param, param_schedule, global_step, name):

if (not self._ignore_schedules and param_schedule.vals

and any(step > 0 for step in param_schedule.steps)):

param = _scheduled_value(param, param_schedule, global_step, name)

tf.summary.scalar(name, param)

return param

def _pad(input_image, image_shape, factor):

"""Pad input_image such that H and W are divisible by factor."""

with tf.name_scope("pad"):

height, width = image_shape[0], image_shape[1]

pad_height = (factor - (height % factor)) % factor

pad_width = (factor - (width % factor)) % factor

return tf.pad(input_image,

[[0, 0], [0, pad_height], [0, pad_width], [0, 0]],

"REFLECT")

class HiFiC(object):

"""HiFiC Model class."""

def init(self,

config,

mode: ModelMode,

lpips_weight_path=None,

auto_encoder_ckpt_dir=None,

create_image_summaries=True):

"""Instantiate model.

Args:

config: A config, see configs.py

mode: Model mode.

lpips_weight_path: path to where LPIPS weights are stored or should be

stored. See helpers.ensure_lpips_weights_exist.

auto_encoder_ckpt_dir: If given, instantiate auto-encoder and probability

model from latest checkpoint in this folder.

create_image_summaries: Whether to create image summaries. Turn off to

save disk space.

"""

self._mode = mode

self._config = config

self._model_type = config.model_type

self._create_image_summaries = create_image_summaries

if not isinstance(self._model_type, ModelType):

raise ValueError("Invalid model_type: [{}]".format(

self._config.model_type))

self._auto_encoder_ckpt_path = None

self._auto_encoder_savers = None

if auto_encoder_ckpt_dir:

latest_ckpt = tf.train.latest_checkpoint(auto_encoder_ckpt_dir)

if not latest_ckpt:

raise ValueError(f"Did not find checkpoint in {auto_encoder_ckpt_dir}!")

self._auto_encoder_ckpt_path = latest_ckpt

if self.training and not lpips_weight_path:

lpips_weight_path = "lpips_weight__net-lin_alex_v0.1.pb"

self._lpips_weight_path = lpips_weight_path

self._transform_layers = []

self._entropy_layers = []

self._layers = None

self._encoder = None

self._decoder = None

self._discriminator = None

self._gan_loss_function = None

self._lpips_loss_weight = None

self._lpips_loss = None

self._entropy_model = None

self._optimize_entropy_vars = True

self._global_step_disc = None  # global_step used for D training

self._setup_discriminator = (

self._model_type == ModelType.COMPRESSION_GAN

and (self.training or self.validation))  # No disc for evaluation.

if self._setup_discriminator:

self._num_steps_disc = self._config.num_steps_disc

if self._num_steps_disc == 0:

raise ValueError("model_type=={} but num_steps_disc == 0.".format(

self._model_type))

tf.logging.info(

"GAN Training enabled. Training discriminator for {} steps.".format(

self._num_steps_disc))

else:

self._num_steps_disc = 0

self.input_spec = {

"input_image":

tf.keras.layers.InputSpec(

dtype=tf.uint8,

shape=(None, None, None, 3))}

if self._setup_discriminator:

# This is an optional argument to build_model. If training a

# discriminator, this is expected to contain multiple sub-batches.

# See build_input for details.

self.input_spec["input_images_d_steps"] = tf.keras.layers.InputSpec(

dtype=tf.uint8,

shape=(None, None, None, 3))

self._gan_loss_function = compare_gan_loss_lib.non_saturating

self._loss_scaler = _LossScaler(

self._config,

ignore_schedules=not self.training and not self.validation)

self._train_op = None

self._hooks = []

@property

def training(self):

"""True if in training mode."""

return self._mode == ModelMode.TRAINING

@property

def validation(self):

"""True if in validation mode."""

return self._mode == ModelMode.VALIDATION

@property

def evaluation(self):

"""True if in evaluation mode."""

return self._mode == ModelMode.EVALUATION

@property

def train_op(self):

return self._train_op

@property

def hooks(self):

return self._hooks

def _add_hook(self, hook):

self._hooks.append(hook)

@property

def num_steps_disc(self):

return self._num_steps_disc

def build_input(self,

batch_size,

crop_size,

images_glob=None,

tfds_arguments: TFDSArguments = None):

"""Build input dataset."""

if not (images_glob or tfds_arguments):

raise ValueError("Need images_glob or tfds_arguments!")

if self._setup_discriminator:

# Unroll dataset for GAN training. If we unroll for N steps,

# we want to fetch (N+1) batches for every step, where 1 batch

# will be used for G training, and the remaining N batches for D training.

batch_size *= (self._num_steps_disc + 1)

if self._setup_discriminator:

# Split the (N+1) batches into two arguments for build_model.

def _batch_to_dict(batch):

images = batch["image"]

offsets = batch["offset_h"]

heights = batch["img_h"]

widths = batch["img_w"]

num_sub_batches = self._num_steps_disc + 1

sub_batch_size = batch_size // num_sub_batches

splits = [sub_batch_size, sub_batch_size * self._num_steps_disc]


    input_image, input_images_d_steps = tf.split(images, splits)

    input_offset, input_offsets_d_steps = tf.split(offsets, splits)

    input_height, input_heights_d_steps = tf.split(heights, splits)

    input_width, input_widths_d_steps = tf.split(widths, splits)

    return dict(input_image=input_image,

                input_images_d_steps=input_images_d_steps,

                input_offset=input_offset,

                input_offsets_d_steps=input_offsets_d_steps,

                input_height=input_height,

                input_heights_d_steps=input_heights_d_steps,

                input_width=input_width,

                input_widths_d_steps=input_widths_d_steps)

else:

  def _batch_to_dict(batch):

    return dict(input_image=batch["image"],

                input_offset=batch["offset_h"],

                input_height=batch["img_h"],

                input_width=batch["img_w"])

dataset = self._get_dataset(batch_size, crop_size,

                            images_glob, tfds_arguments)

return dataset.map(_batch_to_dict)

def _get_dataset(self, batch_size, crop_size,

images_glob, tfds_arguments: TFDSArguments):

"""Build TFDS dataset.

Args:

batch_size: int, batch_size.

crop_size: int, will random crop to this (crop_size, crop_size)

images_glob:

tfds_arguments: argument for TFDS.

Returns:

Instance of tf.data.Dataset.

"""

crop_size_float = tf.constant(crop_size, tf.float32) if crop_size else None

smallest_fac = tf.constant(0.75, tf.float32)

biggest_fac = tf.constant(0.95, tf.float32)

with tf.name_scope("tfds"):

if images_glob:

images = sorted(glob.glob(images_glob))

tf.logging.info(

f"Using images_glob={images_glob} ({len(images)} images)")

filenames = tf.data.Dataset.from_tensor_slices(images)


    def _decode_image(filename):

      """Decodifica imagem (suporta PNG, JPG, JPEG)."""

      image_string = tf.read_file(filename)

      # tf.image.decode_image detecta automaticamente o formato

      image = tf.image.decode_image(image_string, channels=3)

      image.set_shape([None, None, 3])

      return image

    

    dataset = filenames.map(_decode_image)

  else:

    tf.logging.info(f"Using TFDS={tfds_arguments}")

    builder = tfds.builder(

        tfds_arguments.dataset_name, data_dir=tfds_arguments.downloads_dir)

    builder.download_and_prepare()

    split = "train" if self.training else "validation"

    dataset = builder.as_dataset(split=split)

  def _preprocess(features):

    if images_glob:

      image = features

    else:

      image = features[tfds_arguments.features_key]

    if not crop_size:

      image_shape = tf.shape(image)

      return {"image": image, "offset_h": 0, "img_h": image_shape[0], "img_w": image_shape[1]}

    tf.logging.info("Scaling down %s and cropping to %d x %d", image,

                    crop_size, crop_size)

    with tf.name_scope("random_scale"):

      # Scale down by at least `biggest_fac` and at most `smallest_fac` to

      # remove JPG artifacts. This code also handles images that have one

      # side  shorter than crop_size. In this case, we always upscale such

      # that this side becomes the same as `crop_size`. Overall, images

      # returned will never be smaller than `crop_size`.

      image_shape = tf.cast(tf.shape(image), tf.float32)

      height, width = image_shape[0], image_shape[1]

      smallest_side = tf.math.minimum(height, width)

      # The smallest factor such that the downscaled image is still bigger

      # than `crop_size`. Will be bigger than 1 for images smaller than

      # `crop_size`.

      image_smallest_fac = crop_size_float / smallest_side

      min_fac = tf.math.maximum(smallest_fac, image_smallest_fac)

      max_fac = tf.math.maximum(min_fac, biggest_fac)

      scale = tf.random_uniform([],

                                minval=min_fac,

                                maxval=max_fac,

                                dtype=tf.float32,

                                seed=42,

                                name=None)

      image = tf.image.resize_images(

          image, [tf.ceil(scale * height),

                  tf.ceil(scale * width)])

      

      scaled_shape = tf.shape(image)

      img_h = scaled_shape[0]

      img_w = scaled_shape[1]

    with tf.name_scope("random_crop"):

      max_offset_h = img_h - crop_size

      max_offset_w = img_w - crop_size

      

      # Gaussian distribution for offset_h to favor center crops (equator)

      # Center of crop (offset_h + crop_size/2) should be close to center of image (img_h/2)

      # So offset_h should be close to (img_h - crop_size) / 2 = max_offset_h / 2

      max_offset_h_float = tf.cast(max_offset_h, tf.float32)

      mean_h = max_offset_h_float / 2.0

      # Standard deviation set so that +/- 2 sigmas cover the whole range [0, max_offset_h]

      stddev_h = max_offset_h_float / 4.0 

      

      offset_h_float = tf.random_normal([], mean=mean_h, stddev=stddev_h)

      offset_h = tf.cast(tf.round(offset_h_float), tf.int32)

      offset_h = tf.clip_by_value(offset_h, 0, max_offset_h)

      

      offset_w = tf.random_uniform([], minval=0, maxval=max_offset_w + 1, dtype=tf.int32)

      

      image = tf.image.crop_to_bounding_box(image, offset_h, offset_w, crop_size, crop_size)

      

    return {"image": image, "offset_h": offset_h, "img_h": img_h, "img_w": img_w}

  dataset = dataset.map(

      _preprocess, num_parallel_calls=DATASET_NUM_PARALLEL)

  dataset = dataset.batch(batch_size, drop_remainder=True)

  if not self.evaluation:

    # Make sure we don't run out of data

    dataset = dataset.repeat()

    dataset = dataset.shuffle(buffer_size=DATASET_SHUFFLE_BUFFER)

  dataset = dataset.prefetch(buffer_size=DATASET_PREFETCH_BUFFER)

  return dataset

def build_model(self, input_image, input_images_d_steps=None,

input_offset=None, input_height=None, input_width=None,

input_offsets_d_steps=None, input_heights_d_steps=None, input_widths_d_steps=None):

"""Build model and losses and train_ops.

Args:

input_image: A single (B, H, W, C) image, in [0, 255]

input_images_d_steps: If training a discriminator, this is expected to

be a (B*N, H, W, C) stack of images, where N=number of sub batches.

See build_input.

Returns:

output_image and bitstrings if self.evaluation else None.

"""

if input_images_d_steps is None:

input_images_d_steps = []

input_offsets_d_steps = []

input_heights_d_steps = []

input_widths_d_steps = []

else:

input_images_d_steps.set_shape(

self.input_spec["input_images_d_steps"].shape)

input_images_d_steps = tf.split(input_images_d_steps, self.num_steps_disc)


  input_offsets_d_steps = tf.split(input_offsets_d_steps, self.num_steps_disc)

  input_heights_d_steps = tf.split(input_heights_d_steps, self.num_steps_disc)

  input_widths_d_steps = tf.split(input_widths_d_steps, self.num_steps_disc)

if self.evaluation and input_images_d_steps:

  raise ValueError("Only need input_image for eval! {}".format(

      input_images_d_steps))

input_image.set_shape(self.input_spec["input_image"].shape)

self.build_transforms()

if self.training:

  # self._lpips_loss = LPIPSLoss(self._lpips_weight_path)

  self._lpips_loss = True # Placeholder to enable perceptual loss logic

  self._lpips_loss_weight = self._config.loss_config.lpips_weight

if self._setup_discriminator:

  self.build_discriminator()

# Global step needs to be created for train, val and eval.

global_step = tf.train.get_or_create_global_step()

# Compute output graph.

nodes_gen, bpp_pair, bitstrings = \

  self._compute_compression_graph(input_image, input_offset, input_height, input_width)

if self.evaluation:

  tf.logging.info("Evaluation mode: build_model done.")

  reconstruction = tf.clip_by_value(nodes_gen.reconstruction, 0, 255.)

  return reconstruction, bitstrings

nodes_disc = []  # list of Nodes, one for every sub-batch of disc

for i, (sub_batch, sub_offset, sub_height, sub_width) in enumerate(zip(input_images_d_steps, input_offsets_d_steps, input_heights_d_steps, input_widths_d_steps)):

  with tf.name_scope("sub_batch_disc_{}".format(i)):

    nodes, _, _ = self._compute_compression_graph(

        sub_batch, sub_offset, sub_height, sub_width, create_summaries=False)

    nodes_disc.append(nodes)

if self._auto_encoder_ckpt_path:

  self._prepare_auto_encoder_restore()

# The following is inspired by compare_gan/gans/modular_gan.py:

# Let's say we want to train the discriminator for D steps for every 1 step

# of generator training. We do the unroll_graph=True options:

# The features given to the model_fn are split into

# D + 1 sub-batches. The code then creates D train_ops for the

# discriminator, each feeding a different sub-batch of features

# into the discriminator.

# The train_op for the generator then depends on all these D train_ops

# and uses the last (D+1 th) sub-batch.

# Note that the graph is only created once.

d_train_ops = []

if self._setup_discriminator:

  tf.logging.info("Unrolling graph for discriminator")

  self._global_step_disc = tf.get_variable(

      "global_step_disc", [], dtype=global_step.dtype, trainable=False)

  with tf.name_scope("steps"):

    tf.summary.scalar("global_step", global_step)

    tf.summary.scalar("global_step_disc", self._global_step_disc)

  # Create optimizer once, and then call minimize on it multiple times

  # within self._train_discriminator.

  disc_optimizer = self._make_discriminator_optimizer(

      self._global_step_disc)

  for i, nodes in enumerate(nodes_disc):

    with tf.name_scope("train_disc_{}".format(i + 1)):

      with tf.control_dependencies(d_train_ops):

        d_train_ops.append(

            self._train_discriminator(

                nodes, disc_optimizer, create_summaries=(i == 0)))

# Depend on `d_train_ops`, which ensures all `self._num_steps_disc` steps of

# the discriminator will run before the generator training op.

with tf.control_dependencies(d_train_ops):

  train_op = self._train_generator(nodes_gen, bpp_pair, global_step)

if self.training:

  self._train_op = train_op

def prepare_for_arithmetic_coding(self, sess):

"""Run's the update op of the EntropyBottleneck."""

update_op = self._entropy_model.updates[0]

sess.run(update_op)

def restore_trained_model(self, sess, ckpt_dir):

"""Restore a trained model for evaluation."""

saver = tf.train.Saver()

latest_ckpt = tf.train.latest_checkpoint(ckpt_dir)

tf.logging.info("Restoring %s...", latest_ckpt)

saver.restore(sess, latest_ckpt)

def restore_autoencoder(self, sess):

"""Restore encoder, decoder and probability model from checkpoint."""

assert self._auto_encoder_savers

for saver in self._auto_encoder_savers:

tf.logging.info("Restoring %s...", saver)

saver.restore(sess, self._auto_encoder_ckpt_path)

def _prepare_auto_encoder_restore(self):

"""Prepare the savers needed to restore encoder, decoder, entropy_model."""

assert self._auto_encoder_savers is None

self._auto_encoder_savers = []

for name, layer in [

("entropy_model", self._entropy_model),

("encoder", self._encoder),

("decoder", self._decoder)]:

self.auto_encoder_savers.append(

tf.train.Saver(layer.variables, name=f"restore{name}"))

def build_transforms(self):

"""Instantiates all transforms used by this model."""

self._encoder = archs.Encoder()

self._decoder = archs.Decoder()

self._transform_layers.append(self._encoder)

self._transform_layers.append(self._decoder)

self._entropy_model = archs.Hyperprior()

self._transform_layers.extend(self._entropy_model.transform_layers)

self._entropy_layers.extend(self._entropy_model.entropy_layers)

self._layers = self._transform_layers + self._entropy_layers

def build_discriminator(self):

"""Instantiates discriminator."""

self._discriminator = archs.Discriminator()

def _compute_compression_graph(self, input_image, offset_h=None, img_h=None, img_w=None, create_summaries=True):

"""Compute a forward pass through encoder and decoder.

Args:

input_image: Input image, range [0, 255]

create_summaries: Whether to create summaries

Returns:

tuple Nodes, BppPair

"""

with tf.name_scope("image_shape"):

image_shape = tf.shape(input_image)[1:-1]  # Get H, W.

if self.evaluation:

num_downscaling = self._encoder.num_downsampling_layers

factor = 2 ** num_downscaling

tf.logging.info("Padding to {}".format(factor))

input_image = _pad(input_image, image_shape, factor)

with tf.name_scope("scale_down"):

input_image_scaled = 


tf.cast(input_image, tf.float32) / 255.

info = self._get_encoder_out(input_image_scaled, image_shape)

decoder_in = info.decoded

total_nbpp = info.total_nbpp

total_qbpp = info.total_qbpp

bitstream_tensors = info.bitstream_tensors

reconstruction, reconstruction_scaled = 


self._compute_reconstruction(

decoder_in, image_shape, input_image_scaled.shape)

if create_summaries and self._create_image_summaries:

tf.summary.image(

"input_image", tf.saturate_cast(input_image, tf.uint8), max_outputs=1)

tf.summary.image(

"reconstruction",

tf.saturate_cast(reconstruction, tf.uint8),

max_outputs=1)

nodes = Nodes(input_image, input_image_scaled,

reconstruction, reconstruction_scaled,

latent_quantized=decoder_in,

offset_h=offset_h,

img_h=img_h,

img_w=img_w)

return nodes, BppPair(total_nbpp, total_qbpp), bitstream_tensors

def _get_encoder_out(self,

input_image_scaled,

image_shape) -> archs.HyperInfo:

"""Compute encoder transform."""

encoder_out = self._encoder(input_image_scaled, training=self.training)

return self._entropy_model(encoder_out,

image_shape=image_shape,

mode=self._mode)

def _compute_reconstruction(self, decoder_in, image_shape, output_shape):

"""Compute pass through decoder.

Args:

decoder_in: Input to decoder transform.

image_shape: Tuple (height, width) of the image_shape

output_shape: Desired output shape.

Returns:

Tuple (reconstruction (in [0, 255],

reconstruction_scaled (in [0, 1]),

residual_scaled (in [-1, 1]) if it exists else None).

"""

reconstruction_scaled = self._decoder(

decoder_in, training=self.training)

with tf.name_scope("undo_padding"):

height, width = image_shape[0], image_shape[1]

reconstruction_scaled = reconstruction_scaled[:, :height, :width, :]

reconstruction_scaled.set_shape(output_shape)

with tf.name_scope("re_scale"):

reconstruction = reconstruction_scaled * 255.

return reconstruction, reconstruction_scaled

def _create_rd_loss(self, nodes: Nodes, bpp_pair: BppPair, step):

"""Computes noisy/quantized rd-loss and creates summaries."""

with tf.name_scope("loss"):

distortion_loss = self._compute_distortion_loss(nodes)

rd_loss = self._loss_scaler.get_rd_loss(distortion_loss, bpp_pair, step)

tf.summary.scalar("distortion_loss", distortion_loss)

tf.summary.scalar("rd_loss", rd_loss)

return rd_loss

def _compute_distortion_loss(self, nodes: Nodes):

input_image, reconstruction = nodes.input_image, nodes.reconstruction

with tf.name_scope("distortion"):

input_image = tf.cast(input_image, tf.float32)

reconstruction = tf.cast(reconstruction, tf.float32)

sq_err = tf.math.squared_difference(input_image, reconstruction)

  if nodes.offset_h is not None and nodes.img_h is not None:

    crop_shape = tf.shape(input_image)

    crop_height = crop_shape[1]

    crop_width = crop_shape[2]

    

    weights = _tf_weights(nodes.img_h, nodes.img_w, nodes.offset_h, crop_height, crop_width)

    

    weighted_sq_err = sq_err * weights

    distortion_loss = tf.reduce_sum(weighted_sq_err) / (tf.reduce_sum(weights) * 3.0)

  else:

    distortion_loss = tf.reduce_mean(sq_err)

  

  return distortion_loss

def _compute_perceptual_loss(self, nodes: Nodes):

input_image_scaled = nodes.input_image_scaled

reconstruction_scaled = nodes.reconstruction_scaled

# First the fake images, then the real! Otherwise no gradients.


# SSIM calculation

# tf.image.ssim expects images in range [0, max_val]

# nodes.input_image_scaled is [0, 1]

# nodes.reconstruction_scaled is [0, 1]


# User specified parameters: K1 = 0.01, K2 = 0.03, k = 11, sigma = 1.5

# These are defaults for tf.image.ssim


if nodes.offset_h is not None and nodes.img_h is not None:

    # Implementação manual do SSIM para obter o mapa, já que tf.image.ssim não suporta return_index_map no TF 1.15

    

    def _ssim_map(img1, img2, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03):

        img1 = tf.convert_to_tensor(img1)

        img2 = tf.convert_to_tensor(img2)

        

        # Constantes

        c1 = (k1 * max_val) ** 2

        c2 = (k2 * max_val) ** 2

        

        # Filtro Gaussiano

        # Nota: tf.image.ssim usa um filtro gaussiano 1D aplicado separadamente em H e W

        # Aqui vamos simplificar usando tf.nn.depthwise_conv2d com um kernel gaussiano 2D ou similar

        # Para reproduzir exatamente o tf.image.ssim, precisaríamos criar o kernel gaussiano

        

        # Criando kernel gaussiano

        x = tf.range(filter_size, dtype=tf.float32)

        x = x - tf.cast(filter_size // 2, tf.float32)

        gauss = tf.exp(-(x**2) / (2 * filter_sigma**2))

        gauss = gauss / tf.reduce_sum(gauss)

        

        # Kernel 2D separável (H, 1, 1, 1) e (1, W, 1, 1) seria o ideal, mas depthwise espera (H, W, In, Multiplier)

        # Vamos criar um kernel (filter_size, filter_size, 1, 1)

        gauss_kernel_1d = gauss[:, tf.newaxis] # (11, 1)

        gauss_kernel_2d = tf.matmul(gauss_kernel_1d, tf.transpose(gauss_kernel_1d)) # (11, 11)

        gauss_kernel = gauss_kernel_2d[:, :, tf.newaxis, tf.newaxis] # (11, 11, 1, 1)

        

        # Replicar para 3 canais (depthwise conv)

        kernel = tf.tile(gauss_kernel, [1, 1, 3, 1]) # (11, 11, 3, 1)

        

        def _conv(img):

            return tf.nn.depthwise_conv2d(img, kernel, strides=[1, 1, 1, 1], padding='VALID')

        

        # Padding para manter o tamanho (VALID reduz o tamanho, SAME introduz bordas artificiais)

        # tf.image.ssim usa 'VALID' internamente mas computa a média apenas na área válida?

        # Na verdade, tf.image.ssim reduz as dimensões da imagem resultante.

        # Vamos usar SAME para manter o tamanho e alinhar com os pesos, ou VALID e cortar os pesos.

        # O código original do evaluate.py usa 'valid'.

        # Se usarmos 'VALID', o mapa de saída será menor que a entrada.

        # Precisamos ajustar os pesos também.

        

        # Vamos usar SAME para simplificar o alinhamento com os pesos globais

        # Mas cuidado com bordas.

        

        # Melhor abordagem: Usar VALID como no paper/implementação padrão e cortar os pesos.

        pad = filter_size // 2

        

        # Médias

        mu1 = _conv(img1)

        mu2 = _conv(img2)

        

        mu1_sq = mu1 * mu1

        mu2_sq = mu2 * mu2

        mu1_mu2 = mu1 * mu2

        

        sigma1_sq = _conv(img1 * img1) - mu1_sq

        sigma2_sq = _conv(img2 * img2) - mu2_sq

        sigma12 = _conv(img1 * img2) - mu1_mu2

        

        # SSIM map

        numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)

        denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)

        ssim_map = numerator / denominator

        

        return ssim_map

    ssim_map = _ssim_map(input_image_scaled, reconstruction_scaled)

    

    # Ajustar pesos para o tamanho do mapa SSIM (devido à convolução VALID)

    # O mapa SSIM é menor por (filter_size - 1) pixels em cada dimensão

    filter_size = 11

    pad = filter_size // 2

    

    crop_shape = tf.shape(input_image_scaled)

    crop_height = crop_shape[1]

    crop_width = crop_shape[2]

    

    # Pesos originais para o crop inteiro

    weights_full = _tf_weights(nodes.img_h, nodes.img_w, nodes.offset_h, crop_height, crop_width)

    

    # Cortar os pesos para corresponder à área válida do SSIM

    # weights_full shape: (Batch, H, W, 1)

    weights_valid = weights_full[:, pad : crop_height - pad, pad : crop_width - pad, :]

    

    weighted_ssim = ssim_map * weights_valid

    mean_ssim = tf.reduce_sum(weighted_ssim) / (tf.reduce_sum(weights_valid) * 3.0)

    ssim_loss = 1.0 - mean_ssim

else:

    ssim_value = tf.image.ssim(input_image_scaled, reconstruction_scaled, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)

    ssim_loss = 1.0 - tf.reduce_mean(ssim_value)


return ssim_loss

def _create_gan_loss(self,

d_out: archs.DiscOutSplit,

create_summaries=True,

mode="g_loss"):

"""Create GAN loss using compare_gan."""

if mode not in ("g_loss", "d_loss"):

raise ValueError("Invalid mode: {}".format(mode))

assert self._gan_loss_function is not None

# Called within either train_disc or train_gen namescope.

with tf.name_scope("gan_loss"):

d_loss, _, _, g_loss = compare_gan_loss_lib.get_losses(

# Note: some fn's need other args.

fn=self._gan_loss_function,

d_real=d_out.d_real,

d_fake=d_out.d_fake,

d_real_logits=d_out.d_real_logits,

d_fake_logits=d_out.d_fake_logits)

loss = d_loss if mode == "d_loss" else g_loss

if create_summaries:

tf.summary.scalar("d_loss", d_loss)

tf.summary.scalar("g_loss", g_loss)

return loss

def _train_discriminator(self, nodes: Nodes, optimizer, create_summaries):

"""Creates a train_op for the discriminator.

Args:

nodes: Instance of Nodes, the nodes of the model to feed to D.

optimizer: Discriminator optimizer. Passed in because it will be re-used

in the different discriminator steps.

create_summaries: If True, create summaries.

Returns:

A training op if training, else no_op.

"""

d_out = self._compute_discriminator_out(

nodes,

create_summaries,

gradients_to_generator=False)  # Only train discriminator!

d_loss = self._create_gan_loss(d_out, create_summaries, mode="d_loss")

if not self.training:

return tf.no_op()

self._add_hook(tf.train.NanTensorHook(d_loss))

# Getting the variables here because they don't exist before calling

# _compute_discriminator_out for the first time!

disc_vars = self._discriminator.trainable_variables

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

with tf.control_dependencies(update_ops):

with tf.name_scope("min_d"):

train_op_d = optimizer.minimize(

d_loss, self._global_step_disc, disc_vars)

return train_op_d

def _train_generator(self, nodes: Nodes, bpp_pair: BppPair, step):

"""Create training op for generator.

This also create the optimizers for the encoder/decoder and entropy

layers.

Args:

nodes: The output of the model to create a R-D loss and feed to D.

bpp_pair: Instance of BppPair.

step: the global step of G.

Returns:

A training op if training, else None

"""

rd_loss = self._create_rd_loss(nodes, bpp_pair, step)

with tf.name_scope("train_gen"):

if self._setup_discriminator:

d_outs = self._compute_discriminator_out(nodes,

create_summaries=False,

gradients_to_generator=True)

g_loss = self._create_gan_loss(d_outs, create_summaries=True,

mode="g_loss")

scaled_g_loss = self._loss_scaler.get_scaled_g_loss(g_loss)

tf.summary.scalar("scaled_g_loss", scaled_g_loss)

loss_enc_dec_entropy = rd_loss + scaled_g_loss

else:

loss_enc_dec_entropy = rd_loss

if self._lpips_loss is not None:

tf.logging.info("Using SSIM...")

perceptual_loss = self._compute_perceptual_loss(nodes)

weighted_perceptual_loss = 


self._lpips_loss_weight * perceptual_loss

tf.summary.scalar("weighted_ssim",

weighted_perceptual_loss)

loss_enc_dec_entropy += weighted_perceptual_loss

tf.summary.scalar("loss_enc_dec_entropy", loss_enc_dec_entropy)

if self.training:

self._add_hook(tf.train.NanTensorHook(loss_enc_dec_entropy))

if self.validation:

return None

entropy_vars, transform_vars, _ = self._get_and_check_variables()

# Train G.

with tf.name_scope("min_g"):

train_op = self._make_enc_dec_entropy_train_op(

step, loss_enc_dec_entropy, entropy_vars, transform_vars)

return train_op

def _compute_discriminator_out(self,

nodes: Nodes,

create_summaries,

gradients_to_generator=True

) -> archs.DiscOutSplit:

"""Get discriminator outputs."""

with tf.name_scope("disc"):

input_image = nodes.input_image_scaled

reconstruction = nodes.reconstruction_scaled

if not gradients_to_generator:

reconstruction = tf.stop_gradient(reconstruction)

discriminator_in = tf.concat([input_image, reconstruction], axis=0)

# Condition D.

latent = tf.stop_gradient(nodes.latent_quantized)

latent = tf.concat([latent, latent], axis=0)

discriminator_in = (discriminator_in, latent)

disc_out_all = self._discriminator(discriminator_in,

training=self.training)

d_real, d_fake = tf.split(disc_out_all.d_all, 2)

d_real_logits, d_fake_logits = tf.split(disc_out_all.d_all_logits, 2)

disc_out_split = archs.DiscOutSplit(d_real, d_fake,

d_real_logits, d_fake_logits)

if create_summaries:

tf.summary.scalar("d_real", tf.reduce_mean(disc_out_split.d_real))

tf.summary.scalar("d_fake", tf.reduce_mean(disc_out_split.d_fake))

return disc_out_split

def _get_and_check_variables(self):

"""Make sure we train the right variables."""

entropy_vars = list(

itertools.chain.from_iterable(

x.trainable_variables for x in self._entropy_layers))

transform_vars = list(

itertools.chain.from_iterable(x.trainable_variables

for x in self._transform_layers))

# Just getting these for book-keeping

transform_vars_non_trainable = list(

itertools.chain.from_iterable(x.variables

for x in self._transform_layers))

disc_vars = (self._discriminator.trainable_variables

if self._setup_discriminator

else [])

# Check that we didn't miss any variables.

all_trainable = set(tf.trainable_variables())

all_known = set(transform_vars + entropy_vars + disc_vars)

if ((all_trainable != all_known) and

all_trainable != set(transform_vars_non_trainable) | all_known):

all_known |= set(transform_vars_non_trainable)

missing_in_trainable = all_known - all_trainable

missing_in_known = all_trainable - all_known

non_trainable_vars_str = 


"\n".join(sorted(v.name for v in transform_vars_non_trainable))

raise ValueError("Did not capture all variables! " +

" Missing in trainable: " + str(missing_in_trainable) +

" Missing in known: " + str(missing_in_known) +

" \n\nNon trainable transform vars: " +

non_trainable_vars_str)

return entropy_vars, transform_vars, disc_vars

def _make_enc_dec_entropy_train_op(self,

step,

loss,

entropy_vars,

transform_vars):

"""Create optimizers for encoder/decoder and entropy model."""

minimize_ops = []

assert len(self._entropy_model.losses) == 1

for i, (name, vs, l) in enumerate(

[("transform", transform_vars, loss),

("entropy", entropy_vars, loss),

("aux", entropy_vars, self._entropy_model.losses[0])

]):

optimizer = tf.train.AdamOptimizer(

learning_rate=_scheduled_value(

self.config.lr,

self.config.lr_schedule,

step,

"lr" + name,

summary=True),

name="adam" + name)

minimize = optimizer.minimize(

l, var_list=vs,

global_step=step if i == 0 else None)  # Only update step once.

minimize_ops.append(minimize)

return tf.group(minimize_ops, name="enc_dec_ent_train_op")

def _make_discriminator_optimizer(self, step):

if not self.training:

return None

return tf.train.AdamOptimizer(

learning_rate=_scheduled_value(

self._config.lr,

self._config.lr_schedule,

step,

"lr_disc",

summary=True),

name="adam_disc")

class LPIPSLoss(object):

"""Calcualte LPIPS loss."""

def init(self, weight_path):

helpers.ensure_lpips_weights_exist(weight_path)

def wrap_frozen_graph(graph_def, inputs, outputs):

def _imports_graph_def():

tf.graph_util.import_graph_def(graph_def, name="")

wrapped_import = tf.wrap_function(_imports_graph_def, [])

import_graph = wrapped_import.graph

return wrapped_import.prune(

tf.nest.map_structure(import_graph.as_graph_element, inputs),

tf.nest.map_structure(import_graph.as_graph_element, outputs))

# Pack LPIPS network into a tf function

graph_def = tf.GraphDef()

with open(weight_path, "rb") as f:

graph_def.ParseFromString(f.read())

self._lpips_func = tf.function(

wrap_frozen_graph(

graph_def, inputs=("0:0", "1:0"), outputs="Reshape_10:0"))

def call(self, fake_image, real_image):

"""Assuming inputs are in [0, 1]."""

# Move inputs to [-1, 1] and NCHW format.

def _transpose_to_nchw(x):

return tf.transpose(x, (0, 3, 1, 2))

fake_image = _transpose_to_nchw(fake_image * 2 - 1.0)

real_image = _transpose_to_nchw(real_image * 2 - 1.0)

loss = self._lpips_func(fake_image, real_image)

return tf.reduce_mean(loss)  # Loss is N111, take mean to get scalar.

def scheduled_value(value, schedule, step, name, summary=False):

"""Create a tensor whose value depends on global step.

Args:

value: The value to adapt.

schedule: Dictionary. Expects 'steps' and 'vals'.

step: The global_step to find to.

name: Name of the value.

summary: Boolean, whether to add a summary for the scheduled value.

Returns:

tf.Tensor.

"""

with tf.name_scope("schedule" + name):

if len(schedule["steps"]) + 1 != len(schedule["vals"]):

raise ValueError("Schedule expects one more value than steps.")

steps = [int(s) for s in schedule["steps"]]

steps = tf.stack(steps + [step + 1])

idx = tf.where(step < steps)[0, 0]

value = value * tf.convert_to_tensor(schedule["vals"])[idx]

if summary:

tf.summary.scalar(name, value)

return value

<\model_gauss_ssim.py>

<train.py>


Copyright 2020 Google LLC. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");

you may not use this file except in compliance with the License.

You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software

distributed under the License is distributed on an "AS IS" BASIS,

WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

See the License for the specific language governing permissions and

limitations under the License.

==============================================================================

"""Training code for HiFiC."""

import argparse

import sys

import os

import tensorflow.compat.v1 as tf

from hific import configs

from hific import helpers

from hific import model


Show custom tf.logging calls.

tf.logging.set_verbosity(tf.logging.INFO)


Configurações para compatibilidade com RTX 3090

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

os.environ['TF_ENABLE_CUBLAS_TENSOR_OP_MATH_FP32'] = '1'

SAVE_CHECKPOINT_STEPS = 1000

def train(config_name, ckpt_dir, num_steps: int, auto_encoder_ckpt_dir,

batch_size, crop_size, lpips_weight_path, create_image_summaries,

tfds_arguments: model.TFDSArguments, local_image_dir=None):

"""Train the model."""

config = configs.get_config(config_name)

hific = model.HiFiC(config, helpers.ModelMode.TRAINING, lpips_weight_path,

auto_encoder_ckpt_dir, create_image_summaries)


import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''

Se local_image_dir for especificado, usar imagens locais ao invés de TFDS

if local_image_dir:

# Suporta PNG, JPG e JPEG

import glob as glob_module

png_images = glob_module.glob(f"{local_image_dir}/.png")

jpg_images = glob_module.glob(f"{local_image_dir}/.jpg")

jpeg_images = glob_module.glob(f"{local_image_dir}/*.jpeg")

all_images = png_images + jpg_images + jpeg_images


if not all_images:

  raise ValueError(f'No images found in {local_image_dir}. '

                  'Make sure there are *.png, *.jpg, or *.jpeg files.')


tf.logging.info(f'Found {len(all_images)} images in {local_image_dir}')

tf.logging.info(f'  PNG: {len(png_images)}, JPG: {len(jpg_images)}, '

               f'JPEG: {len(jpeg_images)}')


# Usar padrão que pega todos os formatos

images_glob = f"{local_image_dir}/*.[pjpJ][npNP][gGgE]*"

dataset = hific.build_input(batch_size, crop_size,

                            images_glob=images_glob)

else:

dataset = hific.build_input(batch_size, crop_size,

tfds_arguments=tfds_arguments)

iterator = tf.data.make_one_shot_iterator(dataset)

get_next = iterator.get_next()

hific.build_model(**get_next)

train_op = hific.train_op

hooks = hific.hooks + [tf.train.StopAtStepHook(last_step=num_steps)]

global_step = tf.train.get_or_create_global_step()

tf.logging.info(f'\nStarting MonitoredTrainingSession at {ckpt_dir}\n')


Configuração de sessão para RTX 3090

session_config = tf.ConfigProto()

session_config.gpu_options.allow_growth = True

session_config.gpu_options.per_process_gpu_memory_fraction = 0.95


Força o uso de operações mais compatíveis

session_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.OFF

with tf.train.MonitoredTrainingSession(

checkpoint_dir=ckpt_dir,

save_checkpoint_steps=SAVE_CHECKPOINT_STEPS,

hooks=hooks,

config=session_config) as sess:

if auto_encoder_ckpt_dir:

hific.restore_autoencoder(sess)

tf.logging.info('Session setup, starting training...')

while True:

if sess.should_stop():

break

global_step_np, _ = sess.run([global_step, train_op])

if global_step_np == 0:

tf.logging.info('First iteration passed.')

if global_step_np > 1 and global_step_np % 100 == 1:

tf.logging.info(f'Iteration {global_step_np}')

tf.logging.info('Training session closed.')

def parse_args(argv):

"""Parses command line arguments."""

parser = argparse.ArgumentParser(

formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--config', required=True,

choices=configs.valid_configs(),

help='The config to use.')

parser.add_argument('--ckpt_dir', required=True,

help=('Path to the folder where checkpoints should be '

'stored. Passing the same folder twice will resume '

'training.'))

parser.add_argument('--num_steps', default='1M',

help=('Number of steps to train for. Supports M and k '

'postfix for "million" and "thousand", resp.'))

parser.add_argument('--init_autoencoder_from_ckpt_dir',

metavar='AUTOENC_CKPT_DIR',

help=('If given, restore encoder, decoder, and '

'probability model from the latest checkpoint in '

'AUTOENC_CKPT_DIR. See README.md.'))

parser.add_argument('--batch_size', type=int, default=8,

help='Batch size for training.')

parser.add_argument('--crop_size', type=int, default=256,

help='Crop size for input pipeline (square crop).')

parser.add_argument('--crop_height', type=int, default=None,

help='Crop height (overrides crop_size if used with crop_width).')

parser.add_argument('--crop_width', type=int, default=None,

help='Crop width (overrides crop_size if used with crop_height).')

parser.add_argument('--lpips_weight_path',

help=('Where to store the LPIPS weights. Defaults to '

'current directory'))

parser.add_argument('--local_image_dir',

help=('Path to local directory containing training images '

'(*.png, *.jpg, *.jpeg). If provided, this will be used instead of '

'TFDS dataset. Example: tensorflow_datasets/my_images'))

helpers.add_tfds_arguments(parser)

parser.add_argument(

'--no-image-summaries',

dest='image_summaries',

action='store_false',

help='Disable image summaries.')

parser.set_defaults(image_summaries=True)

args = parser.parse_args(argv[1:])

if args.ckpt_dir == args.init_autoencoder_from_ckpt_dir:

raise ValueError('--init_autoencoder_from_ckpt_dir should not point to '

'the same folder as --ckpt_dir. If you simply want to '

'continue training the model in --ckpt_dir, you do not '

'have to pass --init_autoencoder_from_ckpt_dir, as '

'continuing training is the default.')

args.num_steps = _parse_num_steps(args.num_steps)

return args

def _parse_num_steps(steps):

try:

return int(steps)

except ValueError:

pass

if steps.endswith('M'):

return int(steps[:-1]) * 1000000

if steps.endswith('k'):

return int(steps[:-1]) * 1000

raise ValueError('Invalid num_steps value: {steps}')

def main(args):

crop_size = args.crop_size

if args.crop_height and args.crop_width:

crop_size = (args.crop_height, args.crop_width)

train(args.config, args.ckpt_dir, args.num_steps,

args.init_autoencoder_from_ckpt_dir, args.batch_size, crop_size,

args.lpips_weight_path, args.image_summaries,

helpers.parse_tfds_arguments(args), args.local_image_dir)

if name == 'main':

main(parse_args(sys.argv))

<\train.py>

<evaluate.py>


Copyright 2020 Google LLC. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");

you may not use this file except in compliance with the License.

You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software

distributed under the License is distributed on an "AS IS" BASIS,

WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

See the License for the specific language governing permissions and

limitations under the License.

==============================================================================

"""Eval models trained with train.py.

NOTE: To evaluate models used in the paper, use tfci.py! See README.md.

"""

import argparse

import collections

import glob

import itertools

import os

import sys

import csv

import numpy as np

from scipy import signal

from PIL import Image

import tensorflow.compat.v1 as tf

import tensorflow_compression as tfc

from hific import configs

from hific import helpers

from hific import model


Show custom tf.logging calls.

tf.logging.set_verbosity(tf.logging.INFO)

def eval_trained_model(config_name,

ckpt_dir,

out_dir,

images_glob,

tfds_arguments: helpers.TFDSArguments,

local_image_dir=None,

max_images=None):

"""Evaluate a trained model."""

tf.reset_default_graph()

with tf.Graph().as_default():

config = configs.get_config(config_name)

hific = model.HiFiC(config, helpers.ModelMode.EVALUATION)

# Se local_image_dir for especificado, usar imagens locais ao invés de TFDS

if local_image_dir:

# Suporta PNG, JPG e JPEG

import glob as glob_module

png_images = glob_module.glob(f"{local_image_dir}/.png")

jpg_images = glob_module.glob(f"{local_image_dir}/.jpg")

jpeg_images = glob_module.glob(f"{local_image_dir}/*.jpeg")

all_images = png_images + jpg_images + jpeg_images


  if not all_images:

    raise ValueError(f'No images found in {local_image_dir}. '

                    'Make sure there are *.png, *.jpg, or *.jpeg files.')

  

  tf.logging.info(f'Found {len(all_images)} images in {local_image_dir}')

  tf.logging.info(f'  PNG: {len(png_images)}, JPG: {len(jpg_images)}, '

                f'JPEG: {len(jpeg_images)}')

  

  # Usar padrão que pega todos os formatos

  images_glob = f"{local_image_dir}/*.[pjpJ][npNP][gGgE]*"

  dataset = hific.build_input(

      batch_size=1,

      crop_size=None,

      images_glob=images_glob)

  image_names = get_image_names(images_glob)

else:

  # Note: Automatically uses the validation split for TFDS.

  dataset = hific.build_input(

      batch_size=1,

      crop_size=None,

      images_glob=images_glob,

      tfds_arguments=tfds_arguments)

  image_names = get_image_names(images_glob)

iterator = tf.data.make_one_shot_iterator(dataset)

get_next_image = iterator.get_next()

input_image = get_next_image['input_image']

output_image, bitstring = hific.build_model(**get_next_image)

input_image = tf.cast(tf.round(input_image[0, ...]), tf.uint8)

output_image = tf.cast(tf.round(output_image[0, ...]), tf.uint8)


# Add SSIM calculation to graph

ssim_tensor = tf.image.ssim(tf.expand_dims(input_image, 0), tf.expand_dims(output_image, 0), max_val=255)

os.makedirs(out_dir, exist_ok=True)

accumulated_metrics = collections.defaultdict(list)

with tf.Session() as sess:

  hific.restore_trained_model(sess, ckpt_dir)

  hific.prepare_for_arithmetic_coding(sess)

  for i in itertools.count(0):

    if max_images and i == max_images:

      break

    try:

      inp_np, otp_np, bitstring_np, ssim_val = \

        sess.run([input_image, output_image, bitstring, ssim_tensor])

      h, w, c = inp_np.shape

      assert c == 3

      bpp = get_arithmetic_coding_bpp(

          bitstring, bitstring_np, num_pixels=h * w)

      metrics = {'psnr': get_psnr(inp_np, otp_np),

                'ws-psnr': get_ws_psnr(inp_np, otp_np),

                'ssim': get_ssim(inp_np, otp_np),

                'ws_ssim': get_ws_ssim(inp_np, otp_np),

                'mse': get_mse(inp_np, otp_np),

                'ws-mse': get_ws_mse(inp_np, otp_np),

                'bpp_real': bpp}

      metrics_str = ' / '.join(f'{metric}: {value:.5f}'

                              for metric, value in metrics.items())

      print(f'Image {i: 4d}: {metrics_str}, saving in {out_dir}...')

      for metric, value in metrics.items():

        accumulated_metrics[metric].append(value)

      # Save images.

      name = image_names.get(i, f'img_{i:010d}')

      Image.fromarray(inp_np).save(

          os.path.join(out_dir, f'{name}_inp.png'))

      Image.fromarray(otp_np).save(

          os.path.join(out_dir, f'{name}_otp_{bpp:.3f}.png'))

    except tf.errors.OutOfRangeError:

      print('No more inputs.')

      break

means = {metric: np.mean(values) for metric, values in accumulated_metrics.items()}

print('\n'.join(f'{metric}: {val}' for metric, val in means.items()))

print('Done!')

return means

def get_arithmetic_coding_bpp(bitstring, bitstring_np, num_pixels):

"""Calculate bitrate we obtain with arithmetic coding."""


TODO(fab-jul): Add compress and decompress methods.

packed = tfc.PackedTensors()

packed.pack(tensors=bitstring, arrays=bitstring_np)

return len(packed.string) * 8 / num_pixels

def weights(height, width): # calculo da matriz de pesos, otimizada

phis = np.arange(height+1)np.pi/height

deltaTheta = 2np.pi/width

column = deltaTheta * (-np.cos(phis[1:]) + np.cos(phis[:-1]))

return np.repeat(column[:, np.newaxis], width, 1)

def get_psnr(inp, otp):

mse = np.mean(np.square(inp.astype(np.float32) - otp.astype(np.float32)))

psnr = 20. * np.log10(255.) - 10. * np.log10(mse)

return psnr

def get_ws_psnr(img1, img2, max_val=255.): # cálculo em 3 canais, otimizada

img1 = np.float64(img1)

img2 = np.float64(img2)


height, width = img1.shape[0], img1.shape[1]

# calcula os pesos e expande os pesos para shape (height, width, 1)

w = weights(height, width)

w_expanded = w[:, :, np.newaxis] # (height, width, 1)


# calcula o WS-MSE para todos os canais (olhar o código WSMSE.py)

squared_diff = (img1 - img2) ** 2

weighted_squared_diff = squared_diff * w_expanded

wmse_three_channel = np.sum(np.sum(weighted_squared_diff, 0), 0) / (4 * np.pi)

# calcula PSNR para cada canal

wmse_three_channel = np.where(wmse_three_channel == 0, 1e-10, wmse_three_channel) # evita divisão por zero, pois iria para infinito (ainda fica com um valor muito alto)

wspsnr_three_channel = 10 * np.log10(max_val**2 / wmse_three_channel)

return np.mean(wspsnr_three_channel)

def get_ssim(img1, img2, K1=.01, K2=.03, L=255):

def __fspecial_gauss(size, sigma):

x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

g = np.exp(-((x2 + y2)/(2.0*sigma**2)))

return g/g.sum()

img1 = np.float64(img1)

img2 = np.float64(img2)

k = 11

sigma = 1.5

window = __fspecial_gauss(k, sigma)

C1 = (K1L)**2

C2 = (K2L)**2

ssim_channels = np.zeros(3)

for c in range(3):

channel1 = img1[:, :, c]

channel2 = img2[:, :, c]


  mu1 = signal.convolve2d(channel1, window, 'valid')

  mu2 = signal.convolve2d(channel2, window, 'valid')

  

  mu1_sq = mu1 * mu1

  mu2_sq = mu2 * mu2

  mu1_mu2 = mu1 * mu2

  

  sigma1_sq = signal.convolve2d(channel1 * channel1, window, 'valid') - mu1_sq

  sigma2_sq = signal.convolve2d(channel2 * channel2, window, 'valid') - mu2_sq

  sigma12 = signal.convolve2d(channel1 * channel2, window, 'valid') - mu1_mu2

  

  numerator = (2*mu1_mu2 + C1) * (2*sigma12 + C2)

  denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

  ssim_map = numerator / denominator

  

  ssim_channels[c] = np.mean(ssim_map)

return np.mean(ssim_channels)

def get_ws_ssim(img1, img2, K1=.01, K2=.03, L=255):

def __fspecial_gauss(size, sigma):

x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

g = np.exp(-((x2 + y2)/(2.0*sigma**2)))

return g/g.sum()

img1 = np.float64(img1)

img2 = np.float64(img2)

k = 11

sigma = 1.5

window = __fspecial_gauss(k, sigma)

window2 = np.zeros_like(window)

window2[k//2, k//2] = 1

C1 = (K1L)**2

C2 = (K2L)**2

height, width = img1.shape[0], img1.shape[1]

W = weights(height, width)

Wi = signal.convolve2d(W, window2, 'valid')

weight_sum = np.sum(Wi)

wsssim_channels = np.zeros(3)

for c in range(3):

channel1 = img1[:, :, c]

channel2 = img2[:, :, c]


  mu1 = signal.convolve2d(channel1, window, 'valid')

  mu2 = signal.convolve2d(channel2, window, 'valid')

  

  mu1_sq = mu1 * mu1

  mu2_sq = mu2 * mu2

  mu1_mu2 = mu1 * mu2

  

  sigma1_sq = signal.convolve2d(channel1 * channel1, window, 'valid') - mu1_sq

  sigma2_sq = signal.convolve2d(channel2 * channel2, window, 'valid') - mu2_sq

  sigma12 = signal.convolve2d(channel1 * channel2, window, 'valid') - mu1_mu2

  

  numerator = (2*mu1_mu2 + C1) * (2*sigma12 + C2)

  denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

  ssim_map = (numerator / denominator) * Wi

  

  wsssim_channels[c] = np.sum(ssim_map) / weight_sum

return np.mean(wsssim_channels)

def get_mse(inp, otp):

mse = np.mean(np.square(inp.astype(np.float32) - otp.astype(np.float32)))

return mse

def get_ws_mse(img1, img2):

img1 = np.float64(img1)

img2 = np.float64(img2)


height, width = img1.shape[0], img1.shape[1]


# calcula os pesos e expande os pesos para shape (height, width, 1)

w = weights(height, width)

w_expanded = w[:, :, np.newaxis] # (height, width, 1)


# (img1 - img2)^2 tem shape (height, width, 3)

# w_expanded tem shape (height, width, 1)

# a multiplicação faz broadcast automaticamente

squared_diff = (img1 - img2) ** 2

weighted_squared_diff = squared_diff * w_expanded


# soma sobre altura e largura, mantendo os canais separados

r = 1

wmse_three_channel = np.sum(np.sum(weighted_squared_diff, 0), 0) / (4 * np.pi * r)

# média dos 3 canais

return np.mean(wmse_three_channel)

def get_image_names(images_glob):

if not images_glob:

return {}

return {i: os.path.splitext(os.path.basename(p))[0]

for i, p in enumerate(sorted(glob.glob(images_glob)))}

def parse_args(argv):

"""Parses command line arguments."""

parser = argparse.ArgumentParser(

formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--config', required=True, nargs='+',

choices=configs.valid_configs(),

help='The config(s) to use.')

parser.add_argument('--ckpt_dir', required=True, nargs='+',

help=('Path(s) to the folder where checkpoints of the '

'trained model are.'))

parser.add_argument('--out_dir', required=True, nargs='+', help='Where to save outputs (one per model).')

parser.add_argument('--group', nargs='+', help='Group name for each model.')

parser.add_argument('--results_csv', default='evaluation_results.csv', help='Output CSV file for results.')

parser.add_argument('--images_glob', help='If given, use TODO')

parser.add_argument('--local_image_dir',

help=('Path to local directory containing images to evaluate '

'(*.png, *.jpg, *.jpeg). If provided, this will be used '

'instead of TFDS dataset. Example: my_images/validation'))

parser.add_argument('--max_images', type=int,

help='Maximum number of images to evaluate. If not specified, '

'evaluates all images.')

helpers.add_tfds_arguments(parser)

args = parser.parse_args(argv[1:])

return args

def main(args):


Ensure arguments are lists (nargs='+' makes them lists)

configs_list = args.config

ckpt_dirs = args.ckpt_dir

out_dirs = args.out_dir

groups = args.group if args.group else ['Default'] * len(configs_list)

if not (len(configs_list) == len(ckpt_dirs) == len(out_dirs)):

raise ValueError("All model arguments (config, ckpt_dir, out_dir) must have the same length.")

if len(groups) != len(configs_list):

if len(groups) == 1:

groups = groups * len(configs_list)

else:

raise ValueError("Group argument length must match other arguments or be 1.")

all_results = {}

for conf, ckpt, out, grp in zip(configs_list, ckpt_dirs, out_dirs, groups):

print(f"Evaluating model: Config={conf}, Group={grp}, Ckpt={ckpt}")

metrics = eval_trained_model(conf, ckpt, out,

args.images_glob,

helpers.parse_tfds_arguments(args),

args.local_image_dir,

args.max_images)


  model_id = os.path.basename(ckpt.rstrip('/'))

  if not model_id: model_id = "model"

  

  if model_id in all_results:

      model_id = f"{model_id}_{conf}"

  

  all_results[model_id] = {'group': grp, **metrics}

if args.results_csv:

model_names = list(all_results.keys())

if model_names:

first_metrics = all_results[model_names[0]]

metric_names = list(first_metrics.keys())

if 'group' in metric_names:

metric_names.remove('group')

metric_names = ['group'] + metric_names


      with open(args.results_csv, 'w', newline='') as csvfile:

          writer = csv.writer(csvfile)

          header = ['Metric'] + model_names

          writer.writerow(header)

          

          for metric in metric_names:

              row = [metric]

              for model in model_names:

                  row.append(all_results[model].get(metric, 'N/A'))

              writer.writerow(row)

  print(f"Results saved to {args.results_csv}")

if name == 'main':

main(parse_args(sys.argv))

<\evaluate.py>

<configs.py>


Copyright 2020 Google LLC. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");

you may not use this file except in compliance with the License.

You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software

distributed under the License is distributed on an "AS IS" BASIS,

WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

See the License for the specific language governing permissions and

limitations under the License.

==============================================================================

"""Configurations for HiFiC."""

from hific import helpers

_CONFIGS = {

# HiFiC variants used in the paper: Low, Mid, High target bitrates.

'hific_lo': helpers.Config(

model_type=helpers.ModelType.COMPRESSION_GAN,

lambda_schedule=helpers.Config(

vals=[2., 1.],

steps=[50000]),

lr=1e-4,

lr_schedule=helpers.Config(

vals=[1., 0.1],

steps=[500000]),

num_steps_disc=1,

loss_config=helpers.Config(

CP=0.1 * 1.5 ** 1,

C=0.1 * 2. ** -5,

CD=0.75,

target=0.14,  # HiFiCLo target (rt=0.14 in paper)

lpips_weight=1.,

target_schedule=helpers.Config(

vals=[0.20/0.14, 1.],

steps=[50000]),

lmbda_a=0.1 * 2. ** -6,

lmbda_b=0.1 * 2. ** 1,

)

),

'hific_mi': helpers.Config(

model_type=helpers.ModelType.COMPRESSION_GAN,

lambda_schedule=helpers.Config(

vals=[2., 1.],

steps=[50000]),

lr=1e-4,

lr_schedule=helpers.Config(

vals=[1., 0.1],

steps=[500000]),

num_steps_disc=1,

loss_config=helpers.Config(

CP=0.1 * 1.5 ** 2,

C=0.1 * 2. ** -5,

CD=0.75,

target=0.3,  # HiFiCMi target (rt=0.3 in paper)

lpips_weight=1.,

target_schedule=helpers.Config(

vals=[0.20/0.14, 1.],

steps=[50000]),

lmbda_a=0.1 * 2. ** -6,

lmbda_b=0.1 * 2. ** 1,

)

),

'hific_hi': helpers.Config(

model_type=helpers.ModelType.COMPRESSION_GAN,

lambda_schedule=helpers.Config(

vals=[2., 1.],

steps=[50000]),

lr=1e-4,

lr_schedule=helpers.Config(

vals=[1., 0.1],

steps=[500000]),

num_steps_disc=1,

loss_config=helpers.Config(

CP=0.1 * 1.5 ** 3,

C=0.1 * 2. ** -5,

CD=0.75,

target=0.45,  # HiFiCHi target (rt=0.45 in paper)

lpips_weight=1.,

target_schedule=helpers.Config(

vals=[0.20/0.14, 1.],

steps=[50000]),

lmbda_a=0.1 * 2. ** -6,

lmbda_b=0.1 * 2. ** 1,

)

),

# Baseline (no GAN) variants matching HiFiC targets used for initialization.

'mselpips_lo': helpers.Config(

model_type=helpers.ModelType.COMPRESSION,

lambda_schedule=helpers.Config(

vals=[2., 1.],

steps=[50000]),

lr=1e-4,

lr_schedule=helpers.Config(

vals=[1., 0.1],

steps=[500000]),

num_steps_disc=None,

loss_config=helpers.Config(

CP=None,

C=0.1 * 2. ** -5,

CD=0.75,

target=0.14,

lpips_weight=1.,

target_schedule=helpers.Config(

vals=[0.20/0.14, 1.],

steps=[50000]),

lmbda_a=0.1 * 2. ** -6,

lmbda_b=0.1 * 2. ** 1,

)

),

'mselpips_mi': helpers.Config(

model_type=helpers.ModelType.COMPRESSION,

lambda_schedule=helpers.Config(

vals=[2., 1.],

steps=[50000]),

lr=1e-4,

lr_schedule=helpers.Config(

vals=[1., 0.1],

steps=[500000]),

num_steps_disc=None,

loss_config=helpers.Config(

CP=None,

C=0.1 * 2. ** -5,

CD=0.75,

target=0.3,

lpips_weight=1.,

target_schedule=helpers.Config(

vals=[0.20/0.14, 1.],

steps=[50000]),

lmbda_a=0.1 * 2. ** -6,

lmbda_b=0.1 * 2. ** 1,

)

),

'mselpips_hi': helpers.Config(

model_type=helpers.ModelType.COMPRESSION,

lambda_schedule=helpers.Config(

vals=[2., 1.],

steps=[50000]),

lr=1e-4,

lr_schedule=helpers.Config(

vals=[1., 0.1],

steps=[500000]),

num_steps_disc=None,

loss_config=helpers.Config(

CP=None,

C=0.1 * 2. ** -5,

CD=0.75,

target=0.45,

lpips_weight=1.,

target_schedule=helpers.Config(

vals=[0.20/0.14, 1.],

steps=[50000]),

lmbda_a=0.1 * 2. ** -6,

lmbda_b=0.1 * 2. ** 1,

)

),

}

def get_config(config_name):

if config_name not in _CONFIGS:

raise ValueError(f'Unknown config_name={config_name} not in '

f'{_CONFIGS.keys()}')

return _CONFIGS[config_name]

def valid_configs():

return list(_CONFIGS.keys())

<\configs.py>

</CODE>

<RESULTS>

Metric,mse_lpips_lo_200k,mse_lpips_mi_200k,mse_lpips_hi_200k,mse_ssim_lo_200k,mse_ssim_mi_200k,mse_ssim_hi_200k,WSmse_WSssim_lo_200k,WSmse_WSssim_mi_200k,WSmse_WSssim_hi_200k,gauss_WSmse_WSssim_lo_200k,gauss_WSmse_WSssim_mi_200k,gauss_WSmse_WSssim_hi_200k,SWHDC_WSmse_WSssim_lo_200k,SWHDC_WSmse_WSssim_mi_200k,SWHDC_WSmse_WSssim_hi_200k,mse_ssim_256x512_lo_200k,mse_ssim_256x512_mi_200k,mse_ssim_256x512_hi_200k,WSmse_WSssim_256x512_lo_200k,WSmse_WSssim_256x512_mi_200k,WSmse_WSssim_256x512_hi_200k,SWHDC_WSmse_WSssim_256x512_lo_200k,SWHDC_WSmse_WSssim_256x512_mi_200k,SWHDC_WSmse_WSssim_256x512_hi_200k,SWHDC_learn_WSmse_WSssim_lo_200k,SWHDC_learn_WSmse_WSssim_mi_200k,SWHDC_learn_WSmse_WSssim_hi_200k

group,LPIPS,LPIPS,LPIPS,SSIM,SSIM,SSIM,WSSSIM,WSSSIM,WSSSIM,GAUSS,GAUSS,GAUSS,SWHDC,SWHDC,SWHDC,SSIM256x512,SSIM256x512,SSIM256x512,WSSSIM256x512,WSSSIM256x512,WSSSIM256x512,SWHDC256x512,SWHDC256x512,SWHDC256x512,SWHDCLEARN256x512,SWHDCLEARN256x512,SWHDCLEARN256x512

psnr,27.63819013415111,29.308886949006048,30.66909792401724,27.913005057755438,29.558445715688993,30.413809402568468,27.805743161622015,29.63422320662273,30.736582302831938,27.996176186982122,29.508231663488676,30.5115021703475,27.802710596505133,29.5142542757425,30.730481846912035,28.244009915772406,29.58713303385509,30.786671383960375,28.020583955231634,29.7589329161081,30.626751009407965,28.19950877009166,29.744981988055834,30.664117161217657,28.183869321925768,29.79582055388225,30.73607644218537

ws-psnr,26.75298874198341,28.40914879436348,29.756460779395617,27.039847462214937,28.663391952895054,29.56900233311966,26.985156933784747,28.7786488785105,29.845107789081066,27.15315339422858,28.681455454498362,29.674860862430915,27.03322073772802,28.68725735367188,29.884980087844134,27.36046487924531,28.803042155164853,29.92514093583303,27.51039868783179,29.15673619881093,30.18473966304009,27.815128110090133,29.48477460181453,30.455698674714005,27.73705370597001,29.377518870047142,30.372350438019422

ssim,0.8023522830332124,0.8583090589600909,0.894935298266264,0.8332037846367867,0.8813714733247463,0.9067605033479753,0.830755110160382,0.8823331199691309,0.9064107364901146,0.833066025640922,0.8801339246894381,0.9045449000546684,0.8283950262710899,0.8781157971956316,0.9051523161014511,0.8455561926745594,0.8930220050619834,0.9140298571539636,0.8403718517968285,0.8865953902730839,0.9111752037741129,0.8432572878017363,0.8895733702386306,0.908045792528386,0.8416115597462186,0.8929752592571957,0.9109087405434615

ws_ssim,0.7782766554895135,0.8407483139634203,0.8823245476415337,0.8105637613520231,0.8655677131360832,0.8947723301702527,0.810137512503441,0.8677993815316485,0.8949019589406197,0.8125777642104988,0.8663886956584678,0.892851627508121,0.8094880472939517,0.8646441094947962,0.8944734397837234,0.8243167981236657,0.8784002171456853,0.9028045601956745,0.8236627580258842,0.8805697932011652,0.9025788351991269,0.8348274284676975,0.8889110849863723,0.9083685527084006,0.8314326602091731,0.8875819881206423,0.9069790358820036

mse,122.82638,84.00736,61.51694,114.700455,78.681145,63.843975,117.02519,77.24814,60.19753,112.399895,79.25818,62.957882,117.11452,79.51496,60.52094,105.67318,76.61544,58.7876,110.99051,76.31557,60.721935,107.875296,75.3718,60.793726,109.287094,75.63739,61.12455

ws-mse,148.5549308750784,101.94586644764426,74.87415514467656,138.86566983742648,95.70764944291884,77.0355480716496,139.92682531518827,93.07460427926766,73.07907940727411,134.96870636282932,94.99510478748556,75.81720827305297,138.5603448988665,95.05998729504759,72.61389899620377,128.3884676992669,91.91030225384654,71.24749619448203,123.5716386618822,84.5495885297098,66.66851482679057,118.04101083697395,79.93824040563357,63.90804480280144,120.05861276779719,82.22596726010429,65.97608845025233

bpp_real,0.13492635091145833,0.26144866943359374,0.4152008056640625,0.14495646158854167,0.26772969563802085,0.3843994140625,0.14034525553385416,0.26784006754557294,0.3880223592122396,0.14137776692708334,0.2591959635416667,0.38290659586588544,0.13837992350260417,0.25867919921875,0.3876307169596354,0.15258992513020833,0.2885248819986979,0.42391510009765626,0.1578033447265625,0.2948603312174479,0.41244405110677085,0.16011962890625,0.2869160970052083,0.391900634765625,0.15065384928385417,0.29060272216796874,0.399849365234375

</RESULTS>

<ABSTRACT>

Omnidirectional (360\textdegree{}) images are becoming increasingly popular in virtual reality and immersive applications. The equirectangular projection format, however, introduces severe distortions, especially near the poles, and the visually salient equatorial region demands higher fidelity. In this paper, we adapt the generative image compression model HiFiC (High-Fidelity Image Compression) to the equirectangular domain, resulting in HiFiC360. We investigate three key modifications: (i) replacing the perspective-trained LPIPS loss with structural similarity (SSIM) and its latitude-weighted variant (WS-SSIM), (ii) incorporating a spherical-aware convolution (SWHDC) that respects the equirectangular geometry, and (iii) modifying the training crop strategy to favor the equatorial region. Models are trained on the SUN360 dataset and evaluated on CTC360 images using standard and weighted distortion metrics. Quantitative results show that our best configuration --- a fixed-weights SWHDC with WS-SSIM loss --- achieves substantial improvements in weighted PSNR and weighted SSIM metrics over the baseline HiFIC, while maintaining competitive bitrates and similar training time.

</ABSTRACT>

<INSTRUCTIONS>

Elaborate the sections for the paper 'HiFiC360: Adapting High Fidelity Compression for Equirectangular Images': 'Introduction', 'Datasets' 'Methodology', 'Results and Analysis' and 'Conclusion and Future Work', given the content described. Your output should be in LaTeX format.


Write more content for these steps:

Introduction: (i) A paragraph talking about Omnidirectional images, that they, however, uses a lot of memory and, therefore, needs compression. Talk about the distortions in equirectangular (ERP) images. (ii) A paragraph talking about learned image compression, and HiFIC that was designed for conventional perspective images; its components, such as the LPIPS perceptual loss and standard convolutions, do not account for the specific geometry of equirectangular content. (iii) Talk briefly about our ablation study, converting the two loss terms: distortion loss (MSE) and the perceptual loss (LPIPS) to MSE + SSIM and WS-MSE + WS-SSIM. Also talk about our experiment with Gaussian Distribution (which didn't had any impact) and the convolutional adaptations (SWHDC and SWHDC with learned weights). After, write about the 1024x512 pixel datasets, and that the model train by slicing crops among the image, and that we trained models using 256x512 size (which will get the entire latitude of the images).


Datasets: (i) Write a subsection about the SUN360 training set; a collection of 48319 images, including indoor scenes, landscapes, and urban environments. Write that is good to train a compression model like this one with many and different images. (ii) Write a subsection about the CTC360 dataset, which contains 30 images of indoor and urban environments

</INSTRUCTIONS>