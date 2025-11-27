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


# Show custom tf.logging calls.
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
      png_images = glob_module.glob(f"{local_image_dir}/*.png")
      jpg_images = glob_module.glob(f"{local_image_dir}/*.jpg")
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
                    'ssim': ssim_val[0],
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
  # TODO(fab-jul): Add `compress` and `decompress` methods.
  packed = tfc.PackedTensors()
  packed.pack(tensors=bitstring, arrays=bitstring_np)
  return len(packed.string) * 8 / num_pixels

def weights(height, width): # calculo da matriz de pesos, otimizada
    phis = np.arange(height+1)*np.pi/height
    deltaTheta = 2*np.pi/width
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

def get_ws_ssim(img1, img2, K1=.01, K2=.03, L=255):
  def __fspecial_gauss(size, sigma):
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()

  img1 = np.float64(img1)
  img2 = np.float64(img2)
  
  k = 11
  sigma = 1.5
  window = __fspecial_gauss(k, sigma)
  window2 = np.zeros_like(window)
  window2[k//2, k//2] = 1 

  C1 = (K1*L)**2
  C2 = (K2*L)**2
  
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
  # Ensure arguments are lists (nargs='+' makes them lists)
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


if __name__ == '__main__':
  main(parse_args(sys.argv))
