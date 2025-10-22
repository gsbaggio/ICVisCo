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
                      auto_encoder_ckpt_dir, create_image_summaries)
  
  import os
  os.environ['CUDA_VISIBLE_DEVICES'] = ''

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

  # Configuração de sessão para RTX 3090
  session_config = tf.ConfigProto()
  session_config.gpu_options.allow_growth = True
  session_config.gpu_options.per_process_gpu_memory_fraction = 0.95
  # Força o uso de operações mais compatíveis
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
                      help='Crop size for input pipeline.')
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
  train(args.config, args.ckpt_dir, args.num_steps,
        args.init_autoencoder_from_ckpt_dir, args.batch_size, args.crop_size,
        args.lpips_weight_path, args.image_summaries,
        helpers.parse_tfds_arguments(args), args.local_image_dir)


if __name__ == '__main__':
  main(parse_args(sys.argv))
