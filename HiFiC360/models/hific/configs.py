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
"""Configurations for HiFiC."""

from hific import helpers


_CONFIGS = {
    'hific': helpers.Config(
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
            # Constrain rate:
            #   Loss = C * (1/lambda * R + CD * D) + CP * P
            #       where
            #          lambda = lambda_a if current_bpp > target
            #                   lambda_b otherwise.
            CP=0.1 * 1.5 ** 1,  # Sweep over 0.1 * 1.5 ** x
            C=0.1 * 2. ** -5,
            CD=0.75,
            target=0.14,  # This is $r_t$ in the paper.
            lpips_weight=1.,
            target_schedule=helpers.Config(
                vals=[0.20/0.14, 1.],  # Factor is independent of target.
                steps=[50000]),
            lmbda_a=0.1 * 2. ** -6,
            lmbda_b=0.1 * 2. ** 1,
            )
        ),
    'mselpips': helpers.Config(
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
            # Constrain rate:
            #   Loss = C * (1/lambda * R + CD * D) + CP * P
            #       where
            #          lambda = lambda_a if current_bpp > target
            #                   lambda_b otherwise.
            CP=None,
            C=0.1 * 2. ** -5,
            CD=0.75,
            target=0.14,  # This is $r_t$ in the paper.
            lpips_weight=1.,
            target_schedule=helpers.Config(
                vals=[0.20/0.14, 1.],  # Factor is independent of target.
                steps=[50000]),
            lmbda_a=0.1 * 2. ** -6,
            lmbda_b=0.1 * 2. ** 1,
            )
        ),
    'hific-360': helpers.Config(
        model_type=helpers.ModelType.COMPRESSION_GAN,
        lambda_schedule=helpers.Config(
            vals=[2., 1.],
            steps=[50000]),
        lr=8e-5,  # Slightly lower learning rate for 360 images
        lr_schedule=helpers.Config(
            vals=[1., 0.1],
            steps=[500000]),
        num_steps_disc=1,
        loss_config=helpers.Config(
            # Optimized for 360-degree images with latitude-weighted LPIPS
            CP=0.1 * 1.5 ** 1,
            C=0.1 * 2. ** -5,
            CD=0.65,  # Slightly reduced distortion weight
            target=0.16,  # Slightly higher target for 360 images
            lpips_weight=1.2,  # Increased LPIPS weight for better perceptual quality
            target_schedule=helpers.Config(
                vals=[0.20/0.16, 1.],
                steps=[50000]),
            lmbda_a=0.1 * 2. ** -6,
            lmbda_b=0.1 * 2. ** 1,
            )
        ),
    'hific-360-lo': helpers.Config(
        model_type=helpers.ModelType.COMPRESSION_GAN,
        lambda_schedule=helpers.Config(
            vals=[2., 1.],
            steps=[50000]),
        lr=8e-5,
        lr_schedule=helpers.Config(
            vals=[1., 0.1],
            steps=[500000]),
        num_steps_disc=1,
        loss_config=helpers.Config(
            # Low bitrate configuration for 360 images (~0.08-0.12 bpp)
            CP=0.1 * 1.5 ** 1,
            C=0.1 * 2. ** -5,
            CD=0.6,  # Lower distortion weight for more compression
            target=0.10,  # Low target bitrate
            lpips_weight=1.0,  # Standard LPIPS weight
            target_schedule=helpers.Config(
                vals=[0.15/0.10, 1.],
                steps=[50000]),
            lmbda_a=0.1 * 2. ** -7,  # Stronger rate constraint
            lmbda_b=0.1 * 2. ** 0,
            )
        ),
    'hific-360-mi': helpers.Config(
        model_type=helpers.ModelType.COMPRESSION_GAN,
        lambda_schedule=helpers.Config(
            vals=[2., 1.],
            steps=[50000]),
        lr=8e-5,
        lr_schedule=helpers.Config(
            vals=[1., 0.1],
            steps=[500000]),
        num_steps_disc=1,
        loss_config=helpers.Config(
            # Medium bitrate configuration for 360 images (~0.16-0.24 bpp)
            CP=0.1 * 1.5 ** 1,
            C=0.1 * 2. ** -5,
            CD=0.65,  # Balanced distortion weight
            target=0.20,  # Medium target bitrate
            lpips_weight=1.2,  # Enhanced LPIPS weight
            target_schedule=helpers.Config(
                vals=[0.25/0.20, 1.],
                steps=[50000]),
            lmbda_a=0.1 * 2. ** -6,  # Balanced rate constraint
            lmbda_b=0.1 * 2. ** 1,
            )
        ),
    'hific-360-hi': helpers.Config(
        model_type=helpers.ModelType.COMPRESSION_GAN,
        lambda_schedule=helpers.Config(
            vals=[2., 1.],
            steps=[50000]),
        lr=8e-5,
        lr_schedule=helpers.Config(
            vals=[1., 0.1],
            steps=[500000]),
        num_steps_disc=1,
        loss_config=helpers.Config(
            # High bitrate configuration for 360 images (~0.30-0.50 bpp)
            CP=0.1 * 1.5 ** 1,
            C=0.1 * 2. ** -5,
            CD=0.75,  # Higher distortion weight for quality
            target=0.35,  # High target bitrate
            lpips_weight=1.5,  # Maximum LPIPS weight for best quality
            target_schedule=helpers.Config(
                vals=[0.45/0.35, 1.],
                steps=[50000]),
            lmbda_a=0.1 * 2. ** -5,  # Relaxed rate constraint
            lmbda_b=0.1 * 2. ** 2,
            )
        ),
    'mselpips-360': helpers.Config(
        model_type=helpers.ModelType.COMPRESSION,
        lambda_schedule=helpers.Config(
            vals=[2., 1.],
            steps=[50000]),
        lr=8e-5,  # Slightly lower learning rate for 360 images
        lr_schedule=helpers.Config(
            vals=[1., 0.1],
            steps=[500000]),
        num_steps_disc=None,
        loss_config=helpers.Config(
            # MSE + LPIPS for 360-degree images
            CP=None,
            C=0.1 * 2. ** -5,
            CD=0.65,  # Slightly reduced distortion weight
            target=0.16,  # Slightly higher target for 360 images
            lpips_weight=1.2,  # Increased LPIPS weight
            target_schedule=helpers.Config(
                vals=[0.20/0.16, 1.],
                steps=[50000]),
            lmbda_a=0.1 * 2. ** -6,
            lmbda_b=0.1 * 2. ** 1,
            )
        ),
    'mselpips-360-lo': helpers.Config(
        model_type=helpers.ModelType.COMPRESSION,
        lambda_schedule=helpers.Config(
            vals=[2., 1.],
            steps=[50000]),
        lr=8e-5,
        lr_schedule=helpers.Config(
            vals=[1., 0.1],
            steps=[500000]),
        num_steps_disc=None,
        loss_config=helpers.Config(
            # Low bitrate MSE + LPIPS for 360 images
            CP=None,
            C=0.1 * 2. ** -5,
            CD=0.6,
            target=0.12,
            lpips_weight=1.0,
            target_schedule=helpers.Config(
                vals=[0.18/0.12, 1.],
                steps=[50000]),
            lmbda_a=0.1 * 2. ** -7,
            lmbda_b=0.1 * 2. ** 0,
            )
        ),
    'mselpips-360-hi': helpers.Config(
        model_type=helpers.ModelType.COMPRESSION,
        lambda_schedule=helpers.Config(
            vals=[2., 1.],
            steps=[50000]),
        lr=8e-5,
        lr_schedule=helpers.Config(
            vals=[1., 0.1],
            steps=[500000]),
        num_steps_disc=None,
        loss_config=helpers.Config(
            # High bitrate MSE + LPIPS for 360 images
            CP=None,
            C=0.1 * 2. ** -5,
            CD=0.75,
            target=0.30,
            lpips_weight=1.4,
            target_schedule=helpers.Config(
                vals=[0.40/0.30, 1.],
                steps=[50000]),
            lmbda_a=0.1 * 2. ** -5,
            lmbda_b=0.1 * 2. ** 2,
            )
        )
}


def get_config(config_name):
  if config_name not in _CONFIGS:
    raise ValueError(f'Unknown config_name={config_name} not in '
                     f'{_CONFIGS.keys()}')
  return _CONFIGS[config_name]


def valid_configs():
  return list(_CONFIGS.keys())

