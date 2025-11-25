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

