# Copyright 2024 Gabriel Baggio. All Rights Reserved.
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
"""LPIPS loss adapted for 360-degree images with latitude-based weighting."""

import numpy as np
import tensorflow.compat.v1 as tf

from hific import helpers

# Disable TF2 behavior for compatibility with HiFiC
tf.disable_v2_behavior()


class LPIPS360Loss(object):
  """Calculate LPIPS loss adapted for 360-degree images with latitude weighting."""

  def __init__(self, weight_path, latitude_weight_type='cosine', pole_weight=0.5):
    """Initialize LPIPS 360 loss.
    
    Args:
      weight_path: Path to LPIPS weights file.
      latitude_weight_type: Type of latitude weighting ('cosine', 'linear', 'quadratic').
      pole_weight: Weight factor for polar regions (0.0 to 1.0).
                  Lower values reduce importance of polar regions.
    """
    helpers.ensure_lpips_weights_exist(weight_path)
    
    self._latitude_weight_type = latitude_weight_type
    self._pole_weight = pole_weight

    def wrap_frozen_graph(graph_def, inputs, outputs):
      def _imports_graph_def():
        tf.graph_util.import_graph_def(graph_def, name="")

      wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
      import_graph = wrapped_import.graph
      return wrapped_import.prune(
          tf.nest.map_structure(import_graph.as_graph_element, inputs),
          tf.nest.map_structure(import_graph.as_graph_element, outputs))

    # Pack LPIPS network into a tf function
    graph_def = tf.compat.v1.GraphDef()
    with open(weight_path, "rb") as f:
      graph_def.ParseFromString(f.read())
    self._lpips_func = tf.function(
        wrap_frozen_graph(
            graph_def, inputs=("0:0", "1:0"), outputs="Reshape_10:0"))

  def _create_latitude_weights(self, height, width):
    y_coords = tf.linspace(-np.pi/2, np.pi/2, height)
    
    latitude_weights = tf.cos(y_coords)

    latitude_weights = latitude_weights * (1.0 - self._pole_weight) + self._pole_weight
    
    latitude_weights = tf.expand_dims(latitude_weights, axis=1)
    latitude_weights = tf.tile(latitude_weights, [1, width])
    
    return latitude_weights

  def _compute_weighted_lpips_map(self, fake_image, real_image):

    def _transpose_to_nchw(x):
      return tf.transpose(x, (0, 3, 1, 2))
    
    fake_image_formatted = _transpose_to_nchw(fake_image * 2 - 1.0)
    real_image_formatted = _transpose_to_nchw(real_image * 2 - 1.0)
    
    lpips_output = self._lpips_func(fake_image_formatted, real_image_formatted)
    
    if len(lpips_output.shape) == 4:
      # Convert from NCHW to NHWC if needed
      if lpips_output.shape[1] == 1:
        lpips_map = tf.transpose(lpips_output, (0, 2, 3, 1))
      else:
        lpips_map = lpips_output
    else:
      return tf.reduce_mean(lpips_output)
    
    return lpips_map

  def __call__(self, fake_image, real_image):
    batch_size = tf.shape(fake_image)[0]
    height = tf.shape(fake_image)[1]
    width = tf.shape(fake_image)[2]
    
    # Try to get spatial LPIPS map
    try:
      lpips_map = self._compute_weighted_lpips_map(fake_image, real_image)
      
      if len(lpips_map.shape) >= 2:
        latitude_weights = self._create_latitude_weights(height, width)
        
        latitude_weights = tf.expand_dims(latitude_weights, axis=0)  # [1, height, width]
        latitude_weights = tf.tile(latitude_weights, [batch_size, 1, 1])  # [batch, height, width]
        
        if len(lpips_map.shape) == 4:
          latitude_weights = tf.expand_dims(latitude_weights, axis=-1)  # [batch, height, width, 1]
        
        # Apply latitude weighting
        weighted_lpips_map = lpips_map * latitude_weights
        
        # Compute weighted average
        total_weight = tf.reduce_sum(latitude_weights)
        weighted_loss = tf.reduce_sum(weighted_lpips_map) / total_weight
        
        return weighted_loss
        
    except Exception as e:
      tf.logging.warning(f"Could not compute spatial LPIPS map: {e}. Falling back to standard LPIPS.")
    
    def _transpose_to_nchw(x):
      return tf.transpose(x, (0, 3, 1, 2))
    
    fake_image_formatted = _transpose_to_nchw(fake_image * 2 - 1.0)
    real_image_formatted = _transpose_to_nchw(real_image * 2 - 1.0)
    lpips_scalar = self._lpips_func(fake_image_formatted, real_image_formatted)
    
    latitude_weights = self._create_latitude_weights(height, width)
    avg_latitude_weight = tf.reduce_mean(latitude_weights)
    
    # Apply the average weighting factor
    weighted_loss = tf.reduce_mean(lpips_scalar) * avg_latitude_weight
    
    return weighted_loss


class LPIPS360LossFactory(object):
  """Factory for creating LPIPS 360 loss instances with different configurations."""
  
  @staticmethod
  def create_equator_focused_loss(weight_path, pole_weight=0.3):
    """Create LPIPS loss that focuses more on equatorial regions."""
    return LPIPS360Loss(weight_path, 'cosine', pole_weight)
  
  @staticmethod
  def create_uniform_loss(weight_path):
    """Create LPIPS loss with uniform weighting (standard LPIPS)."""
    return LPIPS360Loss(weight_path, 'linear', pole_weight=1.0)
  
  @staticmethod
  def create_custom_loss(weight_path, latitude_weight_type, pole_weight):
    """Create LPIPS loss with custom parameters."""
    return LPIPS360Loss(weight_path, latitude_weight_type, pole_weight)