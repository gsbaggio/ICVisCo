#!/usr/bin/env python3
"""
Wrapper script for running HiFiC 360 with proper TensorFlow v1 compatibility.

This script ensures TensorFlow v1 behavior is enabled before importing any modules.
"""

import os
import sys

# Set up paths first
current_dir = os.path.dirname(os.path.abspath(__file__))
compression_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(current_dir))), 
    'compression', 'models'
)
sys.path.insert(0, current_dir)
sys.path.insert(0, compression_path)

# Force TensorFlow v1 behavior before any imports
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Monkey patch tensorflow to make v1 the default
import tensorflow as tf_orig
tf_orig.AUTO_REUSE = tf.AUTO_REUSE
tf_orig.variable_scope = tf.variable_scope
tf_orig.get_variable = tf.get_variable

# Now we can safely import the training script
if __name__ == '__main__':
    from train import main, parse_args
    main(parse_args(sys.argv))