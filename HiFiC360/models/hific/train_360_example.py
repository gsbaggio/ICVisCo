#!/usr/bin/env python3
"""
Example script for training HiFiC with 360-degree LPIPS loss.

This script demonstrates how to use the new LPIPS loss adapted for 360-degree images
with latitude-based weighting.
"""

import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(
        description='Train HiFiC with 360-degree LPIPS loss',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Required arguments
    parser.add_argument('--config', required=True,
                        help='Model configuration (e.g., hific-lo, hific-mi, hific-hi)')
    parser.add_argument('--ckpt_dir', required=True,
                        help='Directory to save checkpoints')
    parser.add_argument('--data_dir', required=True,
                        help='Directory containing 360-degree training images')
    
    # Training parameters
    parser.add_argument('--num_steps', default='100k',
                        help='Number of training steps')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size (reduced default for 360 images)')
    parser.add_argument('--crop_size', type=int, default=512,
                        help='Crop size (increased for 360 images)')
    
    # 360-degree LPIPS parameters
    parser.add_argument('--latitude_weight_type', default='cosine',
                        choices=['cosine', 'linear', 'quadratic'],
                        help='Type of latitude weighting')
    parser.add_argument('--pole_weight', type=float, default=0.3,
                        help='Weight for polar regions (0.0-1.0, lower = less importance)')
    
    # Optional parameters
    parser.add_argument('--init_autoencoder_from_ckpt_dir',
                        help='Initialize from pretrained autoencoder')
    parser.add_argument('--lpips_weight_path',
                        help='Path to LPIPS weights')
    
    args = parser.parse_args()
    
    # Build training command
    train_cmd = [
        'python', 'train.py',
        '--config', args.config,
        '--ckpt_dir', args.ckpt_dir,
        '--num_steps', args.num_steps,
        '--batch_size', str(args.batch_size),
        '--crop_size', str(args.crop_size),
        '--use_lpips_360',  # Enable 360-degree LPIPS loss
        '--latitude_weight_type', args.latitude_weight_type,
        '--pole_weight', str(args.pole_weight),
        '--tfds_data_dir', args.data_dir,
        '--tfds_dataset', 'custom_360_dataset'  # You may need to adjust this
    ]
    
    # Add optional arguments
    if args.init_autoencoder_from_ckpt_dir:
        train_cmd.extend(['--init_autoencoder_from_ckpt_dir', args.init_autoencoder_from_ckpt_dir])
    
    if args.lpips_weight_path:
        train_cmd.extend(['--lpips_weight_path', args.lpips_weight_path])
    
    print("Training command:")
    print(" ".join(train_cmd))
    print()
    
    # Run training
    try:
        subprocess.run(train_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        sys.exit(1)
    
    print("Training completed successfully!")


if __name__ == '__main__':
    main()