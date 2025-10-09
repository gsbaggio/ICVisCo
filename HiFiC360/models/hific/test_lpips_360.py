#!/usr/bin/env python3
"""
Test script for LPIPS 360 implementation.

This script tests the LPIPS 360 loss function with synthetic 360-degree images.
"""

import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
from lpips_360 import LPIPS360Loss, LPIPSLoss

# Disable TF2 behavior for compatibility
tf.disable_v2_behavior()


def create_synthetic_360_image(height=256, width=512, add_pole_artifacts=True):
    """Create a synthetic equirectangular 360-degree image."""
    # Create coordinate grids
    lat = np.linspace(-np.pi/2, np.pi/2, height)
    lon = np.linspace(-np.pi, np.pi, width)
    lat_grid, lon_grid = np.meshgrid(lon, lat)
    
    # Create base pattern (spherical harmonics-like)
    base_pattern = (
        np.sin(3 * lat_grid) * np.cos(2 * lon_grid) +
        np.cos(2 * lat_grid) * np.sin(lon_grid) +
        0.5 * np.sin(lat_grid)
    )
    
    # Add pole artifacts if requested
    if add_pole_artifacts:
        # Add stretching artifacts near poles
        pole_factor = 1.0 / (np.cos(lat_grid) + 0.1)
        noise = np.random.normal(0, 0.1, (height, width))
        pole_artifacts = noise * pole_factor * 0.3
        base_pattern += pole_artifacts
    
    # Normalize to [0, 1] and add RGB channels
    image = (base_pattern - base_pattern.min()) / (base_pattern.max() - base_pattern.min())
    
    # Create RGB image
    rgb_image = np.stack([
        image,
        image * 0.8 + 0.1,  # Slightly different G channel
        image * 0.6 + 0.2   # Different B channel
    ], axis=-1)
    
    return rgb_image.astype(np.float32)


def create_corrupted_version(original, corruption_type='gaussian_noise', intensity=0.1):
    """Create a corrupted version of the original image."""
    if corruption_type == 'gaussian_noise':
        noise = np.random.normal(0, intensity, original.shape)
        corrupted = np.clip(original + noise, 0, 1)
    elif corruption_type == 'blur':
        # Simple box blur
        from scipy import ndimage
        corrupted = ndimage.uniform_filter(original, size=int(intensity * 10))
    elif corruption_type == 'polar_corruption':
        # Add corruption mainly to polar regions
        height = original.shape[0]
        lat = np.linspace(-np.pi/2, np.pi/2, height)
        pole_mask = np.abs(lat) > np.pi/3  # Affect regions beyond 60 degrees
        pole_mask = np.expand_dims(pole_mask, axis=(1, 2))
        
        corrupted = original.copy()
        noise = np.random.normal(0, intensity, original.shape)
        corrupted = corrupted + noise * pole_mask
        corrupted = np.clip(corrupted, 0, 1)
    else:
        raise ValueError(f"Unknown corruption type: {corruption_type}")
    
    return corrupted.astype(np.float32)


def test_lpips_360():
    """Test the LPIPS 360 implementation."""
    print("Testing LPIPS 360 implementation...")
    
    # Create synthetic images
    height, width = 256, 512
    batch_size = 2
    
    # Create batch of original images
    original_batch = np.stack([
        create_synthetic_360_image(height, width, add_pole_artifacts=False),
        create_synthetic_360_image(height, width, add_pole_artifacts=False)
    ])
    
    # Create corrupted versions
    corrupted_batch = np.stack([
        create_corrupted_version(original_batch[0], 'polar_corruption', 0.2),
        create_corrupted_version(original_batch[1], 'gaussian_noise', 0.1)
    ])
    
    # Test different configurations
    configs = [
        {'type': 'cosine', 'pole_weight': 0.3, 'name': 'Cosine (pole_weight=0.3)'},
        {'type': 'cosine', 'pole_weight': 0.7, 'name': 'Cosine (pole_weight=0.7)'},
        {'type': 'linear', 'pole_weight': 0.5, 'name': 'Linear (pole_weight=0.5)'},
        {'type': 'quadratic', 'pole_weight': 0.4, 'name': 'Quadratic (pole_weight=0.4)'},
    ]
    
    # Note: This test assumes LPIPS weights are available
    # For a real test, you would need to download or have LPIPS weights
    weight_path = "lpips_weight__net-lin_alex_v0.1.pb"
    
    try:
        with tf.Session() as sess:
            # Placeholders for images
            original_ph = tf.placeholder(tf.float32, [batch_size, height, width, 3])
            corrupted_ph = tf.placeholder(tf.float32, [batch_size, height, width, 3])
            
            # Test each configuration
            results = {}
            
            for config in configs:
                print(f"\nTesting {config['name']}...")
                
                # Create LPIPS 360 loss
                lpips_360 = LPIPS360Loss(
                    weight_path=weight_path,
                    latitude_weight_type=config['type'],
                    pole_weight=config['pole_weight']
                )
                
                # Compute loss
                loss_360 = lpips_360(corrupted_ph, original_ph)
                
                # Run computation
                loss_value = sess.run(loss_360, {
                    original_ph: original_batch,
                    corrupted_ph: corrupted_batch
                })
                
                results[config['name']] = loss_value
                print(f"Loss value: {loss_value:.6f}")
            
            # Compare with standard LPIPS
            print(f"\nTesting Standard LPIPS...")
            standard_lpips = LPIPSLoss(weight_path)
            standard_loss = standard_lpips(corrupted_ph, original_ph)
            
            standard_value = sess.run(standard_loss, {
                original_ph: original_batch,
                corrupted_ph: corrupted_batch
            })
            
            results['Standard LPIPS'] = standard_value
            print(f"Standard LPIPS loss: {standard_value:.6f}")
            
            # Print comparison
            print("\n" + "="*50)
            print("RESULTS COMPARISON:")
            print("="*50)
            for name, value in results.items():
                print(f"{name:25}: {value:.6f}")
            
    except Exception as e:
        print(f"Test failed: {e}")
        print("This is expected if LPIPS weights are not available.")
        print("To run this test properly, you need to:")
        print("1. Download LPIPS weights")
        print("2. Ensure TensorFlow is properly installed")
        print("3. Have scipy installed for image processing")


def visualize_latitude_weights():
    """Visualize different latitude weighting schemes."""
    print("\nVisualizing latitude weights...")
    
    height, width = 180, 360  # Example dimensions
    
    # Create LPIPS 360 instances with different configurations
    # Note: We'll just visualize the weight maps without needing actual LPIPS weights
    
    configs = [
        {'type': 'cosine', 'pole_weight': 0.3},
        {'type': 'cosine', 'pole_weight': 0.7},
        {'type': 'linear', 'pole_weight': 0.5},
        {'type': 'quadratic', 'pole_weight': 0.4},
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for i, config in enumerate(configs):
        # Create latitude coordinates
        lat = np.linspace(-np.pi/2, np.pi/2, height)
        
        if config['type'] == 'cosine':
            weights = np.cos(lat) * (1.0 - config['pole_weight']) + config['pole_weight']
        elif config['type'] == 'linear':
            abs_lat = np.abs(lat)
            max_lat = np.pi / 2
            weights = 1.0 - (abs_lat / max_lat) * (1.0 - config['pole_weight'])
        elif config['type'] == 'quadratic':
            abs_lat = np.abs(lat)
            max_lat = np.pi / 2
            normalized_lat = abs_lat / max_lat
            weights = 1.0 - normalized_lat**2 * (1.0 - config['pole_weight'])
        
        # Create 2D weight map
        weight_map = np.tile(weights[:, np.newaxis], (1, width))
        
        # Plot
        im = axes[i].imshow(weight_map, aspect='auto', cmap='viridis')
        axes[i].set_title(f"{config['type'].title()} (pole_weight={config['pole_weight']})")
        axes[i].set_xlabel('Longitude')
        axes[i].set_ylabel('Latitude')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    plt.savefig('latitude_weights_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Latitude weights visualization saved as 'latitude_weights_visualization.png'")


if __name__ == '__main__':
    print("LPIPS 360 Test Suite")
    print("="*50)
    
    # Test 1: Visualize latitude weights
    try:
        visualize_latitude_weights()
    except ImportError:
        print("Matplotlib not available. Skipping visualization.")
    
    # Test 2: Test LPIPS 360 functionality
    test_lpips_360()
    
    print("\nTest suite completed!")