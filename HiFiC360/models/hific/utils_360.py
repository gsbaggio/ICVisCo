#!/usr/bin/env python3
"""
Utility functions for HiFiC 360-degree image compression.

This module provides helper functions for working with 360-degree images
and the LPIPS 360 loss function.
"""

import numpy as np
import matplotlib.pyplot as plt


def validate_360_image_format(image):
    """
    Validate that an image is in proper 360-degree equirectangular format.
    
    Args:
        image: numpy array of shape [H, W, C] or [B, H, W, C]
        
    Returns:
        tuple: (is_valid, message)
    """
    if len(image.shape) == 3:
        h, w, c = image.shape
        batch_size = 1
    elif len(image.shape) == 4:
        batch_size, h, w, c = image.shape
    else:
        return False, f"Invalid shape: {image.shape}. Expected [H,W,C] or [B,H,W,C]"
    
    # Check aspect ratio (should be 2:1 for equirectangular)
    aspect_ratio = w / h
    if not (1.8 <= aspect_ratio <= 2.2):  # Allow some tolerance
        return False, f"Aspect ratio {aspect_ratio:.2f} is not close to 2:1 for equirectangular projection"
    
    # Check value range
    if image.min() < 0 or image.max() > 1:
        return False, f"Image values should be in [0,1], got range [{image.min():.3f}, {image.max():.3f}]"
    
    # Check channels
    if c != 3:
        return False, f"Expected 3 channels (RGB), got {c}"
    
    return True, f"Valid 360° image: {batch_size} image(s) of {h}x{w}x{c}"


def analyze_latitude_distribution(image, num_bands=8):
    """
    Analyze the distribution of content across latitude bands.
    
    Args:
        image: numpy array of shape [H, W, C]
        num_bands: number of latitude bands to analyze
        
    Returns:
        dict: analysis results
    """
    h, w, c = image.shape
    
    # Create latitude bands
    band_height = h // num_bands
    bands = []
    
    for i in range(num_bands):
        start_idx = i * band_height
        end_idx = min((i + 1) * band_height, h)
        
        # Extract band
        band = image[start_idx:end_idx, :, :]
        
        # Calculate statistics
        mean_intensity = np.mean(band)
        std_intensity = np.std(band)
        edge_strength = np.mean(np.abs(np.gradient(np.mean(band, axis=2))))
        
        # Calculate latitude range
        lat_start = -90 + (i * 180 / num_bands)
        lat_end = -90 + ((i + 1) * 180 / num_bands)
        
        bands.append({
            'latitude_range': (lat_start, lat_end),
            'mean_intensity': mean_intensity,
            'std_intensity': std_intensity,
            'edge_strength': edge_strength,
            'pixel_count': band.size
        })
    
    return {
        'bands': bands,
        'polar_bands': [bands[0], bands[-1]],  # Top and bottom bands (poles)
        'equatorial_bands': bands[num_bands//2 - 1:num_bands//2 + 1]  # Middle bands (equator)
    }


def visualize_latitude_analysis(analysis, title="Latitude Band Analysis"):
    """
    Visualize the results of latitude distribution analysis.
    
    Args:
        analysis: results from analyze_latitude_distribution
        title: plot title
    """
    bands = analysis['bands']
    
    # Extract data for plotting
    latitudes = [np.mean(band['latitude_range']) for band in bands]
    mean_intensities = [band['mean_intensity'] for band in bands]
    std_intensities = [band['std_intensity'] for band in bands]
    edge_strengths = [band['edge_strength'] for band in bands]
    
    # Create plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Mean intensity
    axes[0].plot(latitudes, mean_intensities, 'bo-')
    axes[0].set_xlabel('Latitude (degrees)')
    axes[0].set_ylabel('Mean Intensity')
    axes[0].set_title('Mean Intensity by Latitude')
    axes[0].grid(True)
    
    # Standard deviation
    axes[1].plot(latitudes, std_intensities, 'ro-')
    axes[1].set_xlabel('Latitude (degrees)')
    axes[1].set_ylabel('Standard Deviation')
    axes[1].set_title('Intensity Variation by Latitude')
    axes[1].grid(True)
    
    # Edge strength
    axes[2].plot(latitudes, edge_strengths, 'go-')
    axes[2].set_xlabel('Latitude (degrees)')
    axes[2].set_ylabel('Edge Strength')
    axes[2].set_title('Edge Content by Latitude')
    axes[2].grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def generate_latitude_weights_comparison():
    """
    Generate a comparison of different latitude weighting schemes.
    
    Returns:
        dict: weight values for different schemes
    """
    height = 256
    lat = np.linspace(-np.pi/2, np.pi/2, height)
    
    schemes = {}
    
    # Cosine weighting with different pole weights
    for pole_weight in [0.1, 0.3, 0.5, 0.7, 1.0]:
        weights = np.cos(lat) * (1.0 - pole_weight) + pole_weight
        schemes[f'cosine_pole_{pole_weight}'] = weights
    
    # Linear weighting
    abs_lat = np.abs(lat)
    max_lat = np.pi / 2
    for pole_weight in [0.1, 0.3, 0.5, 0.7]:
        weights = 1.0 - (abs_lat / max_lat) * (1.0 - pole_weight)
        schemes[f'linear_pole_{pole_weight}'] = weights
    
    # Quadratic weighting
    normalized_lat = abs_lat / max_lat
    for pole_weight in [0.1, 0.3, 0.5, 0.7]:
        weights = 1.0 - normalized_lat**2 * (1.0 - pole_weight)
        schemes[f'quadratic_pole_{pole_weight}'] = weights
    
    return schemes, lat


def plot_weight_schemes():
    """Plot comparison of different weighting schemes."""
    schemes, lat = generate_latitude_weights_comparison()
    
    plt.figure(figsize=(12, 8))
    
    # Plot cosine schemes
    plt.subplot(2, 2, 1)
    for name, weights in schemes.items():
        if name.startswith('cosine'):
            pole_weight = name.split('_')[-1]
            plt.plot(np.degrees(lat), weights, label=f'pole_weight={pole_weight}')
    plt.xlabel('Latitude (degrees)')
    plt.ylabel('Weight')
    plt.title('Cosine Weighting')
    plt.legend()
    plt.grid(True)
    
    # Plot linear schemes
    plt.subplot(2, 2, 2)
    for name, weights in schemes.items():
        if name.startswith('linear'):
            pole_weight = name.split('_')[-1]
            plt.plot(np.degrees(lat), weights, label=f'pole_weight={pole_weight}')
    plt.xlabel('Latitude (degrees)')
    plt.ylabel('Weight')
    plt.title('Linear Weighting')
    plt.legend()
    plt.grid(True)
    
    # Plot quadratic schemes
    plt.subplot(2, 2, 3)
    for name, weights in schemes.items():
        if name.startswith('quadratic'):
            pole_weight = name.split('_')[-1]
            plt.plot(np.degrees(lat), weights, label=f'pole_weight={pole_weight}')
    plt.xlabel('Latitude (degrees)')
    plt.ylabel('Weight')
    plt.title('Quadratic Weighting')
    plt.legend()
    plt.grid(True)
    
    # Compare all with pole_weight=0.3
    plt.subplot(2, 2, 4)
    pole_weight = 0.3
    
    # Cosine
    weights_cos = np.cos(lat) * (1.0 - pole_weight) + pole_weight
    plt.plot(np.degrees(lat), weights_cos, 'b-', label='Cosine', linewidth=2)
    
    # Linear
    abs_lat = np.abs(lat)
    max_lat = np.pi / 2
    weights_lin = 1.0 - (abs_lat / max_lat) * (1.0 - pole_weight)
    plt.plot(np.degrees(lat), weights_lin, 'r--', label='Linear', linewidth=2)
    
    # Quadratic
    normalized_lat = abs_lat / max_lat
    weights_quad = 1.0 - normalized_lat**2 * (1.0 - pole_weight)
    plt.plot(np.degrees(lat), weights_quad, 'g:', label='Quadratic', linewidth=2)
    
    plt.xlabel('Latitude (degrees)')
    plt.ylabel('Weight')
    plt.title(f'Comparison (pole_weight={pole_weight})')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('weight_schemes_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def estimate_compression_benefit(original_analysis, compressed_analysis):
    """
    Estimate the potential benefit of latitude-weighted loss.
    
    Args:
        original_analysis: analysis of original 360° image
        compressed_analysis: analysis of compressed 360° image
        
    Returns:
        dict: estimated benefits
    """
    orig_bands = original_analysis['bands']
    comp_bands = compressed_analysis['bands']
    
    # Calculate errors by latitude band
    band_errors = []
    for orig_band, comp_band in zip(orig_bands, comp_bands):
        error = abs(orig_band['mean_intensity'] - comp_band['mean_intensity'])
        band_errors.append({
            'latitude_range': orig_band['latitude_range'],
            'error': error
        })
    
    # Calculate weighted error for different schemes
    lat_centers = [np.mean(band['latitude_range']) for band in orig_bands]
    lat_rad = np.radians(lat_centers)
    
    errors = [band['error'] for band in band_errors]
    
    # Standard (uniform) error
    uniform_error = np.mean(errors)
    
    # Cosine-weighted error (pole_weight=0.3)
    pole_weight = 0.3
    cos_weights = np.cos(lat_rad) * (1.0 - pole_weight) + pole_weight
    cos_weights /= np.mean(cos_weights)  # Normalize
    cosine_error = np.average(errors, weights=cos_weights)
    
    return {
        'uniform_error': uniform_error,
        'cosine_weighted_error': cosine_error,
        'potential_benefit': (uniform_error - cosine_error) / uniform_error * 100,
        'band_errors': band_errors
    }


if __name__ == '__main__':
    print("HiFiC 360 Utilities Test")
    print("=" * 40)
    
    # Test with synthetic 360° image
    height, width = 256, 512
    synthetic_image = np.random.rand(height, width, 3)
    
    # Validate format
    is_valid, message = validate_360_image_format(synthetic_image)
    print(f"Image validation: {is_valid}")
    print(f"Message: {message}")
    
    # Analyze latitude distribution
    analysis = analyze_latitude_distribution(synthetic_image)
    print(f"\nLatitude analysis completed for {len(analysis['bands'])} bands")
    
    # Generate weight comparison plots
    try:
        plot_weight_schemes()
        print("Weight scheme comparison plot saved as 'weight_schemes_comparison.png'")
    except ImportError:
        print("Matplotlib not available. Skipping plots.")
    
    print("\nUtilities test completed!")