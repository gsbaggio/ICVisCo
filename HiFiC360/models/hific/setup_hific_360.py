#!/usr/bin/env python3
"""
Setup script to prepare the environment for HiFiC 360.

This script helps set up the correct Python paths and validates the installation.
"""

import os
import sys
import subprocess

def setup_python_paths():
    """Set up Python paths for both HiFiC360 and compression directories."""
    print("Setting up Python paths...")
    
    # Get current directory (should be HiFiC360/models/hific)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Current directory: {current_dir}")
    
    # Calculate paths
    hific_360_path = current_dir
    compression_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(current_dir))), 
        'compression', 'models'
    )
    
    print(f"HiFiC 360 path: {hific_360_path}")
    print(f"Compression path: {compression_path}")
    
    # Check if paths exist
    if os.path.exists(compression_path):
        print("✓ Compression path exists")
    else:
        print("✗ Compression path not found")
        print("Please make sure you're running this from HiFiC360/models/hific/")
        return False
    
    # Add to Python path
    if hific_360_path not in sys.path:
        sys.path.insert(0, hific_360_path)
    if compression_path not in sys.path:
        sys.path.insert(0, compression_path)
    
    print("✓ Python paths configured")
    return True


def create_run_script():
    """Create a run script with proper environment setup."""
    print("\nCreating run script...")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    compression_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(current_dir))), 
        'compression', 'models'
    )
    
    run_script_content = f'''#!/bin/bash
# HiFiC 360 Run Script
# This script sets up the environment and runs HiFiC 360 training

# Activate conda environment
echo "Activating conda environment 'hific'..."
conda activate hific

# Set Python path to include both directories
export PYTHONPATH="{current_dir}:{compression_path}:$PYTHONPATH"

# Function to run training with 360 support
run_hific_360() {{
    echo "Running HiFiC 360 training..."
    echo "Conda environment: hific"
    echo "PYTHONPATH: $PYTHONPATH"
    echo ""
    
    python "{current_dir}/train.py" "$@"
}}

# Function to run validation
run_validation() {{
    echo "Running HiFiC 360 validation..."
    python "{current_dir}/simple_validation.py"
}}

# Function to run the original validation (with imports)
run_full_validation() {{
    echo "Running full HiFiC 360 validation..."
    python "{current_dir}/validate_implementation.py"
}}

# Function to show usage
show_usage() {{
    echo "HiFiC 360 Run Script"
    echo "Usage:"
    echo "  $0 train [training args]   - Run training with 360 support"
    echo "  $0 validate               - Run simple validation tests"
    echo "  $0 fullvalidate           - Run full validation with imports"
    echo "  $0 example                - Show example training command"
    echo ""
    echo "Example training command:"
    echo "  $0 train --config hific-360 --ckpt_dir ./checkpoints --use_lpips_360 --latitude_weight_type cosine --pole_weight 0.3"
}}

# Main script logic
case "$1" in
    "train")
        shift
        run_hific_360 "$@"
        ;;
    "validate")
        run_validation
        ;;
    "fullvalidate")
        run_full_validation
        ;;
    "example")
        echo "Example HiFiC 360 training command:"
        echo ""
        echo "$0 train \\\\"
        echo "  --config hific-360 \\\\"
        echo "  --ckpt_dir ./checkpoints/hific360_test \\\\"
        echo "  --num_steps 10k \\\\"
        echo "  --batch_size 4 \\\\"
        echo "  --crop_size 512 \\\\"
        echo "  --use_lpips_360 \\\\"
        echo "  --latitude_weight_type cosine \\\\"
        echo "  --pole_weight 0.3"
        echo ""
        echo "For more options: $0 train --help"
        ;;
    *)
        show_usage
        ;;
esac
'''
    
    run_script_path = os.path.join(current_dir, 'run_hific_360.sh')
    with open(run_script_path, 'w') as f:
        f.write(run_script_content)
    
    # Make executable
    os.chmod(run_script_path, 0o755)
    
    print(f"✓ Run script created: {run_script_path}")
    return True


def test_basic_imports():
    """Test basic imports with proper paths."""
    print("\nTesting basic imports...")
    
    try:
        # Test numpy
        import numpy as np
        print("✓ NumPy imported")
    except ImportError:
        print("✗ NumPy not available")
        return False
    
    try:
        # Test TensorFlow
        import tensorflow.compat.v1 as tf
        print("✓ TensorFlow imported")
    except ImportError:
        print("✗ TensorFlow not available")
        print("  Install with: pip install tensorflow")
        return False
    
    try:
        # Test HiFiC imports
        from hific import helpers
        from hific import configs
        print("✓ HiFiC modules imported")
    except ImportError as e:
        print(f"✗ HiFiC imports failed: {e}")
        return False
    
    return True


def check_dependencies():
    """Check if all required dependencies are available."""
    print("\nChecking dependencies...")
    
    required_packages = [
        'numpy',
        'tensorflow',
    ]
    
    optional_packages = [
        'matplotlib',
        'scipy',
    ]
    
    missing_required = []
    missing_optional = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (required)")
            missing_required.append(package)
    
    for package in optional_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"! {package} (optional)")
            missing_optional.append(package)
    
    if missing_required:
        print(f"\nMissing required packages: {missing_required}")
        print("Install with: pip install " + " ".join(missing_required))
        return False
    
    if missing_optional:
        print(f"\nMissing optional packages: {missing_optional}")
        print("Install with: pip install " + " ".join(missing_optional))
    
    return True


def main():
    """Run the setup process."""
    print("HiFiC 360 Setup")
    print("=" * 40)
    
    # Step 1: Setup paths
    if not setup_python_paths():
        print("✗ Failed to setup Python paths")
        return False
    
    # Step 2: Create run script
    if not create_run_script():
        print("✗ Failed to create run script")
        return False
    
    # Step 3: Check dependencies
    if not check_dependencies():
        print("✗ Missing required dependencies")
        return False
    
    # Step 4: Test imports
    if not test_basic_imports():
        print("✗ Import tests failed")
        return False
    
    print("\n" + "=" * 40)
    print("✅ SETUP COMPLETE!")
    print("=" * 40)
    print("\nNext steps:")
    print("1. Run validation: python simple_validation.py")
    print("2. Or use the run script: ./run_hific_360.sh validate")
    print("3. For training: ./run_hific_360.sh example")
    print("4. For help: ./run_hific_360.sh")
    
    return True


if __name__ == '__main__':
    success = main()
    if not success:
        print("\n❌ Setup failed. Please check the errors above.")
        sys.exit(1)
    else:
        print("\n🎉 Setup successful! You can now use HiFiC 360.")
        sys.exit(0)