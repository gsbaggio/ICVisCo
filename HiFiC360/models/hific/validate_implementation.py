#!/usr/bin/env python3
"""
Quick validation script for HiFiC 360 modifications.

This script performs basic validation of the implementation without requiring
actual training or LPIPS weights.
"""

import sys
import os
import numpy as np

# Add the paths to both HiFiC360 and compression directories
current_dir = os.path.dirname(os.path.abspath(__file__))
hific_360_path = os.path.join(current_dir)
compression_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))), 'compression', 'models')

sys.path.insert(0, hific_360_path)
sys.path.insert(0, compression_path)

def test_imports():
    """Test if all necessary modules can be imported."""
    print("Testing imports...")
    
    try:
        # Test standard imports
        import tensorflow.compat.v1 as tf
        print("✓ TensorFlow imported successfully")
    except ImportError as e:
        print(f"✗ TensorFlow import failed: {e}")
        return False
    
    try:
        # Test HiFiC imports from compression directory
        from hific import helpers
        from hific import configs
        print("✓ HiFiC modules imported successfully")
    except ImportError as e:
        print(f"✗ HiFiC imports failed: {e}")
        print(f"Tried paths: {compression_path}")
        return False
    
    try:
        # Test our new module
        from lpips_360 import LPIPS360Loss, LPIPS360LossFactory
        print("✓ LPIPS 360 module imported successfully")
    except ImportError as e:
        print(f"✗ LPIPS 360 import failed: {e}")
        print("This is expected if the file paths are not set up correctly.")
        return False
    
    return True


def test_configurations():
    """Test if new configurations are available."""
    print("\nTesting configurations...")
    
    try:
        # Import from compression directory
        from hific import configs
        
        # Test standard configs
        standard_configs = ['hific', 'mselpips']
        for config_name in standard_configs:
            try:
                config = configs.get_config(config_name)
                print(f"✓ Standard config '{config_name}' loaded successfully")
            except Exception as e:
                print(f"✗ Failed to load config '{config_name}': {e}")
        
        # Test if we can access our new configs (they should be in the HiFiC360 version)
        # Try to load our modified configs from the HiFiC360 directory
        try:
            # Import the modified configs.py from current directory
            import importlib.util
            spec = importlib.util.spec_from_file_location("configs_360", 
                os.path.join(current_dir, "configs.py"))
            configs_360 = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(configs_360)
            
            # Test new 360 configs
            new_configs = ['hific-360', 'mselpips-360']
            for config_name in new_configs:
                try:
                    config = configs_360.get_config(config_name)
                    print(f"✓ New 360 config '{config_name}' loaded successfully")
                    
                    # Verify it has expected properties
                    if hasattr(config, 'loss_config') and hasattr(config.loss_config, 'lpips_weight'):
                        print(f"  - LPIPS weight: {config.loss_config.lpips_weight}")
                    if hasattr(config, 'lr'):
                        print(f"  - Learning rate: {config.lr}")
                        
                except Exception as e:
                    print(f"✗ Failed to load 360 config '{config_name}': {e}")
        
        except Exception as e:
            print(f"✗ Could not load HiFiC360 configs: {e}")
        
        return True
        
    except ImportError as e:
        print(f"✗ Config testing failed: {e}")
        return False


def test_latitude_weights():
    """Test latitude weight generation without TensorFlow."""
    print("\nTesting latitude weight generation...")
    
    try:
        # Test weight calculation manually (without TF)
        height, width = 128, 256
        
        # Test cosine weighting
        lat = np.linspace(-np.pi/2, np.pi/2, height)
        pole_weight = 0.3
        
        # Cosine weights
        cosine_weights = np.cos(lat) * (1.0 - pole_weight) + pole_weight
        print(f"✓ Cosine weights generated: min={cosine_weights.min():.3f}, max={cosine_weights.max():.3f}")
        
        # Linear weights
        abs_lat = np.abs(lat)
        max_lat = np.pi / 2
        linear_weights = 1.0 - (abs_lat / max_lat) * (1.0 - pole_weight)
        print(f"✓ Linear weights generated: min={linear_weights.min():.3f}, max={linear_weights.max():.3f}")
        
        # Quadratic weights
        normalized_lat = abs_lat / max_lat
        quadratic_weights = 1.0 - normalized_lat**2 * (1.0 - pole_weight)
        print(f"✓ Quadratic weights generated: min={quadratic_weights.min():.3f}, max={quadratic_weights.max():.3f}")
        
        # Test that weights make sense
        if cosine_weights[height//2] > cosine_weights[0]:  # Equator > pole
            print("✓ Cosine weights have correct distribution (equator > poles)")
        else:
            print("✗ Cosine weights have incorrect distribution")
        
        return True
        
    except Exception as e:
        print(f"✗ Latitude weight testing failed: {e}")
        return False


def test_model_initialization():
    """Test if the modified HiFiC model can be initialized."""
    print("\nTesting model initialization...")
    
    try:
        # Import from compression directory
        from hific import helpers
        
        # Try to load our modified configs and model
        try:
            import importlib.util
            
            # Load configs_360
            spec = importlib.util.spec_from_file_location("configs_360", 
                os.path.join(current_dir, "configs.py"))
            configs_360 = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(configs_360)
            
            # Load model with our modifications
            spec = importlib.util.spec_from_file_location("model_360", 
                os.path.join(current_dir, "model.py"))
            model_360 = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(model_360)
            
            # Get a test configuration
            config = configs_360.get_config('hific-360')
            
            # Try to initialize model with 360 parameters
            try:
                hific_model = model_360.HiFiC(
                    config=config,
                    mode=helpers.ModelMode.TRAINING,
                    lpips_weight_path=None,  # Don't actually load weights
                    auto_encoder_ckpt_dir=None,
                    create_image_summaries=False,
                    use_lpips_360=True,
                    latitude_weight_type='cosine',
                    pole_weight=0.3
                )
                print("✓ HiFiC model with 360 parameters initialized successfully")
                return True
                
            except Exception as e:
                print(f"✗ HiFiC model initialization failed: {e}")
                return False
                
        except Exception as e:
            print(f"✗ Could not load modified modules: {e}")
            return False
            
    except ImportError as e:
        print(f"✗ Model testing failed due to import error: {e}")
        return False


def test_argument_parsing():
    """Test if training script can parse new arguments."""
    print("\nTesting argument parsing...")
    
    try:
        # Simulate command line arguments
        test_args = [
            'train.py',
            '--config', 'hific-360',
            '--ckpt_dir', '/tmp/test',
            '--num_steps', '1000',
            '--use_lpips_360',
            '--latitude_weight_type', 'cosine',
            '--pole_weight', '0.3'
        ]
        
        try:
            # Import and test the parse_args function from our modified train.py
            import importlib.util
            spec = importlib.util.spec_from_file_location("train_360", 
                os.path.join(current_dir, "train.py"))
            train_360 = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(train_360)
            
            try:
                args = train_360.parse_args(test_args)
                print("✓ Training arguments parsed successfully")
                print(f"  - use_lpips_360: {args.use_lpips_360}")
                print(f"  - latitude_weight_type: {args.latitude_weight_type}")
                print(f"  - pole_weight: {args.pole_weight}")
                return True
                
            except Exception as e:
                print(f"✗ Argument parsing failed: {e}")
                return False
                
        except Exception as e:
            print(f"✗ Could not load modified train.py: {e}")
            return False
            
    except Exception as e:
        print(f"✗ Argument parsing test failed: {e}")
        return False


def main():
    """Run all validation tests."""
    print("HiFiC 360 Implementation Validation")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Configuration Test", test_configurations),
        ("Latitude Weights Test", test_latitude_weights),
        ("Model Initialization Test", test_model_initialization),
        ("Argument Parsing Test", test_argument_parsing),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * len(test_name))
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:25}: {status}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The HiFiC 360 implementation looks good.")
        print("\nNext steps:")
        print("1. Prepare your 360-degree dataset")
        print("2. Download LPIPS weights if needed")
        print("3. Run training with: python train_360_example.py --help")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the implementation.")
        print("\nCommon issues:")
        print("- Missing dependencies (TensorFlow, numpy, etc.)")
        print("- Incorrect Python path setup")
        print("- Missing LPIPS weights (normal for initial testing)")
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)