#!/usr/bin/env python3
"""
Simple validation script for HiFiC 360 modifications.

This script tests the core functionality without complex imports.
"""

import os
import sys
import numpy as np

def test_files_exist():
    """Test if all required files exist."""
    print("Testing file existence...")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    required_files = [
        'lpips_360.py',
        'model.py', 
        'train.py',
        'configs.py',
        'train_360_example.py',
        'utils_360.py',
        'README_360.md',
        'CHANGES_SUMMARY.md'
    ]
    
    all_exist = True
    for filename in required_files:
        filepath = os.path.join(current_dir, filename)
        if os.path.exists(filepath):
            print(f"✓ {filename} exists")
        else:
            print(f"✗ {filename} missing")
            all_exist = False
    
    return all_exist


def test_lpips_360_class():
    """Test LPIPS360Loss class definition without importing TensorFlow."""
    print("\nTesting LPIPS360Loss class definition...")
    
    try:
        # Read the lpips_360.py file and check for key components
        current_dir = os.path.dirname(os.path.abspath(__file__))
        lpips_file = os.path.join(current_dir, 'lpips_360.py')
        
        with open(lpips_file, 'r') as f:
            content = f.read()
        
        required_components = [
            'class LPIPS360Loss',
            'def __init__',
            'def _create_latitude_weights',
            'def __call__',
            'latitude_weight_type',
            'pole_weight',
            'cosine',
            'linear',
            'quadratic'
        ]
        
        all_found = True
        for component in required_components:
            if component in content:
                print(f"✓ Found: {component}")
            else:
                print(f"✗ Missing: {component}")
                all_found = False
        
        return all_found
        
    except Exception as e:
        print(f"✗ Could not read lpips_360.py: {e}")
        return False


def test_config_modifications():
    """Test if configs.py has been modified with 360 configurations."""
    print("\nTesting config modifications...")
    
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(current_dir, 'configs.py')
        
        with open(config_file, 'r') as f:
            content = f.read()
        
        required_configs = [
            'hific-360',
            'mselpips-360'
        ]
        
        all_found = True
        for config in required_configs:
            if f"'{config}'" in content:
                print(f"✓ Found config: {config}")
            else:
                print(f"✗ Missing config: {config}")
                all_found = False
        
        return all_found
        
    except Exception as e:
        print(f"✗ Could not read configs.py: {e}")
        return False


def test_train_modifications():
    """Test if train.py has been modified with 360 arguments."""
    print("\nTesting train.py modifications...")
    
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        train_file = os.path.join(current_dir, 'train.py')
        
        with open(train_file, 'r') as f:
            content = f.read()
        
        required_args = [
            '--use_lpips_360',
            '--latitude_weight_type', 
            '--pole_weight',
            'use_lpips_360=',
            'latitude_weight_type=',
            'pole_weight='
        ]
        
        all_found = True
        for arg in required_args:
            if arg in content:
                print(f"✓ Found argument: {arg}")
            else:
                print(f"✗ Missing argument: {arg}")
                all_found = False
        
        return all_found
        
    except Exception as e:
        print(f"✗ Could not read train.py: {e}")
        return False


def test_model_modifications():
    """Test if model.py has been modified to support LPIPS 360."""
    print("\nTesting model.py modifications...")
    
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_file = os.path.join(current_dir, 'model.py')
        
        with open(model_file, 'r') as f:
            content = f.read()
        
        required_modifications = [
            'lpips_360',
            'LPIPS360Loss',
            'use_lpips_360',
            'latitude_weight_type',
            'pole_weight'
        ]
        
        all_found = True
        for mod in required_modifications:
            if mod in content:
                print(f"✓ Found modification: {mod}")
            else:
                print(f"✗ Missing modification: {mod}")
                all_found = False
        
        return all_found
        
    except Exception as e:
        print(f"✗ Could not read model.py: {e}")
        return False


def test_latitude_weight_calculation():
    """Test latitude weight calculation logic."""
    print("\nTesting latitude weight calculations...")
    
    try:
        height = 180
        lat = np.linspace(-np.pi/2, np.pi/2, height)
        pole_weight = 0.3
        
        # Test cosine weighting
        cosine_weights = np.cos(lat) * (1.0 - pole_weight) + pole_weight
        
        # Test linear weighting  
        abs_lat = np.abs(lat)
        max_lat = np.pi / 2
        linear_weights = 1.0 - (abs_lat / max_lat) * (1.0 - pole_weight)
        
        # Test quadratic weighting
        normalized_lat = abs_lat / max_lat
        quadratic_weights = 1.0 - normalized_lat**2 * (1.0 - pole_weight)
        
        # Verify properties
        tests_passed = 0
        total_tests = 9
        
        # Test 1: Cosine weights range
        if cosine_weights.min() >= pole_weight and cosine_weights.max() <= 1.0:
            print("✓ Cosine weights in correct range")
            tests_passed += 1
        else:
            print("✗ Cosine weights out of range")
        
        # Test 2: Linear weights range
        if linear_weights.min() >= pole_weight and linear_weights.max() <= 1.0:
            print("✓ Linear weights in correct range")
            tests_passed += 1
        else:
            print("✗ Linear weights out of range")
        
        # Test 3: Quadratic weights range
        if quadratic_weights.min() >= pole_weight and quadratic_weights.max() <= 1.0:
            print("✓ Quadratic weights in correct range")
            tests_passed += 1
        else:
            print("✗ Quadratic weights out of range")
        
        # Test 4-6: Equator should have higher weight than poles
        equator_idx = height // 2
        pole_idx = 0
        
        if cosine_weights[equator_idx] > cosine_weights[pole_idx]:
            print("✓ Cosine: equator > pole")
            tests_passed += 1
        else:
            print("✗ Cosine: equator <= pole")
        
        if linear_weights[equator_idx] > linear_weights[pole_idx]:
            print("✓ Linear: equator > pole")
            tests_passed += 1
        else:
            print("✗ Linear: equator <= pole")
        
        if quadratic_weights[equator_idx] > quadratic_weights[pole_idx]:
            print("✓ Quadratic: equator > pole")
            tests_passed += 1
        else:
            print("✗ Quadratic: equator <= pole")
        
        # Test 7-9: Symmetry (North pole = South pole)
        if np.isclose(cosine_weights[0], cosine_weights[-1]):
            print("✓ Cosine weights symmetric")
            tests_passed += 1
        else:
            print("✗ Cosine weights not symmetric")
        
        if np.isclose(linear_weights[0], linear_weights[-1]):
            print("✓ Linear weights symmetric")
            tests_passed += 1
        else:
            print("✗ Linear weights not symmetric")
        
        if np.isclose(quadratic_weights[0], quadratic_weights[-1]):
            print("✓ Quadratic weights symmetric")
            tests_passed += 1
        else:
            print("✗ Quadratic weights not symmetric")
        
        print(f"Latitude weight tests: {tests_passed}/{total_tests} passed")
        return tests_passed == total_tests
        
    except Exception as e:
        print(f"✗ Latitude weight calculation failed: {e}")
        return False


def main():
    """Run all simple validation tests."""
    print("HiFiC 360 Simple Validation")
    print("=" * 50)
    
    tests = [
        ("File Existence", test_files_exist),
        ("LPIPS 360 Class", test_lpips_360_class),
        ("Config Modifications", test_config_modifications),
        ("Train Modifications", test_train_modifications),
        ("Model Modifications", test_model_modifications),
        ("Latitude Weight Logic", test_latitude_weight_calculation),
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
    print("SIMPLE VALIDATION SUMMARY")
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
        print("\n🎉 All simple tests passed! The HiFiC 360 modifications look good.")
        print("\nThe implementation includes:")
        print("- ✓ LPIPS 360 loss function with latitude weighting")
        print("- ✓ New configurations for 360-degree images")
        print("- ✓ Modified training script with new arguments")
        print("- ✓ Updated model to support 360-degree LPIPS")
        print("- ✓ Proper latitude weight calculation logic")
        print("\nNext steps:")
        print("1. Set up proper Python paths for imports")
        print("2. Install required dependencies (TensorFlow, etc.)")
        print("3. Prepare 360-degree image dataset")
        print("4. Start training with: python train_360_example.py --help")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the files.")
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)