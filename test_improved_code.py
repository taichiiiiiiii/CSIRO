#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script to validate improved training and inference code
"""

import sys
import traceback
import torch
import numpy as np
from pathlib import Path

# Test configuration
TEST_BATCH_SIZE = 2
TEST_IMG_SIZE = 512
TEST_N_SAMPLES = 4

def test_imports():
    """Test if all required imports work"""
    print("Testing imports...")
    try:
        import cv2
        import pandas as pd
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        import timm
        from timm.utils import ModelEmaV2
        from sklearn.model_selection import StratifiedGroupKFold
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_model_creation():
    """Test if the model can be created"""
    print("\nTesting model creation...")
    try:
        # Import model class from improved_training
        sys.path.insert(0, str(Path(__file__).parent))
        from improved_training import BiomassModel, LocalMambaBlock
        
        # Create model
        model = BiomassModel("vit_huge_plus_patch16_dinov3.lvd1689m", pretrained=False)
        print(f"✅ Model created successfully")
        
        # Test forward pass with dummy data
        batch_size = 2
        dummy_left = torch.randn(batch_size, 3, 512, 512)
        dummy_right = torch.randn(batch_size, 3, 512, 512)
        
        with torch.no_grad():
            output = model((dummy_left, dummy_right))
        
        expected_shape = (batch_size, 5)  # 5 targets
        assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
        print(f"✅ Forward pass successful, output shape: {output.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        traceback.print_exc()
        return False


def test_dataset():
    """Test if the dataset works correctly"""
    print("\nTesting dataset...")
    try:
        from improved_training import BiomassDataset, get_train_transforms, CFG
        import pandas as pd
        import numpy as np
        
        # Create dummy dataframe
        dummy_df = pd.DataFrame({
            'image_path': ['test1.jpg', 'test2.jpg'],
            'Dry_Green_g': [100.0, 200.0],
            'Dry_Dead_g': [50.0, 75.0],
            'Dry_Clover_g': [25.0, 30.0],
            'GDM_g': [125.0, 230.0],
            'Dry_Total_g': [175.0, 305.0]
        })
        
        # Create dummy transform (simplified)
        transform = get_train_transforms(512)
        
        # Note: We can't fully test dataset without actual images
        print("✅ Dataset class loaded successfully")
        print("   (Full dataset test requires actual image files)")
        return True
        
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        traceback.print_exc()
        return False


def test_swa_components():
    """Test SWA-specific components"""
    print("\nTesting SWA components...")
    try:
        from torch.optim.swa_utils import AveragedModel, SWALR
        from improved_training import BiomassModel
        
        # Create base model
        model = BiomassModel("vit_huge_plus_patch16_dinov3.lvd1689m", pretrained=False)
        
        # Create SWA model
        swa_model = AveragedModel(model)
        print("✅ SWA model created")
        
        # Create optimizer and SWA scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        swa_scheduler = SWALR(optimizer, swa_lr=1e-5)
        print("✅ SWA scheduler created")
        
        # Test update
        swa_model.update_parameters(model)
        print("✅ SWA model update successful")
        
        return True
        
    except Exception as e:
        print(f"❌ SWA test failed: {e}")
        traceback.print_exc()
        return False


def test_tta_functions():
    """Test TTA functions from inference code"""
    print("\nTesting TTA functions...")
    try:
        from improved_inference import apply_tta_transform, predict_with_tta, BiomassModel
        
        # Create dummy tensors
        dummy_img = torch.randn(1, 3, 512, 512)
        
        # Test each transform
        transforms = ["original", "hflip", "vflip", "rotate90", "rotate270"]
        for transform_name in transforms:
            transformed = apply_tta_transform(dummy_img, transform_name)
            assert transformed.shape == dummy_img.shape, f"Transform {transform_name} changed shape"
            print(f"  ✅ {transform_name} transform works")
        
        # Test predict_with_tta
        model = BiomassModel("vit_huge_plus_patch16_dinov3.lvd1689m", pretrained=False)
        model.eval()
        
        device = torch.device("cpu")
        model = model.to(device)
        dummy_left = torch.randn(1, 3, 512, 512).to(device)
        dummy_right = torch.randn(1, 3, 512, 512).to(device)
        
        with torch.no_grad():
            pred = predict_with_tta(model, dummy_left, dummy_right, device, transforms[:2])  # Test with 2 transforms
        
        assert pred.shape == (1, 5), f"TTA prediction shape wrong: {pred.shape}"
        print("✅ TTA prediction successful")
        
        return True
        
    except Exception as e:
        print(f"❌ TTA test failed: {e}")
        traceback.print_exc()
        return False


def test_multi_scale():
    """Test multi-scale training components"""
    print("\nTesting multi-scale components...")
    try:
        from improved_training import get_train_transforms, CFG
        
        # Test transforms with different sizes
        sizes = [448, 512, 576]
        for size in sizes:
            transform = get_train_transforms(size)
            print(f"  ✅ Transform for size {size} created")
        
        print("✅ Multi-scale transforms working")
        return True
        
    except Exception as e:
        print(f"❌ Multi-scale test failed: {e}")
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("🧪 TESTING IMPROVED CODE")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Model Creation", test_model_creation),
        ("Dataset", test_dataset),
        ("SWA Components", test_swa_components),
        ("TTA Functions", test_tta_functions),
        ("Multi-scale", test_multi_scale)
    ]
    
    results = []
    for test_name, test_func in tests:
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:20s}: {status}")
        if not success:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("The improved code should work correctly.")
    else:
        print("⚠️ SOME TESTS FAILED")
        print("Please check the error messages above.")
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)