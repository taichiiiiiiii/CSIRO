#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Code structure validation for improved training and inference scripts
Checks syntax and logical issues without running the actual code
"""

import ast
import sys
from pathlib import Path


def check_syntax(file_path):
    """Check if Python file has valid syntax"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        ast.parse(code)
        return True, "Syntax is valid"
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"


def check_imports_structure(file_path):
    """Check import statements structure"""
    issues = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Check for required imports for the features
    required_imports = {
        'improved_training.py': [
            'from torch.optim.swa_utils import AveragedModel, SWALR, update_bn',
            'import albumentations as A',
            'from timm.utils import ModelEmaV2'
        ],
        'improved_inference.py': [
            'import torch',
            'import torch.nn as nn',
            'import timm'
        ]
    }
    
    filename = Path(file_path).name
    if filename in required_imports:
        code = ''.join(lines)
        for req_import in required_imports[filename]:
            if req_import not in code:
                issues.append(f"Missing required import: {req_import}")
    
    return len(issues) == 0, issues


def check_class_definitions(file_path):
    """Check if required classes are defined"""
    issues = []
    required_classes = {
        'improved_training.py': ['BiomassModel', 'LocalMambaBlock', 'BiomassDataset', 'CFG'],
        'improved_inference.py': ['BiomassModel', 'LocalMambaBlock', 'TestDataset', 'CFG']
    }
    
    filename = Path(file_path).name
    if filename in required_classes:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        tree = ast.parse(code)
        defined_classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        
        for req_class in required_classes[filename]:
            if req_class not in defined_classes:
                issues.append(f"Missing required class: {req_class}")
    
    return len(issues) == 0, issues


def check_function_definitions(file_path):
    """Check if required functions are defined"""
    issues = []
    required_functions = {
        'improved_training.py': [
            'get_train_transforms',
            'get_val_transforms',
            'train_epoch',
            'valid_epoch',
            'train_fold_improved',
            'biomass_loss',
            'weighted_r2_score'
        ],
        'improved_inference.py': [
            'apply_tta_transform',
            'predict_with_tta',
            'run_inference_improved'
        ]
    }
    
    filename = Path(file_path).name
    if filename in required_functions:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        tree = ast.parse(code)
        defined_functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        
        for req_func in required_functions[filename]:
            if req_func not in defined_functions:
                issues.append(f"Missing required function: {req_func}")
    
    return len(issues) == 0, issues


def check_swa_implementation(file_path):
    """Check SWA implementation in training code"""
    if Path(file_path).name != 'improved_training.py':
        return True, []
    
    issues = []
    with open(file_path, 'r', encoding='utf-8') as f:
        code = f.read()
    
    # Check for SWA-specific code patterns
    swa_patterns = [
        'swa_model = AveragedModel',
        'SWALR',
        'update_bn',
        'SWA_START_EPOCH',
        'swa_model.update_parameters'
    ]
    
    for pattern in swa_patterns:
        if pattern not in code:
            issues.append(f"Missing SWA component: {pattern}")
    
    return len(issues) == 0, issues


def check_tta_implementation(file_path):
    """Check TTA implementation in inference code"""
    if Path(file_path).name != 'improved_inference.py':
        return True, []
    
    issues = []
    with open(file_path, 'r', encoding='utf-8') as f:
        code = f.read()
    
    # Check for TTA-specific code patterns
    tta_patterns = [
        'apply_tta_transform',
        'hflip',
        'vflip',
        'rotate90',
        'TTA_TRANSFORMS'
    ]
    
    for pattern in tta_patterns:
        if pattern not in code:
            issues.append(f"Missing TTA component: {pattern}")
    
    return len(issues) == 0, issues


def check_multiscale_implementation(file_path):
    """Check multi-scale implementation in training code"""
    if Path(file_path).name != 'improved_training.py':
        return True, []
    
    issues = []
    with open(file_path, 'r', encoding='utf-8') as f:
        code = f.read()
    
    # Check for multi-scale specific patterns
    multiscale_patterns = [
        'IMG_SIZES',
        '[448, 512, 576]',
        'img_size = CFG.IMG_SIZES[epoch % len(CFG.IMG_SIZES)]'
    ]
    
    for pattern in multiscale_patterns:
        if pattern not in code:
            issues.append(f"Missing multi-scale component: {pattern}")
    
    return len(issues) == 0, issues


def validate_file(file_path):
    """Run all validations on a file"""
    print(f"\n📄 Validating {Path(file_path).name}")
    print("=" * 50)
    
    checks = [
        ("Syntax", check_syntax),
        ("Imports", check_imports_structure),
        ("Classes", check_class_definitions),
        ("Functions", check_function_definitions),
        ("SWA Implementation", check_swa_implementation),
        ("TTA Implementation", check_tta_implementation),
        ("Multi-scale", check_multiscale_implementation)
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        passed, result = check_func(file_path)
        if passed:
            print(f"✅ {check_name}: PASS")
        else:
            print(f"❌ {check_name}: FAIL")
            if isinstance(result, list):
                for issue in result:
                    print(f"   - {issue}")
            else:
                print(f"   - {result}")
            all_passed = False
    
    return all_passed


def main():
    """Main validation function"""
    print("=" * 60)
    print("🔍 CODE STRUCTURE VALIDATION")
    print("=" * 60)
    
    # Files to validate
    files = [
        Path(__file__).parent / "improved_training.py",
        Path(__file__).parent / "improved_inference.py"
    ]
    
    all_valid = True
    for file_path in files:
        if not file_path.exists():
            print(f"\n❌ File not found: {file_path}")
            all_valid = False
            continue
        
        if not validate_file(file_path):
            all_valid = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    
    if all_valid:
        print("✅ All code structure checks passed!")
        print("\nThe improved code has the correct structure and should work")
        print("in an environment with all required packages installed.")
        print("\nRequired packages:")
        print("  - torch >= 2.0")
        print("  - timm >= 1.0.0")
        print("  - albumentations")
        print("  - opencv-python-headless")
        print("  - scikit-learn")
        print("  - pandas")
        print("  - numpy")
        print("  - matplotlib")
        print("  - tqdm")
    else:
        print("❌ Some validation checks failed!")
        print("Please review the issues above.")
    
    return 0 if all_valid else 1


if __name__ == "__main__":
    sys.exit(main())