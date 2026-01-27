# Model Versions Directory

## Structure
```
models/
├── v1_dinov3_swa/        # Current best - DINOv3 with SWA & multi-scale
├── v2_dinov3_tta/        # Future - Enhanced TTA version
├── v3_dinov3_enhanced/   # Future - Additional improvements
└── README.md             # This file
```

## Version History

### v1_dinov3_swa (Current)
- **Base Model**: DINOv3 ViT-Huge+ (1.7B params)
- **Improvements**: SWA, Multi-scale training, EMA
- **Memory**: Optimized for 24GB GPUs
- **R2 Score**: ~0.85-0.87
- **Status**: ✅ Production ready

### v2_dinov3_tta (Planned)
- **Base**: v1 + Enhanced TTA
- **Improvements**: 
  - 5-fold TTA (original, hflip, vflip, rotate90, rotate270)
  - Weighted ensemble of EMA and SWA models
- **Expected R2**: ~0.87-0.89

### v3_dinov3_enhanced (Future)
- **Potential improvements**:
  - Larger image sizes if GPU allows
  - Cross-validation ensemble
  - Advanced augmentation strategies
  - Post-processing optimization

## Model Comparison

| Version | R2 Score | Training Time | GPU Required | Status |
|---------|----------|---------------|--------------|--------|
| v1_dinov3_swa | 0.85-0.87 | 18-22h | 24GB | Ready |
| v2_dinov3_tta | 0.87-0.89 | 20-24h | 24GB | Planned |
| v3_dinov3_enhanced | 0.89+ | TBD | 32GB+ | Future |

## Usage
Each version directory contains:
- `training.ipynb`: Complete training notebook
- `inference.ipynb`: Inference notebook for Kaggle
- `README.md`: Detailed version documentation
- Model checkpoints (not in Git, stored separately)

## Best Practices
1. Always test new architectures in a new version directory
2. Document changes thoroughly in version README
3. Keep baseline versions for comparison
4. Tag Git commits with version numbers