# Version 1: DINOv3 + SWA + Multi-scale Training

## Overview
This version implements the following improvements over the baseline:
- **SWA (Stochastic Weight Averaging)** starting at epoch 25
- **Multi-scale training** with image sizes [384, 448, 512]
- **EMA (Exponential Moving Average)** with decay 0.995
- **Memory optimization** for 24GB GPUs
- **T4×2 optimized inference** for Kaggle notebooks

## Key Features
- Model: DINOv3 ViT-Huge+ (1.7B parameters)
- LocalMambaBlock fusion layers
- Dual-stream processing (left/right image split)
- Mixed precision training (AMP)
- Gradient checkpointing enabled

## Training Configuration
```python
# Memory-optimized settings
IMG_SIZES = [384, 448, 512]  # Multi-scale
BASE_IMG_SIZE = 448
BATCH_SIZE = 1
GRAD_ACC = 8  # Effective batch size = 8
SWA_START_EPOCH = 25
NUM_WORKERS = 0  # Jupyter-safe
```

## Inference Configuration (T4×2 Optimized)
```python
# T4×2 specific settings
IMG_SIZE = 448  # Reduced from 512
TTA_TRANSFORMS = ["original", "hflip"]  # Reduced from 5
USE_EMA = True
USE_SWA = False  # Memory saving
GPU0_FOLDS = [0, 2, 4]
GPU1_FOLDS = [1, 3]
USE_FP16 = True
```

## Performance

### Training
- GPU requirement: 24GB VRAM minimum (RTX A5000, V100)
- Training time: ~18-22 hours on RTX A5000
- R² Score: 0.85-0.87

### Inference
| GPU Setup | Image Size | TTA | Time | R² Score |
|-----------|------------|-----|------|----------|
| A5000 (24GB) | 512px | 5 types | 15min | 0.85-0.87 |
| T4×2 (15GB×2) | 448px | 2 types | 15min | 0.84-0.86 |
| T4×1 (15GB) | 448px | 2 types | 30min | 0.84-0.86 |
| P100 (16GB) | 384px | None | 25min | 0.82-0.84 |

## Files
- `training.ipynb`: Full training notebook with SWA
- `inference.ipynb`: T4×2 optimized inference (main)
- `inference_t4x2_standalone.ipynb`: Alternative T4×2 version
- `inference_p100.ipynb`: P100-specific version

## Memory Usage (Inference)

### T4 (15GB) per GPU
```
Model: 3.5GB (FP16)
Activations: 5GB (448px)
Buffer: 2GB
Total: ~10.5GB / 15GB ✅
```

### GPU Assignment Strategy
- **GPU0**: Processes folds [0, 2, 4] sequentially
- **GPU1**: Processes folds [1, 3] sequentially
- Each GPU loads one model at a time
- Automatic cache clearing every 20 images

## Usage

### Training (RunPod/Colab)
```bash
# Run on 24GB+ GPU
1. Open training.ipynb
2. Install dependencies
3. Download data via Kaggle API
4. Run all cells (18-22 hours)
```

### Inference (Kaggle Notebook)
```bash
# Automatic T4×2 detection
1. Upload models as Kaggle dataset
2. Open inference.ipynb
3. Run all cells
4. Submit predictions
```

## Key Optimizations for T4×2
1. **Parallel Processing**: Different folds on different GPUs
2. **Memory Management**: 
   - FP16 inference
   - Reduced image size (448px)
   - Limited TTA (2 transforms)
   - Regular cache clearing
3. **Smart Loading**: CPU → GPU transfer
4. **Automatic Fallback**: Works on single GPU if needed

## Checkpoints
- Training output: `/workspace/checkpoints_improved/`
- Model files: 
  - `best_ema_fold{0-4}.pth` (primary)
  - `best_swa_fold{0-4}.pth` (optional)
- Size per model: ~3.4GB

## Notes
- The inference notebook automatically detects GPU configuration
- Falls back to single GPU mode if only one GPU is available
- Compatible with Kaggle's T4×2 and P100 environments