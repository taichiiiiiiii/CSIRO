# Version 1: DINOv3 + SWA + Multi-scale Training

## Overview
This version implements the following improvements over the baseline:
- **SWA (Stochastic Weight Averaging)** starting at epoch 25
- **Multi-scale training** with image sizes [384, 448, 512]
- **EMA (Exponential Moving Average)** with decay 0.995
- **Memory optimization** for 24GB GPUs

## Key Features
- Model: DINOv3 ViT-Huge+ (1.7B parameters)
- LocalMambaBlock fusion layers
- Dual-stream processing (left/right image split)
- Mixed precision training (AMP)
- Gradient checkpointing enabled

## Configuration
```python
IMG_SIZES = [384, 448, 512]  # Multi-scale
BASE_IMG_SIZE = 448
BATCH_SIZE = 1
GRAD_ACC = 8  # Effective batch size = 8
SWA_START_EPOCH = 25
```

## Performance
- Expected R2: ~0.85-0.87
- Training time: ~18-22 hours on RTX A5000
- GPU requirement: 24GB VRAM minimum

## Files
- `training.ipynb`: Training notebook with all improvements
- `inference.ipynb`: Inference with TTA support

## Usage
1. Run `training.ipynb` to train the model (creates both EMA and SWA checkpoints)
2. Upload checkpoints to Kaggle as dataset
3. Run `inference.ipynb` on Kaggle for predictions

## Checkpoints Location
- Training: `/workspace/checkpoints_improved/`
- Models: `best_ema_fold{0-4}.pth`, `best_swa_fold{0-4}.pth`