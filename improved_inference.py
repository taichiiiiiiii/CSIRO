#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CSIRO Biomass Inference - Improved Version with TTA
Improvements:
1. Test Time Augmentation (TTA) with multiple transforms
2. Support for both EMA and SWA models
3. Better ensemble weighting
"""

import os
import gc
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm
from tqdm import tqdm

warnings.filterwarnings('ignore')


# ============================================================
# CONFIGURATION
# ============================================================
class CFG:
    TARGETS = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]
    DATA_DIR = Path("/kaggle/input/csiro-biomass")
    MODEL_DIR = Path("/kaggle/input/csiro-improved-models")  # Path to improved models
    
    IMG_SIZE = 512
    N_FOLDS = 5
    BACKBONE = "vit_huge_plus_patch16_dinov3.lvd1689m"
    
    BATCH_SIZE = 1
    NUM_WORKERS = 0
    
    # TTA settings
    TTA_ENABLED = True
    TTA_TRANSFORMS = ["original", "hflip", "vflip", "rotate90", "rotate270"]
    
    # Model types to use
    USE_EMA = True
    USE_SWA = True
    
    # Ensemble weights (can be adjusted based on validation scores)
    FOLD_WEIGHTS = [1.0, 0.9, 0.95, 1.1, 0.95]  # Per-fold weights
    MODEL_TYPE_WEIGHTS = {"ema": 0.6, "swa": 0.4}  # EMA vs SWA weights
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# MODEL
# ============================================================
class LocalMambaBlock(nn.Module):
    def __init__(self, dim, kernel_size=5, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim)
        self.gate = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        x = x * torch.sigmoid(self.gate(x))
        x = self.dwconv(x.transpose(1, 2)).transpose(1, 2)
        x = self.proj(x)
        return shortcut + self.drop(x)


class BiomassModel(nn.Module):
    def __init__(self, model_name, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="")
        nf = self.backbone.num_features
        
        if hasattr(self.backbone, "set_grad_checkpointing"):
            self.backbone.set_grad_checkpointing(True)
        
        self.fusion = nn.Sequential(
            LocalMambaBlock(nf, kernel_size=5, dropout=0.1),
            LocalMambaBlock(nf, kernel_size=5, dropout=0.1)
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        self.head_green = nn.Sequential(
            nn.Linear(nf, nf//2), nn.GELU(), nn.Dropout(0.2), 
            nn.Linear(nf//2, 1), nn.Softplus()
        )
        self.head_dead = nn.Sequential(
            nn.Linear(nf, nf//2), nn.GELU(), nn.Dropout(0.2), 
            nn.Linear(nf//2, 1), nn.Softplus()
        )
        self.head_clover = nn.Sequential(
            nn.Linear(nf, nf//2), nn.GELU(), nn.Dropout(0.2), 
            nn.Linear(nf//2, 1), nn.Softplus()
        )

    def forward(self, x):
        left, right = x
        x_l = self.backbone(left)
        x_r = self.backbone(right)
        x = self.fusion(torch.cat([x_l, x_r], dim=1))
        x = self.pool(x.transpose(1, 2)).flatten(1)
        green = self.head_green(x)
        dead = self.head_dead(x)
        clover = self.head_clover(x)
        gdm = green + clover
        total = green + clover + dead
        return torch.cat([green, dead, clover, gdm, total], dim=1)


# ============================================================
# TTA TRANSFORMS
# ============================================================
def apply_tta_transform(img, transform_name):
    """Apply Test Time Augmentation transform"""
    if transform_name == "original":
        return img
    elif transform_name == "hflip":
        return img.flip(-1)  # Horizontal flip
    elif transform_name == "vflip":
        return img.flip(-2)  # Vertical flip
    elif transform_name == "rotate90":
        return torch.rot90(img, k=1, dims=[-2, -1])
    elif transform_name == "rotate270":
        return torch.rot90(img, k=3, dims=[-2, -1])
    else:
        return img


def inverse_tta_transform(pred, transform_name):
    """Inverse transform for predictions (if needed for spatial predictions)"""
    # For scalar predictions, no inverse transform needed
    return pred


@torch.no_grad()
def predict_with_tta(model, left, right, device, tta_transforms=None):
    """Make predictions with Test Time Augmentation"""
    if tta_transforms is None or not CFG.TTA_ENABLED:
        tta_transforms = ["original"]
    
    predictions = []
    
    for transform_name in tta_transforms:
        # Apply transform to both left and right images
        left_aug = apply_tta_transform(left, transform_name)
        right_aug = apply_tta_transform(right, transform_name)
        
        # Make prediction
        with torch.cuda.amp.autocast():
            pred = model((left_aug, right_aug))
        
        # Apply inverse transform if needed (for spatial outputs)
        pred = inverse_tta_transform(pred, transform_name)
        predictions.append(pred)
    
    # Average predictions
    final_pred = torch.mean(torch.stack(predictions), dim=0)
    return final_pred


# ============================================================
# DATASET
# ============================================================
class TestDataset(Dataset):
    def __init__(self, df, data_dir, transform):
        self.df = df.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.data_dir / row["image_path"]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        left = img.crop((0, 0, w // 2, h))
        right = img.crop((w // 2, 0, w, h))
        left = self.transform(left)
        right = self.transform(right)
        return left, right, row["image_path"]


def collate_fn(batch):
    lefts = torch.stack([b[0] for b in batch])
    rights = torch.stack([b[1] for b in batch])
    paths = [b[2] for b in batch]
    return lefts, rights, paths


# ============================================================
# INFERENCE FUNCTION
# ============================================================
def run_inference_improved():
    """Run inference with TTA and improved ensemble"""
    
    print("=" * 60)
    print("🚀 CSIRO Biomass Inference - Improved Version")
    print("=" * 60)
    print(f"Device: {CFG.DEVICE}")
    print(f"TTA Enabled: {CFG.TTA_ENABLED}")
    if CFG.TTA_ENABLED:
        print(f"TTA Transforms: {CFG.TTA_TRANSFORMS}")
    print(f"Use EMA: {CFG.USE_EMA}")
    print(f"Use SWA: {CFG.USE_SWA}")
    print("=" * 60)
    
    # Load test data
    test_df = pd.read_csv(CFG.DATA_DIR / "test.csv")
    test_wide = test_df[["image_path"]].drop_duplicates().reset_index(drop=True)
    print(f"\n📊 Test images: {len(test_wide)}")
    
    # Prepare dataset and loader
    test_tfms = T.Compose([
        T.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = TestDataset(test_wide, CFG.DATA_DIR, test_tfms)
    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.BATCH_SIZE,
        shuffle=False,
        num_workers=CFG.NUM_WORKERS,
        collate_fn=collate_fn
    )
    
    # Collect predictions
    all_predictions = []
    all_weights = []
    all_paths = None
    
    # Process each fold and model type
    for fold in range(CFG.N_FOLDS):
        fold_predictions = []
        fold_weights = []
        
        # Process EMA model
        if CFG.USE_EMA:
            ema_path = CFG.MODEL_DIR / f"best_ema_fold{fold}.pth"
            if ema_path.exists():
                print(f"\n{'='*40}")
                print(f"Fold {fold} - EMA Model")
                print(f"{'='*40}")
                
                # Load model
                model = BiomassModel(CFG.BACKBONE, pretrained=False)
                state_dict = torch.load(ema_path, map_location="cpu")
                
                # Handle DataParallel wrapper if present
                if list(state_dict.keys())[0].startswith("module."):
                    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                
                model.load_state_dict(state_dict)
                model = model.to(CFG.DEVICE)
                model.eval()
                
                # Make predictions
                preds = []
                paths = []
                
                for left, right, p in tqdm(test_loader, desc=f"Fold {fold} EMA"):
                    left = left.to(CFG.DEVICE)
                    right = right.to(CFG.DEVICE)
                    
                    # Predict with TTA
                    pred = predict_with_tta(
                        model, left, right, 
                        CFG.DEVICE, CFG.TTA_TRANSFORMS if CFG.TTA_ENABLED else None
                    )
                    
                    preds.append(pred.cpu().numpy())
                    paths.extend(p)
                
                preds = np.vstack(preds)
                fold_predictions.append(preds)
                fold_weights.append(CFG.MODEL_TYPE_WEIGHTS["ema"])
                
                if all_paths is None:
                    all_paths = paths
                
                print(f"EMA predictions shape: {preds.shape}")
                
                # Clear memory
                del model, state_dict
                torch.cuda.empty_cache()
                gc.collect()
        
        # Process SWA model
        if CFG.USE_SWA:
            swa_path = CFG.MODEL_DIR / f"best_swa_fold{fold}.pth"
            if swa_path.exists():
                print(f"\n{'='*40}")
                print(f"Fold {fold} - SWA Model")
                print(f"{'='*40}")
                
                # Load model
                model = BiomassModel(CFG.BACKBONE, pretrained=False)
                state_dict = torch.load(swa_path, map_location="cpu")
                
                # Handle DataParallel wrapper if present
                if list(state_dict.keys())[0].startswith("module."):
                    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                
                model.load_state_dict(state_dict)
                model = model.to(CFG.DEVICE)
                model.eval()
                
                # Make predictions
                preds = []
                
                for left, right, p in tqdm(test_loader, desc=f"Fold {fold} SWA"):
                    left = left.to(CFG.DEVICE)
                    right = right.to(CFG.DEVICE)
                    
                    # Predict with TTA
                    pred = predict_with_tta(
                        model, left, right, 
                        CFG.DEVICE, CFG.TTA_TRANSFORMS if CFG.TTA_ENABLED else None
                    )
                    
                    preds.append(pred.cpu().numpy())
                
                preds = np.vstack(preds)
                fold_predictions.append(preds)
                fold_weights.append(CFG.MODEL_TYPE_WEIGHTS["swa"])
                
                print(f"SWA predictions shape: {preds.shape}")
                
                # Clear memory
                del model, state_dict
                torch.cuda.empty_cache()
                gc.collect()
        
        # Combine fold predictions with model type weights
        if fold_predictions:
            fold_weights = np.array(fold_weights)
            fold_weights = fold_weights / fold_weights.sum()
            
            fold_pred = np.sum([p * w for p, w in zip(fold_predictions, fold_weights)], axis=0)
            all_predictions.append(fold_pred * CFG.FOLD_WEIGHTS[fold])
            all_weights.append(CFG.FOLD_WEIGHTS[fold])
    
    # Final ensemble
    print(f"\n{'='*40}")
    print(f"Creating Final Ensemble")
    print(f"{'='*40}")
    
    total_weight = sum(all_weights)
    ensemble = np.sum(all_predictions, axis=0) / total_weight
    print(f"Ensemble shape: {ensemble.shape}")
    print(f"Models used: {len(all_predictions)}")
    
    # Create predictions DataFrame
    preds_wide = pd.DataFrame(ensemble, columns=CFG.TARGETS)
    preds_wide.insert(0, 'image_path', all_paths)
    
    print(f"\nPredictions summary:")
    print(preds_wide[CFG.TARGETS].describe())
    
    # Convert to long format
    preds_long = preds_wide.melt(
        id_vars=['image_path'],
        value_vars=CFG.TARGETS,
        var_name='target_name',
        value_name='target'
    )
    
    # Merge with test_df to get sample_id
    submission = pd.merge(
        test_df[['sample_id', 'image_path', 'target_name']],
        preds_long,
        on=['image_path', 'target_name'],
        how='left'
    )
    
    # Keep only required columns
    submission = submission[['sample_id', 'target']]
    
    # Check for missing values
    missing_count = submission['target'].isna().sum()
    if missing_count > 0:
        print(f"\n⚠️ Warning: {missing_count} missing predictions!")
        submission['target'] = submission['target'].fillna(0.0)
    else:
        print("\n✅ All predictions matched!")
    
    # Sort by sample_id and clip negative values
    submission = submission.sort_values('sample_id').reset_index(drop=True)
    submission['target'] = submission['target'].clip(lower=0)
    
    # Save submission
    submission.to_csv("submission_improved.csv", index=False)
    
    print(f"\n✅ Saved: submission_improved.csv")
    print(f"Shape: {submission.shape}")
    print(f"\nFirst 10 rows:")
    print(submission.head(10))
    print(f"\nTarget statistics:")
    print(submission['target'].describe())
    
    # Validation
    print(f"\n" + "="*50)
    print("VALIDATION")
    print("="*50)
    print(f"Expected rows: {len(test_df)}")
    print(f"Actual rows: {len(submission)}")
    print(f"Match: {len(submission) == len(test_df)}")
    print(f"No missing: {not submission['target'].isna().any()}")
    print(f"All positive: {(submission['target'] >= 0).all()}")
    
    return submission


# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    submission = run_inference_improved()
    print("\n🎉 Inference complete!")