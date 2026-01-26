#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CSIRO Biomass Training - Improved Version with SWA and Multi-scale
Improvements:
1. SWA (Stochastic Weight Averaging) - Last 5 epochs
2. Multi-scale training (512±64)
3. Better logging and monitoring
"""

import os
import gc
import math
import random
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

import cv2
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from timm.utils import ModelEmaV2
from sklearn.model_selection import StratifiedGroupKFold
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================
class CFG:
    TARGETS = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]
    N_FOLDS = 5
    BACKBONE = "vit_huge_plus_patch16_dinov3.lvd1689m"
    
    # Multi-scale settings
    IMG_SIZES = [448, 512, 576]  # 512±64
    BASE_IMG_SIZE = 512
    
    # Paths
    DATA_DIR = Path("/workspace/csiro-biomass")
    TRAIN_CSV = DATA_DIR / "train.csv"
    TRAIN_IMAGE_DIR = DATA_DIR / "train"
    CHECKPOINT_DIR = Path("/workspace/checkpoints_improved")
    
    SEED = 42
    BATCH_SIZE = 2
    GRAD_ACC = 4
    NUM_WORKERS = 4
    EPOCHS = 30
    WARMUP_EPOCHS = 3
    PATIENCE = 5
    
    # SWA settings
    SWA_START_EPOCH = 25  # Start SWA at epoch 25
    SWA_LR = 1e-5  # SWA learning rate
    
    LR_BACKBONE = 5e-5
    LR_HEAD = 1e-4
    WD = 1e-2
    EMA_DECAY = 0.995
    
    LOSS_WEIGHTS = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    R2_WEIGHTS = np.array([0.1, 0.1, 0.1, 0.2, 0.5])
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


CFG.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def seed_everything(seed=CFG.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


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
# DATASET & TRANSFORMS
# ============================================================
class BiomassDataset(Dataset):
    def __init__(self, df, transforms, image_dir):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.image_dir = Path(image_dir)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.image_dir / Path(row["image_path"]).name
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        left = img[:, :w//2]
        right = img[:, w//2:]
        
        if self.transforms:
            seed = random.randint(0, 99999)
            random.seed(seed)
            np.random.seed(seed)
            left = self.transforms(image=left)["image"]
            random.seed(seed)
            np.random.seed(seed)
            right = self.transforms(image=right)["image"]
        
        labels = torch.tensor([row[t] for t in CFG.TARGETS], dtype=torch.float32)
        return left, right, labels


def get_train_transforms(img_size=None):
    """Get training transforms with dynamic image size"""
    if img_size is None:
        img_size = CFG.BASE_IMG_SIZE
    
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms(img_size=None):
    """Get validation transforms with dynamic image size"""
    if img_size is None:
        img_size = CFG.BASE_IMG_SIZE
        
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# ============================================================
# LOSS & METRICS
# ============================================================
def biomass_loss(outputs, labels, weights=None):
    losses = nn.SmoothL1Loss(reduction="none", beta=5.0)(outputs, labels).mean(dim=0)
    if weights is None:
        return losses.mean()
    w = torch.as_tensor(weights, device=losses.device, dtype=losses.dtype)
    return (losses * w / w.sum()).sum()


def weighted_r2_score(y_true, y_pred):
    r2s = []
    for i in range(y_true.shape[1]):
        ss_res = np.sum((y_true[:, i] - y_pred[:, i]) ** 2)
        ss_tot = np.sum((y_true[:, i] - np.mean(y_true[:, i])) ** 2)
        r2s.append(1 - ss_res / ss_tot if ss_tot > 0 else 0.0)
    r2s = np.array(r2s)
    return np.sum(r2s * CFG.R2_WEIGHTS) / np.sum(CFG.R2_WEIGHTS), r2s


# ============================================================
# TRAINING FUNCTIONS
# ============================================================
scaler = torch.cuda.amp.GradScaler()


def train_epoch(model, loader, optimizer, device, epoch, ema=None, swa_model=None):
    """Training epoch with multi-scale support"""
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    # Select image size for this epoch (cycle through sizes)
    img_size = CFG.IMG_SIZES[epoch % len(CFG.IMG_SIZES)]
    pbar = tqdm(loader, desc=f"Train [Size: {img_size}]", leave=False)
    
    for i, (left, right, labels) in enumerate(pbar):
        left = left.to(device)
        right = right.to(device)
        labels = labels.to(device)
        
        with torch.cuda.amp.autocast():
            outputs = model((left, right))
            loss = biomass_loss(outputs, labels, CFG.LOSS_WEIGHTS) / CFG.GRAD_ACC
        
        scaler.scale(loss).backward()
        total_loss += loss.item() * left.size(0) * CFG.GRAD_ACC
        
        if (i + 1) % CFG.GRAD_ACC == 0 or (i + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            if ema:
                ema.update(model)
                
            # Update SWA model if in SWA phase
            if swa_model is not None and epoch >= CFG.SWA_START_EPOCH:
                swa_model.update_parameters(model)
                
            optimizer.zero_grad()
        
        pbar.set_postfix({"loss": f"{loss.item() * CFG.GRAD_ACC:.4f}"})
    
    return total_loss / len(loader.dataset)


@torch.no_grad()
def valid_epoch(model, loader, device):
    """Validation epoch"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for left, right, labels in tqdm(loader, desc="Valid", leave=False):
        left = left.to(device)
        right = right.to(device)
        labels = labels.to(device)
        
        with torch.cuda.amp.autocast():
            outputs = model((left, right))
            loss = biomass_loss(outputs, labels, CFG.LOSS_WEIGHTS)
        
        total_loss += loss.item() * left.size(0)
        all_preds.append(outputs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    
    preds = np.vstack(all_preds)
    labels = np.vstack(all_labels)
    r2, per_r2 = weighted_r2_score(labels, preds)
    return total_loss / len(loader.dataset), r2, per_r2


def build_optimizer(model):
    backbone_ids = {id(p) for p in model.backbone.parameters()}
    backbone_params = [p for p in model.parameters() if p.requires_grad and id(p) in backbone_ids]
    head_params = [p for p in model.parameters() if p.requires_grad and id(p) not in backbone_ids]
    return optim.AdamW([
        {"params": backbone_params, "lr": CFG.LR_BACKBONE, "weight_decay": CFG.WD},
        {"params": head_params, "lr": CFG.LR_HEAD, "weight_decay": CFG.WD},
    ])


def build_scheduler(optimizer, num_epochs):
    def lr_lambda(epoch):
        if epoch < CFG.WARMUP_EPOCHS:
            return (epoch + 1) / CFG.WARMUP_EPOCHS
        progress = (epoch - CFG.WARMUP_EPOCHS) / (num_epochs - CFG.WARMUP_EPOCHS)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


# ============================================================
# MAIN TRAINING FUNCTION WITH SWA
# ============================================================
def train_fold_improved(fold, df_wide):
    print(f"\n{'='*60}")
    print(f"🚀 FOLD {fold} / {CFG.N_FOLDS-1} - Improved Version")
    print(f"   Features: SWA, Multi-scale Training")
    print(f"{'='*60}")
    
    # K-Fold split
    sgkf = StratifiedGroupKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
    splits = list(sgkf.split(df_wide, df_wide["State"], groups=df_wide["Sampling_Date"]))
    train_idx, val_idx = splits[fold]
    
    train_df = df_wide.iloc[train_idx].reset_index(drop=True)
    val_df = df_wide.iloc[val_idx].reset_index(drop=True)
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")
    
    # Clear memory
    torch.cuda.empty_cache()
    gc.collect()
    
    # Model
    model = BiomassModel(CFG.BACKBONE, pretrained=True).to(CFG.DEVICE)
    ema = ModelEmaV2(model, decay=CFG.EMA_DECAY)
    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer, CFG.EPOCHS)
    
    # SWA model (initialized later)
    swa_model = None
    swa_scheduler = None
    swa_start = CFG.SWA_START_EPOCH
    
    # Training loop
    best_r2 = -float("inf")
    best_swa_r2 = -float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_r2": [], "img_size": []}
    
    for epoch in range(CFG.EPOCHS):
        # Dynamic image size selection
        img_size = CFG.IMG_SIZES[epoch % len(CFG.IMG_SIZES)]
        
        # Create datasets with current image size
        train_ds = BiomassDataset(
            train_df, 
            get_train_transforms(img_size), 
            CFG.TRAIN_IMAGE_DIR
        )
        val_ds = BiomassDataset(
            val_df, 
            get_val_transforms(CFG.BASE_IMG_SIZE),  # Always validate at base size
            CFG.TRAIN_IMAGE_DIR
        )
        
        train_loader = DataLoader(
            train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True,
            num_workers=CFG.NUM_WORKERS, pin_memory=True, drop_last=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=CFG.BATCH_SIZE, shuffle=False,
            num_workers=CFG.NUM_WORKERS, pin_memory=True
        )
        
        print(f"\nEpoch {epoch+1}/{CFG.EPOCHS} [Image Size: {img_size}]")
        
        # Initialize SWA if we've reached the SWA start epoch
        if epoch == swa_start and swa_model is None:
            print(f"  🔄 Starting SWA at epoch {epoch+1}")
            swa_model = AveragedModel(model)
            swa_scheduler = SWALR(optimizer, swa_lr=CFG.SWA_LR)
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, CFG.DEVICE, epoch, ema, swa_model)
        
        # Step scheduler
        if epoch < swa_start:
            scheduler.step()
        elif swa_scheduler is not None:
            swa_scheduler.step()
        
        # Validate with EMA model
        val_loss, val_r2, per_r2 = valid_epoch(ema.module, val_loader, CFG.DEVICE)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_r2"].append(val_r2)
        history["img_size"].append(img_size)
        
        lr = optimizer.param_groups[0]["lr"]
        print(f"  Loss: {train_loss:.4f}/{val_loss:.4f} | R2: {val_r2:.4f} | LR: {lr:.2e}")
        print(f"  R2: G={per_r2[0]:.3f} D={per_r2[1]:.3f} C={per_r2[2]:.3f} GDM={per_r2[3]:.3f} T={per_r2[4]:.3f}")
        
        # Save best EMA model
        if val_r2 > best_r2:
            best_r2 = val_r2
            patience_counter = 0
            save_path = CFG.CHECKPOINT_DIR / f"best_ema_fold{fold}.pth"
            torch.save(ema.module.state_dict(), save_path)
            print(f"  ✅ Saved EMA: {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= CFG.PATIENCE and epoch < swa_start:
                print(f"\n⚠️ Early stopping at epoch {epoch+1}")
                break
        
        # Validate SWA model if available
        if swa_model is not None and epoch >= swa_start:
            # Update batch norm statistics
            update_bn(train_loader, swa_model, device=CFG.DEVICE)
            swa_loss, swa_r2, swa_per_r2 = valid_epoch(swa_model, val_loader, CFG.DEVICE)
            print(f"  SWA R2: {swa_r2:.4f}")
            
            if swa_r2 > best_swa_r2:
                best_swa_r2 = swa_r2
                save_path = CFG.CHECKPOINT_DIR / f"best_swa_fold{fold}.pth"
                torch.save(swa_model.state_dict(), save_path)
                print(f"  ✅ Saved SWA: {save_path}")
    
    print(f"\n🏆 Fold {fold} Results:")
    print(f"   Best EMA R2: {best_r2:.4f}")
    if best_swa_r2 > -float("inf"):
        print(f"   Best SWA R2: {best_swa_r2:.4f}")
    
    # Plot training history
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss plot
    axes[0, 0].plot(history["train_loss"], label="Train")
    axes[0, 0].plot(history["val_loss"], label="Val")
    axes[0, 0].set_title(f"Fold {fold} Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # R2 plot
    axes[0, 1].plot(history["val_r2"], "g")
    axes[0, 1].axhline(best_r2, color="r", linestyle="--", label=f"Best EMA: {best_r2:.4f}")
    if best_swa_r2 > -float("inf"):
        axes[0, 1].axhline(best_swa_r2, color="b", linestyle="--", label=f"Best SWA: {best_swa_r2:.4f}")
    axes[0, 1].axvline(swa_start, color="gray", linestyle=":", label=f"SWA Start")
    axes[0, 1].set_title(f"Fold {fold} R2")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("R2 Score")
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Image size plot
    axes[1, 0].plot(history["img_size"], "o-")
    axes[1, 0].set_title("Image Size Schedule")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Image Size")
    axes[1, 0].grid(True)
    
    # Learning rate plot
    lrs = [optimizer.param_groups[0]["lr"] for _ in range(len(history["train_loss"]))]
    axes[1, 1].plot(lrs)
    axes[1, 1].set_title("Learning Rate Schedule")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Learning Rate")
    axes[1, 1].set_yscale("log")
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(CFG.CHECKPOINT_DIR / f"fold{fold}_history.png", dpi=100)
    plt.show()
    
    # Cleanup
    del model, ema, swa_model, optimizer, scheduler, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()
    
    return best_r2, best_swa_r2


# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    seed_everything()
    
    print("=" * 60)
    print("📋 Improved Training Configuration")
    print("=" * 60)
    print(f"N_FOLDS       = {CFG.N_FOLDS}")
    print(f"BACKBONE      = {CFG.BACKBONE}")
    print(f"IMG_SIZES     = {CFG.IMG_SIZES}")
    print(f"BATCH_SIZE    = {CFG.BATCH_SIZE} x {CFG.GRAD_ACC} = {CFG.BATCH_SIZE * CFG.GRAD_ACC}")
    print(f"SWA_START     = Epoch {CFG.SWA_START_EPOCH}")
    print(f"DEVICE        = {CFG.DEVICE}")
    print("=" * 60)
    
    # Load data
    print("\n📊 Loading data...")
    df_long = pd.read_csv(CFG.TRAIN_CSV)
    df_wide = df_long.pivot(index="image_path", columns="target_name", values="target").reset_index()
    df_wide = df_wide[["image_path"] + CFG.TARGETS]
    meta = df_long[["image_path", "Sampling_Date", "State"]].drop_duplicates()
    df_wide = df_wide.merge(meta, on="image_path", how="left")
    print(f"✅ Loaded {len(df_wide)} images")
    
    # Train all folds
    all_scores = []
    start_time = time.time()
    
    for fold in range(CFG.N_FOLDS):
        fold_start = time.time()
        ema_score, swa_score = train_fold_improved(fold, df_wide)
        fold_time = (time.time() - fold_start) / 3600
        all_scores.append((fold, ema_score, swa_score, fold_time))
        print(f"\n⏱️ Fold {fold} completed in {fold_time:.2f} hours")
    
    total_time = (time.time() - start_time) / 3600
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 60)
    for fold, ema_score, swa_score, t in all_scores:
        print(f"  Fold {fold}: EMA R2={ema_score:.4f}, SWA R2={swa_score:.4f} ({t:.2f}h)")
    print(f"\n⏱️ Total time: {total_time:.2f} hours")
    print(f"📁 Checkpoints saved to: {CFG.CHECKPOINT_DIR}")