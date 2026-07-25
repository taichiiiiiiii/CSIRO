# Version 2: Spatial-Aware Pooling

## Overview

v1 の主要な課題だった **AdaptiveAvgPool1d による空間情報の消失** を解決する実験版。

主な変更点:
1. **SpatialAwarePooling**: 平均プーリングを空間構造を保持するプーリングに置換
2. **物理制約付き損失関数**: GDM = Green + Clover、Total = Green + Dead + Clover の関係を損失に組み込み
3. **軽量ステレオ融合**: 左右画像の特徴を軽量な融合層で統合

## Files

| File | 内容 | 位置づけ |
|------|------|----------|
| `training.ipynb` | 基本の学習ノートブック | 基本版 |
| `training_optimized.ipynb` | メモリ最適化版（Gradient Checkpointing、Memory Efficient Attention、動的バッチサイズ） | **推奨** |
| `inference.ipynb` | T4×2 向け推論ノートブック | 推論用 |

## Status

🧪 **実験版** — 学習未完了のため実測スコアはありません。
提出用には安定版の `v1_dinov3_swa` を使用してください。

## Requirements

- ルートの `requirements.txt` を参照
- 学習: 24GB VRAM 以上の GPU を推奨
- Kaggle API 認証は環境変数（`KAGGLE_USERNAME` / `KAGGLE_KEY`）または
  `~/.kaggle/kaggle.json` で設定（[SECURITY.md](../../SECURITY.md) 参照）
