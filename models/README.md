# Model Versions Directory

モデルはバージョンごとに独立したディレクトリで管理します。

## Structure

```
models/
├── v1_dinov3_swa/        # v1: DINOv3 + SWA + マルチスケール学習（安定版）
├── v2_spatial_pooling/   # v2: SpatialAwarePooling による空間情報保持（実験版）
├── v3_complete_fix/      # v3: アーキテクチャ総合修正（最新・実験版）
└── README.md             # This file
```

## Version Comparison

| Version | 主な変更点 | Val R²（実測） | ステータス |
|---------|-----------|----------------|------------|
| v1_dinov3_swa | SWA / EMA / マルチスケール学習 | 0.85–0.87 | ✅ 安定版 |
| v2_spatial_pooling | AdaptiveAvgPool1d → SpatialAwarePooling、物理制約付き損失 | 未検証 | 🧪 実験版 |
| v3_complete_fix | CrossAttention ステレオ融合、Kaggle互換Mamba、物理制約統一 | 未検証 | 🧪 実験版（最新） |

> **Note**: v2 / v3 のスコアは学習未完了のため実測値がありません。
> 検証が完了するまでは v1 を提出用のベースラインとして使用してください。

## 各バージョンの内容

### v1_dinov3_swa（安定版）
- **Base Model**: DINOv3 ViT-Huge+（timm: `vit_huge_plus_patch16_dinov3.lvd1689m`）
- **改善点**: SWA、マルチスケール学習 [384, 448, 512]、EMA (decay 0.995)
- **メモリ**: 24GB GPU 向けに最適化
- 詳細は `v1_dinov3_swa/README.md` を参照

### v2_spatial_pooling（実験版）
- v1 の AdaptiveAvgPool1d を SpatialAwarePooling に置換し空間情報を保持
- 物理制約付き損失関数、軽量ステレオ融合
- 詳細は `v2_spatial_pooling/README.md` を参照

### v3_complete_fix（実験版・最新）
- CrossAttention によるステレオ相互作用、Kaggle 互換 Mamba ブロック
- 学習・推論での物理制約の統一、安全なチェックポイントロード
- **`training_fixed.ipynb` / `inference_fixed.ipynb` が正式版**（無印は旧ドラフト）
- 詳細は `v3_complete_fix/README.md` を参照

## Usage

各バージョンのディレクトリ構成:
- `training*.ipynb`: 学習ノートブック
- `inference*.ipynb`: 推論ノートブック（Kaggle 提出用）
- `README.md`: バージョン詳細ドキュメント
- チェックポイント（`*.pth`）は Git 管理外（別途保管）

## Best Practices

1. 新しいアーキテクチャは必ず新しいバージョンディレクトリで試す
2. 変更内容は各バージョンの README に記録する
3. 比較のためベースラインバージョンを残す
4. 実測スコアと期待値（未検証）を区別して記載する
5. チェックポイントのロードは必ず `torch.load(..., weights_only=True)` を使う（[SECURITY.md](../SECURITY.md) 参照）
