# Version 3: Complete Architecture Fix

## Overview

v1/v2 で判明したアーキテクチャ上の問題をまとめて修正する実験版。

主な変更点:
1. **SpatialAwarePooling**: 空間情報を保持（v2 から継承）
2. **KaggleCompatibleMambaBlock**: 外部依存なしの State Space Model 近似（GRUベース）
3. **CrossAttentionStereoFusion**: 左右画像の相互作用を CrossAttention で学習
4. **物理制約の統一**: 学習と推論で同一の制約適用（80:20 混合）
5. **安全なチェックポイントロード**: `weights_only=True` + ラップ形式チェックポイントの展開 + 形状互換性チェック

## Files

| File | 内容 | 位置づけ |
|------|------|----------|
| `training_fixed.ipynb` | 修正済み学習ノートブック（Kaggle互換Mamba、次元統一、`save_model()` によるメタデータ付き保存） | ✅ **正式版** |
| `inference_fixed.ipynb` | 修正済み推論ノートブック（安全なロード、エラーハンドリング強化） | ✅ **正式版** |
| `inference_optimized.ipynb` | 推論最適化版（バッチ推論、FP16、TTA削減、fold毎の逐次ロード） | 高速版 |
| `training.ipynb` | 初期ドラフト（外部Mamba依存あり） | ⚠️ 旧版・参考用 |
| `inference.ipynb` | 初期ドラフトの推論 | ⚠️ 旧版・参考用 |

> **Note**: `training.ipynb` / `inference.ipynb`（無印）は `*_fixed.ipynb` に
> 置き換えられた旧ドラフトです。履歴参照用に残していますが、実行には
> `*_fixed.ipynb` を使用してください。

## Checkpoint Format

`training_fixed.ipynb` の `save_model()` は以下のラップ形式で保存します:

```python
{
    "model_state_dict": ...,   # モデル重み
    "config": {...},           # backbone / 次元設定
    "optimizer_state_dict": ..., # 任意
    "epoch": ..., "score": ...,  # 任意
}
```

推論ノートブックはラップ形式・生 state_dict のどちらでもロードできます。

## Status

🧪 **実験版（最新）** — 学習未完了のため実測スコアはありません。
提出用には安定版の `v1_dinov3_swa` を使用してください。

## Requirements

- ルートの `requirements.txt` を参照
- 学習: 24GB VRAM 以上の GPU を推奨、推論: T4×2 / P100 対応
- Kaggle API 認証は環境変数（`KAGGLE_USERNAME` / `KAGGLE_KEY`）または
  `~/.kaggle/kaggle.json` で設定（[SECURITY.md](../../SECURITY.md) 参照）
