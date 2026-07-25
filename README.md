# CSIRO Biomass Prediction Competition

## 📊 プロジェクト概要

ステレオ画像から牧草バイオマス（乾物重量）を予測する
[Kaggle コンペティション](https://www.kaggle.com/competitions/csiro-biomass)
の実験リポジトリ。DINOv3 ベースの ViT バックボーンに、ステレオ融合・
状態空間モデル風の特徴融合・物理制約を組み合わせて 5 つのターゲット
（Dry_Green_g / Dry_Dead_g / Dry_Clover_g / GDM_g / Dry_Total_g）を予測します。

## 🚀 主な特徴

- **DINOv3 ViT-Huge+**（約1.7Bパラメータ）バックボーン
- **デュアルストリーム処理**: 左右画像を分割し独立に特徴抽出
- **LocalMambaBlock / Mamba風融合層** による特徴統合
- **物理制約**: GDM = Green + Clover、Total = Green + Dead + Clover
- **SWA / EMA / マルチスケール学習 / TTA** による精度向上
- **メモリ最適化**: 24GB GPU で学習、Kaggle T4×2 / P100 で推論可能

## 📁 ディレクトリ構造

```
CSIRO/
├── models/
│   ├── v1_dinov3_swa/           # v1: SWA + マルチスケール学習（✅ 安定版）
│   │   ├── training.ipynb       # 学習
│   │   └── inference*.ipynb     # GPU別に最適化した推論
│   ├── v2_spatial_pooling/      # v2: 空間プーリング改善（🧪 実験版）
│   │   ├── training.ipynb       # 学習（基本）
│   │   ├── training_optimized.ipynb  # メモリ最適化版（推奨）
│   │   └── inference.ipynb      # 推論
│   ├── v3_complete_fix/         # v3: アーキテクチャ総合修正（🧪 実験版・最新）
│   │   ├── training_fixed.ipynb # 学習（正式版）
│   │   ├── inference_fixed.ipynb # 推論（正式版）
│   │   ├── inference_optimized.ipynb # 推論最適化版
│   │   └── training.ipynb / inference.ipynb  # 旧ドラフト（参考用）
│   └── README.md                # バージョン管理方針・比較表
├── requirements.txt             # 依存ライブラリ
├── SECURITY.md                  # セキュリティポリシー（認証情報・モデルロード）
└── README.md                    # This file
```

各バージョンの詳細は `models/README.md` および各ディレクトリの README を参照してください。

## ⚙️ セットアップ

### 1. 依存ライブラリ

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

> **Note**: DINOv3 バックボーン（`vit_huge_plus_patch16_dinov3.lvd1689m`）には
> DINOv3 対応版の timm（`>= 1.0.20`、2025年8月以降のリリース）が必要です。

### 2. Kaggle API 認証

**認証情報をノートブックに直接書かないでください**（詳細は [SECURITY.md](SECURITY.md)）。
以下のいずれかで設定します:

```bash
# 方法1: 環境変数
export KAGGLE_USERNAME=<your_username>
export KAGGLE_KEY=<your_api_key>

# 方法2: kaggle.json を配置
mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
```

### 3. GPU 環境

| 用途 | GPU | 備考 |
|------|-----|------|
| 学習 | 24GB VRAM 以上（RTX A5000 / A100 など） | 5-fold 全体で約18–22時間 |
| 推論 | Kaggle T4×2（15GB×2）/ P100（16GB） | 約15–30分 |

## 🔬 実行方法

### 安定版（v1_dinov3_swa）

1. **学習**: `models/v1_dinov3_swa/training.ipynb` を実行
   - 5-fold Cross Validation、SWA + EMA
   - 出力: `best_ema_fold{0-4}.pth`, `best_swa_fold{0-4}.pth`
2. **推論**: `models/v1_dinov3_swa/inference.ipynb` を実行（Kaggle T4×2）
   - TTA + fold アンサンブル → `submission.csv`

### 実験版（v3_complete_fix）

1. **学習**: `models/v3_complete_fix/training_fixed.ipynb`
2. **推論**: `models/v3_complete_fix/inference_fixed.ipynb`（高速版は `inference_optimized.ipynb`）

## 🏗️ モデルアーキテクチャ

### ベースモデル: DINOv3

- **モデル名**: `vit_huge_plus_patch16_dinov3.lvd1689m`（timm）
- **アーキテクチャ**: Vision Transformer Huge+（パッチ16、特徴1280次元、約1.7Bパラメータ）
- **事前学習**: LVD-1689M（約1.7B枚の画像による自己教師あり学習）

### 共通構造（v1）

```
Input（左右に分割したステレオ画像）
    ↓
DINOv3 ViT-Huge+ Backbone（左右で共有）
    ↓
LocalMambaBlock ×2（ゲート付き特徴融合）
    ↓
Pooling → Multi-Head（green / dead / clover）
    ↓
導出: GDM = green + clover, Total = green + dead + clover
```

- **予測ヘッド**: Linear → GELU → Dropout(0.2) → Linear → Softplus（非負値保証）
- **LocalMambaBlock**: LayerNorm → Gated Linear → Depthwise Conv1D(k=5) → Linear + 残差接続

### v3 での変更点

```
CrossAttentionStereoFusion（左右画像の相互作用）
    ↓
KaggleCompatibleMambaBlock ×2（外部依存なしのSSM近似）
    ↓
SpatialAwarePooling（空間情報保持）
    ↓
Multi-Head Prediction + 物理制約（80:20混合）
```

```python
# 物理制約の適用（v3、学習・推論で統一）
pred_gdm   = 0.8 * model_gdm   + 0.2 * (green + clover)
pred_total = 0.8 * model_total + 0.2 * (green + dead + clover)
```

## 📈 学習設定（v1）

### データ拡張（Albumentations）
- 幾何: HorizontalFlip / VerticalFlip / RandomRotate90（各 p=0.5）、ShiftScaleRotate（shift=0.1, scale=0.1, rotate=15°, p=0.5）
- 色彩: ColorJitter（brightness/contrast/saturation=0.1, hue=0.05, p=0.5）
- 正規化: ImageNet 標準（mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]）

### 最適化
- **AdamW**（Backbone: 4e-5 / Head: 8e-5、Weight Decay: 1e-2）
- **バッチ**: サイズ1 × 勾配蓄積8ステップ（実効バッチ8）、Gradient Clipping（max_norm=1.0）
- **スケジューラ**: Warmup 3エポック → Cosine Annealing、SWA は epoch 25 から（lr=1e-5）
- **損失**: SmoothL1Loss（β=5.0）、ターゲット均等重み [0.2×5]
- **正則化**: Dropout 0.2（ヘッド）/ 0.1（融合層）、EMA（decay=0.995）
- **メモリ**: Gradient Checkpointing、AMP（FP16）、マルチスケール [384, 448, 512]

### Cross-Validation
- **StratifiedGroupKFold** 5分割（層化: State、グループ: Sampling_Date で時系列リーク防止）
- **評価**: 重み付き R²（重み [0.1, 0.1, 0.1, 0.2, 0.5]、Dry_Total_g を最重視）
- **Early Stopping**: patience 7

## 🧪 バージョン履歴

| Version | 主な変更点 | Val R²（実測） | ステータス |
|---------|-----------|----------------|------------|
| v1_dinov3_swa | SWA / EMA / マルチスケール学習 | 0.85–0.87 | ✅ 安定版 |
| v2_spatial_pooling | SpatialAwarePooling、物理制約付き損失 | 未検証 | 🧪 実験版 |
| v3_complete_fix | CrossAttention融合、Kaggle互換Mamba、制約統一 | 未検証 | 🧪 実験版（最新） |

> **Note**: v2 / v3 は学習未完了のため実測スコアがありません（R² は定義上 1 を
> 超えないため、過去記載の「R² 0.95–1.04」等は期待値の誤記です）。提出には
> v1 を使用してください。

## 🔒 セキュリティ

- API キー・認証情報はコミット禁止（`.gitignore` で除外、詳細は [SECURITY.md](SECURITY.md)）
- チェックポイントのロードは全ノートブックで `torch.load(..., weights_only=True)` を使用
- 出所不明の `.pth` ファイルは読み込まない

## 🔍 今後の改善案

1. v2 / v3 の学習完了と実測スコアでの比較
2. 異なるアーキテクチャのアンサンブル
3. Pseudo Labeling（半教師あり学習）
4. 気象データ・時系列情報の統合

## 📄 ライセンス

本リポジトリのコードは [Kaggle コンペティション規約](https://www.kaggle.com/competitions/csiro-biomass/rules) に従って使用してください。

## 👤 作者

[@taichiiiiiiii](https://github.com/taichiiiiiiii)

## 🔗 参考資料

- [CSIRO Biomass Kaggle Competition](https://www.kaggle.com/competitions/csiro-biomass)
- [DINOv3 Paper (arXiv:2508.10104)](https://arxiv.org/abs/2508.10104)
- [Mamba Paper (arXiv:2312.00752)](https://arxiv.org/abs/2312.00752)

---
最終更新: 2026年7月25日
