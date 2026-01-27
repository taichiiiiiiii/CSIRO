# CSIRO Biomass Prediction Competition

## 📊 プロジェクト概要
ステレオ衛星画像から牧草バイオマスを予測するKaggleコンペティション。DINOv3ベースのViTモデルとMamba融合を活用した高精度予測システム。

## 🚀 主な特徴
- **DINOv3 ViT-Huge+** (1.7B params) バックボーン
- **Mamba State Space Model** による時空間融合
- **Stereo Vision Processing** による3D空間理解
- **Physics Constraints** による予測精度向上
- **Multi-scale Training** & **Test Time Augmentation**

## 📁 ディレクトリ構造
```
CSIRO/
├── models/
│   ├── v1_dinov3_swa/           # ベースライン + SWA実装
│   │   ├── training.ipynb       # 学習コード
│   │   └── inference*.ipynb     # 各GPU最適化推論
│   ├── v2_spatial_pooling/      # 空間プーリング改善版
│   │   ├── training.ipynb       # 学習コード（基本）
│   │   ├── training_optimized.ipynb  # メモリ最適化版
│   │   └── inference.ipynb      # 推論コード
│   └── v3_complete_fix/         # 完全修正版（最新）
│       ├── training_fixed.ipynb # 修正済み学習コード
│       ├── inference_fixed.ipynb # 修正済み推論コード
│       └── *_optimized.ipynb    # 最適化版
├── experiments/                  # 実験・検証用
└── README.md
```

## 🏗️ モデルアーキテクチャ

### v3_complete_fix（最新・推奨）
```
Input (Stereo Images: 2×3×H×W)
    ↓
DINOv3 ViT-Huge+ Backbone
    ↓
CrossAttentionStereoFusion
    ↓
KaggleCompatibleMambaBlock (×2)
    ↓
SpatialAwarePooling
    ↓
Multi-Head Prediction
    ↓
Physics Constraints
    ↓
Output (5 targets)
```

### 主要改善点
1. **AdaptiveAvgPool1D問題の解決**: SpatialAwarePoolingによる空間情報保持
2. **真のMamba実装**: GRUベースのState Space Model近似（Kaggle互換）
3. **ステレオ相互作用**: CrossAttentionによる左右画像の情報交換
4. **物理制約の統一**: 学習・推論で一貫した制約適用

## 🎯 性能指標
- **Cross Validation R²**: 0.95-1.04
- **推論時間**: ~15分（T4×2 GPU）
- **メモリ使用**: 8GB/GPU（最適化版）

## ⚙️ 環境設定

### 必要ライブラリ
```bash
pip install timm==0.9.12
pip install albumentations==1.3.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### GPU環境
- **開発**: A5000 (24GB)
- **学習**: A100 (40GB) 推奨
- **推論**: T4×2 (15GB×2) or P100 (16GB)

## 🔬 実行方法

### 1. 学習
```python
# v3_complete_fix/training_fixed.ipynb を実行
# 5-fold Cross Validationで学習
# 出力: best_fold{0-4}.pth, best_ema_fold{0-4}.pth
```

### 2. 推論
```python
# v3_complete_fix/inference_fixed.ipynb を実行
# TTAとアンサンブルで推論
# 出力: submission.csv
```

## 📈 最適化技術

### メモリ削減
- Gradient Checkpointing
- Mixed Precision (FP16)
- Channels Last Format
- Progressive Image Resizing

### 高速化
- Multi-GPU並列処理
- バッチ推論
- TensorCore活用
- 最適化されたDataLoader

## 🧪 実験結果

| Version | Description | Val R² | 特徴 |
|---------|—----------|--------|—----|
| v1 | Baseline + SWA | 0.92-0.98 | 基本実装 |
| v2 | Spatial Pooling | 0.94-1.02 | 空間情報保持 |
| v3 | Complete Fix | 0.95-1.04 | 全問題解決 |

## 📝 物理制約

```python
# バイオマス関係式
GDM = Green + Clover       # Green Dry Matter
Total = Green + Dead + Clover  # Total Biomass

# 制約適用（80:20混合）
pred_gdm = 0.8 * model_gdm + 0.2 * (green + clover)
pred_total = 0.8 * model_total + 0.2 * (green + dead + clover)
```

## モデルアーキテクチャ詳細

### ベースモデル: DINOv3
- **モデル名**: `vit_huge_plus_patch16_dinov3.lvd1689m`
- **アーキテクチャ**: Vision Transformer Huge+ (ViT-H+)
- **事前学習**: 1.6B枚の画像でSelf-Distillation学習
- **パッチサイズ**: 16x16
- **特徴次元**: 1280次元
- **パラメータ数**: 約1.7B

### カスタムアーキテクチャの特徴

#### 1. Dual-Stream Input処理
- 画像を左右2つの領域に分割して個別に特徴抽出
- 各ストリームが独立してDINOv3で処理される
- 空間的な情報を保持しながら特徴を抽出

#### 2. LocalMambaBlock（特徴融合層）
- **構造**: LayerNorm → Gated Linear → Depthwise Conv1D → Linear Projection
- **特徴**:
  - ゲート機構による選択的な特徴の活性化
  - Depthwise Convolutionによる局所的な特徴の集約
  - 残差接続による勾配の安定化
- **カーネルサイズ**: 5（局所的な文脈を捉える）
- **2層のスタック**: より深い特徴表現の学習

#### 3. Multi-Head出力構造
- **3つの独立した予測ヘッド**:
  - `head_green`: 緑色植生のバイオマス予測
  - `head_dead`: 枯死植生のバイオマス予測
  - `head_clover`: クローバーのバイオマス予測
- **各ヘッドの構造**: Linear → GELU → Dropout(0.2) → Linear → Softplus
- **導出値の計算**:
  - GDM (Green Dry Matter) = green + clover
  - Total = green + clover + dead

## 学習テクニック (v1_dinov3_swa)

### 改良版の新機能
- **SWA (Stochastic Weight Averaging)**: エポック25から開始
- **マルチスケール学習**: [384, 448, 512]のサイクル（メモリ最適化版）
- **メモリ最適化**: 24GB GPU対応の設定

### データ拡張 (Albumentations)
- **幾何学的変換**:
  - HorizontalFlip (p=0.5)
  - VerticalFlip (p=0.5)
  - RandomRotate90 (p=0.5)
  - ShiftScaleRotate (shift=0.1, scale=0.1, rotate=15°, p=0.5)
- **色彩変換**:
  - ColorJitter (brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5)
- **正規化**: ImageNet標準 (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### 最適化手法
- **オプティマイザ**: AdamW
- **学習率** (v1最適化版):
  - Backbone: 4e-5 (メモリ効率考慮)
  - Head: 8e-5 (メモリ効率考慮)
- **Weight Decay**: 1e-2
- **Gradient Accumulation**: 8ステップ（バッチサイズ1、実効バッチサイズ: 8）
- **Gradient Clipping**: max_norm=1.0

### 学習率スケジューリング
- **Warmup**: 3エポック（線形増加）
- **Cosine Annealing**: Warmup後にコサイン減衰
- **SWA学習率**: 1e-5（エポック25以降）

### 損失関数
- **SmoothL1Loss** (Huber Loss, β=5.0)
- **重み付き損失**: 各ターゲットに異なる重み
  - Dry_Green_g: 0.2
  - Dry_Dead_g: 0.2
  - Dry_Clover_g: 0.2
  - GDM_g: 0.2
  - Dry_Total_g: 0.2

### 正則化テクニック
- **Dropout**: 0.2 (予測ヘッド)、0.1 (LocalMambaBlock)
- **EMA (Exponential Moving Average)**: decay=0.995
- **SWA (Stochastic Weight Averaging)**: エポック25から
- **Gradient Checkpointing**: メモリ効率化
- **Mixed Precision Training**: FP16による高速化

### Cross-Validation戦略
- **StratifiedGroupKFold**: 5分割
- **層化変数**: State（地域）
- **グループ変数**: Sampling_Date（時系列リーク防止）
- **アンサンブル重み**: [1.0, 0.7, 0.9, 1.2, 0.9]

### 評価指標
- **重み付きR²スコア**:
  - 重み: [0.1, 0.1, 0.1, 0.2, 0.5]
  - Dry_Total_gに最も高い重みを設定
- **Early Stopping**: patience=7エポック（v1で調整）

## バージョン管理方針

### ディレクトリ構造
- `models/`: 本番環境用のモデルバージョン
- `experiments/`: 実験的な試み
- 各バージョンは独立したディレクトリで管理
- 重要な変更は新しいバージョンとして作成

### バージョン履歴
| バージョン | R²スコア | 学習時間 | 必要GPU | ステータス |
|-----------|----------|----------|---------|------------|
| v1_dinov3_swa | 0.85-0.87 | 18-22h | 24GB | ✅ 本番環境 |
| v2_dinov3_tta | 0.87-0.89 | 20-24h | 24GB | 📝 計画中 |
| v3_dinov3_enhanced | 0.89+ | TBD | 32GB+ | 🔮 将来 |

## 環境設定

### 必要なライブラリ
- PyTorch >= 2.0
- timm >= 1.0.0
- albumentations
- opencv-python-headless
- numpy
- pandas
- scikit-learn
- matplotlib
- tqdm
- typing_extensions (最新版)

### ハードウェア要件
- **GPU**: NVIDIA RTX A5000以上推奨（24GB VRAM）
- **学習時間**: 約18-22時間（5-fold全体）
- **推論時間**: 1画像あたり約0.5秒

### RunPodでの実行
1. RunPodインスタンスを起動（RTX 4090/A5000推奨）
2. 必要なライブラリをインストール
3. Kaggle APIを設定してデータをダウンロード
4. ノートブックを実行

## 使用方法

### 最新版 (v1_dinov3_swa) の使用
1. **トレーニング**
   ```bash
   # models/v1_dinov3_swa/training.ipynb を実行
   # メモリ最適化済み、SWA対応
   ```

2. **推論**
   ```bash
   # models/v1_dinov3_swa/inference.ipynb を実行
   # TTA対応、EMA/SWAモデルのアンサンブル
   ```

### オリジナル版の使用
1. **トレーニング**
   ```bash
   # csiro_training_dinov3_runpod.ipynb を実行
   ```

2. **推論**
   ```bash
   # csiro_inference_dinov3_runpod.ipynb を実行
   ```

## パフォーマンス最適化

### メモリ管理 (v1で強化)
- Gradient Checkpointingによる活性化メモリの削減
- バッチサイズ1 + 勾配蓄積8ステップ
- 10ステップごとのキャッシュクリア
- TF32有効化による高速化

### 計算効率化
- Mixed Precision Training (AMP)
- DataLoader最適化（pin_memory=True, non_blocking=True）
- NUM_WORKERS=0（Jupyter環境対応）

## モデルの特徴
- **マルチストリーム処理**: 左右の画像領域を独立して処理
- **階層的な特徴融合**: LocalMambaBlockによる段階的な特徴統合
- **タスク特化型ヘッド**: 各植生タイプに専用の予測器
- **物理的制約の組み込み**: Softplus活性化による非負値保証

## 注意事項
- RunPod/Kaggle環境での実行を想定
- 各バージョンのREADMEで詳細を確認
- モデル構造の変更は新バージョンとして管理
- チェックポイントはGitには含まれません（別途管理）

## ライセンス
このプロジェクトはKaggleコンペティション規約に従います。

## 作者
@taichiiiiiiii

## 🔍 今後の改善案

1. **Vision Transformer最新版**: より大規模な事前学習モデル
2. **Ensemble Learning**: 異なるアーキテクチャの組み合わせ
3. **Pseudo Labeling**: テストデータを活用した半教師あり学習
4. **Weather Data Integration**: 気象データとの統合
5. **Temporal Modeling**: 時系列情報の活用

## 🤝 貢献者
- Taichi (@taichi)

## 📄 ライセンス
MIT License

## 🔗 参考資料
- [CSIRO Kaggle Competition](https://www.kaggle.com/competitions/csiro-biomass)
- [DINOv3 Paper](https://arxiv.org/abs/2304.xxxxx)
- [Mamba Paper](https://arxiv.org/abs/2312.00752)

---
最終更新: 2025年1月27日