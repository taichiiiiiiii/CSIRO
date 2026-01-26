# CSIRO Kaggle Competition

## 概要
このプロジェクトは、CSIRO Kaggleコンペティション用のコードリポジトリです。DINOv3を使用した画像分類タスクに取り組んでいます。

## プロジェクト構造
```
CSIRO/
├── README.md
├── csiro_training_dinov3_runpod.ipynb   # モデルトレーニング用ノートブック
└── csiro_inference_dinov3_runpod.ipynb  # 推論用ノートブック
```

## ノートブックの説明

### csiro_training_dinov3_runpod.ipynb
DINOv3モデルのトレーニングを行うためのノートブック。RunPod環境での実行に最適化されています。

主な機能：
- データの前処理
- DINOv3モデルの設定とトレーニング
- モデルの評価と保存

### csiro_inference_dinov3_runpod.ipynb
トレーニング済みモデルを使用して推論を実行するためのノートブック。

主な機能：
- トレーニング済みモデルのロード
- テストデータに対する予測
- 結果の出力と提出ファイルの生成

## モデルアーキテクチャ

### ベースモデル: DINOv3
- **モデル名**: `vit_huge_plus_patch16_dinov3.lvd1689m`
- **アーキテクチャ**: Vision Transformer Huge+ (ViT-H+)
- **事前学習**: 1.6B枚の画像でSelf-Distillation学習
- **パッチサイズ**: 16x16
- **特徴次元**: 1280次元

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

## 学習テクニック

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
- **学習率**:
  - Backbone: 5e-5 (より慎重な更新)
  - Head: 1e-4 (より積極的な学習)
- **Weight Decay**: 1e-2
- **Gradient Accumulation**: 4ステップ（実効バッチサイズ: 8）
- **Gradient Clipping**: max_norm=1.0

### 学習率スケジューリング
- **Warmup**: 3エポック（線形増加）
- **Cosine Annealing**: Warmup後にコサイン減衰
- **数式**: `lr = base_lr * 0.5 * (1 + cos(π * progress))`

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
- **Early Stopping**: patience=5エポック

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

### ハードウェア要件
- **GPU**: NVIDIA RTX A5000以上推奨（24GB VRAM）
- **学習時間**: 約20時間（5-fold全体）
- **推論時間**: 1画像あたり約0.5秒

### RunPodでの実行
1. RunPodインスタンスを起動（RTX 4090/A5000推奨）
2. 必要なライブラリをインストール
3. Kaggle APIを設定してデータをダウンロード
4. ノートブックを実行

## 使用方法

1. **トレーニング**
   ```bash
   # csiro_training_dinov3_runpod.ipynb を実行
   # 5-fold CVで約20時間
   ```

2. **推論**
   ```bash
   # csiro_inference_dinov3_runpod.ipynb を実行
   # 5モデルのアンサンブル予測
   ```

## パフォーマンス最適化

### メモリ管理
- Gradient Checkpointingによる活性化メモリの削減
- バッチサイズの動的調整
- 推論後の明示的なメモリ解放（`torch.cuda.empty_cache()`）

### 計算効率化
- Mixed Precision Training (AMP)
- DataLoader最適化（pin_memory=True）
- 並列データ処理（num_workers=4）

## モデルの特徴
- **マルチストリーム処理**: 左右の画像領域を独立して処理
- **階層的な特徴融合**: LocalMambaBlockによる段階的な特徴統合
- **タスク特化型ヘッド**: 各植生タイプに専用の予測器
- **物理的制約の組み込み**: Softplus活性化による非負値保証

## 注意事項
- RunPod環境での実行を想定しています
- GPUメモリに応じてバッチサイズの調整が必要な場合があります
- データセットへのパスは環境に応じて適切に設定してください

## ライセンス
このプロジェクトはKaggleコンペティション規約に従います。

## 作者
@taichiiiiiiii

## 更新履歴
- 2026-01-26: 初回コミット、基本的なトレーニングと推論ノートブックを追加