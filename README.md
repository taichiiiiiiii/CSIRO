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

## 環境設定

### 必要なライブラリ
- PyTorch
- torchvision
- transformers
- numpy
- pandas
- scikit-learn
- matplotlib

### RunPodでの実行
1. RunPodインスタンスを起動
2. 必要なライブラリをインストール
3. ノートブックを実行

## 使用方法

1. **トレーニング**
   ```bash
   # csiro_training_dinov3_runpod.ipynb を実行
   ```

2. **推論**
   ```bash
   # csiro_inference_dinov3_runpod.ipynb を実行
   ```

## モデルについて
DINOv3 (Self-Distillation with No Labels v3)は、自己教師あり学習による視覚表現学習手法です。ラベルなしデータから強力な特徴表現を学習することができます。

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