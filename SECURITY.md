# セキュリティポリシー / Security Policy

## 機密情報の取り扱い

### 絶対にコミットしてはいけないもの
- Kaggle APIキー（`kaggle.json`、`KAGGLE_KEY`）
- 各種トークン・パスワード・秘密鍵（`.env`、`*.pem`、`*.key` など）

これらは `.gitignore` で除外していますが、ノートブックのセルや出力に直接
書き込むと除外できません。**認証情報をノートブックに書かないでください。**

### Kaggle API の認証方法（推奨順）
1. 環境変数を使う:
   ```bash
   export KAGGLE_USERNAME=<your_username>
   export KAGGLE_KEY=<your_api_key>
   ```
2. `kaggle.json` を `~/.kaggle/kaggle.json` に配置する（リポジトリ外）:
   ```bash
   mkdir -p ~/.kaggle
   cp kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

本リポジトリの学習ノートブックは上記いずれかが設定済みであることを前提とし、
未設定の場合はエラーで停止します（キーの書き込みは行いません）。

### もしキーをコミットしてしまったら
1. **直ちにキーを無効化する**: [Kaggle Account Settings](https://www.kaggle.com/settings) → API → "Expire API Token"
2. 新しいトークンを発行する
3. `git filter-repo` 等で履歴からも削除する（コミット取り消しだけでは履歴に残ります）

## モデルチェックポイントの取り扱い

`torch.load()` は内部で pickle を使用するため、**信頼できないチェックポイント
ファイルを読み込むと任意コードが実行される危険があります**。

- 本リポジトリのノートブックはすべて `torch.load(..., weights_only=True)` を
  使用しています（PyTorch 1.13 以降で利用可能、2.6 以降はデフォルト）。
- 新しくロード処理を書く場合も必ず `weights_only=True` を指定してください。
- 出所不明のチェックポイント（共有された `.pth` など）は読み込まないでください。

## 依存ライブラリ

- インストールは PyPI / PyTorch 公式インデックスからのみ行ってください。
- バージョンは `requirements.txt` に準拠してください。

## 脆弱性の報告

問題を発見した場合は、公開Issueではなくリポジトリオーナー宛に
非公開で連絡してください。
