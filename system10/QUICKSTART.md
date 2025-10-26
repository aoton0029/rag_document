# クイックスタートガイド

このガイドでは、RAG評価フレームワークをすぐに使い始める方法を説明します。

## 前提条件

- Python 3.9以上がインストールされている
- Dockerがインストールされている（データベース使用時）
- Ollamaがインストールされている（ローカルLLM使用時）

## ステップ1: 環境セットアップ

### 1.1 仮想環境の作成と有効化

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 1.2 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

## ステップ2: データベースのセットアップ（オプション）

データベースを使用する場合は、Dockerで起動します。

```bash
# すべてのデータベースを起動
docker-compose up -d

# 起動確認
docker-compose ps
```

各サービスが起動するまで1-2分待ちます。

### データベースの確認

```bash
# Milvus (ベクトルストア)
curl http://localhost:9091/healthz

# MongoDB (ドキュメントストア)
# MongoDBクライアントで接続
# mongodb://admin:password@localhost:27017

# Redis (インデックスストア)
redis-cli ping
# PONG が返ればOK

# Neo4j (グラフストア)
# ブラウザでアクセス
# http://localhost:7474
# ユーザー名: neo4j, パスワード: password
```

## ステップ3: Ollamaのセットアップ（オプション）

ローカルLLMを使用する場合は、Ollamaをセットアップします。

### 3.1 Ollamaのインストール

[Ollama公式サイト](https://ollama.ai/)からダウンロードしてインストール

### 3.2 必要なモデルのダウンロード

```bash
# LLMモデル（約20GB）
ollama pull qwen3:32b

# 埋め込みモデル（約5GB）
ollama pull qwen3-embedding:8b

# 小さいモデルでテスト（約4GB）
ollama pull qwen3:8b
```

### 3.3 Ollamaの起動確認

```bash
# Ollamaサーバーが起動しているか確認
ollama list

# テストリクエスト
ollama run qwen3:8b "Hello"
```

## ステップ4: サンプルデータの準備

```bash
# dataディレクトリを作成
mkdir -p data\documents

# サンプルPDFを配置
# 実際のPDFファイルを data/documents/ に配置してください
```

## ステップ5: 簡単な実行例

### 5.1 シンプルな例（データベースなし）

```bash
# examples/simple_example.pyを編集してPDFパスを設定
# pdf_path = "data/documents/your_file.pdf"

python examples\simple_example.py
```

### 5.2 完全な例（データベース使用）

```bash
python examples\full_example_with_db.py
```

### 5.3 チャンキング手法の比較

```bash
python examples\compare_chunking.py
```

## ステップ6: メイン実験ランナーの使用

### 6.1 評価クエリファイルの作成

```bash
# queries.txtを作成
echo この文書の主題は何ですか？ > queries.txt
echo 重要なポイントを教えてください。 >> queries.txt
echo 著者の結論は何ですか？ >> queries.txt
```

### 6.2 単一パターンで実験実行

```bash
python main.py --data data\documents --pattern baseline_001 --queries queries.txt
```

### 6.3 全パターンで実験実行

```bash
python main.py --data data\documents --queries queries.txt
```

## ステップ7: 結果の確認

実験結果は`results/`ディレクトリに保存されます。

```bash
# ログを確認
type results\logs\experiment.log

# メトリクスを確認
dir results\metrics\

# 比較レポートを確認（全パターン実行時）
type results\comparison_report.json
```

## トラブルシューティング

### Ollamaに接続できない

```bash
# Ollamaサーバーが起動しているか確認
ollama serve

# 別のターミナルで実行
ollama list
```

### データベースに接続できない

```bash
# Dockerコンテナの状態を確認
docker-compose ps

# 再起動
docker-compose restart

# ログを確認
docker-compose logs milvus-standalone
docker-compose logs mongodb
docker-compose logs redis
```

### メモリ不足エラー

設定ファイルでバッチサイズやチャンクサイズを小さくします。

```yaml
# config/embedding_configs.yaml
ollama_embedding:
  batch_size: 16  # 32から16に削減
```

```yaml
# config/chunking_configs.yaml
token_based:
  chunk_size: 256  # 512から256に削減
```

### パッケージのインポートエラー

```bash
# 仮想環境が有効化されているか確認
where python
# 仮想環境のパスが表示されるはず

# パッケージを再インストール
pip install --upgrade -r requirements.txt
```

## 次のステップ

1. **設定のカスタマイズ**: `config/`ディレクトリの設定ファイルを編集
2. **独自のテストパターン追加**: `config/test_patterns.yaml`に新しいパターンを追加
3. **評価指標の追加**: `src/evaluation/evaluator.py`をカスタマイズ
4. **大規模データでの実験**: より多くのPDFで実験を実行

## よくある質問

**Q: GPUは必要ですか？**
A: 必須ではありませんが、大規模モデルを使用する場合は推奨されます。CPU版のモデルも使用できます。

**Q: クラウドLLMを使用できますか？**
A: はい。OpenAI APIなどを使用する場合は、設定ファイルとコードを適宜修正してください。

**Q: 日本語以外の言語に対応していますか？**
A: はい。llama_indexは多言語対応しており、使用するLLMと埋め込みモデル次第で様々な言語に対応できます。

**Q: どのくらいの時間がかかりますか？**
A: データ量とモデルによりますが、小規模データ（数MB）であれば数分、大規模データ（数GB）の場合は数時間かかることがあります。

## サポート

問題が発生した場合は、以下を確認してください：

1. `results/logs/experiment.log`のログファイル
2. Dockerコンテナのログ（`docker-compose logs`）
3. Pythonのバージョン（`python --version`）
4. インストールされたパッケージ（`pip list`）

それでも解決しない場合は、GitHubのIssueを作成してください。
