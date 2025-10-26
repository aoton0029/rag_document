# RAG評価フレームワーク

llama_indexを用いたRAGシステムの包括的な評価フレームワーク。チャンキングから評価までを一つの試行として、複数の手法の組み合わせによる実験パターンを作成・評価します。

## 概要

このフレームワークは、以下の機能を提供します：

- **自動化された実験フレームワーク**: 複数手法を系統的に比較し、最適構成を特定
- **再現性のある評価プロセス**: 結果を定量的に保存・可視化
- **論文PDF・製品マニュアルPDF対応**: 実運用に近い条件での精度評価
- **柔軟な設定管理**: YAMLベースの設定ファイルで全てのパラメータを制御

## 主な機能

### データ処理パイプライン
- PDF読み取り（pymupdf4llm、llama_index readers）
- メタデータ抽出（タイトル、キーワード、要約など）
- 複数のチャンキング手法（固定長トークン、セマンティック、見出しベースなど）
- トークナイズとトークン長制御

### RAGシステム構築
- 複数の埋め込みモデル対応（Ollama、HuggingFace、LangChain）
- 多様なベクトルストア（Milvus、Faiss）
- ドキュメントストア（MongoDB）
- インデックスストア（Redis）
- グラフストア（Neo4j）
- 様々なインデックスタイプ（VectorStore、Summary、Tree、KnowledgeGraphなど）

### 検索と生成
- 複数のRetriever戦略（Vector、Keyword、Hybrid、Rerankerなど）
- 柔軟なQueryEngine（Retriever、Router、MultiStepなど）
- ResponseSynthesizer（Refine、Compact、TreeSummarizeなど）

### 評価と監視
- 自動評価メトリクス（ROUGE、BLEU、BERTScore、F1など）
- llama_index評価（Faithfulness、Relevancy、Correctness）
- 検索指標（Recall@k、MRR、Precision@k）
- 実験ログとメトリクスの時系列保存
- 結果の可視化とレポート生成

## ディレクトリ構造

```
rag-evaluation-framework/
├── README.md                  # このファイル
├── requirements.txt           # Python依存パッケージ
├── main.py                   # メイン実験ランナー
├── config/                   # 設定ファイル
│   ├── chunking_configs.yaml
│   ├── embedding_configs.yaml
│   ├── llm_configs.yaml
│   ├── tokenizer_configs.yaml
│   ├── evaluation_configs.yaml
│   ├── domain_configs.yaml
│   └── test_patterns.yaml
├── src/                      # ソースコード
│   ├── database/            # データベースクライアント
│   │   ├── database_manager.py
│   │   ├── mongodb_client.py
│   │   ├── redis_client.py
│   │   ├── neo4j_client.py
│   │   └── milvus_client.py
│   ├── data_generation/     # データ取り込み
│   │   ├── document_loader.py
│   │   └── metadata_extractor.py
│   ├── chunking/            # チャンキング
│   │   └── chunking.py
│   ├── embedding/           # 埋め込み
│   │   └── embedding.py
│   ├── indexing/            # インデックス構築
│   │   └── index_builder.py
│   ├── retrieval/           # 検索
│   │   └── retriever.py
│   ├── query/               # クエリエンジン
│   │   └── query_engine.py
│   ├── responsesynthesizer/ # レスポンス合成
│   │   └── response_synthesizer.py
│   ├── evaluation/          # 評価
│   │   └── evaluator.py
│   ├── monitoring/          # 監視・ログ
│   │   └── monitoring.py
│   └── utils/               # ユーティリティ
│       ├── config_manager.py
│       └── token_utils.py
├── tests/                    # テストコード
├── data/                     # データファイル
└── results/                  # 実験結果
    ├── logs/                # 実験ログ
    ├── metrics/             # メトリクス
    └── configs/             # 設定スナップショット
```

## セットアップ

### 必要要件

- Python 3.9以上
- Docker（データベース用、オプション）
- Ollama（ローカルLLM/Embedding用、オプション）

### インストール

1. リポジトリをクローン
```bash
git clone <repository-url>
cd rag-evaluation-framework
```

2. 仮想環境を作成・有効化
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. 依存パッケージをインストール
```bash
pip install -r requirements.txt
```

4. データベースをセットアップ（Docker使用の場合）
```bash
# docker-compose.ymlを作成してから
docker-compose up -d
```

### データベース設定

以下のデータベースが必要です（オプション、StorageContextで選択可能）：

- **Milvus**: ベクトルストア（デフォルト: localhost:19530）
- **MongoDB**: ドキュメントストア（デフォルト: localhost:27017）
- **Redis**: インデックスストア（デフォルト: localhost:6379）
- **Neo4j**: グラフストア（デフォルト: bolt://localhost:7687）

設定は`config/test_patterns.yaml`の`storage`セクションで変更できます。

## 使い方

### 基本的な使用方法

1. **データを準備**
```bash
# PDFファイルをdata/ディレクトリに配置
mkdir -p data/documents
cp your_documents.pdf data/documents/
```

2. **評価クエリを準備**（オプション）
```bash
# queries.txtファイルに1行1クエリで記述
echo "この文書の主要なポイントは？" > queries.txt
echo "著者の結論は何ですか？" >> queries.txt
```

3. **実験を実行**

単一パターンで実行:
```bash
python main.py --data data/documents --pattern baseline_001 --queries queries.txt
```

全パターンで実行:
```bash
python main.py --data data/documents --queries queries.txt
```

### コマンドライン引数

- `--data`: データファイル/ディレクトリパス（必須）
- `--pattern`: 実行するテストパターン名（オプション、省略時は全パターン実行）
- `--queries`: 評価クエリファイルパス（オプション）
- `--config-dir`: 設定ファイルディレクトリ（デフォルト: config）
- `--results-dir`: 結果出力ディレクトリ（デフォルト: results）

### プログラムからの使用

```python
from main import ExperimentRunner

# 実験ランナー作成
runner = ExperimentRunner(
    config_dir="config",
    results_dir="results"
)

# 単一実験実行
result = runner.run_experiment(
    pattern_name="baseline_001",
    data_path="data/documents",
    queries=["質問1", "質問2"]
)

print(f"実験ID: {result['run_id']}")
print(f"結果: {result['results']}")
```

## 設定ファイル

### test_patterns.yaml

実験パターンを定義します。各パターンは以下の要素を含みます：

```yaml
test_patterns:
  my_experiment:
    name: "実験名"
    description: "実験の説明"
    enabled: true
    config:
      chunking:
        method: "token_based"
        chunk_size: 512
        chunk_overlap: 50
      embedding:
        backend: "ollama"
        model: "qwen3-embedding:8b"
      indexing:
        type: "vector"
        vector_store: "milvus"
      retrieval:
        retriever_type: "vector"
        similarity_top_k: 10
      llm:
        model: "qwen3:32b"
        backend: "ollama"
        temperature: 0.0
      query:
        response_mode: "compact"
      storage:
        use_vector_store: true
        use_docstore: true
        use_index_store: true
```

### chunking_configs.yaml

チャンキング手法を定義します：

- `token_based`: 固定長トークンチャンキング
- `semantic_splitter`: セマンティックセグメント
- `sentence_splitter`: 文単位分割
- `markdown_header`: Markdownヘッダーベース
- など

### embedding_configs.yaml

埋め込みモデルを定義します：

- Ollamaモデル（qwen3-embedding:8b など）
- HuggingFaceモデル
- LangChainモデル

### llm_configs.yaml

LLMモデルを定義します：

- Ollamaモデル（qwen3:32b など）
- vLLMモデル
- OpenAI互換モデル

## 実験結果

実験結果は`results/`ディレクトリに保存されます：

### ログファイル
- `results/logs/`: 実験ログ（JSON形式）
- 各実験の設定、メトリクス、エラー情報

### メトリクス
- `results/metrics/`: 評価メトリクス（CSV、JSON）
- ROUGE、BERTScore、Faithfulness、Relevancyなど

### 設定スナップショット
- `results/configs/`: 実験時の設定を完全保存
- 再現性を保証

### 比較レポート
- `results/comparison_report.json`: 全実験の比較結果

## 評価指標

### 自動メトリクス
- **ROUGE**: テキスト重複度
- **BLEU**: 機械翻訳品質
- **BERTScore**: 意味的類似度
- **Exact Match (EM)**: 完全一致率
- **F1**: 適合率と再現率の調和平均

### llama_index評価
- **Faithfulness**: 生成の忠実性
- **Relevancy**: 検索の関連性
- **Correctness**: 回答の正確性

### 検索指標
- **Recall@k**: 上位k件の再現率
- **MRR**: 平均逆順位
- **Precision@k**: 上位k件の適合率

## 拡張方法

### 新しいチャンキング手法を追加

1. `src/chunking/chunking.py`に新しいクラスを追加
2. `config/chunking_configs.yaml`に設定を追加
3. `ChunkerFactory`に登録

### 新しい埋め込みモデルを追加

1. `src/embedding/embedding.py`に新しいアダプターを追加
2. `config/embedding_configs.yaml`に設定を追加
3. `EmbeddingFactory`に登録

### 新しい評価指標を追加

1. `src/evaluation/evaluator.py`にメソッドを追加
2. `config/evaluation_configs.yaml`に設定を追加

## トラブルシューティング

### データベース接続エラー

```
接続エラー: Could not connect to Milvus
```

**解決方法**: データベースが起動していることを確認
```bash
docker ps  # コンテナが起動しているか確認
```

### メモリ不足エラー

```
RuntimeError: CUDA out of memory
```

**解決方法**: バッチサイズやチャンクサイズを小さくする
```yaml
embedding:
  batch_size: 16  # デフォルト32から削減
```

### LLM接続エラー

```
ConnectionError: Could not connect to Ollama
```

**解決方法**: Ollamaが起動していることを確認
```bash
ollama serve  # Ollamaサーバーを起動
ollama list   # 利用可能なモデルを確認
```

## ベストプラクティス

1. **小規模データで検証**: 最初は少量のデータで設定をテスト
2. **設定のバージョン管理**: 設定ファイルをGitで管理
3. **実験の命名規則**: わかりやすいパターン名を使用
4. **結果の定期バックアップ**: `results/`ディレクトリをバックアップ
5. **ログの確認**: エラー時は`results/logs/experiment.log`を確認

## ライセンス

MIT License

## 貢献

プルリクエストを歓迎します。大きな変更の場合は、まずissueを開いて変更内容を議論してください。

## 引用

このフレームワークを研究で使用する場合は、以下のように引用してください：

```bibtex
@software{rag_evaluation_framework,
  title = {RAG Evaluation Framework},
  year = {2025},
  author = {Your Name},
  url = {https://github.com/your-repo}
}
```

## 関連リンク

- [llama_index Documentation](https://docs.llamaindex.ai/)
- [Milvus Documentation](https://milvus.io/docs)
- [MongoDB Documentation](https://docs.mongodb.com/)
- [Redis Documentation](https://redis.io/documentation)
- [Neo4j Documentation](https://neo4j.com/docs/)

## サポート

問題が発生した場合は、以下の方法でサポートを受けられます：

- GitHubのIssueを作成
- ディスカッションフォーラムに投稿
- メール: support@example.com
