# RAG評価フレームワーク

LlamaIndexを使用したRAG（Retrieval-Augmented Generation）システムの包括的な評価フレームワークです。論文PDF対応のチャンキング、複数の埋め込みモデル、LLMモデルの組み合わせによるテストパターンの評価結果を比較し、RAGの最適手法を見つけることを目的としています。

## 🚀 特徴

- **論文PDF特化**: 表紙、目次、見出しを考慮した階層的チャンキング
- **多様な埋め込みモデル**: OpenAI、HuggingFace、Ollama、日本語特化モデル対応
- **複数のベクターストア**: Chroma、Milvus、Qdrant対応
- **RAGAS評価**: 忠実性、関連性、精度、再現率の自動評価
- **実験管理**: パターン比較、統計分析、結果可視化

## 📁 プロジェクト構造

```
system8/
├── README.md
├── requirements.txt
├── main.py                 # メインアプリケーション
├── config/                 # 設定ファイル
│   ├── chunking_configs.yaml
│   ├── embedding_configs.yaml
│   ├── evaluation_configs.yaml
│   ├── test_patterns.yaml
│   └── domain_configs.yaml
├── src/                    # ソースコード
│   ├── chunking/          # チャンキング戦略
│   ├── embedding/         # 埋め込みモデル
│   ├── indexing/          # インデキシング
│   ├── retrieval/         # 検索機能
│   ├── evaluation/        # 評価指標
│   ├── responsesynthesizer/ # レスポンス生成
│   ├── data_generation/   # データ生成
│   ├── monitoring/        # ログ・監視
│   └── utils/             # ユーティリティ
├── tests/                 # テストファイル
├── data/                  # データディレクトリ
└── results/               # 実験結果
```

## 🛠️ セットアップ

### 1. 依存関係のインストール

```bash
pip install -r requirements.txt
```

### 2. 環境変数の設定（オプション）

```bash
# OpenAI APIを使用する場合
export OPENAI_API_KEY="your-api-key"

# Ollamaを使用する場合（ローカルで実行）
ollama serve
ollama pull llama2
ollama pull nomic-embed-text
```

### 3. 設定の確認

設定ファイルは `config/` ディレクトリにあります。必要に応じて修正してください。

## 🔧 使用方法

### 基本的な使用例

```bash
# 単一パターンの実行
python main.py --document data/sample_paper.pdf --pattern pattern_1

# 複数パターンの比較
python main.py --document data/sample_paper.pdf --patterns pattern_1 pattern_2 pattern_3

# 出力ファイルの指定
python main.py --document data/sample_paper.pdf --output results/my_experiment.json
```

### 利用可能なテストパターン

#### 基本パターン
- `pattern_1`: OpenAI Ada + GPT-3.5
- `pattern_2`: Sentence Transformers + Ollama
- `pattern_3`: 日本語最適化パターン

#### 論文特化パターン  
- `academic_basic`: 論文構造対応基本版
- `academic_advanced`: 論文構造対応高度版
- `multilingual_academic`: 多言語論文対応

#### 高度なパターン
- `ensemble_1`: 埋め込みアンサンブル
- `multi_stage`: 多段階検索
- `graph_enhanced`: グラフ強化RAG

## 📊 評価指標

### RAGAS メトリクス
- **Faithfulness**: 生成回答のソース文書への忠実性
- **Answer Relevancy**: 質問に対する回答の関連性  
- **Context Precision**: 検索コンテキストの精度
- **Context Recall**: 検索コンテキストの再現率

### 追加メトリクス
- **ROUGE スコア**: 要約品質評価
- **BERTScore**: セマンティック類似度
- **検索評価**: Precision@K, Recall@K, MAP, MRR

## 🎯 論文PDF対応機能

### 階層的チャンキング
- セクション検出（Abstract, Introduction, Methodology, etc.）
- 重要度スコアによる重み付け
- 参考文献の個別処理

### 文書構造認識
- 見出しレベルの検出
- 表紙・目次の分離
- 図表キャプションの処理

## 🔬 実験例

### 1. チャンキング戦略の比較

```python
# 異なるチャンキング戦略を比較
patterns = ["fixed_size_pattern", "semantic_pattern", "hierarchical_pattern"]
python main.py --document paper.pdf --patterns {" ".join(patterns)}
```

### 2. 埋め込みモデルの比較

```python
# 異なる埋め込みモデルを比較  
patterns = ["openai_pattern", "huggingface_pattern", "japanese_pattern"]
python main.py --document paper.pdf --patterns {" ".join(patterns)}
```

### 3. カスタムパターンの追加

`config/test_patterns.yaml` に新しいパターンを追加:

```yaml
custom_patterns:
  my_pattern:
    name: "My Custom Pattern"
    chunking: "hierarchical"
    embedding: "huggingface/intfloat/multilingual-e5-large"  
    llm: "ollama/llama2"
    retrieval: "semantic_search"
```

## 📈 結果の分析

実験結果は JSON 形式で保存され、以下の情報が含まれます:

```json
{
  "experiment_type": "comparison",
  "individual_results": {
    "pattern_1": {
      "evaluation_results": {
        "faithfulness": 0.85,
        "answer_relevancy": 0.78,
        "context_precision": 0.82
      }
    }
  },
  "comparison_analysis": {
    "best_patterns": {
      "faithfulness": {"pattern": "pattern_1", "score": 0.85}
    }
  }
}
```

## 🧪 テスト

```bash
# 単体テストの実行
python -m pytest tests/

# 特定のテストの実行
python -m pytest tests/test_rag_framework.py::TestChunking
```

## 🛠️ カスタマイズ

### 新しいチャンキング戦略の追加

1. `src/chunking/` に新しいチャンカークラスを作成
2. `chunker_factory.py` に登録
3. `config/chunking_configs.yaml` に設定を追加

### 新しい埋め込みモデルの追加

1. `src/embedding/` に新しい埋め込みクラスを作成  
2. `embedding_factory.py` に登録
3. `config/embedding_configs.yaml` に設定を追加

## 🚧 制限事項

- Ollamaモデルは事前にダウンロードが必要
- 一部の評価指標は英語コンテンツで最適化
- 大きなPDFファイルは処理時間がかかる場合がある

## 📝 ライセンス

MIT License

## 🤝 貢献

プルリクエストやイシューは歓迎です。大きな変更を行う前に、まずイシューを作成して議論してください。

## 📞 サポート

問題が発生した場合は、以下を確認してください:

1. 依存関係が正しくインストールされているか
2. 必要な環境変数が設定されているか  
3. Ollamaサービスが起動しているか（使用する場合）
4. ログファイル（`logs/rag_evaluation.log`）でエラーの詳細を確認