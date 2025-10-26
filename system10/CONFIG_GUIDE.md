# 設定ガイド

このドキュメントでは、RAG評価フレームワークの設定ファイルの詳細を説明します。

## 設定ファイルの概要

全ての設定ファイルは`config/`ディレクトリに配置されています。

```
config/
├── chunking_configs.yaml      # チャンキング手法の定義
├── embedding_configs.yaml     # 埋め込みモデルの定義
├── llm_configs.yaml          # LLMモデルの定義
├── tokenizer_configs.yaml    # トークナイザーの定義
├── evaluation_configs.yaml   # 評価設定
├── domain_configs.yaml       # ドメイン固有設定
└── test_patterns.yaml        # 実験パターンの定義
```

## test_patterns.yaml - 実験パターン定義

最も重要な設定ファイルです。実験の全体的な構成を定義します。

### 基本構造

```yaml
test_patterns:
  pattern_name:
    name: "表示名"
    description: "説明"
    enabled: true  # この実験を有効にするか
    config:
      # ここに各モジュールの設定を記述
```

### 完全な例

```yaml
test_patterns:
  my_experiment:
    name: "カスタム実験"
    description: "説明文"
    enabled: true
    config:
      # メタデータ抽出設定
      metadata:
        enabled: true
        extract_title: true
        extract_keywords: true
        extract_summary: false
        use_llm: false
      
      # チャンキング設定
      chunking:
        method: "token_based"  # chunking_configs.yamlで定義された手法名
        chunk_size: 512
        chunk_overlap: 50
      
      # 埋め込み設定
      embedding:
        backend: "ollama"  # ollama, huggingface, langchain
        model: "qwen3-embedding:8b"
        dimension: 1024
        batch_size: 32
      
      # インデックス設定
      indexing:
        type: "vector"  # vector, summary, tree, keyword, knowledge_graph
        vector_store: "milvus"
      
      # ストレージ設定
      storage:
        use_vector_store: true
        use_docstore: true
        use_index_store: true
        use_graph_store: false
        mongodb:
          host: "localhost"
          port: 27017
          database: "rag_system"
        redis:
          host: "localhost"
          port: 6379
        milvus:
          host: "localhost"
          port: 19530
          collection: "my_vectors"
        neo4j:
          uri: "bolt://localhost:7687"
      
      # 検索設定
      retrieval:
        retriever_type: "vector"  # vector, keyword, hybrid
        similarity_top_k: 10
        similarity_cutoff: 0.7
      
      # LLM設定
      llm:
        backend: "ollama"  # ollama, vllm
        model: "qwen3:32b"
        temperature: 0.0
        max_tokens: 1024
      
      # クエリエンジン設定
      query:
        type: "retriever"  # retriever, router, multi_step
        response_mode: "compact"  # compact, refine, tree_summarize
      
      # 評価設定
      evaluation:
        enabled: true
        metrics:
          - "faithfulness"
          - "relevancy"
          - "correctness"
```

## chunking_configs.yaml - チャンキング手法定義

### トークンベースチャンキング

```yaml
chunking_methods:
  token_based:
    type: "token_text_splitter"
    chunk_size: 512        # チャンクサイズ（トークン数）
    chunk_overlap: 50      # オーバーラップ（トークン数）
    separator: "\n\n"      # 主要セパレータ
    backup_separators: ["\n", ".", " "]  # バックアップセパレータ
```

### センテンスベースチャンキング

```yaml
  sentence_splitter:
    type: "sentence_splitter"
    chunk_size: 512
    chunk_overlap: 50
    paragraph_separator: "\n\n"
    secondary_chunking_regex: "[^,.;。！？]+[,.;。！？]?"
```

### セマンティックチャンキング

```yaml
  semantic_splitter:
    type: "semantic_splitter"
    buffer_size: 1           # バッファサイズ
    breakpoint_percentile_threshold: 95  # 分割閾値（パーセンタイル）
    embed_model: "sentence-transformers/all-MiniLM-L6-v2"
```

### Markdownヘッダーベース

```yaml
  markdown_header:
    type: "markdown_header_splitter"
    headers_to_split_on:
      - ["#", "Header 1"]
      - ["##", "Header 2"]
      - ["###", "Header 3"]
    strip_headers: false
```

## embedding_configs.yaml - 埋め込みモデル定義

### Ollama埋め込み

```yaml
embeddings:
  ollama_embedding:
    backend: "ollama"
    model_name: "qwen3-embedding:8b"
    base_url: "http://localhost:11434"
    request_timeout: 120.0
    dimension: 1024
```

### HuggingFace埋め込み

```yaml
  huggingface_embedding:
    backend: "huggingface"
    model_name: "sentence-transformers/all-MiniLM-L6-v2"
    device: "cpu"  # cpu, cuda
    max_length: 512
    normalize_embeddings: true
```

### OpenAI埋め込み

```yaml
  openai_embedding:
    backend: "openai"
    model_name: "text-embedding-3-small"
    api_key: "${OPENAI_API_KEY}"  # 環境変数から取得
    dimension: 1536
```

## llm_configs.yaml - LLMモデル定義

### Ollama LLM

```yaml
llms:
  ollama_qwen3_32b:
    backend: "ollama"
    model: "qwen3:32b"
    temperature: 0.0
    max_tokens: 1024
    base_url: "http://localhost:11434"
    request_timeout: 120.0
```

### vLLM

```yaml
  vllm_qwen3:
    backend: "vllm"
    model: "Qwen/Qwen3-32B"
    temperature: 0.0
    max_tokens: 1024
    tensor_parallel_size: 2  # GPU数
```

### OpenAI互換

```yaml
  openai_gpt4:
    backend: "openai"
    model: "gpt-4-turbo-preview"
    temperature: 0.0
    max_tokens: 1024
    api_key: "${OPENAI_API_KEY}"
```

## evaluation_configs.yaml - 評価設定

```yaml
evaluation:
  # llama_index評価器設定
  llama_index_evaluators:
    faithfulness:
      enabled: true
      raise_error: false
    
    relevancy:
      enabled: true
      raise_error: false
    
    correctness:
      enabled: true
      raise_error: false
  
  # 外部メトリクス設定
  external_metrics:
    rouge:
      enabled: true
      rouge_types: ["rouge1", "rouge2", "rougeL"]
    
    bert_score:
      enabled: true
      model_type: "bert-base-multilingual-cased"
    
    bleu:
      enabled: false
  
  # 検索評価設定
  retrieval_metrics:
    recall_at_k:
      enabled: true
      k_values: [5, 10, 20]
    
    precision_at_k:
      enabled: true
      k_values: [5, 10, 20]
    
    mrr:
      enabled: true
```

## domain_configs.yaml - ドメイン固有設定

```yaml
domains:
  # 論文PDF用設定
  academic_paper:
    name: "学術論文"
    file_types: [".pdf"]
    metadata_fields:
      - "title"
      - "authors"
      - "abstract"
      - "keywords"
      - "publication_date"
    
    chunking_preference: "markdown_header"
    chunk_size: 1024
    
    extraction:
      extract_citations: true
      extract_equations: true
      extract_figures: true
  
  # 製品マニュアル用設定
  product_manual:
    name: "製品マニュアル"
    file_types: [".pdf", ".docx"]
    metadata_fields:
      - "product_name"
      - "version"
      - "date"
    
    chunking_preference: "markdown_header"
    chunk_size: 512
    
    extraction:
      extract_tables: true
      extract_diagrams: true
```

## 環境変数の使用

設定ファイルで環境変数を参照できます：

```yaml
llm:
  api_key: "${OPENAI_API_KEY}"
  organization: "${OPENAI_ORG_ID}"
```

`.env`ファイルに定義：

```bash
OPENAI_API_KEY=sk-xxx
OPENAI_ORG_ID=org-xxx
```

## 設定のバリデーション

設定の妥当性をチェック：

```python
from src.utils.config_manager import ConfigManager

config_manager = ConfigManager()

# 設定の検証
is_valid = config_manager.validate_config(config_dict)

# エラーメッセージ取得
errors = config_manager.get_validation_errors()
```

## 設定のベストプラクティス

1. **小さく始める**: まずデフォルト設定で試す
2. **段階的に変更**: 一度に1つのパラメータを変更
3. **命名規則**: わかりやすいパターン名を使用
4. **コメント追加**: 設定の意図を記述
5. **バージョン管理**: 設定ファイルをGitで管理

## トラブルシューティング

### 設定の読み込みエラー

```
YAMLError: Invalid syntax
```

- YAMLの構文を確認（インデント、コロン、ハイフンなど）
- オンラインYAMLバリデーターでチェック

### モデルが見つからない

```
ModelNotFoundError: Model 'xxx' not found
```

- `ollama list`でモデルを確認
- `ollama pull model_name`でダウンロード

### メモリ不足

```
CUDA out of memory
```

- `chunk_size`を小さくする（1024 → 512）
- `batch_size`を小さくする（32 → 16）
- `similarity_top_k`を小さくする（10 → 5）

## 参考資料

- [llama_index設定ドキュメント](https://docs.llamaindex.ai/en/stable/module_guides/supporting_modules/settings.html)
- [YAML構文リファレンス](https://yaml.org/spec/1.2/spec.html)
