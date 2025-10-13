# Advanced RAG System

llama_index、Ollama、Redis、MongoDB、Milvus、Neo4jを使用した高度なRAG（Retrieval-Augmented Generation）システムです。

## 特徴

- **マルチストレージ対応**: 複数のデータベースを統合したStorageContext
  - **Milvus**: ベクトルストア（vector_store）
  - **Redis**: インデックスストア（index_store） 
  - **MongoDB**: ドキュメントストア（docstore）
  - **Neo4j**: グラフストア（graph_store）

- **Ollama統合**: LLMと埋め込みモデルの両方でOllamaを使用
  - 完全にローカルで動作
  - 様々なモデルをサポート

- **Advanced RAG機能**:
  - ハイブリッド検索（ベクトル + グラフ）
  - セマンティックチャンク分割
  - 再ランキング機能
  - ソース引用機能

## システム要件

### 必須サービス

1. **Ollama** (http://localhost:11434)
2. **Milvus** (http://localhost:19530)
3. **Redis** (localhost:6379)
4. **MongoDB** (localhost:27017)
5. **Neo4j** (bolt://localhost:7687)

### 推奨モデル

- LLM: `llama3.1:8b` または `llama3.2`
- Embedding: `nomic-embed-text` または `mxbai-embed-large`

## インストール

### 1. 依存関係のインストール

```bash
pip install -r requirements.txt
```

### 2. 環境設定

```bash
# .envファイルを作成
cp .env.example .env

# 必要に応じて設定を編集
notepad .env
```

### 3. 外部サービスの起動

#### Ollama
```bash
# Ollamaをインストール（https://ollama.com/）
ollama serve

# 必要なモデルをダウンロード
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

#### Milvus (Dockerを使用)
```bash
# Milvus Standalone
curl -L https://github.com/milvus-io/milvus/releases/download/v2.3.0/milvus-standalone-docker-compose.yml -o docker-compose.yml
docker-compose up -d
```

#### Redis
```bash
# Dockerを使用
docker run -d --name redis-server -p 6379:6379 redis:latest
```

#### MongoDB
```bash
# Dockerを使用
docker run -d --name mongodb -p 27017:27017 mongo:latest
```

#### Neo4j
```bash
# Dockerを使用
docker run -d --name neo4j \\
    -p 7474:7474 -p 7687:7687 \\
    -e NEO4J_AUTH=neo4j/password \\
    neo4j:latest
```

## 使用方法

### 基本的な使用方法

```python
from main import advanced_rag_system
import asyncio

async def main():
    # システムの初期化
    await advanced_rag_system.initialize_system()
    
    # ドキュメントの読み込み
    result = advanced_rag_system.load_documents([
        "path/to/your/documents",
        "path/to/specific/file.pdf"
    ])
    
    # 質問応答
    response = advanced_rag_system.query("あなたの質問をここに入力")
    print(response['answer'])

asyncio.run(main())
```

### コマンドライン使用方法

#### インタラクティブセッション
```bash
python main.py --interactive
```

#### ドキュメント読み込み + 質問
```bash
python main.py --documents "data/docs" --query "AIについて教えて"
```

#### エンジン指定
```bash
python main.py --documents "data/docs" --query "質問" --engine hybrid
```

#### セマンティックチャンク使用
```bash
python main.py --documents "data/docs" --semantic-chunking
```

### デモの実行

```bash
# 完全なデモ
python examples/demo.py

# クイック接続テスト
python examples/demo.py --quick-test
```

## 設定

### config/config.py

システムの設定は`config/config.py`で管理されています：

```python
# 例: Ollamaの設定変更
config.ollama.llm_model = "llama3.2"
config.ollama.embed_model = "mxbai-embed-large"

# 例: RAGの設定変更
config.rag.chunk_size = 1024
config.rag.similarity_top_k = 10
```

### 環境変数での設定

`.env`ファイルで設定を上書きできます：

```
OLLAMA_LLM_MODEL=llama3.2
MILVUS_HOST=192.168.1.100
REDIS_PASSWORD=your_password
```

## API仕様

### AdvancedRAGSystem

#### `initialize_system(check_connections=True)`
システムを初期化します。

#### `load_documents(sources, use_semantic_chunking=False)`
ドキュメントを読み込んでインデックス化します。

#### `query(question, engine_type="hybrid", include_sources=True)`
質問応答を実行します。

#### `start_interactive_session()`
対話型セッションを開始します。

### 利用可能なエンジン

- `vector`: ベクトル検索のみ
- `graph`: グラフ検索のみ
- `hybrid`: ベクトル + グラフのハイブリッド検索

## ディレクトリ構造

```
system7/
├── config/
│   ├── __init__.py
│   └── config.py           # 設定ファイル
├── src/
│   ├── __init__.py
│   ├── storage_setup.py    # ストレージ設定
│   ├── model_setup.py      # モデル設定
│   ├── rag_engine.py       # RAGエンジン
│   └── document_processor.py # ドキュメント処理
├── examples/
│   └── demo.py             # デモスクリプト
├── data/                   # ドキュメントデータ
├── logs/                   # ログファイル
├── main.py                 # メインモジュール
├── requirements.txt        # 依存関係
├── .env.example           # 環境変数テンプレート
└── README.md              # このファイル
```

## 対応ファイル形式

- PDF (`.pdf`)
- Word (`.docx`, `.doc`)
- テキスト (`.txt`)
- Markdown (`.md`)

## トラブルシューティング

### 接続エラー

1. **Ollama接続エラー**
   ```bash
   # Ollamaサービスの状態確認
   curl http://localhost:11434/api/tags
   ```

2. **Milvus接続エラー**
   ```bash
   # Milvusの状態確認
   docker ps | grep milvus
   ```

3. **Redis/MongoDB/Neo4j接続エラー**
   ```bash
   # 各サービスの状態確認
   docker ps
   ```

### モデル関連エラー

1. **モデルが見つからない**
   ```bash
   # 利用可能なモデルの確認
   ollama list
   
   # モデルのダウンロード
   ollama pull model_name
   ```

2. **埋め込み次元エラー**
   - `config.py`でMilvusの次元設定を確認
   - 埋め込みモデルの出力次元と一致させる

### パフォーマンス最適化

1. **チャンクサイズの調整**
   ```python
   config.rag.chunk_size = 512  # デフォルト
   config.rag.chunk_overlap = 50
   ```

2. **検索結果数の調整**
   ```python
   config.rag.similarity_top_k = 5
   ```

3. **セマンティックチャンクの使用**
   ```bash
   python main.py --semantic-chunking
   ```

## ライセンス

MIT License

## 貢献

プルリクエストやイシューの報告は歓迎します。

## 参考資料

- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [Ollama Documentation](https://github.com/ollama/ollama)
- [Milvus Documentation](https://milvus.io/docs)
- [Redis Documentation](https://redis.io/docs)
- [MongoDB Documentation](https://docs.mongodb.com/)
- [Neo4j Documentation](https://neo4j.com/docs/)