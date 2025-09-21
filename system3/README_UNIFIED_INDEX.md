以下は、llamaindex（現行API概念：Document / Node / Index / Retriever / ServiceContext 等）を使ったRAGシステムの詳細設計（ドキュメント保存フロー、RAG/類似検索フロー）です。ソースコードは含めない。運用・性能・安全面の考慮点も含む。

# 目的
- ドキュメントを取り込み、埋め込みを付与してベクトルDBに保存する。
- 保存済みデータを使って高速かつ根拠付きの回答（RAG）と類似検索を行う。

# 主要コンポーネント
- Ingest Service：ファイルアップロード／パーシング／前処理／チャンク化／埋め込み生成／インデックス登録を行う。
- **Milvus (Vector Store)**：ベクトル検索とANN検索を担う。コレクション単位でのスケーリングが可能。
- **MongoDB (Document DB)**：原文メタ（raw_documents コレクション）、チャンクデータ、バージョン管理、権限情報、生データを格納。
- **Neo4j (Graph DB)**：ドキュメント間の関係性、エンティティ間の関係、知識グラフを管理。
- **Redis (KV Store & Index Store)**：検索結果キャッシュ、セッション管理、高速インデックス、頻出クエリキャッシュを担う。
- llamaindex Index/ServiceContext：Retriever/QueryEngine を組み立て、LLM呼び出し（LLMPredictor）と結合する。
- LLM：回答生成と（必要なら）リランキングに使用。
- API 層：クエリ受付、RAGワークフロー実行、結果返却（本文＋出典）。
- Monitoring / Audit：検索ログ、コスト、品質指標を収集。

# 統合ID管理システム
## ID生成戦略
- **unified_id**: UUID4形式の統合識別子（全DB共通のプライマリキー）
- **global_sequence**: 分散環境での一意性を保証するスノーフレークID
- **correlation_id**: リクエスト単位でのトレーシング用ID

## ID生成フロー
1. ドキュメント受信時に unified_id を生成（例：`doc_uuid4_timestamp`）
2. チャンク分割時に unified_chunk_id を生成（例：`{doc_unified_id}_chunk_{index}`）
3. 各DB操作時に同一IDを使用してデータ整合性を確保

# データモデル（MongoDB コレクション設計）
## raw_documents コレクション
```json
{
  "_id": ObjectId,
  "unified_id": "uuid_string", // 全DB共通の統合ID
  "doc_id": "uuid_string", // 後方互換用（deprecated予定）
  "global_sequence": "number", // スノーフレークID
  "source_uri": "URL/パス",
  "title": "string",
  "author": "string", 
  "created_at": ISODate,
  "language": "ja|en|...",
  "original_format": "pdf|html|txt|docx",
  "checksum": "sha256_hash",
  "version": "string",
  "file_size": "number",
  "metadata": {
    "category": "string",
    "tags": ["tag1", "tag2"],
    "access_level": "public|private|restricted"
  },
  "processing_status": "pending|processing|completed|failed",
  "indexed_at": ISODate,
  "index_status": {
    "mongodb": "completed|failed",
    "milvus": "completed|failed", 
    "neo4j": "completed|failed",
    "redis": "completed|failed"
  },
  "raw_data": "生データ"
}
```

## chunks コレクション
```json
{
  "_id": ObjectId,
  "unified_chunk_id": "uuid_string", // 全DB共通のチャンク統合ID
  "chunk_id": "uuid_string", // 後方互換用（deprecated予定）
  "doc_unified_id": "uuid_string", // raw_documents への統合ID参照
  "doc_id": "uuid_string", // 後方互換用
  "text": "string",
  "chunk_index": "number",
  "offset_start": "number",
  "offset_end": "number", 
  "milvus_vector_id": "string", // Milvusでのベクトルレコード統合ID
  "neo4j_node_id": "string", // Neo4jでのノード統合ID
  "redis_cache_key": "string", // Redisでのキャッシュキー
  "metadata": {
    "section_title": "string",
    "page": "number",
    "language": "string",
    "confidence": "number",
    "channel": "string",
    "entities": ["entity1", "entity2"] // Neo4j連携用
  },
  "embedding_generated_at": ISODate,
  "created_at": ISODate,
  "index_status": {
    "mongodb": "completed|failed",
    "milvus": "completed|failed", 
    "neo4j": "completed|failed",
    "redis": "completed|failed"
  }
}
```

## distributed_index_registry コレクション（新規追加）
```json
{
  "_id": ObjectId,
  "unified_id": "uuid_string", // ドキュメントまたはチャンクの統合ID
  "entity_type": "document|chunk",
  "indexes": {
    "mongodb": {
      "collection": "string",
      "index_keys": ["unified_id", "created_at", "language"],
      "status": "creating|completed|failed",
      "created_at": ISODate
    },
    "milvus": {
      "collection": "string", 
      "index_type": "IVF_FLAT|HNSW",
      "vector_id": "string",
      "status": "creating|completed|failed",
      "created_at": ISODate
    },
    "neo4j": {
      "node_labels": ["Document", "Chunk"],
      "relationship_types": ["CONTAINS", "RELATES_TO"],
      "node_id": "string",
      "status": "creating|completed|failed", 
      "created_at": ISODate
    },
    "redis": {
      "key_pattern": "string",
      "index_name": "string", 
      "ttl": "number",
      "status": "creating|completed|failed",
      "created_at": ISODate
    }
  },
  "overall_status": "pending|creating|completed|failed",
  "created_at": ISODate,
  "updated_at": ISODate
}
```

# 使用するライブラリ
- llama-index
- llama-index-core
- llama-index-llms-ollama
- llama-index-embeddings-ollama
- **llama-index-vector-stores-milvus** # Milvus連携
- **llama-index-storage-docstore-mongodb** # MongoDB Document Store
- **llama-index-storage-kvstore-redis** # Redis KV Store  
- **llama-index-storage-index-store-redis** # Redis Index Store
- **llama-index-graph-stores-neo4j** # Neo4j Graph Store
- **llama-index-packs-neo4j-query-engine** # Neo4j Query Engine
- llama-index-readers-file
- llama-index-readers-json
- llama-index-readers-mongodb
- llama-index-extractors-entity
- llama-index-tools-database
- llama-index-tools-neo4j
- llama-index-experimental
- llama-index-llms-huggingface
- llama-index-embeddings-ollama
- llama-index-embeddings-langchain
- llama-index-readers-database
- llama-index-readers-file
- llama-index-readers-json
- llama-index-readers-milvus
- llama-index-readers-mongodb
- llama-index-readers-obsidian
- llama-index-readers-whisper
- llama-index-readers-graphdb-cypher
- llama-index-vector-stores-milvus
- llama-index-extractors-entity
- llama-index-storage-index-store-elasticsearch
- llama-index-storage-index-store-mongodb
- llama-index-storage-index-store-redis
- llama-index-storage-docstore-elasticsearch
- llama-index-storage-docstore-mongodb
- llama-index-storage-docstore-redis
- llama-index-storage-kvstore-elasticsearch
- llama-index-storage-kvstore-mongodb
- llama-index-storage-kvstore-redis
- llama-index-tools-requests
- llama-index-tools-mcp
- llama-index-tools-neo4j
- llama-index-tools-database
- llama-index-graph-stores-neo4j
- llama-index-packs-neo4j-query-engine
- llama-index-multi-modal-llms-ollama

# ドキュメント保存フロー（統合StorageContext + 分散インデックス）

## StorageContext統合設計
### 統合StorageContext構成
```python
# 概念設計（コードではない）
unified_storage_context = StorageContext(
    docstore=MongoDocumentStore(collection="unified_documents"),  # MongoDB統合
    vector_store=MilvusVectorStore(collection="unified_vectors"),  # Milvus統合
    graph_store=Neo4jGraphStore(database="unified_graph"),        # Neo4j統合  
    index_store=RedisIndexStore(namespace="unified_indexes"),     # Redis統合
    property_graph_store=Neo4jPropertyGraphStore()               # Neo4j拡張
)
```

### 統合インデックス種類
- **VectorStoreIndex**: Milvusベクトル検索 + MongoDB/Redisメタデータ
- **DocumentSummaryIndex**: MongoDB要約 + Neo4j関係性
- **KnowledgeGraphIndex**: Neo4j知識グラフ + 全DB連携
- **PropertyGraphIndex**: Neo4j詳細グラフ + 統合ID追跡
- **ComposableGraph**: 全インデックス統合クエリエンジン

## 修正フロー設計

1. 統合ID生成・受け取り
   - 統合ID生成（unified_id, global_sequence, correlation_id）
   - ソース受領（アップロード、URL、DB取り込み）
   - **MongoDB**: raw_documents に統合IDで初期レコード作成（processing_status: "pending"）
   - **Redis**: 処理状況を統合IDでキャッシュ
   - distributed_index_registry に統合IDエントリ作成
   - **StorageContext初期化**: 統合IDベースで全DB接続確立

2. 前処理（統合ID継承）
   - OCR（必要時）、エンコーディング正規化
   - 言語検出、不要ノイズ除去（ヘッダ/フッタ/ナビ）
   - **MongoDB**: 統合IDで processing_status を "processing" に更新
   - PII検出/マスキング（ポリシーに基づく）
   - **Redis**: 前処理状況を統合IDでキャッシュ

3. 分割（チャンク統合ID生成）
   - unified_chunk_id 生成（{doc_unified_id}_chunk_{index}形式）
   - トークンベース／文ベースの分割（例：chunk_size=800 tokens, overlap=50）
   - セマンティック境界を保つ工夫（見出しで分割）
   - **Document/Node生成**: llamaindex Document/Nodeオブジェクトに統合IDを埋め込み
   - **Neo4j**: 統合IDでエンティティ抽出してノード・関係性を構築
   - distributed_index_registry にチャンク統合IDエントリ追加

4. 埋め込み生成（統合ID追跡）
   - 一括バッチ化で埋め込みを生成（APIコール回数最小化）
   - **Redis**: 統合IDで埋め込みキャッシュ（同一チャンクの再計算回避）
   - 埋め込みモデルはユースケースに応じて選択（OpenAI/hf/ベクトル化モデル）
   - **Document/Node更新**: 埋め込みベクトルを統合IDで関連付け

5. **統合StorageContext保存・インデックス作成**
   ```python
   # 概念フロー（コードではない）
   # 5-1. 全DBデータ保存（統合IDベース）
   unified_storage_context.docstore.add_documents(documents)     # MongoDB保存
   unified_storage_context.vector_store.add(nodes_with_vectors) # Milvus保存  
   unified_storage_context.graph_store.upsert_triplets(triplets) # Neo4j保存
   
   # 5-2. 統合インデックス並列作成
   vector_index = VectorStoreIndex(nodes, storage_context=unified_storage_context)
   summary_index = DocumentSummaryIndex(documents, storage_context=unified_storage_context)  
   graph_index = KnowledgeGraphIndex(documents, storage_context=unified_storage_context)
   property_graph_index = PropertyGraphIndex(documents, storage_context=unified_storage_context)
   
   # 5-3. 統合インデックス永続化
   vector_index.storage_context.persist(persist_dir=f"./storage/{unified_id}")
   summary_index.storage_context.persist(persist_dir=f"./storage/{unified_id}")
   graph_index.storage_context.persist(persist_dir=f"./storage/{unified_id}")
   property_graph_index.storage_context.persist(persist_dir=f"./storage/{unified_id}")
   ```

6. **統合インデックス状態管理・検証**
   - distributed_index_registry にStorageContext統合インデックス状態を記録
   ```json
   {
     "unified_id": "doc_uuid_string",
     "storage_context": {
       "vector_index": {"status": "completed", "index_id": "vector_idx_uuid"},
       "summary_index": {"status": "completed", "index_id": "summary_idx_uuid"},
       "graph_index": {"status": "completed", "index_id": "graph_idx_uuid"},
       "property_graph_index": {"status": "completed", "index_id": "prop_graph_idx_uuid"},
       "persist_dir": "./storage/doc_uuid_string"
     },
     "overall_status": "completed"
   }
   ```
   - **MongoDB**: 統合IDで processing_status を "completed" に更新
   - サンプリングで統合ID検索精度確認、重複チェック、ログ出力
   - 全インデックス作成完了でoverall_status を "completed" に更新

7. **統合ComposableGraph作成**
   ```python
   # 概念設計（コードではない）
   composable_graph = ComposableGraph.from_indices(
       vector_index,      # Milvus高速ベクトル検索
       summary_index,     # MongoDB要約情報  
       graph_index,       # Neo4j知識グラフ
       property_graph_index, # Neo4j詳細プロパティ
       index_summaries=[
           "ベクトル類似検索用インデックス",
           "ドキュメント要約検索用インデックス", 
           "知識グラフ関係性検索用インデックス",
           "詳細プロパティグラフ検索用インデックス"
       ]
   )
   # 統合クエリエンジン構築
   unified_query_engine = composable_graph.as_query_engine()
   ```

8. 完了・監査ログ
   - **MongoDB**: query_logs に統合IDでの処理履歴を保存
   - **Redis**: 統合IDでの処理結果をキャッシュ
   - 全DB横断での統合ID一貫性チェック
   - **StorageContext永続化確認**: 全インデックスの永続化状態確認

## RAG（質問応答）フロー（統合StorageContext活用）

1. 入力クエリの正規化
   - correlation_id 生成でリクエスト追跡
   - **Redis**: セッション情報とクエリ履歴を統合IDベースで取得
   - トークン化、言語判定、簡易前処理（不要文字除去）

2. **統合StorageContext読み込み**
   ```python
   # 概念フロー（コードではない）
   # 既存統合インデックス読み込み
   storage_context = StorageContext.from_defaults(persist_dir="./storage")
   vector_index = load_index_from_storage(storage_context, index_id="vector_idx")
   summary_index = load_index_from_storage(storage_context, index_id="summary_idx")  
   graph_index = load_index_from_storage(storage_context, index_id="graph_idx")
   property_graph_index = load_index_from_storage(storage_context, index_id="prop_graph_idx")
   
   # ComposableGraph再構築
   composable_graph = ComposableGraph.from_indices(
       vector_index, summary_index, graph_index, property_graph_index
   )
   unified_query_engine = composable_graph.as_query_engine()
   ```

3. 統合キャッシュ確認
   - **Redis**: 統合IDベースで類似クエリの結果をチェック
   - ヒットした場合は統合IDトレース付きで高速レスポンス

4. **統合インデックスクエリ実行**
   ```python
   # 概念フロー（コードではない）
   # 統合クエリエンジンで全DB横断検索
   response = unified_query_engine.query(
       query_str,
       response_mode="tree_summarize",  # 全インデックス結果統合
       use_async=True,                  # 並列検索実行
       similarity_top_k=10,            # ベクトル検索上位K
       include_graph_context=True,     # グラフ関係性含む
       include_summary_context=True    # 要約情報含む
   )
   ```

5. 統合レスポンス処理
   - **MongoDB**: query_logs に統合IDでの詳細ログを保存
   - **Redis**: 統合IDレスポンスをキャッシュ
   - 出典整形（unified_chunk_id -> doc_unified_id mapping、ページ情報）
   - 全インデックスからの根拠情報統合

# 類似検索フロー（統合StorageContext連携）

1. **統合StorageContext読み込み**: 永続化された全インデックス復元
2. **統合クエリ実行**: ComposableGraphで全DB横断類似検索
3. **結果統合**: ベクトル類似度 + グラフ関係性 + 要約情報の統合スコアリング
4. **Redis統合キャッシュ**: 統合ID結果をキャッシュして返却

# 統合StorageContext管理・検索パラメータ

## StorageContext設計原則
- **統合ID基盤**: 全インデックスで統一された統合ID参照
- **永続化戦略**: 統合IDディレクトリでの階層化永続化
- **増分更新**: 既存StorageContextへの新規ドキュメント追加最適化
- **並列処理**: 複数インデックス作成・検索の並列実行

## インデックス更新戦略
- **増分インデックス**: 新規ドキュメントの既存インデックスへの追加
- **リバランス**: 定期的な全インデックス最適化・再構築  
- **バージョン管理**: StorageContext変更履歴とロールバック機能

# データ一貫性・整合性管理（統合StorageContext基盤）

## StorageContext整合性チェック
- 全インデックス間での統合ID参照整合性確認
- インデックス欠損・不整合の自動検出・修復
- distributed_index_registry とStorageContext状態の同期確認

## 障害時StorageContext復旧
- 部分インデックス障害時の他インデックス情報活用
- 統合IDベースでの段階的インデックス復旧
- StorageContext永続化データからの高速復元

# 運用・監視（統合StorageContext統合）

## StorageContext監視
- 全インデックス横断でのクエリ性能監視
- インデックスサイズ・更新頻度の統合監視
- ComposableGraph応答時間・精度の継続監視

## 統合メトリクス
- StorageContextベースの統合検索精度測定
- 全DB協調でのレスポンス品質指標
- インデックス最適化効果の定量評価

# 結論（統合StorageContext設計のポイント）
- **統合インデックス管理**: StorageContextで全DB統合インデックス一元化
- **ComposableGraph活用**: 複数インデックス横断での高精度検索実現
- **永続化戦略**: 統合IDベース階層化でのインデックス永続化
- **並列処理最適化**: 複数インデックス作成・検索の効率的並列実行
- **整合性保証**: StorageContext + distributed_index_registry での二重整合性管理
- **運用効率向上**: 統合インデックス監視でのシステム全体最適化
